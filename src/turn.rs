//! Turn-taking: per-frame speech decisions in, conversation events out.
//!
//! The `Endpointer` (the shipped `TurnDetector`) reports:
//!
//! - `SpeechStarted` after `onset_ms` of voice (a click or a cough is shorter);
//! - `Sustained` once the utterance has `sustained_ms` of voice — the bar a
//!   barge-in must clear, so a backchannel "mm" does not stop the reply;
//! - `PauseBegan` after `pause_ms` of silence — early enough to start
//!   transcribing while the turn may still be going on;
//! - `SpeechResumed` if voice returns before the turn ends;
//! - `EndOfTurn` after `end_of_turn_ms` of silence, carrying the utterance
//!   (with `pre_roll_ms` of audio from before the onset, so the first
//!   syllable is not clipped), or `Discarded` if it held less than
//!   `min_speech_ms` of voice.
//!
//! Pure — samples in, events out — so the rules are tested on synthetic
//! audio without a microphone, a socket or a speech backend.

use crate::audio::SAMPLE_RATE;
use crate::vad::Vad;

/// Turns audio into turn-taking events. The pipeline drives one per call.
pub trait TurnDetector: Send {
    /// Feed audio of any length; returns the events it caused, in order.
    fn push(&mut self, samples: &[i16]) -> Vec<EndpointEvent>;
    /// The reply started or stopped playing (the echo guard).
    fn set_playback(&mut self, playing: bool);
    /// True between `SpeechStarted` and the turn's end.
    fn in_turn(&self) -> bool;
    /// The utterance so far, for an early or partial transcription.
    fn segment(&self) -> &[i16];
    /// Forget the current utterance.
    fn reset(&mut self);
}

/// Turn-taking timings, in milliseconds of audio.
#[derive(Debug, Clone, Copy)]
pub struct EndpointConfig {
    /// Voice needed to start a turn; a click or a cough is shorter.
    pub onset_ms: u32,
    /// Voice in the utterance before `Sustained` is reported: the bar a
    /// barge-in must clear.
    pub sustained_ms: u32,
    /// Silence that counts as a pause. Keep it below `end_of_turn_ms`, or
    /// the turn ends before a pause is ever reported.
    pub pause_ms: u32,
    /// Silence that ends the turn.
    pub end_of_turn_ms: u32,
    /// Voice an utterance needs to be a turn rather than `Discarded`.
    pub min_speech_ms: u32,
    /// The longest turn; at this length it ends whatever the speaker does.
    pub max_turn_ms: u32,
    /// Audio from before the onset kept at the start of the utterance, so
    /// the first syllable is not clipped.
    pub pre_roll_ms: u32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            onset_ms: 100,
            sustained_ms: 500,
            pause_ms: 200,
            end_of_turn_ms: 700,
            min_speech_ms: 250,
            max_turn_ms: 30_000,
            pre_roll_ms: 300,
        }
    }
}

/// What the turn detector reports. See the module docs for when.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EndpointEvent {
    /// A turn began.
    SpeechStarted,
    /// The utterance has held `sustained_ms` of voice.
    Sustained,
    /// The speaker paused for `pause_ms`.
    PauseBegan,
    /// Voice came back before the turn ended.
    SpeechResumed,
    /// The turn is over.
    EndOfTurn {
        /// The utterance, pre-roll included.
        audio: Vec<i16>,
        /// Milliseconds of voiced audio in the utterance.
        speech_ms: u32,
    },
    /// Speech started but held too little voice to be a turn.
    Discarded,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Idle,
    Speaking,
    Paused,
}

/// Frames incoming audio, asks the `Vad` about each frame, and reports the
/// turn-taking events above.
pub struct Endpointer<V: Vad> {
    config: EndpointConfig,
    vad: V,
    frame_len: usize,
    frame_ms: u32,
    /// Samples waiting to make up a whole frame.
    partial: Vec<i16>,
    /// Recent frames kept while idle, so an utterance starts `pre_roll_ms`
    /// before its onset was confirmed.
    pre_roll: std::collections::VecDeque<Vec<i16>>,
    /// Leaky count of voiced frames while idle: up on voice, down on silence.
    onset_frames: u32,
    phase: Phase,
    segment: Vec<i16>,
    voiced_frames: u32,
    silent_frames: u32,
    sustained_sent: bool,
}

impl<V: Vad> Endpointer<V> {
    /// A detector with these timings, asking `vad` about each frame.
    pub fn new(config: EndpointConfig, vad: V) -> Self {
        let frame_len = vad.frame_len().max(1);
        let frame_ms = (frame_len as u32 * 1000 / SAMPLE_RATE).max(1);
        Self {
            config,
            vad,
            frame_len,
            frame_ms,
            partial: Vec::new(),
            pre_roll: std::collections::VecDeque::new(),
            onset_frames: 0,
            phase: Phase::Idle,
            segment: Vec::new(),
            voiced_frames: 0,
            silent_frames: 0,
            sustained_sent: false,
        }
    }

    fn frames(&self, ms: u32) -> u32 {
        (ms / self.frame_ms).max(1)
    }

    /// The call's reply started or stopped playing.
    pub fn set_playback(&mut self, playing: bool) {
        self.vad.set_playback(playing);
    }

    /// True between `SpeechStarted` and the turn's end.
    pub fn in_turn(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// The utterance so far, for an early or partial transcription.
    pub fn segment(&self) -> &[i16] {
        &self.segment
    }

    /// Milliseconds of voice in the current utterance.
    pub fn speech_ms(&self) -> u32 {
        self.voiced_frames * self.frame_ms
    }

    /// Forget the current utterance and any partial frame — after the call
    /// has decided what to do with it some other way.
    pub fn reset(&mut self) {
        self.partial.clear();
        self.pre_roll.clear();
        self.onset_frames = 0;
        self.phase = Phase::Idle;
        self.segment.clear();
        self.voiced_frames = 0;
        self.silent_frames = 0;
        self.sustained_sent = false;
    }

    /// Feed audio of any length; returns the events it caused, in order.
    pub fn push(&mut self, samples: &[i16]) -> Vec<EndpointEvent> {
        let mut events = Vec::new();
        self.partial.extend_from_slice(samples);
        while self.partial.len() >= self.frame_len {
            let frame: Vec<i16> = self.partial.drain(..self.frame_len).collect();
            self.frame(frame, &mut events);
        }
        events
    }

    fn frame(&mut self, frame: Vec<i16>, events: &mut Vec<EndpointEvent>) {
        let speech = self.vad.is_speech(&frame);
        match self.phase {
            Phase::Idle => {
                self.onset_frames = if speech {
                    self.onset_frames + 1
                } else {
                    self.onset_frames.saturating_sub(1)
                };
                self.pre_roll.push_back(frame);
                let keep = self.frames(self.config.pre_roll_ms) as usize;
                while self.pre_roll.len() > keep {
                    self.pre_roll.pop_front();
                }
                if self.onset_frames >= self.frames(self.config.onset_ms) {
                    self.phase = Phase::Speaking;
                    self.segment = self.pre_roll.drain(..).flatten().collect();
                    self.voiced_frames = self.onset_frames;
                    self.silent_frames = 0;
                    self.onset_frames = 0;
                    events.push(EndpointEvent::SpeechStarted);
                }
            }
            Phase::Speaking | Phase::Paused => {
                self.segment.extend_from_slice(&frame);
                if speech {
                    self.voiced_frames += 1;
                    self.silent_frames = 0;
                    if self.phase == Phase::Paused {
                        self.phase = Phase::Speaking;
                        events.push(EndpointEvent::SpeechResumed);
                    }
                } else {
                    self.silent_frames += 1;
                    if self.phase == Phase::Speaking
                        && self.silent_frames >= self.frames(self.config.pause_ms)
                    {
                        self.phase = Phase::Paused;
                        events.push(EndpointEvent::PauseBegan);
                    }
                }
                if !self.sustained_sent && self.speech_ms() >= self.config.sustained_ms {
                    self.sustained_sent = true;
                    events.push(EndpointEvent::Sustained);
                }
                let turn_ms = self.segment.len() as u32 * 1000 / SAMPLE_RATE;
                let ended = self.silent_frames >= self.frames(self.config.end_of_turn_ms)
                    || turn_ms >= self.config.max_turn_ms;
                if ended {
                    let speech_ms = self.speech_ms();
                    let audio = std::mem::take(&mut self.segment);
                    self.reset();
                    events.push(if speech_ms >= self.config.min_speech_ms {
                        EndpointEvent::EndOfTurn { audio, speech_ms }
                    } else {
                        EndpointEvent::Discarded
                    });
                }
            }
        }
    }
}

impl<V: Vad> TurnDetector for Endpointer<V> {
    fn push(&mut self, samples: &[i16]) -> Vec<EndpointEvent> {
        Endpointer::push(self, samples)
    }
    fn set_playback(&mut self, playing: bool) {
        Endpointer::set_playback(self, playing)
    }
    fn in_turn(&self) -> bool {
        Endpointer::in_turn(self)
    }
    fn segment(&self) -> &[i16] {
        Endpointer::segment(self)
    }
    fn reset(&mut self) {
        Endpointer::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::{EnergyVad, EnergyVadConfig};

    const FRAME: usize = 320; // 20 ms at 16 kHz

    /// `ms` of a 220 Hz tone at `amplitude` — a stand-in for voice.
    fn tone(ms: u32, amplitude: f32) -> Vec<i16> {
        let n = (SAMPLE_RATE * ms / 1000) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (amplitude * 32767.0 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()) as i16
            })
            .collect()
    }

    /// `ms` of low pseudo-random noise — a quiet room.
    fn noise(ms: u32, amplitude: i16) -> Vec<i16> {
        let n = (SAMPLE_RATE * ms / 1000) as usize;
        let mut x: u32 = 0x1234_5678;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x % (2 * amplitude as u32 + 1)) as i32 - amplitude as i32) as i16
            })
            .collect()
    }

    fn endpointer() -> Endpointer<EnergyVad> {
        Endpointer::new(
            EndpointConfig::default(),
            EnergyVad::new(EnergyVadConfig::default()),
        )
    }

    fn feed(ep: &mut Endpointer<EnergyVad>, audio: &[i16]) -> Vec<EndpointEvent> {
        // In 20 ms pieces, as the client sends them.
        audio
            .chunks(FRAME)
            .flat_map(|chunk| ep.push(chunk))
            .collect()
    }

    fn kinds(events: &[EndpointEvent]) -> Vec<&'static str> {
        events
            .iter()
            .map(|e| match e {
                EndpointEvent::SpeechStarted => "started",
                EndpointEvent::Sustained => "sustained",
                EndpointEvent::PauseBegan => "pause",
                EndpointEvent::SpeechResumed => "resumed",
                EndpointEvent::EndOfTurn { .. } => "end",
                EndpointEvent::Discarded => "discarded",
            })
            .collect()
    }

    #[test]
    fn one_utterance_in_a_quiet_room_is_one_turn_with_its_lead_in() {
        let mut ep = endpointer();
        let mut audio = noise(500, 30);
        audio.extend(tone(800, 0.3));
        audio.extend(noise(900, 30));
        let events = feed(&mut ep, &audio);
        assert_eq!(kinds(&events), vec!["started", "sustained", "pause", "end"]);
        let EndpointEvent::EndOfTurn { audio, speech_ms } = events.last().unwrap() else {
            unreachable!()
        };
        assert!((700..=900).contains(speech_ms), "speech_ms {speech_ms}");
        // Pre-roll + speech + the silence that ended it.
        let ms = audio.len() as u32 * 1000 / SAMPLE_RATE;
        assert!(
            ms >= 800 + 700,
            "the utterance keeps its lead-in, got {ms} ms"
        );
    }

    #[test]
    fn a_short_pause_inside_a_sentence_does_not_end_the_turn() {
        let mut ep = endpointer();
        let mut audio = noise(300, 30);
        audio.extend(tone(600, 0.3));
        audio.extend(noise(400, 30)); // a breath, shorter than end_of_turn
        audio.extend(tone(600, 0.3));
        audio.extend(noise(900, 30));
        let events = feed(&mut ep, &audio);
        assert_eq!(
            kinds(&events),
            vec!["started", "sustained", "pause", "resumed", "pause", "end"]
        );
    }

    #[test]
    fn a_click_is_not_speech_and_a_short_sound_is_discarded() {
        let mut ep = endpointer();
        let mut audio = noise(300, 30);
        audio.extend(tone(40, 0.5)); // under onset_ms
        audio.extend(noise(900, 30));
        assert!(feed(&mut ep, &audio).is_empty());

        let mut audio = tone(160, 0.5); // over onset, under min_speech
        audio.extend(noise(900, 30));
        assert_eq!(
            kinds(&feed(&mut ep, &audio)),
            vec!["started", "pause", "discarded"]
        );
    }

    #[test]
    fn a_steady_noise_becomes_the_floor_and_speech_above_it_still_counts() {
        let mut ep = endpointer();
        // A loud fan: well above the default floor, but steady.
        let fan = noise(3000, 600);
        let events = feed(&mut ep, &fan);
        assert!(
            events.iter().all(|e| *e != EndpointEvent::Sustained),
            "a steady noise must not read as a sustained voice: {events:?}"
        );
        ep.reset();
        let mut audio = noise(500, 600);
        audio.extend(tone(800, 0.4));
        audio.extend(noise(900, 600));
        let kinds = kinds(&feed(&mut ep, &audio));
        assert!(kinds.contains(&"end"), "{kinds:?}");
    }

    #[test]
    fn quiet_echo_that_is_speech_while_idle_is_not_speech_while_the_reply_plays() {
        let quiet_voice = 0.009; // about -44 dBFS
        let mut idle = endpointer();
        let mut audio = noise(300, 10);
        audio.extend(tone(700, quiet_voice));
        audio.extend(noise(900, 10));
        assert!(kinds(&feed(&mut idle, &audio)).contains(&"started"));

        let mut playing = endpointer();
        playing.set_playback(true);
        assert!(feed(&mut playing, &audio).is_empty());
    }

    #[test]
    fn a_turn_that_never_pauses_is_cut_at_the_maximum() {
        let mut ep = Endpointer::new(
            EndpointConfig {
                max_turn_ms: 2_000,
                ..EndpointConfig::default()
            },
            EnergyVad::new(EnergyVadConfig::default()),
        );
        let mut audio = noise(300, 30);
        audio.extend(tone(3_000, 0.3));
        let events = feed(&mut ep, &audio);
        assert!(kinds(&events).contains(&"end"));
    }

    #[test]
    fn the_segment_so_far_is_readable_mid_turn() {
        let mut ep = endpointer();
        feed(&mut ep, &noise(200, 30));
        feed(&mut ep, &tone(400, 0.3));
        assert!(ep.in_turn());
        assert!(ep.segment().len() >= (SAMPLE_RATE as usize * 400 / 1000));
        ep.reset();
        assert!(!ep.in_turn());
        assert!(ep.segment().is_empty());
    }
}
