//! Voice activity detection: the `Vad` trait and two detectors.
//!
//! - `EarshotVad` (feature `earshot`, on by default) wraps the `earshot`
//!   crate: a small neural detector in pure Rust with no runtime
//!   dependencies, judging 16 ms frames. The better choice wherever the
//!   input is real speech.
//! - `EnergyVad` is an adaptive noise-floor energy detector with no
//!   dependencies at all. It does well on the echo-cancelled,
//!   noise-suppressed signal a browser or phone delivers, and is what the
//!   tests drive with synthetic tones (which no speech model calls speech).

use crate::audio::frame_dbfs;

/// A per-frame speech decision on 16 kHz mono PCM16.
pub trait Vad: Send {
    /// Samples in the frames this detector judges. The turn detector cuts
    /// the incoming audio to this length.
    fn frame_len(&self) -> usize {
        320
    }
    fn is_speech(&mut self, frame: &[i16]) -> bool;
    /// The call's own reply is playing. A detector should demand more of a
    /// frame then: whatever the echo canceller leaves of the reply is the
    /// likeliest false trigger there is.
    fn set_playback(&mut self, playing: bool);
}

/// Tuning for `EnergyVad`.
#[derive(Debug, Clone, Copy)]
pub struct EnergyVadConfig {
    /// How far above the noise floor a frame must be to count as speech.
    pub margin_db: f32,
    /// A frame quieter than this is never speech, whatever the floor says —
    /// in a silent room the floor sinks and the margin alone would hear
    /// breathing.
    pub min_speech_dbfs: f32,
    /// Added to both while the reply plays.
    pub playback_extra_db: f32,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            margin_db: 12.0,
            min_speech_dbfs: -50.0,
            playback_extra_db: 8.0,
        }
    }
}

/// An adaptive noise-floor energy detector.
///
/// The first `CALIBRATION_FRAMES` set the floor to the quietest level heard
/// (a call opens on a beat of room sound, and a floor guessed rather than
/// measured would hear a loud fan as a voice forever). After that the floor
/// follows non-speech frames — quickly downward, so a noise that stops stops
/// counting at once, and slowly upward — and creeps up very slowly even
/// through "speech", so a noise that starts mid-call is eventually learned
/// rather than heard as one endless sentence.
pub struct EnergyVad {
    config: EnergyVadConfig,
    floor_db: f32,
    playing: bool,
    calibration_left: u32,
}

const CALIBRATION_FRAMES: u32 = 10;

const FLOOR_MIN_DB: f32 = -90.0;
const FLOOR_MAX_DB: f32 = -25.0;

impl EnergyVad {
    pub fn new(config: EnergyVadConfig) -> Self {
        Self {
            config,
            floor_db: FLOOR_MAX_DB,
            playing: false,
            calibration_left: CALIBRATION_FRAMES,
        }
    }

    pub fn floor_db(&self) -> f32 {
        self.floor_db
    }
}

impl Vad for EnergyVad {
    fn is_speech(&mut self, frame: &[i16]) -> bool {
        let level = frame_dbfs(frame);
        if self.calibration_left > 0 {
            self.calibration_left -= 1;
            self.floor_db = self.floor_db.min(level).clamp(FLOOR_MIN_DB, FLOOR_MAX_DB);
            return false;
        }
        let extra = if self.playing {
            self.config.playback_extra_db
        } else {
            0.0
        };
        let threshold = (self.floor_db + self.config.margin_db + extra)
            .max(self.config.min_speech_dbfs + extra);
        let speech = level > threshold;
        let rate = match (speech, level < self.floor_db) {
            (_, true) => 0.3,
            (false, false) => 0.02,
            (true, false) => 0.002,
        };
        self.floor_db += rate * (level - self.floor_db);
        self.floor_db = self.floor_db.clamp(FLOOR_MIN_DB, FLOOR_MAX_DB);
        speech
    }

    fn set_playback(&mut self, playing: bool) {
        self.playing = playing;
    }
}

/// `earshot`'s neural detector: score ≥ `threshold` is speech. While the
/// reply plays the threshold is raised by `playback_extra`, the same echo
/// guard `EnergyVad` applies in decibels.
///
/// The detector judges a *dithered* copy of each frame (±`DITHER` LSB of
/// white noise, about -60 dBFS; the audio sent on for transcription is
/// untouched). Fed exact digital silence — which a browser's noise
/// suppression emits between words — earshot's internal level estimate
/// collapses, and afterwards it scores everything, silence included, as
/// voice (measured: 0.83–0.85 against its 0.5 threshold). A floor of
/// inaudible noise keeps it calibrated.
#[cfg(feature = "earshot")]
pub struct EarshotVad {
    detector: Box<earshot::Detector>,
    dither_state: u32,
    threshold: f32,
    playback_extra: f32,
    playing: bool,
}

#[cfg(feature = "earshot")]
impl EarshotVad {
    /// `earshot` judges frames of exactly this many samples (16 ms).
    pub const FRAME_LEN: usize = 256;
    /// Peak dither added to the detector's copy of a frame, in LSB.
    pub const DITHER: i16 = 32;

    pub fn new(threshold: f32, playback_extra: f32) -> Self {
        Self {
            detector: earshot::Detector::default_boxed(),
            dither_state: 0x9e37_79b9,
            threshold,
            playback_extra,
            playing: false,
        }
    }

    /// The frame's speech score in 0..=1.
    pub fn score(&mut self, frame: &[i16]) -> f32 {
        let mut x = self.dither_state;
        let span = 2 * Self::DITHER as u32 + 1;
        let dithered: Vec<i16> = frame
            .iter()
            .map(|sample| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                sample.saturating_add((x % span) as i16 - Self::DITHER)
            })
            .collect();
        self.dither_state = x;
        self.detector.predict_i16(&dithered)
    }
}

#[cfg(feature = "earshot")]
impl Default for EarshotVad {
    fn default() -> Self {
        Self::new(0.5, 0.2)
    }
}

#[cfg(feature = "earshot")]
impl Vad for EarshotVad {
    fn frame_len(&self) -> usize {
        Self::FRAME_LEN
    }

    fn is_speech(&mut self, frame: &[i16]) -> bool {
        let threshold = if self.playing {
            (self.threshold + self.playback_extra).min(0.99)
        } else {
            self.threshold
        };
        self.score(frame) >= threshold
    }

    fn set_playback(&mut self, playing: bool) {
        self.playing = playing;
    }
}

#[cfg(all(test, feature = "earshot"))]
mod tests {
    use super::*;

    #[test]
    fn earshot_stays_calibrated_through_digital_silence_then_noise() {
        let mut vad = EarshotVad::default();
        assert_eq!(vad.frame_len(), 256);
        let silence = vec![0i16; 256];
        for _ in 0..30 {
            assert!(!vad.is_speech(&silence));
        }
        let mut x: u32 = 0x9e37_79b9;
        let mut noise = || -> Vec<i16> {
            (0..256)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x % 2001) as i16 - 1000
                })
                .collect()
        };
        // earshot adapts to a new noise floor over roughly 300 ms; a sudden
        // noise onset can read as voice until it has (the turn detector's
        // minimum speech length discards that blip).
        for _ in 0..30 {
            vad.is_speech(&noise());
        }
        let heard = (0..60).filter(|_| vad.is_speech(&noise())).count();
        assert!(heard < 6, "steady white noise is not a voice ({heard}/60)");
    }
}
