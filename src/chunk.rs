//! Streaming transcription by chunks, for transcribers that only take whole
//! clips.
//!
//! Whisper-family servers (speaches, faster-whisper, whisper.cpp, the hosted
//! `/audio/transcriptions` APIs) decode a finished clip; none decodes audio
//! incrementally as it arrives. So a turn is streamed to them as a run of
//! *chunks*: each time the speaker pauses, the audio since the last cut is
//! sent off to be transcribed while they carry on talking, and when the turn
//! ends only what follows the last cut — usually nothing — is still to do.
//! Every sample is decoded once, a chunk ends in the speaker's own pause (a
//! word boundary), and the transcript is ready about when the turn is
//! declared over instead of starting then.
//!
//! [`Chunker`] decides the cuts and nothing else: segment in, sample ranges
//! out. The pipeline sends the ranges to the transcriber, one at a time and
//! in order, each with the words before it as context.

use std::ops::Range;

use crate::audio::{frame_dbfs, SAMPLE_RATE};

/// Where turns are cut into chunks, in milliseconds of audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    /// A pause does not cut a chunk shorter than this; it waits to join the
    /// next. A transcriber given a sliver of a word tends to invent one.
    pub min_ms: u32,
    /// Speech that runs this long without a pause is cut anyway, at its
    /// quietest moment, so a long unbroken stretch is not left to the end.
    pub max_ms: u32,
    /// How far back from the newest audio a forced cut looks for that
    /// quietest moment.
    pub search_ms: u32,
    /// Silence kept at the end of the turn's last chunk. The rest of the
    /// end-of-turn silence is dropped: whisper models given a long tail of
    /// nothing are prone to filling it with repeated words.
    pub tail_silence_ms: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            min_ms: 1_000,
            max_ms: 6_000,
            search_ms: 1_500,
            tail_silence_ms: 200,
        }
    }
}

/// 20 ms at the pipeline's rate: the unit cuts are judged in.
const FRAME: usize = (SAMPLE_RATE / 50) as usize;
/// Below the turn's loudest frame by this much counts as quiet when
/// trimming the end of a turn.
const QUIET_BELOW_PEAK_DB: f32 = 35.0;

fn samples(ms: u32) -> usize {
    (SAMPLE_RATE / 1000 * ms) as usize
}

/// Cuts one turn into chunks. Driven by the turn detector's events: speech
/// (`voiced`), a pause (`at_pause`), audio arriving (`overlong`), the end of
/// the turn (`finish`).
#[derive(Debug, Clone)]
pub struct Chunker {
    config: ChunkConfig,
    /// Samples of the segment already handed out.
    committed: usize,
    /// The audio after `committed` holds speech.
    voiced: bool,
}

impl Chunker {
    pub fn new(config: ChunkConfig) -> Self {
        Self {
            config,
            committed: 0,
            voiced: true,
        }
    }

    /// The speaker is talking (again).
    pub fn voiced(&mut self) {
        self.voiced = true;
    }

    /// Samples of the turn handed out so far.
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// A pause began: the chunk to transcribe now, if what has built up
    /// since the last cut is long enough and held speech.
    pub fn at_pause(&mut self, segment: &[i16]) -> Option<Range<usize>> {
        let end = segment.len();
        if !self.voiced || end < self.committed + samples(self.config.min_ms) {
            return None;
        }
        self.voiced = false;
        Some(self.cut(end))
    }

    /// Audio arrived mid-speech: a forced cut, at the quietest frame of the
    /// last `search_ms`, once the uncut audio has run past `max_ms`.
    pub fn overlong(&mut self, segment: &[i16]) -> Option<Range<usize>> {
        let end = segment.len();
        if !self.voiced || end < self.committed + samples(self.config.max_ms) {
            return None;
        }
        let from = end
            .saturating_sub(samples(self.config.search_ms))
            .max(self.committed + samples(self.config.min_ms));
        let at = quietest_frame_end(segment, from, end);
        Some(self.cut(at))
    }

    /// The turn is over: what is left to transcribe, if it held speech, with
    /// the end-of-turn silence trimmed to `tail_silence_ms`.
    pub fn finish(&mut self, segment: &[i16]) -> Option<Range<usize>> {
        if !self.voiced || segment.len() <= self.committed {
            return None;
        }
        let rest = &segment[self.committed..];
        let end = self.committed + trim_trailing_quiet(rest, self.config.tail_silence_ms);
        self.voiced = false;
        (end > self.committed).then(|| self.cut(end))
    }

    fn cut(&mut self, at: usize) -> Range<usize> {
        let range = self.committed..at;
        self.committed = at;
        range
    }
}

/// The end of the quietest whole frame of `segment[from..to]` (frames
/// counted from `from`), or `to` when there is not a whole frame.
fn quietest_frame_end(segment: &[i16], from: usize, to: usize) -> usize {
    let mut best: Option<(f32, usize)> = None;
    let mut at = from;
    while at + FRAME <= to {
        let level = frame_dbfs(&segment[at..at + FRAME]);
        if best.is_none_or(|(quietest, _)| level < quietest) {
            best = Some((level, at + FRAME));
        }
        at += FRAME;
    }
    best.map_or(to, |(_, end)| end)
}

/// How much of `audio` to keep: up to its last frame that is not quiet
/// (relative to its loudest), plus `keep_ms` of what follows.
fn trim_trailing_quiet(audio: &[i16], keep_ms: u32) -> usize {
    let frames: Vec<f32> = audio.chunks(FRAME).map(frame_dbfs).collect();
    let peak = frames.iter().copied().fold(-100.0f32, f32::max);
    let Some(last) = frames
        .iter()
        .rposition(|level| *level > peak - QUIET_BELOW_PEAK_DB && *level > -90.0)
    else {
        return 0;
    };
    ((last + 1) * FRAME + samples(keep_ms)).min(audio.len())
}

/// The last `max_chars` or so of `text`, starting at a word: the context a
/// chunk is transcribed with. Whisper reads it as the text before the clip.
pub fn context_tail(text: &str, max_chars: usize) -> &str {
    let text = text.trim();
    if text.len() <= max_chars {
        return text;
    }
    let mut start = text.len() - max_chars;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    if text[..start].ends_with(' ') {
        return &text[start..];
    }
    match text[start..].find(' ') {
        Some(space) => text[start + space..].trim_start(),
        None => &text[start..],
    }
}

/// Whether `text` is more words than `seconds` of speech can hold, or
/// repeats itself in a run: the two signs of a whisper decode that looped
/// (it does, on a short clip given a prompt, or on a long silence) instead of
/// transcribing.
pub fn implausible(text: &str, seconds: f32) -> bool {
    let words = text.split_whitespace().count() as f32;
    words > MAX_WORDS_PER_SECOND * seconds + 4.0
        || collapse_repeats(text) != text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Brisk speech is about four words a second; past this a transcript is
/// not of the clip.
const MAX_WORDS_PER_SECOND: f32 = 6.0;
/// The longest phrase looked for repeating, in words.
const MAX_REPEAT_WORDS: usize = 12;
/// A phrase said this many times running is a loop, not speech.
const LOOP_RUN: usize = 3;

/// `text` with any phrase repeated `LOOP_RUN` or more times running cut back
/// to one saying of it ("passed passed passed passed" → "passed"). Words are
/// compared without case or punctuation.
pub fn collapse_repeats(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let key = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let keys: Vec<String> = words.iter().map(|w| key(w)).collect();
    let mut kept: Vec<&str> = Vec::with_capacity(words.len());
    let mut at = 0;
    'scan: while at < words.len() {
        for n in 1..=MAX_REPEAT_WORDS.min((words.len() - at) / LOOP_RUN) {
            let unit = &keys[at..at + n];
            let mut runs = 1;
            while at + (runs + 1) * n <= words.len()
                && keys[at + runs * n..at + (runs + 1) * n] == *unit
            {
                runs += 1;
            }
            if runs >= LOOP_RUN {
                kept.extend_from_slice(&words[at..at + n]);
                at += runs * n;
                continue 'scan;
            }
        }
        kept.push(words[at]);
        at += 1;
    }
    kept.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u32) -> usize {
        samples(n)
    }

    fn voice(ms_: u32) -> Vec<i16> {
        (0..ms(ms_))
            .map(|i| ((i as f32 * 0.09).sin() * 8_000.0) as i16)
            .collect()
    }

    fn hush(ms_: u32) -> Vec<i16> {
        vec![0; ms(ms_)]
    }

    fn config() -> ChunkConfig {
        ChunkConfig::default()
    }

    #[test]
    fn a_pause_cuts_what_was_said_since_the_last_cut() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(1_500);
        segment.extend(hush(200));
        assert_eq!(chunker.at_pause(&segment), Some(0..segment.len()));
        // Talking on, and pausing again, cuts only the new part.
        chunker.voiced();
        let first = segment.len();
        segment.extend(voice(1_200));
        segment.extend(hush(200));
        assert_eq!(chunker.at_pause(&segment), Some(first..segment.len()));
    }

    #[test]
    fn a_pause_after_too_little_waits_to_join_the_next_chunk() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(400);
        segment.extend(hush(200));
        assert_eq!(chunker.at_pause(&segment), None, "too short to cut");
        chunker.voiced();
        segment.extend(voice(900));
        segment.extend(hush(200));
        assert_eq!(chunker.at_pause(&segment), Some(0..segment.len()));
    }

    #[test]
    fn a_second_pause_with_nothing_said_between_cuts_nothing() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(1_500);
        segment.extend(hush(200));
        assert!(chunker.at_pause(&segment).is_some());
        segment.extend(hush(1_200));
        assert_eq!(chunker.at_pause(&segment), None);
        assert_eq!(chunker.finish(&segment), None, "nothing left at the end");
    }

    #[test]
    fn long_unbroken_speech_is_cut_at_its_quietest_moment() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(5_000);
        // A dip that is not a pause: quieter, still voice-like.
        let dip_start = segment.len();
        segment.extend(voice(60).iter().map(|s| s / 20));
        segment.extend(voice(940));
        assert_eq!(chunker.overlong(&segment[..ms(5_900)]), None);
        let cut = chunker.overlong(&segment).expect("past max_ms");
        assert_eq!(cut.start, 0);
        assert!(
            cut.end > dip_start && cut.end <= dip_start + ms(60),
            "cut at the dip: {cut:?}, dip at {dip_start}"
        );
        assert_eq!(chunker.committed(), cut.end);
    }

    #[test]
    fn the_end_of_a_turn_hands_out_the_rest_without_its_long_silence() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(1_500);
        segment.extend(hush(200));
        let first = chunker.at_pause(&segment).unwrap();
        chunker.voiced();
        let said_end = segment.len() + ms(600);
        segment.extend(voice(600));
        segment.extend(hush(700));
        let tail = chunker.finish(&segment).expect("a voiced tail");
        assert_eq!(tail.start, first.end);
        assert_eq!(tail.end, said_end + ms(200), "200 ms of the silence kept");
    }

    #[test]
    fn a_turn_with_no_pause_is_one_chunk_at_the_end() {
        let mut chunker = Chunker::new(config());
        let mut segment = voice(900);
        segment.extend(hush(700));
        let tail = chunker.finish(&segment).expect("the whole turn");
        assert_eq!(tail, 0..ms(1_100));
    }

    #[test]
    fn a_looping_decode_is_caught_and_cut_back() {
        assert_eq!(
            collapse_repeats("tell me whether the tests passed passed passed passed"),
            "tell me whether the tests passed"
        );
        assert_eq!(
            collapse_repeats(
                "Check the build. You can check the build, you can check the build, \
                 you can check the build, you can check the build"
            ),
            "Check the build. You can check the build"
        );
        // Said twice is speech.
        assert_eq!(collapse_repeats("no no I meant that"), "no no I meant that");
        assert!(implausible("passed passed passed passed", 2.0));
        assert!(implausible(&"word ".repeat(40), 2.0), "40 words in 2 s");
        assert!(!implausible(
            "Can you check the build on the dev stack?",
            2.5
        ));
    }

    #[test]
    fn context_is_the_end_of_the_text_from_a_word() {
        assert_eq!(context_tail("  short  ", 50), "short");
        assert_eq!(context_tail("one two three four", 9), "four");
        assert_eq!(context_tail("one two three four", 10), "three four");
        assert_eq!(context_tail("ünïcödé wörds hère", 7), "hère");
    }
}
