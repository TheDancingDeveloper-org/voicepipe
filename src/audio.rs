//! PCM16 framing and WAV encoding for 16 kHz mono call audio.

/// The sample rate the pipeline works at: client audio arrives at it, and
/// every `Vad` judges frames at it.
pub const SAMPLE_RATE: u32 = 16_000;

/// Little-endian PCM16 bytes to samples. A trailing odd byte — half a sample —
/// is dropped rather than guessed at.
pub fn pcm16_from_le_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect()
}

/// A mono PCM16 WAV file of `samples` — the container every
/// OpenAI-compatible transcription backend accepts.
pub fn wav_from_pcm16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Frame level in dBFS (0 = full scale; digital silence is clamped to -100).
pub fn frame_dbfs(frame: &[i16]) -> f32 {
    if frame.is_empty() {
        return -100.0;
    }
    let sum: f64 = frame.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    let rms = (sum / frame.len() as f64).sqrt();
    if rms < 1.0 {
        return -100.0;
    }
    (20.0 * (rms / 32768.0).log10()) as f32
}

/// How long a WAV clip plays, read from its header: the `fmt ` chunk's byte
/// rate over the `data` chunk's length (or, for a streaming header that
/// leaves the length unset, everything after it). `None` for anything that
/// is not a PCM WAV.
pub fn wav_duration(bytes: &[u8]) -> Option<std::time::Duration> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12;
    let mut byte_rate: Option<u32> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().ok()?) as usize;
        let body = at + 8;
        if id == b"fmt " && body + 12 <= bytes.len() {
            byte_rate = Some(u32::from_le_bytes(
                bytes[body + 8..body + 12].try_into().ok()?,
            ));
        } else if id == b"data" {
            let available = bytes.len() - body;
            let data = if len == 0 || len > available {
                available
            } else {
                len
            };
            let rate = byte_rate.filter(|r| *r > 0)?;
            return Some(std::time::Duration::from_secs_f64(
                data as f64 / rate as f64,
            ));
        }
        at = body.checked_add(len + (len & 1))?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wav_header_describes_mono_pcm16_at_the_given_rate() {
        let wav = wav_from_pcm16(&[0, 1, -1], 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(wav.len(), 44 + 6);
        assert_eq!(
            pcm16_from_le_bytes(&wav[44..]),
            vec![0, 1, -1],
            "the data chunk round-trips"
        );
    }

    #[test]
    fn half_a_sample_is_dropped_not_guessed() {
        assert_eq!(pcm16_from_le_bytes(&[1, 0, 7]), vec![1]);
    }

    #[test]
    fn a_wav_says_how_long_it_plays_and_other_audio_does_not() {
        let wav = wav_from_pcm16(&vec![0; 16_000], 16_000);
        assert_eq!(wav_duration(&wav), Some(std::time::Duration::from_secs(1)));
        assert_eq!(wav_duration(b"ID3\x03not a wav at all"), None);
    }
}
