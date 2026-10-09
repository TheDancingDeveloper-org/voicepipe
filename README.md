# voicepipe

A pipeline for live, turn-taking voice calls with a language model, in Rust.

Audio streams in. voicepipe decides when the speaker's turn is over,
transcribes it, runs the host's turn, and speaks the reply back a sentence at
a time while it is still being written. It stops the reply the moment the
speaker talks over it, and keeps only what was heard.

Speech-to-text, text-to-speech and the model are traits the host implements,
so the same pipeline runs over local models or hosted APIs. voicepipe owns
what is common to every voice agent: turn-taking, barge-in, ordering and
timing.

## What the host provides

| Trait | The host's part |
|---|---|
| `Stt` | One clip (a 16 kHz mono PCM16 WAV) to its words. `transcribe_after` takes one chunk of a turn still being spoken, plus the words before it, for a transcriber that accepts a prompt. |
| `Tts` | One piece of the reply to one audio clip, in any container. |
| `Llm` | The turn: a `TurnRequest` in, text deltas and tool rounds out through a sink, and an outcome (`Completed`, `Interrupted` or `AwaitingApproval`). It honours a `CancellationToken`, and `truncate_reply` cuts the recorded reply back to what was heard. |
| `Approvals` | Whether an approval card is waiting, and what to do with an utterance heard while one waits. |
| `Vad`, `TurnDetector` | Optional. `EarshotVad` (default feature, pure Rust) and `EnergyVad` ship, with `Endpointer` as the turn detector. |

The transport is two channels, `Inbound` (audio frames and JSON control
frames) and `Outbound` (events and audio clips). Bridging them to a WebSocket
takes a few lines of host code, and the host authenticates the connection
before it bridges.

## What the pipeline does

- **Turn-taking.** Onset, a barge-in bar that a backchannel "mm" does not
  clear, pauses, end of turn, and pre-roll so the first syllable is not
  clipped.
- **Streaming transcription.** The turn is cut into chunks at the speaker's
  pauses and transcribed while they talk on, so the transcript is ready about
  when the turn ends. Whole-clip transcription is the fallback, and a mode of
  its own.
- **Sentence-streamed speech.** The reply is cut into sentences (the first at
  its first clause), synthesized ahead of playback and sent in order. A
  filler line covers a tool round that comes before any text.
- **Barge-in.** Voice over the reply cancels the turn, tells the client to
  drop its queued audio, and asks the host to record only what had started
  playing, as the client reports it or as estimated from clip lengths.
- **Metrics.** Each response reports where its time went, from the end of
  speech to the first audio.

## The approval invariant

A turn may stop at an approval card: a change the host wants a person to
confirm. **Nothing a person says resolves one.** While `Approvals::pending`
reports a card, an utterance is answered with a reminder, handed to
`Approvals::held_utterance`, and never reaches the `Llm`.
`TurnRequest::Resolve` is built only from an `action.resolve` control frame,
which is a button press. No audio or transcript path can construct one, and
the crate's tests assert that a spoken "yes" leaves the card waiting.

## Example

A complete host with stand-in providers. A real one calls its speech
services and model where these return fixed values, and forwards a socket's
frames instead of a synthetic tone.

```rust
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use voicepipe::audio::wav_from_pcm16;
use voicepipe::{
    run, Approvals, BoxFuture, CallConfig, CallServerEvent, Clip, Endpointer, EnergyVad,
    EnergyVadConfig, Inbound, Llm, LlmEvent, LlmSink, Outbound, ProviderError, Providers, Stt,
    Tts, TurnOutcome, TurnRequest,
};

struct Ears;
impl Stt for Ears {
    fn transcribe(&self, _wav: Vec<u8>) -> BoxFuture<'_, Result<String, ProviderError>> {
        Box::pin(async { Ok("what time is it".to_string()) })
    }
}

struct Voice;
impl Tts for Voice {
    fn synthesize(&self, _text: String) -> BoxFuture<'_, Result<Clip, ProviderError>> {
        let wav = wav_from_pcm16(&[0; 8_000], 16_000);
        Box::pin(async move {
            Ok(Clip { content_type: "audio/wav".into(), bytes: wav.into() })
        })
    }
}

struct Echo;
impl Llm for Echo {
    fn turn(
        &self,
        request: TurnRequest,
        sink: LlmSink,
        _cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<TurnOutcome, ProviderError>> {
        Box::pin(async move {
            let reply = match request {
                TurnRequest::Utterance(text) => format!("You said: {text}. It is noon."),
                TurnRequest::Resolve { .. } => "Done.".to_string(),
            };
            sink(LlmEvent::TextDelta(reply.clone()));
            Ok(TurnOutcome::Completed { reply: Some(reply) })
        })
    }

    fn truncate_reply(&self, _heard: String) -> BoxFuture<'_, bool> {
        Box::pin(async { true })
    }
}

struct NoCards;
impl Approvals for NoCards {
    fn pending(&self) -> BoxFuture<'_, Option<serde_json::Value>> {
        Box::pin(async { None })
    }
    fn held_utterance(&self, _text: String) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

#[tokio::main]
async fn main() {
    let config = CallConfig::default();
    let vad = EnergyVad::new(EnergyVadConfig::default());
    let detector = Box::new(Endpointer::new(config.endpoint_config(), vad));
    let providers = Providers {
        stt: Arc::new(Ears),
        tts: Arc::new(Voice),
        llm: Arc::new(Echo),
        approvals: Arc::new(NoCards),
        observer: None,
    };
    let (to_call, inbound) = mpsc::channel(256);
    let (outbound, mut from_call) = mpsc::unbounded_channel();
    tokio::spawn(run("call-1".into(), config, providers, detector, inbound, outbound));

    // Room noise, a second of "speech" (a 220 Hz tone), then a silence long
    // enough to end the turn, in 20 ms frames of little-endian PCM16.
    let samples = (0..48_000).map(|i: i32| {
        let voiced = (8_000..24_000).contains(&i);
        let tone = (i as f32 * 220.0 * std::f32::consts::TAU / 16_000.0).sin() * 9_000.0;
        if voiced { tone as i16 } else { (i * 7_919 % 41 - 20) as i16 }
    });
    let bytes: Vec<u8> = samples.flat_map(i16::to_le_bytes).collect();
    for frame in bytes.chunks(640) {
        to_call.send(Inbound::Audio(frame.to_vec().into())).await.unwrap();
    }

    while let Some(message) = from_call.recv().await {
        match message {
            Outbound::Event(event) => {
                println!("{}", serde_json::to_string(&event).unwrap());
                if matches!(event, CallServerEvent::ResponseDone { .. }) {
                    break;
                }
            }
            Outbound::Audio(clip) => println!("({} bytes of audio)", clip.len()),
        }
    }
}
```

## The wire protocol

One duplex connection per call. The client streams its microphone as binary
frames of little-endian PCM16, mono, 16 kHz, and sends JSON control frames as
text: `session.update`, `response.cancel`, `output_audio.started` /
`output_audio.idle` (playback reports, which make "what was heard" exact),
`action.resolve` (the approval button) and `ping`. The server answers with
JSON events named, where the meaning is the same, after the OpenAI Realtime
API's: `session.created`, `call.state`,
`input_audio_buffer.speech_started`, transcription captions,
`response.created`, `response.text.delta`, `response.audio.start` followed by
one binary frame holding that piece's clip, `response.done` with metrics,
`output_audio.clear`, `conversation.item.truncated`, and
`assistant.pending_action` / `assistant.action_resolved`. The types are in the `protocol` module.

## Status

Pre-1.0. While the version is 0.x, a minor release may break the API or the
wire protocol and a patch release only fixes. The minimum supported Rust
version is 1.88.

voicepipe was extracted from the call path of
[Vogt](https://github.com/TheDancingDeveloper-org/vogt), which is its first
host; its design notes are in [`docs/DESIGN.md`](docs/DESIGN.md).

## Licence

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
