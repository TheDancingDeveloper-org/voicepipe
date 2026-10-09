# voicepipe — design note

`voicepipe` is a transport-agnostic Rust pipeline for live, turn-taking voice
calls. A microphone streams in and the pipeline decides when a turn is over,
transcribes it, runs the host's turn, speaks the reply a sentence at a time,
and stops when the user talks over it. It began as the call path of
[Vogt](https://github.com/TheDancingDeveloper-org/vogt) and holds no type,
configuration key or path of any host.

## The boundary

Every provider is a trait, and the pipeline owns only what is common to any
voice agent: turn-taking, barge-in, ordering and timing.

| Trait | Contract | Shipped implementations |
|---|---|---|
| `Vad` | 16 kHz PCM16 frame → speech or not; told when the reply is playing (echo guard) | `EarshotVad` (default; pure Rust, MIT/Apache, no runtime deps) and `EnergyVad` (adaptive noise floor, zero deps) |
| `TurnDetector` | turns VAD decisions into events: speech started, sustained (the barge-in bar), pause (early-STT trigger), resumed, end of turn, discarded. The end of turn is decided on silence alone; there is no semantic-endpointing hook yet | `Endpointer` (onset, hangover and pre-roll timings) |
| `Stt` | one clip (a 16 kHz WAV) → text; `transcribe_after(clip, context)` takes one chunk of a turn still being spoken plus the words before it (whisper's `prompt`), defaulting to `transcribe`. A recognizer that decodes audio incrementally would be a further method; none of the whisper servers does | host-provided |
| `Llm` | *the host's turn*: a `TurnRequest` (`Utterance` or `Resolve`) in; text deltas and `ToolRound` events out; honours a `CancellationToken`; returns `Completed`, `Interrupted` or `AwaitingApproval { card }` (the card is opaque JSON with a string `id`). It also has `truncate_reply(heard)` | host-provided |
| `Tts` | text → one audio clip (content type and bytes) | host-provided |
| `Approvals` | host callback: is a card waiting (`pending`)? An utterance was held (`held_utterance`). Resolution is *not* here: it is `TurnRequest::Resolve`, built only from a button frame | host-provided |
| transport | two channels: `Inbound` (audio frames, control JSON) and `Outbound` (typed events, clips). Bridging them to a WebSocket is about 25 lines of host code, and the host authenticates *before* bridging | host-provided; there is no `Transport` trait |

The pipeline (`run`) owns:

- endpointing
- a warm-up transcription of silence as the call opens (`warm_stt`), so an idle-unloaded model is loaded before the first turn
- streaming transcription by chunks (`SttMode::Chunked`, the default, in `chunk.rs`): a turn is cut at the speaker's pauses (and, after `max_ms` of unbroken speech, at its quietest frame), the chunks are transcribed one at a time in order while the speaker talks on, each told the words before it, and each finished chunk is a caption. At the end of the turn only the audio after the last cut is left, with the end-of-turn silence trimmed. A failed chunk falls back to one whole-clip transcription
- `SttMode::Whole`, the whole-clip path: early transcription of the turn so far on a pause, discarded if the speaker resumes, and optional partial captions (off unless `partial_interval_ms` is set — each one is a full re-decode)
- the sentence chunker (the first clause is cut early) and `speakable` markdown stripping
- in-order synthesis that runs ahead of playback
- a filler line during tool rounds
- barge-in: cancel the turn, clear playback, and cut the reply to what was heard, using what the client reports or an estimate from clip lengths
- per-turn latency metrics
- the wire protocol: event names shaped like OpenAI Realtime, with clips sent as `response.audio.start` followed by a binary frame

**The approval invariant is structural, not a setting.** While `Approvals::pending()` reports a card:

- an utterance is handed to `Approvals::held_utterance` and answered with the host's reminder;
- the utterance never reaches `Llm`;
- `TurnRequest::Resolve` is constructed *only* from an explicit `action.resolve` control frame from the transport, which is a button press. No audio or transcript path can reach it.

`tests/pipeline.rs` asserts that a spoken "yes" leaves the card pending and makes no model call.

### What earshot is, and what it is not

earshot answers one question: is there voice in these 16 ms? Everything
else on the list above belongs to voicepipe, and earshot is just its default
`Vad`. Wrapping it surfaced a defect. After exact digital silence, which
browser noise suppression emits between words, earshot's level estimate
collapses and it scores everything as voice, silence included (measured
0.83–0.85 against a 0.5 threshold). `EarshotVad` therefore judges a dithered
copy of each frame (±32 LSB, about -60 dBFS). The dither never reaches the
transcriber. earshot also takes about 300 ms to settle on a new noise floor,
and the turn detector's minimum speech length absorbs that blip.

## Dependencies

The crate depends on tokio, tokio-util, serde/serde_json, bytes and (default feature) earshot.

## Prior art (checked 2026-10-07)

| Name | Real? | Licence | Maturity | Use |
|---|---|---|---|---|
| earshot (pykeio) | crate 1.2.2 | MIT OR Apache-2.0 | 290k downloads, active, pure Rust, no runtime deps, claims to beat Silero v6 and TEN VAD | **Default `Vad`** |
| sherpa-onnx (k2-fsa, official crate) | 1.13.8 | Apache-2.0 | 500k downloads, active; streaming ASR, Silero VAD, TTS | candidate optional streaming-`Stt` adapter (heavy C++ dependency, behind a feature) |
| sherpa-rs | 0.6.8 | MIT | last release 2025-10 | superseded by the official crate |
| voice_activity_detector (Silero via ort) | 0.2.1 | non-standard licence | 178k downloads | not adopted (licence, ONNX runtime) |
| TEN VAD / TEN Turn Detection | GitHub | Apache-2.0 **with Agora non-compete conditions**; turn detection also forbids end-user devices and is a Qwen2.5-7B model | 2.3k and 0.6k stars | **Not a dependency**: not OSI-open and incompatible with a permissive crate. At most a user-supplied adapter. Turn detection is too large for a CPU budget anyway |
| ten-vad (drmckay Rust wrapper) | crate 0.1.0 | Apache-2.0 (wrapper only; the model keeps TEN's terms) | 38 downloads | no |
| rustvani | crate 0.4.0-dev | BSD-2-Clause | 34 stars, pre-release | reference only |
| flowcat (AreevAI) | GitHub only | Apache-2.0 | 121 stars, active, Pipecat-compatible runtime | reference only (a runtime, not a library) |
| feros (ferosai) | GitHub only | Apache-2.0 | 111 stars, last push 2026-05 | reference only |
| car-voice (Parslee) | crate 0.55 | Apache-2.0 | part of the CAR framework, local mic oriented | no |
| fono | crate 0.18 | GPL-3.0-only | desktop dictation app | no (licence) |
| any-tts | crate 0.1.3 | MIT OR Apache-2.0 | Candle TTS backends | possible future local `Tts` adapter |
| ADK-Rust (zavora-ai) | crate 2.2.0 | Apache-2.0 | 691 stars; an agent framework with realtime voice | reference for a later realtime-provider path (option B) |
| voice-stream | **not found** (no crate, no repository) | — | — | treated as not real |
| Pipecat / LiveKit Agents | Python | BSD-2-Clause / Apache-2.0 | the reference pipeline designs | design reference: frame pipeline, pluggable turn detector, interruption, minimum endpointing delay |
