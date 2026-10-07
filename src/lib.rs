//! # voxcall
//!
//! A pipeline for live, turn-taking voice calls with a language model: audio
//! streams in, the pipeline decides when the speaker's turn is over,
//! transcribes it, runs the host's turn, and speaks the reply back a
//! sentence at a time while it is still being written — and stops it the
//! moment the speaker talks over it.
//!
//! Every provider is a trait (`Vad`, `TurnDetector`, and — with the pipeline
//! — `Stt`, `Llm`, `Tts`, `Approvals`, `Transport`), so the same pipeline
//! runs over local models or hosted APIs. See `DESIGN.md` for the boundary
//! and the approval invariant: nothing a person *says* resolves an approval.
//!
//! Modules: PCM/WAV framing ([`audio`]), voice activity detection
//! ([`vad`]), turn-taking ([`turn`]), cutting a streamed reply into
//! speakable pieces ([`text`]), the wire protocol ([`protocol`]) and the
//! call itself ([`pipeline`]).

pub mod audio;
pub mod pipeline;
pub mod protocol;
pub mod text;
pub mod turn;
pub mod vad;

pub use audio::SAMPLE_RATE;
pub use pipeline::{
    run, Approvals, BoxFuture, CallConfig, Clip, Inbound, Llm, LlmEvent, LlmSink, Observer,
    Outbound, ProviderError, Providers, ResponseReport, Stt, Tts, TurnOutcome, TurnRequest,
};
pub use protocol::{CallClientEvent, CallMetrics, CallResponseStatus, CallServerEvent, CallState};
pub use text::{speakable, SentenceChunker};
pub use turn::{EndpointConfig, EndpointEvent, Endpointer, TurnDetector};
#[cfg(feature = "earshot")]
pub use vad::EarshotVad;
pub use vad::{EnergyVad, EnergyVadConfig, Vad};
