#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
//!
//! ## Modules
//!
//! PCM/WAV framing ([`audio`]), voice activity detection ([`vad`]),
//! turn-taking ([`turn`]), cutting a turn into chunks to transcribe while it
//! is spoken ([`chunk`]), cutting a streamed reply into speakable pieces
//! ([`text`]), the wire protocol ([`protocol`]) and the call itself
//! ([`pipeline`]).

pub mod audio;
pub mod chunk;
pub mod pipeline;
pub mod protocol;
pub mod text;
pub mod turn;
pub mod vad;

pub use audio::SAMPLE_RATE;
pub use chunk::{ChunkConfig, Chunker};
pub use pipeline::{
    run, Approvals, BoxFuture, CallConfig, Clip, Inbound, Llm, LlmEvent, LlmSink, Observer,
    Outbound, ProviderError, Providers, ResponseReport, Stt, SttMode, Tts, TurnOutcome,
    TurnRequest, MAX_AUDIO_FRAME_BYTES,
};
pub use protocol::{
    CallClientEvent, CallMetrics, CallResponseStatus, CallServerEvent, CallState, PROTOCOL_VERSION,
};
pub use text::{speakable, SentenceChunker};
pub use turn::{EndpointConfig, EndpointEvent, Endpointer, TurnDetector};
#[cfg(feature = "earshot")]
pub use vad::EarshotVad;
pub use vad::{EnergyVad, EnergyVadConfig, Vad};
