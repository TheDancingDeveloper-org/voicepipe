//! The call's wire protocol.
//!
//! One duplex connection per call. The client streams its microphone as
//! binary frames of little-endian PCM16, mono, 16 kHz (any whole number of
//! samples; 20 ms is typical) and sends the JSON control frames below as
//! text. The server answers with the JSON events below; the one binary frame
//! it sends follows a `response.audio.start` and is that piece's audio clip,
//! whole, in the container its `content_type` names.
//!
//! Event names follow the OpenAI Realtime API's where the meaning is the
//! same, so a client written against this protocol reads like one written
//! against that. Authentication is the host's business and happens before
//! the pipeline sees the connection; `auth` is defined here only so a host
//! can parse a first frame with the same types.

use serde::{Deserialize, Serialize};

/// The version of this protocol, sent in `session.created`. It goes up with
/// any change a client could notice: an event or field renamed or removed,
/// or a meaning changed. New events and new fields do not change it; a
/// client ignores what it does not know.
pub const PROTOCOL_VERSION: u32 = 1;

/// What a call client sends, as text frames.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum CallClientEvent {
    /// The conventional first frame, for hosts that authenticate in-band.
    /// The host checks it; the pipeline ignores it.
    #[serde(rename = "auth")]
    Auth {
        /// The client's credential, in whatever form the host takes.
        token: String,
    },
    /// Per-call options for the host's turns.
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// Host-defined options (a model profile, say). Handed to
        /// `Llm::session_update` as they came.
        #[serde(flatten)]
        options: serde_json::Map<String, serde_json::Value>,
    },
    /// Stop the reply now — a tap rather than a barge-in.
    #[serde(rename = "response.cancel")]
    ResponseCancel,
    /// The client began playing piece `index` of `response_id`. What was
    /// heard of an interrupted reply is reckoned from these.
    #[serde(rename = "output_audio.started")]
    OutputAudioStarted {
        /// The response the piece belongs to.
        response_id: String,
        /// The piece, as numbered in its `response.audio.start`.
        index: u32,
    },
    /// The client's playback queue for `response_id` ran dry: nothing of the
    /// reply is coming out of the speaker any more.
    #[serde(rename = "output_audio.idle")]
    OutputAudioIdle {
        /// The response whose audio ran out.
        response_id: String,
    },
    /// A button press on the approval card. Never sent for speech: a spoken
    /// "yes" is not an approval, and the server never treats one as one.
    #[serde(rename = "action.resolve")]
    ActionResolve {
        /// The card's `id`, from its `approval.pending`.
        id: String,
        /// Approve (`true`) or deny.
        approve: bool,
    },
    /// Keep-alive; answered with `pong`.
    #[serde(rename = "ping")]
    Ping,
}

/// Where the call is, for the client's status line.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CallState {
    /// Waiting for the user to speak.
    Listening,
    /// The user is speaking.
    UserSpeaking,
    /// The turn ended; transcribing it and waiting for the model.
    Thinking,
    /// The reply is being spoken.
    Speaking,
    /// A change is waiting on the on-screen card.
    AwaitingApproval,
}

/// How a response ended.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CallResponseStatus {
    /// The reply was given in full.
    Completed,
    /// Cut short by a barge-in or `response.cancel`.
    Interrupted,
    /// The turn proposed a change and stopped at the approval gate.
    PendingApproval,
    /// The host's turn failed; an `error` event says why.
    Failed,
}

/// Where a response's time went, in milliseconds. Every field is `None` when
/// that stage did not happen (no tool round, no audio before a cut, …).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct CallMetrics {
    /// Last voiced audio → the end of the turn was declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_ms: Option<u64>,
    /// End of turn → the transcript was ready (0 when an early
    /// transcription had already finished).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stt_ms: Option<u64>,
    /// Transcript ready → the model's first text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_first_text_ms: Option<u64>,
    /// First piece of text → its audio was ready.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tts_first_ms: Option<u64>,
    /// The headline: last voiced audio → the first reply audio was sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_end_to_first_audio_ms: Option<u64>,
    /// Model rounds that asked for tools.
    #[serde(default)]
    pub tool_rounds: u32,
    /// The first audio sent was the filler phrase, not the answer.
    #[serde(default)]
    pub filler: bool,
}

/// What the server sends, as text frames.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum CallServerEvent {
    /// The call is up. Audio is expected at `sample_rate`.
    #[serde(rename = "session.created")]
    SessionCreated {
        /// [`PROTOCOL_VERSION`] of the server.
        protocol: u32,
        /// The host's name for this call.
        call_id: String,
        /// Samples per second of the audio the client sends.
        sample_rate: u32,
        /// Silence, in milliseconds, that ends the user's turn.
        end_of_turn_ms: u32,
        /// Voice, in milliseconds, that stops a reply when the user talks
        /// over it.
        barge_in_ms: u32,
    },
    /// The call moved to `state`.
    #[serde(rename = "call.state")]
    State {
        /// Where the call is now.
        state: CallState,
    },
    /// The user began a turn.
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted,
    /// The user's turn ended.
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped,
    /// The turn so far, re-transcribed while the user is still speaking. The
    /// whole text each time, not an increment.
    #[serde(rename = "conversation.item.input_audio_transcription.partial")]
    TranscriptionPartial {
        /// The words so far.
        text: String,
    },
    /// The turn's transcript, as the host's turn will get it.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted {
        /// The words.
        text: String,
    },
    /// A response began.
    #[serde(rename = "response.created")]
    ResponseCreated {
        /// The response's id, unique within the call.
        response_id: String,
    },
    /// More of the reply's text.
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta {
        /// The response it belongs to.
        response_id: String,
        /// The new text, to append.
        delta: String,
    },
    /// Piece `index` of the reply; the next frame is its audio, binary.
    #[serde(rename = "response.audio.start")]
    ResponseAudioStart {
        /// The response it belongs to.
        response_id: String,
        /// The piece's place in the response, from 0.
        index: u32,
        /// What the piece says.
        text: String,
        /// The audio's media type (`audio/wav`, `audio/mpeg`, ...).
        content_type: String,
        /// The length of the binary frame that follows.
        bytes: u64,
    },
    /// A response ended.
    #[serde(rename = "response.done")]
    ResponseDone {
        /// The response that ended.
        response_id: String,
        /// How it ended.
        status: CallResponseStatus,
        /// The reply as recorded — for an interrupted reply, what was heard.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Where its time went.
        metrics: CallMetrics,
    },
    /// Stop playing `response_id` now and drop whatever is queued.
    #[serde(rename = "output_audio.clear")]
    OutputAudioClear {
        /// The response to silence.
        response_id: String,
    },
    /// A reply that had finished generating was cut while it was being
    /// spoken; the conversation now records only `text`, what was heard
    /// (empty: nothing of it was, and it was removed).
    #[serde(rename = "conversation.item.truncated")]
    ItemTruncated {
        /// The response that was cut.
        response_id: String,
        /// What was heard of it.
        text: String,
    },
    /// A change the host wants approved: the host's own card, opaque here,
    /// with a string `id`. It happens only if the card's button is pressed
    /// (`action.resolve`).
    #[serde(rename = "approval.pending")]
    ApprovalPending {
        /// The host's card.
        card: serde_json::Value,
    },
    /// A card was approved or denied by its button.
    #[serde(rename = "approval.resolved")]
    ApprovalResolved {
        /// The card's `id`.
        id: String,
        /// Whether it was approved.
        approved: bool,
    },
    /// Something went wrong; the call goes on.
    #[serde(rename = "error")]
    Error {
        /// What, in words.
        message: String,
    },
    /// The answer to `ping`.
    #[serde(rename = "pong")]
    Pong,
}
