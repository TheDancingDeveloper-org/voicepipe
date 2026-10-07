//! The call pipeline: turn-taking, transcription, the host's turn spoken as
//! it streams, barge-in, and the approval invariant.
//!
//! [`run`] drives one call. Audio and control frames arrive on a channel
//! ([`Inbound`]); events and audio clips leave on another ([`Outbound`]).
//! Bridging those to a WebSocket (or anything else) is the host's few lines,
//! as is authenticating the connection before handing it over.
//!
//! The providers are traits: [`Stt`], [`Tts`], [`Llm`] (the host's *turn* —
//! a model call, a tool loop, whatever produces the reply) and
//! [`Approvals`]. The pipeline owns the rest:
//!
//! - **Turn-taking.** A [`TurnDetector`] ends the user's turn. When the user
//!   pauses, the turn so far is transcribed at once, so the transcript is
//!   usually ready when the turn is declared over; while they speak, it is
//!   re-transcribed every `partial_interval_ms` as a live caption.
//! - **The reply.** Streamed text is cut into sentences ([`SentenceChunker`],
//!   the first at its first clause), each synthesized and sent the moment
//!   it exists, in order, while the host is still producing the rest. A
//!   filler line covers a tool round that comes before any text.
//! - **Barge-in.** `barge_in_ms` of voice over a reply cancels the turn,
//!   tells the client to drop its queued audio, and asks the host to cut
//!   the recorded reply back to what had started playing.
//! - **Approvals.** While [`Approvals::pending`] reports a card, an utterance
//!   is answered with `approval_reminder`, handed to
//!   [`Approvals::held_utterance`], and **never reaches the [`Llm`]**. A
//!   card is resolved only by an `action.resolve` control frame — a button
//!   press — which becomes [`TurnRequest::Resolve`]. No audio or transcript
//!   path constructs one. Nothing a person says approves anything.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::audio::{pcm16_from_le_bytes, wav_duration, wav_from_pcm16, SAMPLE_RATE};
use crate::protocol::{
    CallClientEvent, CallMetrics, CallResponseStatus, CallServerEvent, CallState,
};
use crate::text::{speakable, SentenceChunker};
use crate::turn::{EndpointConfig, EndpointEvent, TurnDetector};

/// A boxed, sendable future — the shape every provider method returns, so
/// the traits stay object-safe.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A provider's failure, in words for a log line.
#[derive(Debug, Clone)]
pub struct ProviderError(pub String);

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ProviderError {}

/// One synthesized clip, in whatever container the TTS produced.
#[derive(Debug, Clone)]
pub struct Clip {
    pub content_type: String,
    pub bytes: Bytes,
}

/// Speech to text: one utterance (a 16 kHz mono PCM16 WAV) to its words.
pub trait Stt: Send + Sync {
    fn transcribe(&self, wav: Vec<u8>) -> BoxFuture<'_, Result<String, ProviderError>>;
}

/// Text to speech: one piece of a reply to one clip.
pub trait Tts: Send + Sync {
    fn synthesize(&self, text: String) -> BoxFuture<'_, Result<Clip, ProviderError>>;
}

/// What a turn reports while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmEvent {
    /// A piece of the reply's text, as soon as it exists.
    TextDelta(String),
    /// The turn is about to run tools before answering.
    ToolRound,
}

/// Where a turn's events go. Called from inside the turn.
pub type LlmSink = Arc<dyn Fn(LlmEvent) + Send + Sync>;

/// What a turn is asked to do.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnRequest {
    /// Answer what the user said.
    Utterance(String),
    /// A button resolved approval card `card_id`. Built only from an
    /// `action.resolve` control frame, never from speech.
    Resolve { card_id: String, approve: bool },
}

/// How a turn ended.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnOutcome {
    /// The reply, as the host recorded it.
    Completed { reply: Option<String> },
    /// Cut short by the cancel token; the part produced by then.
    Interrupted { reply: Option<String> },
    /// The turn proposed something that needs approval: the host's card, a
    /// JSON object with a string `id`, shown to the user to press.
    AwaitingApproval { card: Value },
}

/// The host's turn — whatever turns an utterance into a reply.
pub trait Llm: Send + Sync {
    /// Run one turn, reporting text and tool rounds to `sink` as they happen
    /// and stopping as soon as it safely can once `cancel` fires.
    fn turn(
        &self,
        request: TurnRequest,
        sink: LlmSink,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<TurnOutcome, ProviderError>>;

    /// The listener heard only `heard` of the last reply (a prefix of it,
    /// possibly empty): record that, not the whole. Returns whether the
    /// host did.
    fn truncate_reply(&self, heard: String) -> BoxFuture<'_, bool>;

    /// Options the client sent in `session.update`.
    fn session_update(&self, _options: serde_json::Map<String, Value>) {}
}

/// The host's approval cards.
pub trait Approvals: Send + Sync {
    /// The card waiting for a button, if any (a JSON object with `id`).
    fn pending(&self) -> BoxFuture<'_, Option<Value>>;
    /// An utterance heard while a card waited. It was answered with the
    /// reminder and not passed on; the host may record it.
    fn held_utterance(&self, text: String) -> BoxFuture<'_, ()>;
}

/// A finished response, for a host's log line.
#[derive(Debug, Clone)]
pub struct ResponseReport {
    pub response_id: String,
    pub status: CallResponseStatus,
    pub metrics: CallMetrics,
}

/// Told about every finished response (a host's log line).
pub type Observer = Arc<dyn Fn(&ResponseReport) + Send + Sync>;

/// Everything a call is built from.
#[derive(Clone)]
pub struct Providers {
    pub stt: Arc<dyn Stt>,
    pub tts: Arc<dyn Tts>,
    pub llm: Arc<dyn Llm>,
    pub approvals: Arc<dyn Approvals>,
    /// Told about every finished response.
    pub observer: Option<Observer>,
}

/// Timings and fixed lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallConfig {
    /// Silence after speech that ends the user's turn.
    pub end_of_turn_ms: u32,
    /// Voice needed, over a reply, to stop it.
    pub barge_in_ms: u32,
    /// Silence that counts as a pause (and starts an early transcription).
    pub pause_ms: u32,
    /// How often a turn is re-transcribed as a caption; 0 is never.
    pub partial_interval_ms: u32,
    /// Said during a tool round before any text; empty is nothing.
    pub filler: String,
    /// Said for an utterance heard while a card waits.
    pub approval_reminder: String,
    /// Said when a turn stops at a card without having said anything.
    pub proposed_line: String,
    /// Said when a turn fails without having said anything.
    pub failed_line: String,
}

impl Default for CallConfig {
    fn default() -> Self {
        Self {
            end_of_turn_ms: 700,
            barge_in_ms: 500,
            pause_ms: 200,
            partial_interval_ms: 1_500,
            filler: "One moment.".into(),
            approval_reminder: "That change is waiting on your screen. Tap approve or deny there."
                .into(),
            proposed_line: "I've put that change on your screen for you to approve.".into(),
            failed_line: "Sorry, I couldn't get an answer just then.".into(),
        }
    }
}

impl CallConfig {
    /// The turn detector timings these settings imply.
    pub fn endpoint_config(&self) -> EndpointConfig {
        EndpointConfig {
            end_of_turn_ms: self.end_of_turn_ms,
            sustained_ms: self.barge_in_ms,
            pause_ms: self.pause_ms,
            ..EndpointConfig::default()
        }
    }
}

/// From the client.
#[derive(Debug, Clone)]
pub enum Inbound {
    /// Little-endian PCM16, mono, 16 kHz.
    Audio(Bytes),
    /// A text frame: a `CallClientEvent` as JSON.
    Control(String),
}

/// To the client.
#[derive(Debug, Clone)]
pub enum Outbound {
    Event(CallServerEvent),
    /// The clip announced by the `response.audio.start` just before it.
    Audio(Bytes),
}

/// The largest audio frame accepted: two seconds at 16 kHz.
const MAX_AUDIO_FRAME_BYTES: usize = 64 * 1024;
/// A partial caption needs at least this much of the turn to say anything.
const MIN_PARTIAL_MS: u32 = 800;
/// Grace after a reply should have finished playing before a client that
/// does not report its playback is assumed to have finished.
const PLAYBACK_SLACK: Duration = Duration::from_millis(1_500);

/// Run one call until `inbound` closes.
pub async fn run(
    call_id: String,
    config: CallConfig,
    providers: Providers,
    turn_detector: Box<dyn TurnDetector>,
    mut inbound: mpsc::Receiver<Inbound>,
    outbound: mpsc::UnboundedSender<Outbound>,
) {
    let (internal_tx, mut internal_rx) = mpsc::unbounded_channel::<Internal>();
    let ctx = Ctx {
        providers,
        config: Arc::new(config),
        out: outbound,
        internal: internal_tx,
        clips: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
    };
    // Warm the fixed lines so the first time one is needed it is instant.
    {
        let ctx = ctx.clone();
        tokio::spawn(async move {
            for line in [
                ctx.config.filler.clone(),
                ctx.config.approval_reminder.clone(),
            ] {
                if !line.is_empty() {
                    ctx.canned(&line).await;
                }
            }
        });
    }
    let mut call = Call::new(ctx, turn_detector);
    call.ctx.send(CallServerEvent::SessionCreated {
        call_id,
        sample_rate: SAMPLE_RATE,
        end_of_turn_ms: call.ctx.config.end_of_turn_ms,
        barge_in_ms: call.ctx.config.barge_in_ms,
    });
    call.ctx.send(CallServerEvent::State {
        state: CallState::Listening,
    });
    loop {
        tokio::select! {
            message = inbound.recv() => match message {
                Some(Inbound::Audio(bytes)) => call.audio(&bytes),
                Some(Inbound::Control(text)) => call.control(&text),
                None => break,
            },
            Some(internal) = internal_rx.recv() => call.internal(internal),
        }
    }
    call.hang_up();
}

/// What the call's own tasks tell the loop.
enum Internal {
    Partial {
        utterance: u64,
        text: String,
    },
    /// A response's first audio went out.
    AudioStarted {
        response_id: String,
    },
    Finished(Box<Finished>),
}

struct Finished {
    response_id: String,
    /// `None` when the turn turned out not to be one (nothing was said).
    status: Option<CallResponseStatus>,
    text: Option<String>,
    pending: Option<Value>,
    /// The card this response resolved, and how (`None`: it failed). Either
    /// way the card is no longer waiting.
    resolved: Option<(String, Option<bool>)>,
    metrics: CallMetrics,
}

/// What the response task and the loop both see of one response.
#[derive(Default)]
struct Shared {
    /// For each piece sent: where it ends in the recorded reply text, or
    /// `None` for one that is not part of it (a filler, a reminder, text
    /// said before a tool round).
    piece_ends: Vec<Option<usize>>,
    /// What the host writes after its last tool round.
    reply_text: String,
    last_started: Option<u32>,
    /// When each piece should start playing, reckoned from when it was sent
    /// and how long the ones before it play.
    estimated_starts: Vec<Instant>,
    estimated_end: Option<Instant>,
    audio_sent: bool,
    first_audio: Option<Instant>,
    metrics: CallMetrics,
}

impl Shared {
    /// What the listener heard of the recorded reply: up to the end of the
    /// last of its pieces that had started playing — as the client reported
    /// it, or, from a client that reports nothing, as estimated.
    fn heard(&self, client_reports: bool) -> String {
        let last = if client_reports {
            self.last_started
        } else {
            let now = Instant::now();
            self.estimated_starts
                .iter()
                .rposition(|start| *start <= now)
                .map(|i| i as u32)
        };
        let Some(last) = last else {
            return String::new();
        };
        let end = self
            .piece_ends
            .iter()
            .take(last as usize + 1)
            .filter_map(|end| *end)
            .max()
            .unwrap_or(0);
        self.reply_text[..end.min(self.reply_text.len())]
            .trim()
            .to_string()
    }
}

struct ActiveResponse {
    id: String,
    cancel: CancellationToken,
    shared: Arc<Mutex<Shared>>,
    generating: bool,
    playing: bool,
    cut: bool,
}

enum Trigger {
    Speech {
        wav: Vec<u8>,
        eager: Option<JoinHandle<Option<String>>>,
    },
    Resolve {
        card_id: String,
        approve: bool,
    },
}

#[derive(Clone)]
struct Ctx {
    providers: Providers,
    config: Arc<CallConfig>,
    out: mpsc::UnboundedSender<Outbound>,
    internal: mpsc::UnboundedSender<Internal>,
    clips: Arc<tokio::sync::Mutex<HashMap<String, Clip>>>,
}

impl Ctx {
    fn send(&self, event: CallServerEvent) {
        let _ = self.out.send(Outbound::Event(event));
    }

    /// A fixed line's audio, synthesized once per call and reused.
    async fn canned(&self, text: &str) -> Option<Clip> {
        if let Some(clip) = self.clips.lock().await.get(text) {
            return Some(clip.clone());
        }
        let clip = self.providers.tts.synthesize(speakable(text)).await.ok()?;
        self.clips
            .lock()
            .await
            .insert(text.to_string(), clip.clone());
        Some(clip)
    }
}

static RESPONSES: AtomicU64 = AtomicU64::new(1);

struct Call {
    ctx: Ctx,
    detector: Box<dyn TurnDetector>,
    state: CallState,
    utterance: u64,
    speech_end: Option<Instant>,
    eager: Option<JoinHandle<Option<String>>>,
    partial_in_flight: bool,
    last_partial: Instant,
    response: Option<ActiveResponse>,
    /// The previous response's task, still finishing. The next waits for it,
    /// so a cut reply is truncated before a new turn lands.
    finishing: Option<JoinHandle<()>>,
    pending_card: Option<String>,
    client_reports: bool,
}

impl Call {
    fn new(ctx: Ctx, detector: Box<dyn TurnDetector>) -> Self {
        Self {
            ctx,
            detector,
            state: CallState::Listening,
            utterance: 0,
            speech_end: None,
            eager: None,
            partial_in_flight: false,
            last_partial: Instant::now(),
            response: None,
            finishing: None,
            pending_card: None,
            client_reports: false,
        }
    }

    fn set_state(&mut self, state: CallState) {
        if self.state != state {
            self.state = state;
            self.ctx.send(CallServerEvent::State { state });
        }
    }

    fn resting_state(&self) -> CallState {
        if self.pending_card.is_some() {
            CallState::AwaitingApproval
        } else {
            CallState::Listening
        }
    }

    fn restore_state(&mut self) {
        let state = match &self.response {
            Some(r) if r.playing => CallState::Speaking,
            Some(r) if r.generating => CallState::Thinking,
            _ => self.resting_state(),
        };
        self.set_state(state);
    }

    fn audio(&mut self, bytes: &[u8]) {
        if bytes.len() > MAX_AUDIO_FRAME_BYTES {
            return;
        }
        self.expire_estimated_playback();
        let samples = pcm16_from_le_bytes(bytes);
        for event in self.detector.push(&samples) {
            self.endpoint(event);
        }
        self.maybe_partial();
    }

    fn endpoint(&mut self, event: EndpointEvent) {
        match event {
            EndpointEvent::SpeechStarted => {
                self.utterance += 1;
                self.speech_end = None;
                self.last_partial = Instant::now();
                self.ctx.send(CallServerEvent::SpeechStarted);
                if self.response.is_none() {
                    self.set_state(CallState::UserSpeaking);
                }
            }
            EndpointEvent::Sustained => {
                if self.response.is_some() {
                    self.barge_in();
                }
                self.set_state(CallState::UserSpeaking);
            }
            EndpointEvent::PauseBegan => {
                self.speech_end = Instant::now()
                    .checked_sub(Duration::from_millis(self.ctx.config.pause_ms as u64));
                if let Some(old) = self.eager.take() {
                    old.abort();
                }
                let wav = wav_from_pcm16(self.detector.segment(), SAMPLE_RATE);
                let stt = Arc::clone(&self.ctx.providers.stt);
                self.eager = Some(tokio::spawn(async move { transcribe(&*stt, wav).await }));
            }
            EndpointEvent::SpeechResumed => {
                self.speech_end = None;
                if let Some(eager) = self.eager.take() {
                    eager.abort();
                }
            }
            EndpointEvent::EndOfTurn { audio, speech_ms } => {
                self.ctx.send(CallServerEvent::SpeechStopped);
                let eager = self.eager.take();
                // Too short to have been a barge-in, said over a reply that
                // is still going: a backchannel ("mm", "right") or echo.
                if self.response.as_ref().is_some_and(|r| !r.cut)
                    && speech_ms < self.ctx.config.barge_in_ms
                {
                    if let Some(eager) = eager {
                        eager.abort();
                    }
                    self.restore_state();
                    return;
                }
                let speech_end = self.speech_end.take().or_else(|| {
                    Instant::now()
                        .checked_sub(Duration::from_millis(self.ctx.config.end_of_turn_ms as u64))
                });
                let wav = wav_from_pcm16(&audio, SAMPLE_RATE);
                self.start_response(Trigger::Speech { wav, eager }, speech_end);
            }
            EndpointEvent::Discarded => {
                if let Some(eager) = self.eager.take() {
                    eager.abort();
                }
                self.restore_state();
            }
        }
    }

    fn maybe_partial(&mut self) {
        let interval = self.ctx.config.partial_interval_ms;
        if interval == 0
            || self.partial_in_flight
            || !self.detector.in_turn()
            || self.eager.is_some()
        {
            return;
        }
        let segment_ms = self.detector.segment().len() as u32 * 1000 / SAMPLE_RATE;
        if segment_ms < MIN_PARTIAL_MS
            || self.last_partial.elapsed() < Duration::from_millis(interval as u64)
        {
            return;
        }
        self.partial_in_flight = true;
        self.last_partial = Instant::now();
        let wav = wav_from_pcm16(self.detector.segment(), SAMPLE_RATE);
        let stt = Arc::clone(&self.ctx.providers.stt);
        let internal = self.ctx.internal.clone();
        let utterance = self.utterance;
        tokio::spawn(async move {
            let text = transcribe(&*stt, wav).await.unwrap_or_default();
            let _ = internal.send(Internal::Partial { utterance, text });
        });
    }

    fn expire_estimated_playback(&mut self) {
        if self.client_reports {
            return;
        }
        let over = self.response.as_ref().is_some_and(|r| {
            r.playing
                && r.shared
                    .lock()
                    .expect("shared")
                    .estimated_end
                    .is_some_and(|end| Instant::now() > end + PLAYBACK_SLACK)
        });
        if over {
            let id = self
                .response
                .as_ref()
                .map(|r| r.id.clone())
                .unwrap_or_default();
            self.playback_idle(&id);
        }
    }

    fn playback_idle(&mut self, response_id: &str) {
        let Some(response) = self.response.as_mut().filter(|r| r.id == response_id) else {
            return;
        };
        response.playing = false;
        self.detector.set_playback(false);
        if !response.generating {
            self.response = None;
        }
        if !self.detector.in_turn() {
            self.restore_state();
        }
    }

    /// Stop the reply: the user spoke over it, or tapped stop.
    fn barge_in(&mut self) {
        let Some(response) = self.response.take() else {
            return;
        };
        response.cancel.cancel();
        self.ctx.send(CallServerEvent::OutputAudioClear {
            response_id: response.id.clone(),
        });
        self.detector.set_playback(false);
        if !response.generating {
            // The reply had finished and was only still being spoken, so no
            // task is left to cut it back: do it here.
            let heard = response
                .shared
                .lock()
                .expect("shared")
                .heard(self.client_reports);
            let ctx = self.ctx.clone();
            let id = response.id;
            let previous = self.finishing.take();
            self.finishing = Some(tokio::spawn(async move {
                if let Some(previous) = previous {
                    let _ = previous.await;
                }
                if ctx.providers.llm.truncate_reply(heard.clone()).await {
                    ctx.send(CallServerEvent::ItemTruncated {
                        response_id: id,
                        text: heard,
                    });
                }
            }));
        }
        // A still-generating response truncates itself when it sees the
        // cancel, and reports how it ended.
    }

    fn start_response(&mut self, trigger: Trigger, speech_end: Option<Instant>) {
        if self.response.is_some() {
            self.barge_in();
        }
        let id = format!("resp_{}", RESPONSES.fetch_add(1, Ordering::Relaxed));
        let cancel = CancellationToken::new();
        let shared = Arc::new(Mutex::new(Shared::default()));
        self.response = Some(ActiveResponse {
            id: id.clone(),
            cancel: cancel.clone(),
            shared: Arc::clone(&shared),
            generating: true,
            playing: false,
            cut: false,
        });
        self.set_state(CallState::Thinking);
        let ctx = self.ctx.clone();
        let previous = self.finishing.take();
        let client_reports = self.client_reports;
        self.finishing = Some(tokio::spawn(async move {
            if let Some(previous) = previous {
                let _ = previous.await;
            }
            let finished = respond(
                &ctx,
                id,
                trigger,
                cancel,
                shared,
                speech_end,
                client_reports,
            )
            .await;
            let _ = ctx.internal.send(Internal::Finished(Box::new(finished)));
        }));
    }

    fn control(&mut self, text: &str) {
        let Ok(event) = serde_json::from_str::<CallClientEvent>(text) else {
            self.ctx.send(CallServerEvent::Error {
                message: "unrecognised control frame".into(),
            });
            return;
        };
        match event {
            CallClientEvent::Auth { .. } => {}
            CallClientEvent::Ping => self.ctx.send(CallServerEvent::Pong),
            CallClientEvent::SessionUpdate { options } => {
                self.ctx.providers.llm.session_update(options)
            }
            CallClientEvent::ResponseCancel => {
                if let Some(r) = self.response.as_mut() {
                    r.cut = true;
                }
                self.barge_in();
            }
            CallClientEvent::OutputAudioStarted { response_id, index } => {
                self.client_reports = true;
                if let Some(response) = self.response.as_ref().filter(|r| r.id == response_id) {
                    let mut shared = response.shared.lock().expect("shared");
                    shared.last_started = Some(shared.last_started.map_or(index, |i| i.max(index)));
                }
            }
            CallClientEvent::OutputAudioIdle { response_id } => {
                self.client_reports = true;
                self.playback_idle(&response_id);
            }
            CallClientEvent::ActionResolve { id, approve } => {
                // The only road to `TurnRequest::Resolve`.
                if self.pending_card.as_deref() != Some(id.as_str()) {
                    self.ctx.send(CallServerEvent::Error {
                        message: "no such approval card is waiting".into(),
                    });
                    return;
                }
                self.start_response(
                    Trigger::Resolve {
                        card_id: id,
                        approve,
                    },
                    None,
                );
            }
        }
    }

    fn internal(&mut self, internal: Internal) {
        match internal {
            Internal::Partial { utterance, text } => {
                self.partial_in_flight = false;
                if utterance == self.utterance && self.detector.in_turn() && !text.is_empty() {
                    self.ctx
                        .send(CallServerEvent::TranscriptionPartial { text });
                }
            }
            Internal::AudioStarted { response_id } => {
                if let Some(response) = self.response.as_mut().filter(|r| r.id == response_id) {
                    response.playing = true;
                    self.detector.set_playback(true);
                    if !self.detector.in_turn() {
                        self.set_state(CallState::Speaking);
                    }
                }
            }
            Internal::Finished(finished) => self.finished(*finished),
        }
    }

    fn finished(&mut self, finished: Finished) {
        if let Some((id, outcome)) = finished.resolved {
            if self.pending_card.as_deref() == Some(id.as_str()) {
                self.pending_card = None;
            }
            if let Some(approved) = outcome {
                self.ctx
                    .send(CallServerEvent::ActionResolved { id, approved });
            }
        }
        if let Some(card) = finished.pending {
            self.pending_card = card_id(&card);
            self.ctx
                .send(CallServerEvent::PendingAction { action: card });
        }
        if let Some(status) = finished.status {
            if let Some(observer) = &self.ctx.providers.observer {
                observer(&ResponseReport {
                    response_id: finished.response_id.clone(),
                    status,
                    metrics: finished.metrics.clone(),
                });
            }
            self.ctx.send(CallServerEvent::ResponseDone {
                response_id: finished.response_id.clone(),
                status,
                text: finished.text,
                metrics: finished.metrics,
            });
        }
        if let Some(response) = self
            .response
            .as_mut()
            .filter(|r| r.id == finished.response_id)
        {
            response.generating = false;
            if !response.playing {
                self.response = None;
            }
        }
        if !self.detector.in_turn() {
            self.restore_state();
        }
    }

    fn hang_up(&mut self) {
        if let Some(response) = self.response.take() {
            response.cancel.cancel();
        }
        if let Some(eager) = self.eager.take() {
            eager.abort();
        }
    }
}

fn card_id(card: &Value) -> Option<String> {
    card.get("id").and_then(Value::as_str).map(str::to_string)
}

/// Transcribe a turn, or `None` if it said nothing a person would call words.
async fn transcribe(stt: &dyn Stt, wav: Vec<u8>) -> Option<String> {
    let text = stt.transcribe(wav).await.ok()?;
    let text = text.trim().to_string();
    (!is_non_speech(&text)).then_some(text)
}

/// Whisper-family transcribers describe what they heard when it was not
/// speech — `[BLANK_AUDIO]`, `(silence)`, `[Music]` — and a turn made of
/// nothing else is no turn.
pub fn is_non_speech(text: &str) -> bool {
    let mut rest = text.trim();
    loop {
        rest = rest.trim_start();
        let close = match rest.chars().next() {
            Some('[') => ']',
            Some('(') => ')',
            Some('*') => '*',
            _ => break,
        };
        match rest[1..].find(close) {
            Some(end) => rest = &rest[end + 2..],
            None => break,
        }
    }
    !rest.chars().any(char::is_alphanumeric)
}

fn ms_between(start: Option<Instant>, end: Option<Instant>) -> Option<u64> {
    match (start, end) {
        (Some(start), Some(end)) if end >= start => Some((end - start).as_millis() as u64),
        _ => None,
    }
}

struct Piece {
    text: String,
    end: Option<usize>,
    clip: Option<Clip>,
}

#[derive(Default)]
struct Times {
    speech_end: Option<Instant>,
    end_of_turn: Option<Instant>,
    transcript: Option<Instant>,
    first_text: Option<Instant>,
}

/// One response: transcribe (for speech), then the host's turn, spoken as
/// it streams.
async fn respond(
    ctx: &Ctx,
    response_id: String,
    trigger: Trigger,
    cancel: CancellationToken,
    shared: Arc<Mutex<Shared>>,
    speech_end: Option<Instant>,
    client_reports: bool,
) -> Finished {
    let mut times = Times {
        speech_end,
        end_of_turn: speech_end.map(|_| Instant::now()),
        ..Times::default()
    };
    let mut finished = Finished {
        response_id: response_id.clone(),
        status: None,
        text: None,
        pending: None,
        resolved: None,
        metrics: CallMetrics::default(),
    };

    let request = match trigger {
        Trigger::Speech { wav, eager } => {
            let text = tokio::select! {
                _ = cancel.cancelled() => return finished,
                text = async {
                    match eager {
                        Some(eager) => match eager.await {
                            Ok(Some(text)) => Some(text),
                            _ => transcribe(&*ctx.providers.stt, wav).await,
                        },
                        None => transcribe(&*ctx.providers.stt, wav).await,
                    }
                } => text,
            };
            times.transcript = Some(Instant::now());
            let Some(text) = text else {
                return finished;
            };
            ctx.send(CallServerEvent::TranscriptionCompleted { text: text.clone() });
            TurnRequest::Utterance(text)
        }
        Trigger::Resolve { card_id, approve } => TurnRequest::Resolve { card_id, approve },
    };

    ctx.send(CallServerEvent::ResponseCreated {
        response_id: response_id.clone(),
    });

    let (piece_tx, piece_rx) = mpsc::unbounded_channel::<Piece>();
    let speaker = tokio::spawn(speak(
        ctx.clone(),
        response_id.clone(),
        piece_rx,
        cancel.clone(),
        Arc::clone(&shared),
    ));

    // An utterance while a card waits is answered here and goes no further:
    // a card is resolved by its button, and by nothing said.
    if let TurnRequest::Utterance(text) = &request {
        if let Some(card) = ctx.providers.approvals.pending().await {
            ctx.providers.approvals.held_utterance(text.clone()).await;
            // The card may have come from elsewhere; this client shows it too.
            finished.pending = Some(card);
            let reminder = ctx.config.approval_reminder.clone();
            let _ = piece_tx.send(Piece {
                clip: ctx.canned(&reminder).await,
                text: reminder.clone(),
                end: None,
            });
            drop(piece_tx);
            let _ = speaker.await;
            finished.status = Some(if cancel.is_cancelled() {
                CallResponseStatus::Interrupted
            } else {
                CallResponseStatus::Completed
            });
            finished.text = Some(reminder);
            finished.metrics = metrics(&times, &shared);
            return finished;
        }
    }

    let resolve = match &request {
        TurnRequest::Resolve { card_id, approve } => Some((card_id.clone(), *approve)),
        TurnRequest::Utterance(_) => None,
    };
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<LlmEvent>();
    let sink: LlmSink = Arc::new(move |event| {
        let _ = event_tx.send(event);
    });
    let llm = Arc::clone(&ctx.providers.llm);
    let turn = llm.turn(request, sink, cancel.clone());
    tokio::pin!(turn);

    let mut chunker = SentenceChunker::new();
    let mut offset = 0usize;
    let mut spoke_anything = false;
    let mut tool_rounds = 0u32;
    let mut on_event = |event: LlmEvent, times: &mut Times, spoke: &mut bool, rounds: &mut u32| {
        match event {
            LlmEvent::TextDelta(delta) => {
                times.first_text.get_or_insert_with(Instant::now);
                ctx.send(CallServerEvent::ResponseTextDelta {
                    response_id: response_id.clone(),
                    delta: delta.clone(),
                });
                shared.lock().expect("shared").reply_text.push_str(&delta);
                for piece in chunker.push(&delta) {
                    let end = piece_end(
                        &shared.lock().expect("shared").reply_text,
                        &piece,
                        &mut offset,
                    );
                    *spoke = true;
                    let _ = piece_tx.send(Piece {
                        text: piece,
                        end,
                        clip: None,
                    });
                }
            }
            LlmEvent::ToolRound => {
                *rounds += 1;
                // What was said before the tools is spoken, but the recorded
                // reply is only what the host writes after its last round.
                if let Some(piece) = chunker.finish() {
                    *spoke = true;
                    let _ = piece_tx.send(Piece {
                        text: piece,
                        end: None,
                        clip: None,
                    });
                }
                shared.lock().expect("shared").reply_text.clear();
                offset = 0;
                if !*spoke && !ctx.config.filler.is_empty() {
                    *spoke = true;
                    shared.lock().expect("shared").metrics.filler = true;
                    let _ = piece_tx.send(Piece {
                        text: ctx.config.filler.clone(),
                        end: None,
                        clip: None,
                    });
                }
            }
        }
    };
    let result = loop {
        tokio::select! {
            biased;
            Some(event) = event_rx.recv() => {
                on_event(event, &mut times, &mut spoke_anything, &mut tool_rounds);
            }
            result = &mut turn => {
                while let Ok(event) = event_rx.try_recv() {
                    on_event(event, &mut times, &mut spoke_anything, &mut tool_rounds);
                }
                break result;
            }
        }
    };
    drop(on_event);
    if let Some(piece) = chunker.finish() {
        let end = piece_end(
            &shared.lock().expect("shared").reply_text,
            &piece,
            &mut offset,
        );
        spoke_anything = true;
        let _ = piece_tx.send(Piece {
            text: piece,
            end,
            clip: None,
        });
    }
    finished.resolved = resolve.map(|(id, approve)| (id, result.is_ok().then_some(approve)));
    let (status, recorded) = match &result {
        Ok(TurnOutcome::AwaitingApproval { card }) => {
            if !spoke_anything {
                let _ = piece_tx.send(Piece {
                    text: ctx.config.proposed_line.clone(),
                    end: None,
                    clip: None,
                });
            }
            finished.pending = Some(card.clone());
            (CallResponseStatus::PendingApproval, None)
        }
        Ok(TurnOutcome::Interrupted { reply }) => (CallResponseStatus::Interrupted, reply.clone()),
        Ok(TurnOutcome::Completed { reply }) => (CallResponseStatus::Completed, reply.clone()),
        Err(error) => {
            ctx.send(CallServerEvent::Error {
                message: error.to_string(),
            });
            if !spoke_anything {
                let _ = piece_tx.send(Piece {
                    text: ctx.config.failed_line.clone(),
                    end: None,
                    clip: None,
                });
            }
            (CallResponseStatus::Failed, None)
        }
    };
    drop(piece_tx);
    let _ = speaker.await;
    shared.lock().expect("shared").metrics.tool_rounds = tool_rounds;

    // A cut response keeps only what was heard.
    if cancel.is_cancelled() {
        let heard = shared.lock().expect("shared").heard(client_reports);
        if recorded.is_some() && ctx.providers.llm.truncate_reply(heard.clone()).await {
            finished.text = Some(heard);
        }
        finished.status = Some(CallResponseStatus::Interrupted);
    } else {
        finished.text = recorded;
        finished.status = Some(status);
    }
    finished.metrics = metrics(&times, &shared);
    finished
}

/// Where `piece` ends in `text`, searching from `offset` and moving it on.
fn piece_end(text: &str, piece: &str, offset: &mut usize) -> Option<usize> {
    let start = (*offset).min(text.len());
    let found = text[start..].find(piece)?;
    let end = start + found + piece.len();
    *offset = end;
    Some(end)
}

fn metrics(times: &Times, shared: &Mutex<Shared>) -> CallMetrics {
    let shared = shared.lock().expect("shared");
    let mut metrics = shared.metrics.clone();
    metrics.endpoint_ms = ms_between(times.speech_end, times.end_of_turn);
    metrics.stt_ms = ms_between(times.end_of_turn, times.transcript);
    metrics.llm_first_text_ms = ms_between(times.transcript, times.first_text);
    metrics.speech_end_to_first_audio_ms = ms_between(times.speech_end, shared.first_audio);
    metrics
}

/// How long `text` takes to say, for a clip whose container does not say:
/// about fifteen characters a second.
fn spoken_length(text: &str) -> Duration {
    Duration::from_millis(400 + text.chars().count() as u64 * 1000 / 15)
}

/// Synthesize and send pieces in order until the channel closes or the
/// response is cut.
async fn speak(
    ctx: Ctx,
    response_id: String,
    mut pieces: mpsc::UnboundedReceiver<Piece>,
    cancel: CancellationToken,
    shared: Arc<Mutex<Shared>>,
) {
    let mut index = 0u32;
    loop {
        let piece = tokio::select! {
            _ = cancel.cancelled() => return,
            piece = pieces.recv() => match piece {
                Some(piece) => piece,
                None => return,
            },
        };
        let text = speakable(&piece.text);
        if text.is_empty() {
            continue;
        }
        let queued = Instant::now();
        let clip = match piece.clip {
            Some(clip) => Some(clip),
            None if piece.end.is_none() && piece.text == ctx.config.filler => {
                ctx.canned(&piece.text).await
            }
            None => tokio::select! {
                _ = cancel.cancelled() => return,
                clip = ctx.providers.tts.synthesize(text.clone()) => clip.ok(),
            },
        };
        let Some(clip) = clip else {
            continue;
        };
        if cancel.is_cancelled() {
            return;
        }
        let first = {
            let mut shared = shared.lock().expect("shared");
            let now = Instant::now();
            let first = !shared.audio_sent;
            if first {
                shared.audio_sent = true;
                shared.first_audio = Some(now);
                shared.metrics.tts_first_ms = Some(queued.elapsed().as_millis() as u64);
            }
            shared.piece_ends.push(piece.end);
            let start = shared.estimated_end.filter(|end| *end > now).unwrap_or(now);
            let length = wav_duration(&clip.bytes).unwrap_or_else(|| spoken_length(&text));
            shared.estimated_starts.push(start);
            shared.estimated_end = Some(start + length);
            first
        };
        if first {
            let _ = ctx.internal.send(Internal::AudioStarted {
                response_id: response_id.clone(),
            });
        }
        ctx.send(CallServerEvent::ResponseAudioStart {
            response_id: response_id.clone(),
            index,
            text,
            content_type: clip.content_type.clone(),
            bytes: clip.bytes.len() as u64,
        });
        let _ = ctx.out.send(Outbound::Audio(clip.bytes));
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcriber_noise_labels_are_not_a_turn() {
        for text in [
            "",
            "  ",
            "[BLANK_AUDIO]",
            "(silence)",
            "[Music] (wind)",
            "*coughs*",
            "...",
        ] {
            assert!(is_non_speech(text), "{text:?}");
        }
        for text in ["yes", "[laughs] okay then", "(um) stop"] {
            assert!(!is_non_speech(text), "{text:?}");
        }
    }

    #[test]
    fn pieces_are_found_in_order_even_when_they_repeat() {
        let text = "Yes. Yes. Done.";
        let mut offset = 0;
        assert_eq!(piece_end(text, "Yes.", &mut offset), Some(4));
        assert_eq!(piece_end(text, "Yes.", &mut offset), Some(9));
        assert_eq!(piece_end(text, "Done.", &mut offset), Some(15));
        assert_eq!(piece_end(text, "Missing.", &mut offset), None);
    }

    fn shared_with(pieces: &[Option<usize>], text: &str) -> Shared {
        Shared {
            piece_ends: pieces.to_vec(),
            reply_text: text.to_string(),
            ..Shared::default()
        }
    }

    #[test]
    fn what_was_heard_ends_at_the_last_piece_that_started_playing() {
        let mut shared = shared_with(&[None, Some(6), Some(14)], "First. Second.");
        assert_eq!(shared.heard(true), "", "nothing reported, nothing heard");
        shared.last_started = Some(0);
        assert_eq!(shared.heard(true), "", "only the filler had started");
        shared.last_started = Some(1);
        assert_eq!(shared.heard(true), "First.");
        shared.last_started = Some(2);
        assert_eq!(shared.heard(true), "First. Second.");
    }

    #[test]
    fn a_client_that_reports_nothing_is_reckoned_by_the_clock() {
        let now = Instant::now();
        let mut shared = shared_with(&[Some(6), Some(14)], "First. Second.");
        shared.estimated_starts = vec![now - Duration::from_secs(1), now + Duration::from_secs(5)];
        assert_eq!(shared.heard(false), "First.");
    }
}
