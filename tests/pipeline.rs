//! The call pipeline end to end, with fake providers: what a host gets for
//! free — turn-taking, sentence-streamed speech, barge-in that keeps only
//! what was heard — and the invariant it can rely on: nothing said resolves
//! an approval.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use voicepipe::{
    run, Approvals, BoxFuture, CallConfig, Clip, Endpointer, EnergyVad, EnergyVadConfig, Inbound,
    Llm, LlmEvent, LlmSink, Outbound, ProviderError, Providers, Stt, SttMode, Tts, TurnDetector,
    TurnOutcome, TurnRequest, Vad,
};

const RATE: u32 = 16_000;

#[derive(Default)]
struct World {
    transcript: String,
    /// Every clip sent to the transcriber.
    stt_calls: usize,
    /// Scripted answers for the chunks of a streamed turn, in order (`None`
    /// fails that chunk). Once they run out a chunk gets `transcript`.
    chunk_texts: Vec<Option<String>>,
    /// The context each chunk was sent with.
    contexts: Vec<String>,
    /// Whole-clip transcriptions (not chunks), the warm-up included.
    whole_calls: usize,
    /// How long a chunk takes to transcribe.
    chunk_delay_ms: u64,
    /// Chunk transcriptions that ran to the end.
    chunks_done: usize,
    /// How long the model thinks before its first word.
    llm_delay_ms: u64,
    /// Scripted replies, in order: (tool round first?, text, card?).
    replies: Vec<(bool, String, Option<Value>)>,
    requests: Vec<TurnRequest>,
    truncated: Vec<String>,
    held: Vec<String>,
    card: Option<Value>,
    tts_delay_ms: u64,
}

type W = Arc<Mutex<World>>;

struct FakeStt(W);
impl Stt for FakeStt {
    fn transcribe(&self, _wav: Vec<u8>) -> BoxFuture<'_, Result<String, ProviderError>> {
        let text = {
            let mut world = self.0.lock().unwrap();
            world.stt_calls += 1;
            world.whole_calls += 1;
            world.transcript.clone()
        };
        Box::pin(async move { Ok(text) })
    }

    fn transcribe_after(
        &self,
        _wav: Vec<u8>,
        context: String,
    ) -> BoxFuture<'_, Result<String, ProviderError>> {
        let (text, delay) = {
            let mut world = self.0.lock().unwrap();
            world.stt_calls += 1;
            world.contexts.push(context);
            let text = if world.chunk_texts.is_empty() {
                Some(world.transcript.clone())
            } else {
                world.chunk_texts.remove(0)
            };
            (text, world.chunk_delay_ms)
        };
        let world = Arc::clone(&self.0);
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            world.lock().unwrap().chunks_done += 1;
            text.ok_or_else(|| ProviderError::from("chunk refused"))
        })
    }
}

fn wav(ms: u32) -> Bytes {
    let samples = vec![0i16; (24_000 * ms / 1000) as usize];
    Bytes::from(voicepipe::audio::wav_from_pcm16(&samples, 24_000))
}

struct FakeTts(W);
impl Tts for FakeTts {
    fn synthesize(&self, _text: String) -> BoxFuture<'_, Result<Clip, ProviderError>> {
        let delay = self.0.lock().unwrap().tts_delay_ms;
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(Clip {
                content_type: "audio/wav".into(),
                bytes: wav(400),
            })
        })
    }
}

struct FakeLlm(W);
impl Llm for FakeLlm {
    fn turn(
        &self,
        request: TurnRequest,
        sink: LlmSink,
        cancel: CancellationToken,
    ) -> BoxFuture<'_, Result<TurnOutcome, ProviderError>> {
        let world = Arc::clone(&self.0);
        Box::pin(async move {
            let (delay, (tools, text, card)) = {
                let mut w = world.lock().unwrap();
                w.requests.push(request.clone());
                if let TurnRequest::Resolve { .. } = request {
                    w.card = None;
                }
                let reply = if w.replies.is_empty() {
                    (false, "script exhausted".to_string(), None)
                } else {
                    w.replies.remove(0)
                };
                (w.llm_delay_ms, reply)
            };
            tokio::select! {
                _ = cancel.cancelled() => return Ok(TurnOutcome::Interrupted { reply: None }),
                _ = tokio::time::sleep(Duration::from_millis(delay)) => {}
            }
            if tools {
                sink(LlmEvent::ToolRound);
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            if let Some(card) = card {
                world.lock().unwrap().card = Some(card.clone());
                return Ok(TurnOutcome::AwaitingApproval { card });
            }
            let mut said = String::new();
            for word in text.split_inclusive(' ') {
                if cancel.is_cancelled() {
                    return Ok(TurnOutcome::Interrupted {
                        reply: Some(said.trim().to_string()),
                    });
                }
                said.push_str(word);
                sink(LlmEvent::TextDelta(word.to_string()));
                tokio::task::yield_now().await;
            }
            Ok(TurnOutcome::Completed { reply: Some(text) })
        })
    }

    fn truncate_reply(&self, heard: String) -> BoxFuture<'_, bool> {
        self.0.lock().unwrap().truncated.push(heard);
        Box::pin(async { true })
    }
}

struct FakeApprovals(W);
impl Approvals for FakeApprovals {
    fn pending(&self) -> BoxFuture<'_, Option<Value>> {
        let card = self.0.lock().unwrap().card.clone();
        Box::pin(async move { card })
    }
    fn held_utterance(&self, text: String) -> BoxFuture<'_, ()> {
        self.0.lock().unwrap().held.push(text);
        Box::pin(async {})
    }
}

struct Harness {
    world: W,
    tx: mpsc::Sender<Inbound>,
    rx: mpsc::UnboundedReceiver<Outbound>,
    seen: Vec<Value>,
}

fn start(world: World) -> Harness {
    start_with(
        world,
        CallConfig {
            partial_interval_ms: 0,
            ..CallConfig::default()
        },
    )
}

fn start_with(world: World, config: CallConfig) -> Harness {
    let detector = Box::new(Endpointer::new(
        config.endpoint_config(),
        EnergyVad::new(EnergyVadConfig::default()),
    ));
    start_detecting(world, config, detector)
}

fn start_detecting(world: World, config: CallConfig, detector: Box<dyn TurnDetector>) -> Harness {
    let world = Arc::new(Mutex::new(world));
    let providers = Providers {
        stt: Arc::new(FakeStt(Arc::clone(&world))),
        tts: Arc::new(FakeTts(Arc::clone(&world))),
        llm: Arc::new(FakeLlm(Arc::clone(&world))),
        approvals: Arc::new(FakeApprovals(Arc::clone(&world))),
        observer: None,
    };
    let (tx, in_rx) = mpsc::channel(4096);
    let (out_tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(run(
        "call-1".into(),
        config,
        providers,
        detector,
        in_rx,
        out_tx,
    ));
    Harness {
        world,
        tx,
        rx,
        seen: Vec::new(),
    }
}

fn tone(ms: u32, amplitude: f32) -> Vec<i16> {
    (0..(RATE * ms / 1000) as usize)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            (amplitude * 32767.0 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()) as i16
        })
        .collect()
}

fn quiet(ms: u32) -> Vec<i16> {
    (0..(RATE * ms / 1000) as usize)
        .map(|i| ((i * 7919) % 41) as i16 - 20)
        .collect()
}

fn utterance() -> Vec<i16> {
    let mut audio = quiet(300);
    audio.extend(tone(900, 0.3));
    audio.extend(quiet(1_000));
    audio
}

impl Harness {
    async fn speak(&self, audio: &[i16]) {
        for frame in audio.chunks(320) {
            let bytes: Vec<u8> = frame.iter().flat_map(|s| s.to_le_bytes()).collect();
            self.tx.send(Inbound::Audio(bytes.into())).await.unwrap();
        }
    }

    /// Speak at the pace a microphone would: one 20 ms frame per 20 ms.
    async fn speak_live(&self, audio: &[i16]) {
        let mut next = tokio::time::Instant::now();
        for frame in audio.chunks(320) {
            let bytes: Vec<u8> = frame.iter().flat_map(|s| s.to_le_bytes()).collect();
            self.tx.send(Inbound::Audio(bytes.into())).await.unwrap();
            next += Duration::from_millis(20);
            tokio::time::sleep_until(next).await;
        }
    }

    async fn control(&self, event: Value) {
        self.tx
            .send(Inbound::Control(event.to_string()))
            .await
            .unwrap();
    }

    async fn until(&mut self, stop: impl Fn(&Value) -> bool) -> Value {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let item = tokio::time::timeout_at(deadline, self.rx.recv())
                .await
                .unwrap_or_else(|_| panic!("timed out; saw {:#?}", self.seen))
                .expect("pipeline running");
            let event = match item {
                Outbound::Event(event) => serde_json::to_value(&event).unwrap(),
                Outbound::Audio(bytes) => json!({"type": "<audio>", "bytes": bytes.len()}),
            };
            self.seen.push(event.clone());
            if stop(&event) {
                return event;
            }
        }
    }
}

fn is(kind: &'static str) -> impl Fn(&Value) -> bool {
    move |e| e["type"] == kind
}

#[tokio::test(start_paused = true)]
async fn opening_a_call_warms_the_transcriber_before_anyone_speaks() {
    let mut h = start(World::default());
    h.until(is("session.created")).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while h.world.lock().unwrap().stt_calls == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no warm-up transcription"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The warm-up's words go nowhere: no turn, no response.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(h.world.lock().unwrap().stt_calls, 1);
    assert!(h.world.lock().unwrap().requests.is_empty());
    assert!(!h.seen.iter().any(|e| e["type"] == "response.created"));
}

#[tokio::test(start_paused = true)]
async fn a_host_can_turn_the_warm_up_off() {
    let mut h = start_with(
        World::default(),
        CallConfig {
            warm_stt: false,
            ..CallConfig::default()
        },
    );
    h.until(is("session.created")).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(h.world.lock().unwrap().stt_calls, 0);
}

#[tokio::test(start_paused = true)]
async fn a_turn_is_heard_and_answered_a_sentence_at_a_time() {
    let mut h = start(World {
        transcript: "what is running".into(),
        replies: vec![(false, "Two are running. Both are idle.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let done = h.until(is("response.done")).await;
    assert_eq!(done["status"], "completed");
    let pieces: Vec<&Value> = h
        .seen
        .iter()
        .filter(|e| e["type"] == "response.audio.start")
        .collect();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0]["text"], "Two are running.");
    assert!(done["metrics"]["speech_end_to_first_audio_ms"].is_u64());
    assert_eq!(
        h.world.lock().unwrap().requests,
        vec![TurnRequest::Utterance("what is running".into())]
    );
}

#[tokio::test(start_paused = true)]
async fn a_tool_round_before_any_text_is_covered_by_the_filler() {
    let mut h = start(World {
        transcript: "check the build".into(),
        replies: vec![(true, "It passed.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let first = h.until(is("response.audio.start")).await;
    assert_eq!(first["text"], "One moment.");
    let done = h.until(is("response.done")).await;
    assert_eq!(done["metrics"]["filler"], true);
    assert_eq!(done["metrics"]["tool_rounds"], 1);
    // The filler is not part of the reply.
    assert_eq!(done["text"], "It passed.");
}

#[tokio::test(start_paused = true)]
async fn talking_over_the_reply_stops_it_and_keeps_what_was_heard() {
    let mut h = start(World {
        transcript: "tell me everything".into(),
        replies: vec![(
            false,
            "First one here. Second one here. Third one here.".into(),
            None,
        )],
        tts_delay_ms: 300,
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let first = h.until(is("response.audio.start")).await;
    let id = first["response_id"].clone();
    h.control(json!({"type": "output_audio.started", "response_id": id, "index": 0}))
        .await;
    let mut over = quiet(100);
    over.extend(tone(800, 0.5));
    h.speak(&over).await;
    h.until(is("output_audio.clear")).await;
    let done = h
        .until(|e| e["type"] == "response.done" && e["response_id"] == id)
        .await;
    assert_eq!(done["status"], "interrupted");
    assert_eq!(done["text"], "First one here.");
    assert_eq!(
        h.world.lock().unwrap().truncated,
        vec!["First one here.".to_string()]
    );
}

#[tokio::test(start_paused = true)]
async fn a_short_sound_over_the_reply_does_not_stop_it() {
    let mut h = start(World {
        transcript: "go on".into(),
        replies: vec![(false, "One. Two. Three.".into(), None)],
        tts_delay_ms: 200,
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    h.until(is("response.audio.start")).await;
    // "mm": voice, but well under the barge-in bar.
    let mut mm = quiet(50);
    mm.extend(tone(300, 0.5));
    mm.extend(quiet(900));
    h.speak(&mm).await;
    let done = h.until(is("response.done")).await;
    assert_eq!(done["status"], "completed");
    assert!(!h.seen.iter().any(|e| e["type"] == "output_audio.clear"));
}

#[tokio::test(start_paused = true)]
async fn nothing_said_resolves_a_card_and_the_button_does() {
    let card = json!({"id": "card-1", "kind": "anything"});
    let mut h = start(World {
        transcript: "type ls in my shell".into(),
        replies: vec![
            (false, String::new(), Some(card.clone())),
            (false, "Done.".into(), None),
        ],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let shown = h.until(is("approval.pending")).await;
    assert_eq!(shown["card"], card);
    let done = h.until(is("response.done")).await;
    assert_eq!(done["status"], "pending_approval");

    h.world.lock().unwrap().transcript = "yes, approve it".into();
    h.speak(&utterance()).await;
    let reminder = h.until(is("response.done")).await;
    assert!(reminder["text"]
        .as_str()
        .unwrap()
        .contains("on your screen"));
    {
        let w = h.world.lock().unwrap();
        assert_eq!(
            w.requests.len(),
            1,
            "the spoken yes never reached the model"
        );
        assert_eq!(w.held, vec!["yes, approve it".to_string()]);
        assert!(w.card.is_some(), "the card still waits");
    }

    h.control(json!({"type": "action.resolve", "id": "card-1", "approve": true}))
        .await;
    let resolved = h.until(is("approval.resolved")).await;
    assert_eq!(resolved["approved"], true);
    let done = h.until(is("response.done")).await;
    assert_eq!(done["text"], "Done.");
    assert_eq!(
        h.world.lock().unwrap().requests[1],
        TurnRequest::Resolve {
            card_id: "card-1".into(),
            approve: true
        }
    );
}

#[tokio::test(start_paused = true)]
async fn a_button_for_a_card_that_is_not_waiting_is_refused() {
    let mut h = start(World::default());
    h.until(is("session.created")).await;
    h.control(json!({"type": "action.resolve", "id": "nope", "approve": true}))
        .await;
    h.until(is("error")).await;
    assert!(h.world.lock().unwrap().requests.is_empty());
}

/// Two phrases with a pause between them that is shorter than the end of
/// the turn.
fn two_phrases() -> Vec<i16> {
    let mut audio = quiet(300);
    audio.extend(tone(1_200, 0.3));
    audio.extend(quiet(400));
    audio.extend(tone(1_000, 0.3));
    audio.extend(quiet(1_000));
    audio
}

fn warm_off(world: World) -> Harness {
    start_with(
        world,
        CallConfig {
            warm_stt: false,
            ..CallConfig::default()
        },
    )
}

#[tokio::test(start_paused = true)]
async fn a_turn_is_transcribed_in_chunks_while_it_is_spoken() {
    let mut h = warm_off(World {
        chunk_texts: vec![
            Some("check the build".into()),
            Some("on the dev stack".into()),
        ],
        replies: vec![(false, "It passed.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak_live(&two_phrases()).await;
    // The first phrase's words are out as a caption before the turn ends.
    let caption = h
        .until(|e| {
            e["type"] == "conversation.item.input_audio_transcription.partial"
                || e["type"] == "input_audio_buffer.speech_stopped"
        })
        .await;
    assert_eq!(caption["text"], "check the build", "{:#?}", h.seen);
    let done = h.until(is("response.done")).await;
    assert_eq!(done["status"], "completed");
    let w = h.world.lock().unwrap();
    assert_eq!(
        w.requests,
        vec![TurnRequest::Utterance(
            "check the build on the dev stack".into()
        )]
    );
    assert_eq!(w.whole_calls, 0, "no whole-clip transcription");
    assert_eq!(
        w.contexts,
        vec![String::new(), "check the build".to_string()],
        "each chunk is told the words before it"
    );
}

#[tokio::test(start_paused = true)]
async fn the_transcript_is_ready_when_the_turn_ends_not_after() {
    // A slow transcriber: each chunk takes 300 ms. Streamed, the first
    // phrase is done while the second is spoken, and at the end only the
    // second is waited for (its pause is a chunk of its own).
    let mut h = warm_off(World {
        chunk_texts: vec![Some("one".into()), Some("two".into())],
        chunk_delay_ms: 300,
        replies: vec![(false, "Okay.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak_live(&two_phrases()).await;
    let done = h.until(is("response.done")).await;
    let stt_ms = done["metrics"]["stt_ms"].as_u64().expect("stt_ms");
    assert!(stt_ms < 300, "transcript waited {stt_ms} ms after the turn");
    assert_eq!(
        h.world.lock().unwrap().requests,
        vec![TurnRequest::Utterance("one two".into())]
    );
}

#[tokio::test(start_paused = true)]
async fn a_failed_chunk_falls_back_to_the_whole_clip() {
    let mut h = warm_off(World {
        transcript: "the whole thing".into(),
        chunk_texts: vec![None],
        replies: vec![(false, "Right.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&two_phrases()).await;
    let done = h.until(is("response.done")).await;
    assert_eq!(done["status"], "completed");
    let w = h.world.lock().unwrap();
    assert_eq!(w.whole_calls, 1);
    assert_eq!(
        w.requests,
        vec![TurnRequest::Utterance("the whole thing".into())]
    );
}

#[tokio::test(start_paused = true)]
async fn a_chunk_that_heard_no_words_adds_none() {
    let mut h = warm_off(World {
        chunk_texts: vec![Some("[BLANK_AUDIO]".into()), Some("hello".into())],
        replies: vec![(false, "Hi.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&two_phrases()).await;
    h.until(is("response.done")).await;
    assert_eq!(
        h.world.lock().unwrap().requests,
        vec![TurnRequest::Utterance("hello".into())]
    );
}

#[tokio::test(start_paused = true)]
async fn a_spoken_yes_streamed_in_chunks_still_never_approves() {
    let card = json!({"id": "card-9", "kind": "anything"});
    let mut h = warm_off(World {
        transcript: "restart the service".into(),
        replies: vec![(false, String::new(), Some(card))],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    h.until(is("approval.pending")).await;
    h.until(is("response.done")).await;
    h.world.lock().unwrap().chunk_texts = vec![Some("yes".into()), Some("approve it".into())];
    h.speak(&two_phrases()).await;
    let reminder = h.until(is("response.done")).await;
    assert!(reminder["text"]
        .as_str()
        .unwrap()
        .contains("on your screen"));
    let w = h.world.lock().unwrap();
    assert_eq!(
        w.requests.len(),
        1,
        "the spoken yes never reached the model"
    );
    assert_eq!(w.held, vec!["yes approve it".to_string()]);
    assert!(w.card.is_some(), "the card still waits");
}

#[tokio::test(start_paused = true)]
async fn whole_clip_mode_transcribes_the_turn_as_one_clip() {
    let mut h = start_with(
        World {
            transcript: "check the build on the dev stack".into(),
            replies: vec![(false, "It passed.".into(), None)],
            ..World::default()
        },
        CallConfig {
            stt_mode: SttMode::Whole,
            warm_stt: false,
            ..CallConfig::default()
        },
    );
    h.until(is("session.created")).await;
    h.speak(&two_phrases()).await;
    h.until(is("response.done")).await;
    let w = h.world.lock().unwrap();
    assert!(w.contexts.is_empty(), "no chunks in whole-clip mode");
    assert!(w.whole_calls >= 1);
    assert_eq!(
        w.requests,
        vec![TurnRequest::Utterance(
            "check the build on the dev stack".into()
        )]
    );
}

#[tokio::test(start_paused = true)]
async fn a_chunk_that_loops_is_retried_without_its_context() {
    let mut h = warm_off(World {
        chunk_texts: vec![
            Some("check the build".into()),
            // Given the first chunk as its prompt, the second loops on it.
            Some("check the build check the build check the build check the build".into()),
            Some("and the tests".into()),
        ],
        replies: vec![(false, "Okay.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&two_phrases()).await;
    h.until(is("response.done")).await;
    let w = h.world.lock().unwrap();
    assert_eq!(
        w.requests,
        vec![TurnRequest::Utterance(
            "check the build and the tests".into()
        )]
    );
    assert_eq!(
        w.contexts,
        vec![String::new(), "check the build".into(), String::new()],
        "the retry went without the context"
    );
}

#[tokio::test(start_paused = true)]
async fn the_call_opens_with_the_protocol_version() {
    let mut h = start(World::default());
    let created = h.until(is("session.created")).await;
    assert_eq!(created["protocol"], voicepipe::PROTOCOL_VERSION);
    assert_eq!(created["sample_rate"], RATE);
}

#[tokio::test(start_paused = true)]
async fn an_oversized_audio_frame_is_dropped_and_reported_once() {
    let mut h = warm_off(World::default());
    h.until(is("session.created")).await;
    let big = Bytes::from(vec![0u8; voicepipe::MAX_AUDIO_FRAME_BYTES + 2]);
    for _ in 0..3 {
        h.tx.send(Inbound::Audio(big.clone())).await.unwrap();
    }
    h.control(json!({"type": "ping"})).await;
    h.until(is("pong")).await;
    let errors: Vec<_> = h.seen.iter().filter(|e| e["type"] == "error").collect();
    assert_eq!(errors.len(), 1, "{:#?}", h.seen);
}

#[test]
fn a_provider_error_carries_its_source() {
    use std::error::Error;
    let io = std::io::Error::other("connection reset");
    let error = ProviderError::with_source("transcriber unreachable", io);
    assert_eq!(error.to_string(), "transcriber unreachable");
    assert_eq!(error.source().unwrap().to_string(), "connection reset");
    let plain: ProviderError = "timed out".into();
    assert_eq!(plain.message(), "timed out");
    assert!(plain.source().is_none());
}

#[tokio::test(start_paused = true)]
async fn a_short_word_while_the_reply_is_being_thought_of_is_a_turn() {
    let mut h = warm_off(World {
        transcript: "delete the branch".into(),
        replies: vec![
            (false, "Deleting it now.".into(), None),
            (false, "Stopped.".into(), None),
        ],
        llm_delay_ms: 3_000,
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let first = h.until(is("response.created")).await;
    // "no": shorter than the barge-in bar, but nothing is playing yet.
    h.world.lock().unwrap().transcript = "no".into();
    let mut no = quiet(50);
    no.extend(tone(300, 0.3));
    no.extend(quiet(1_000));
    h.speak(&no).await;
    let cut = h
        .until(|e| e["type"] == "response.done" && e["response_id"] == first["response_id"])
        .await;
    assert_eq!(cut["status"], "interrupted");
    let done = h
        .until(|e| e["type"] == "response.done" && e["response_id"] != first["response_id"])
        .await;
    assert_eq!(done["text"], "Stopped.");
    assert_eq!(
        h.world.lock().unwrap().requests,
        vec![
            TurnRequest::Utterance("delete the branch".into()),
            TurnRequest::Utterance("no".into()),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn the_first_reply_of_a_call_is_cut_where_the_client_says() {
    // The client reports piece 1 playing before the clock estimate would
    // have it start: what was heard follows the report, from the first
    // response on.
    let mut h = warm_off(World {
        transcript: "tell me everything".into(),
        replies: vec![(
            false,
            "First one here. Second one here. Third one here.".into(),
            None,
        )],
        tts_delay_ms: 100,
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let first = h.until(is("response.audio.start")).await;
    let id = first["response_id"].clone();
    h.until(|e| e["type"] == "response.audio.start" && e["index"] == 1)
        .await;
    for index in [0, 1] {
        h.control(json!({"type": "output_audio.started", "response_id": id, "index": index}))
            .await;
    }
    let mut over = quiet(100);
    over.extend(tone(800, 0.5));
    h.speak(&over).await;
    let done = h
        .until(|e| e["type"] == "response.done" && e["response_id"] == id)
        .await;
    assert_eq!(done["status"], "interrupted");
    assert_eq!(done["text"], "First one here. Second one here.");
}

#[tokio::test(start_paused = true)]
async fn cancelling_while_transcribing_stops_the_transcription() {
    let mut h = warm_off(World {
        chunk_texts: vec![Some("one".into()), Some("two".into())],
        chunk_delay_ms: 2_000,
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&two_phrases()).await;
    h.until(is("input_audio_buffer.speech_stopped")).await;
    h.control(json!({"type": "response.cancel"})).await;
    let done_at_cancel = h.world.lock().unwrap().chunks_done;
    tokio::time::sleep(Duration::from_secs(10)).await;
    let w = h.world.lock().unwrap();
    assert_eq!(
        w.chunks_done, done_at_cancel,
        "a chunk ran on after the cancel"
    );
    assert!(w.requests.is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_button_resolves_a_card_raised_elsewhere() {
    let card = json!({"id": "card-9"});
    let mut h = warm_off(World {
        card: Some(card),
        replies: vec![(false, "Done.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.control(json!({"type": "action.resolve", "id": "card-9", "approve": false}))
        .await;
    let resolved = h.until(is("approval.resolved")).await;
    assert_eq!(resolved["approved"], false);
    assert_eq!(
        h.world.lock().unwrap().requests,
        vec![TurnRequest::Resolve {
            card_id: "card-9".into(),
            approve: false
        }]
    );
}

#[tokio::test(start_paused = true)]
async fn response_ids_belong_to_their_call() {
    let mut h = warm_off(World {
        transcript: "hello".into(),
        replies: vec![(false, "Hi.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    let created = h.until(is("response.created")).await;
    assert_eq!(created["response_id"], "call-1/resp_1");
}

#[tokio::test(start_paused = true)]
async fn a_silent_client_still_hears_the_reply_end() {
    // No audio after the turn and no playback reports: the reply's
    // estimated end alone moves the call from speaking back to listening.
    let mut h = warm_off(World {
        transcript: "hello".into(),
        replies: vec![(false, "Hi.".into(), None)],
        ..World::default()
    });
    h.until(is("session.created")).await;
    h.speak(&utterance()).await;
    h.until(|e| e["type"] == "call.state" && e["state"] == "speaking")
        .await;
    h.until(|e| e["type"] == "call.state" && e["state"] == "listening")
        .await;
}

/// Speech is anything louder than room noise. Unlike the energy detector it
/// never learns a long steady tone as the noise floor.
struct Loudness;
impl Vad for Loudness {
    fn is_speech(&mut self, frame: &[i16]) -> bool {
        frame.iter().any(|s| s.unsigned_abs() > 1_000)
    }
    fn set_playback(&mut self, _playing: bool) {}
}

#[tokio::test(start_paused = true)]
async fn a_turn_cut_off_at_its_longest_ends_where_it_was_cut() {
    let config = CallConfig {
        warm_stt: false,
        ..CallConfig::default()
    };
    let detector = Box::new(Endpointer::new(config.endpoint_config(), Loudness));
    let mut h = start_detecting(
        World {
            transcript: "a very long story".into(),
            replies: vec![(false, "Go on.".into(), None)],
            ..World::default()
        },
        config,
        detector,
    );
    h.until(is("session.created")).await;
    // With the pre-roll the turn reaches 30 s, the most, inside the tone;
    // the silence after it starts nothing new.
    let mut audio = quiet(300);
    audio.extend(tone(29_900, 0.3));
    audio.extend(quiet(1_500));
    h.speak(&audio).await;
    let done = h.until(is("response.done")).await;
    assert_eq!(done["metrics"]["endpoint_ms"], 0, "{done:#?}");
}
