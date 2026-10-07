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
use voxcall::{
    run, Approvals, BoxFuture, CallConfig, Clip, Endpointer, EnergyVad, EnergyVadConfig, Inbound,
    Llm, LlmEvent, LlmSink, Outbound, ProviderError, Providers, Stt, Tts, TurnOutcome, TurnRequest,
};

const RATE: u32 = 16_000;

#[derive(Default)]
struct World {
    transcript: String,
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
        let text = self.0.lock().unwrap().transcript.clone();
        Box::pin(async move { Ok(text) })
    }
}

fn wav(ms: u32) -> Bytes {
    let samples = vec![0i16; (24_000 * ms / 1000) as usize];
    Bytes::from(voxcall::audio::wav_from_pcm16(&samples, 24_000))
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
            let (tools, text, card) = {
                let mut w = world.lock().unwrap();
                w.requests.push(request.clone());
                if let TurnRequest::Resolve { .. } = request {
                    w.card = None;
                }
                if w.replies.is_empty() {
                    (false, "script exhausted".to_string(), None)
                } else {
                    w.replies.remove(0)
                }
            };
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
    let world = Arc::new(Mutex::new(world));
    let config = CallConfig {
        partial_interval_ms: 0,
        ..CallConfig::default()
    };
    let detector = Box::new(Endpointer::new(
        config.endpoint_config(),
        EnergyVad::new(EnergyVadConfig::default()),
    ));
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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

#[tokio::test]
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
    let shown = h.until(is("assistant.pending_action")).await;
    assert_eq!(shown["action"], card);
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
    let resolved = h.until(is("assistant.action_resolved")).await;
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

#[tokio::test]
async fn a_button_for_a_card_that_is_not_waiting_is_refused() {
    let mut h = start(World::default());
    h.until(is("session.created")).await;
    h.control(json!({"type": "action.resolve", "id": "nope", "approve": true}))
        .await;
    h.until(is("error")).await;
    assert!(h.world.lock().unwrap().requests.is_empty());
}
