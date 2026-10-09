# Changelog

All notable changes to this crate are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the
version is 0.x, a minor release may break the API or the wire protocol, and a
patch release only fixes.

## [Unreleased]

## [0.1.0]

First release.

- `run`: one live call over two channels, with turn-taking, streaming
  transcription by chunks (or whole clips), the reply spoken a sentence at a
  time as it streams, a filler during tool rounds, barge-in that keeps only
  what was heard, and per-response metrics.
- Provider traits `Stt`, `Tts`, `Llm`, `Approvals`, `Vad` and `TurnDetector`.
- `EarshotVad` (default feature `earshot`), `EnergyVad` and `Endpointer`.
- The approval invariant: an approval card is resolved only by an
  `action.resolve` control frame, never by speech.
- The wire protocol in `protocol`.

[Unreleased]: https://github.com/TheDancingDeveloper-org/voicepipe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheDancingDeveloper-org/voicepipe/releases/tag/v0.1.0
