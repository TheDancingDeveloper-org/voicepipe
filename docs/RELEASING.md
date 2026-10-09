# Releasing

Releases are published to crates.io by `.github/workflows/release.yml`
through crates.io trusted publishing. The repository holds no crates.io
token.

## A release

1. On a branch, bump `version` in `Cargo.toml` (and `Cargo.lock`, with
   `cargo update -p voicepipe`), and move the `Unreleased` entries in
   `CHANGELOG.md` under a new `## [x.y.z]` heading. While the version is
   0.x, a minor bump may break the API or the wire protocol, and a patch only
   fixes. Raise `PROTOCOL_VERSION` for any wire change a client could notice.
2. Merge it once CI is green.
3. Tag the merged commit and push the tag:

   ```sh
   git tag -a vX.Y.Z -m "voicepipe X.Y.Z" <commit>
   git push origin vX.Y.Z
   ```

4. `release.yml` checks that the tag matches `Cargo.toml` and that the
   changelog has the version, runs the CI gates on the tagged commit, and
   then waits in the `release` environment. A required reviewer approves the
   deployment in the run's page, and the job publishes.

## One-time setup

- The first release, 0.1.0, is published by hand by the account that is to
  own the crate. That establishes ownership on crates.io.
- On crates.io, under the crate's **Settings → Trusted Publishing**, add a
  GitHub publisher: owner `TheDancingDeveloper-org`, repository `voicepipe`,
  workflow `release.yml`, environment `release`.
- In this repository, the `release` environment has a required reviewer and
  admits only `v*` tags.

Pushing the `v0.1.0` tag after the hand publish is safe: the workflow finds
that version on crates.io and skips the publish step.
