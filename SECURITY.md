# Security policy

## Supported versions

voicepipe is pre-1.0. Security fixes are made against the latest released
minor version only; there is no long-term-support branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

Please use **GitHub's private vulnerability reporting** rather than a public
issue: open the repository's **Security** tab and choose **"Report a
vulnerability"**. That creates a private advisory that only the maintainers
can see, and lets you attach reproduction steps or a patch while the issue
is unfixed.

There is no email or other private channel; GitHub's private reporting is the
route. In a report, include:

- what you found and why it is a security issue, not just a bug;
- steps or a proof of concept to reproduce it;
- the version or commit you tested against;
- any suggested fix, if you have one.

You should get an acknowledgement within a few days. There is no bug-bounty
programme. Please allow a reasonable time to fix a confirmed issue before any
public disclosure.

## What counts

voicepipe is a library. It does not authenticate connections, open sockets or
store anything; the host does all of that before it hands the pipeline a pair
of channels. Worth a report:

- a way for anything a person **says** (audio or its transcript) to resolve
  an approval card, which would break the crate's central invariant;
- a panic, unbounded memory growth or a hang that a client can cause from the
  wire (audio frames or control frames);
- a way for one call's data to reach another call.
