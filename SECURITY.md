# Security policy

## Supported versions

Only the latest release gets security fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:
**Security → Report a vulnerability** on the repository page
(GitHub private vulnerability reporting). You should get an answer within 7 days. Please give
us 90 days, or until a fix is released, before you publish details.

Include the version, how the server was started (`--host`, API key set or not, Docker, reverse
proxy) and steps to reproduce.

## Scope and threat model

MGTranscriber is designed to run on your own computer or home server.

- By default the server listens on `127.0.0.1` only and has no login. It rejects foreign `Host` and
  `Origin` headers (DNS rebinding, cross-site requests).
- Listening on the network (`--host 0.0.0.0`, Docker) **requires** `MGTRANSCRIBER_API_KEY`; without it
  anyone on the network can start jobs, read transcripts and use the GPU. The server warns about this.
- With the server on the network, URLs to private/loopback addresses are refused (SSRF), but local
  file paths can be allowed by configuration; see [docs/API.md](docs/API.md).
- Cloud engines and cloud LLM providers send audio or transcript text to that provider. They are off
  until you add an API key; offline mode disables them entirely.
- Running untrusted media through FFmpeg, PyAV and model code has the usual risks of complex parsers;
  keep those dependencies updated.

Out of scope: problems that need an attacker who already has access to your user account, your `.env`
or the `data/` folder, and denial of service by an authenticated user (the GPU queue is first come,
first served).
