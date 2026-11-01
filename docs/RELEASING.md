# Releasing

1. Update `__version__` in `mgtranscriber/__init__.py` and `version` in `pyproject.toml`, run `uv lock`,
   and commit.
2. `git tag v0.3.0 && git push origin v0.3.0`.
3. Create the GitHub release from the tag by hand (there is no CI). `install.ps1` downloads the source ZIP of
   the latest release (or of `main` when there is none), so nothing else has to be uploaded. Optionally attach:
   - `MGTranscriber-setup-<version>.exe` (Inno Setup, see below),
   - Docker images built locally (`docker compose build`) and pushed to a registry of your choice.

## The setup .exe

The `.exe` is a thin wrapper: it copies the app to `%LOCALAPPDATA%\Programs\mgtranscriber` (no admin
rights), registers it in *Settings → Apps*, creates shortcuts and runs `install.ps1 -NoShortcuts
-NoAppsEntry` in a console window. Uninstalling runs `install.ps1 -Uninstall` (removes `.venv`) and
keeps `output\`, `data\` and `.env`. Build locally (Windows):

```
winget install JRSoftware.InnoSetup
"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" /DAppVersion=0.3.0 installer\mgtranscriber.iss
```

## Code signing

Unsigned `.exe` files (and `install.ps1` run from a download) trigger SmartScreen ("Windows protected
your PC"). Options, as of 10/2026: SignPath Foundation signs open-source projects for free, Azure Artifact
Signing is closed to individuals outside the USA and Canada, and the Microsoft Store is the alternative
when a signed package matters more than a plain `.exe`.

| option | cost | for a natural person in CZ |
|---|---|---|
| **SignPath Foundation** | free for OSS | **yes**; recommended |
| Azure Artifact Signing (Trusted Signing) | $9.99/month | no: individuals only from the USA and Canada |
| OV/EV certificate from a CA | ~$120–620/year, max. 460 days validity | yes, but EV no longer skips SmartScreen |
| Microsoft Store (MSIX) | one-time fee for an individual account | yes; the Store re-signs the package |

### SignPath Foundation

> **Status: not set up.** The repository has no CI, and SignPath only signs builds made by a CI workflow, so the
> steps below cannot be completed until a GitHub Actions release workflow exists. Releases are unsigned until then.

Conditions (https://signpath.org/terms): OSI license (Apache-2.0 qualifies), no proprietary
components in what is signed, only released versions, built from the public repository by a reproducible, documented build,
an uninstaller (the `.exe` has one; `install.ps1 -Uninstall` for the script install), a privacy
statement with opt-out for data sent elsewhere (cloud engines are opt-in), MFA for all maintainers on
GitHub and SignPath, and a **"Code signing policy"** section on the home and download pages, e.g.:

> **Code signing policy.** Free code signing provided by [SignPath.io](https://about.signpath.io),
> certificate by [SignPath Foundation](https://signpath.org). Committers and reviewers: the members of
> this repository. Approvers: <maintainer>. Privacy: this program does not transfer any information to
> other networked systems unless specifically requested by the user (cloud engines, cloud LLMs).

Steps:

1. Make the repository public, enable MFA, add the policy section to the README and the release page.
2. Apply at https://signpath.org/apply (project name, repository URL, what is signed: the setup `.exe`).
3. After approval, SignPath creates an organization and a project. Install the SignPath GitHub app on
   the repository, create a signing policy (`release-signing`), and add the CI user's API token as the
   repository secret `SIGNPATH_API_TOKEN` (once a CI exists).
4. SignPath signs only builds it can verify came from the repository's CI (a GitHub Actions workflow that
   uploads the unsigned `.exe` as an artifact). This project has no CI at the moment, so one has to be added
   before signing can work.

`install.ps1` itself can be signed the same way (Authenticode works for `.ps1`), so that
`irm … | iex` users and execution policy `AllSigned` see a valid signature. Sign it in the same
signing request as the `.exe` (SignPath supports PowerShell scripts in a ZIP artifact).

SmartScreen reputation is still built per file hash, so even signed releases may show a warning for
the first days after a release.

## MCP Registry

`server.json` in the repository root describes the MCP server for the
[MCP Registry](https://registry.modelcontextprotocol.io) (preview as of 10/2026). The registry stores only
this metadata; the code comes from PyPI. Nothing is published automatically.

What to fill in before the first publication:

1. `server.json` already uses `io.github.michal1609/mgtranscriber`; the namespace is verified through GitHub login.
2. The registry entry points at a **PyPI package** (`identifier: mgtranscriber`), so the package has to be on
   PyPI first, and `version` in both places of `server.json` must be the released version
   (`__version__`). Caveat: `pip install mgtranscriber` from PyPI gets the default torch wheels
   (CPU only on Windows); the CUDA build comes from the `cu128` index configured for `uv sync`. Either document
   `uvx`/`uv tool install` with that index, or publish only after this is solved. Check this before announcing.
3. PyPI ownership is verified by a marker in the package's long description: add
   `<!-- mcp-name: io.github.michal1609/mgtranscriber -->` to `README.md` and set `readme = "README.md"` in
   `pyproject.toml` (the name must equal `name` in `server.json`).
4. `description` in `server.json` is limited to 100 characters.

Publish with [`mcp-publisher`](https://github.com/modelcontextprotocol/registry/releases) (binary, or
`brew install mcp-publisher`):

```
mcp-publisher login github      # device code in the browser, authorizes the io.github.<user> namespace
mcp-publisher publish           # reads ./server.json; then look the name up in the registry
```

Every release: update `version` in `server.json` (also `packages[0].version`), upload to PyPI, run
`mcp-publisher publish` again. Validate the file without publishing against the schema named in `$schema`
(for example with `jsonschema`). It is checked by `tests/test_mcp.py::test_server_json`.
