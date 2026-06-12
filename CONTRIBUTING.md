# Contributing

Thanks for your interest in CRISPI / STIMscope. This file is the
short version of "how to work on the codebase." For platform context
and architecture, start with [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md)
and the [wiki](https://github.com/Aharoni-Lab/STIMscope/wiki).

## Before you start

- The repo is **GPL-3.0**. Contributions must be compatible. By
  opening a PR you agree your contribution can be redistributed under
  the same license.
- Hardware / algorithm understanding here is still evolving — phrase
  code comments and docstrings as **"current implementation does X"**,
  not "X is guaranteed." Treat hard-contract language as a smell.
- Do not introduce internal-process breadcrumbs (date-stamped notes,
  ticket-style identifiers, "user-approved" markers) in new code.
  Those belong in commit messages, not source.

## Dev environment

Build the image once, then iterate on the host:

```bash
git clone https://github.com/Aharoni-Lab/STIMscope.git
cd STIMscope
./build.sh                        # auto-detects JetPack version
```

The source tree at `STIMscope/STIMViewer_CRISPI/` is bind-mounted into
the container, so Python edits take effect on the next run — no rebuild
needed for code changes. Rebuild is required only when `requirements.txt`,
`Dockerfile`, `entrypoint.sh`, or `ZMQ_sender_mask/main.cpp` change.

Tests, on the host (faster than rebuilding the container):

```bash
# Install dev deps once
pip install -r requirements-dev.txt PyQt5~=5.15

# Run the layers you touched
pytest -q tests/L1_algorithms/
pytest -q tests/L2_orchestration/
pytest -q tests/L3_hardware/
pytest -q tests/L3_5_split_first/
pytest -q tests/L5_UI/
```

CI runs all of the above plus `L3_projector`, `L4_orchestration`,
bandit, and ruff on every push to main and every PR. Hardware-only
paths (CuPy GPU, real IDS Peak, real DMD over I²C, real GPIO) run on a
Jetson via `make test` and are out of scope for CI.

## Workflow

1. **Open or claim an issue first.** Bigger than a typo: file a
   [bug report](https://github.com/Aharoni-Lab/STIMscope/issues/new?template=bug_report.yml)
   or [feature request](https://github.com/Aharoni-Lab/STIMscope/issues/new?template=feature_request.yml).
   Comment to claim — avoids two people doing the same work.
2. **Branch off `main`.** Naming convention: `<short-topic>`, e.g.
   `roi-drag-fix`, `calibration-cleanup`, `wiki-install-edits`.
3. **Commit messages.** First line under 72 chars, imperative mood
   ("fix camera trigger latency", not "fixed it"). Reference the issue
   if it's not already in the PR description.
4. **Run the test layers you touched** before opening the PR — don't
   rely on CI to catch obvious things.
5. **Open a PR against `main`.** The PR template will prompt you for
   the summary, type of change, linked issue, and test plan. Fill it
   out — don't blank it.
6. **CI must be green** before merge. Bandit medium+ severity gates
   the build (anything `bandit` flags must be either fixed or marked
   `# nosec <test-id>` with a one-line rationale).
7. **Squash-merge is the only merge mode.** Branches are auto-deleted
   after merge. Squash-merge keeps history scannable; if you need
   multiple logical units, open multiple PRs.

## Code conventions

- **Formatter**: Black, 88-char line length.
- **Linter**: Ruff with rules `E, F, B, UP, SIM`. Currently advisory
  in CI — don't intentionally introduce new violations.
- **Type hints** in `core/` and all new code. `tests/` is exempt.
- **Hardware code must degrade gracefully**: if `ids_peak` or
  `Jetson.GPIO` is missing, the codepath logs a warning and falls
  back to no-op or simulation. Production code should never raise
  `ImportError` at module load just because hardware isn't present.
- **No `from <module> import *`**. No re-exports unless there's a
  documented backward-compat reason.

## When in doubt

- Architecture question: [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)
- Test layer conventions: same doc, "Test layers" section
- How a feature is wired: the wiki's [Architecture](https://github.com/Aharoni-Lab/STIMscope/wiki/Architecture) and [Hardware-Interfaces](https://github.com/Aharoni-Lab/STIMscope/wiki/Hardware-Interfaces) pages
- Stuck on a bug: [Troubleshooting](https://github.com/Aharoni-Lab/STIMscope/wiki/Troubleshooting) first, then file a bug-report issue with the template

## Licensing

By contributing, you certify that:

1. Your contribution is your original work, or you have the rights
   to submit it under GPL-3.0.
2. You agree the contribution may be distributed under GPL-3.0
   alongside the rest of the project.

There's no CLA. Standard inbound = outbound: your PR commits become
part of the GPL-3.0 codebase.
