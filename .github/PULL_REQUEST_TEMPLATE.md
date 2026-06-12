## Summary

<!-- One or two sentences describing what this PR does and why. -->

## Type of change

- [ ] Bug fix (non-breaking, restores intended behavior)
- [ ] Feature (non-breaking, adds new capability)
- [ ] Refactor (non-breaking, no functional change)
- [ ] Breaking change (changes existing API / config / wire format)
- [ ] Documentation only
- [ ] CI / build / dev tooling only

## Linked issue

<!-- Closes #NNN, or "n/a" if no tracked issue. -->

## Test plan

<!-- What you ran locally. Tick what applies; list what was skipped and why. -->

- [ ] `pytest -q tests/L1_algorithms/` — passes
- [ ] `pytest -q tests/L2_orchestration/` — passes
- [ ] `pytest -q tests/L3_hardware/` (mocked) — passes
- [ ] `pytest -q tests/L3_5_split_first/` — passes (Qt offscreen)
- [ ] `pytest -q tests/L5_UI/` — passes (Qt offscreen)
- [ ] `make bandit` — clean at medium+
- [ ] Manual smoke test in the GUI on the Jetson
- [ ] Hardware regression check (only if PR touches camera / projector / calibration / recording)

## Notes for the reviewer

<!--
Anything non-obvious:
- Why this approach was chosen over the alternative
- Any hardware-only behavior that wasn't reproduced in CI
- Any open follow-up tracked in docs/IMPLEMENTATION_NOTES.md or as a new issue
-->
