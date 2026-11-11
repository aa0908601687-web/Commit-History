# V6 coverage boundary audit — updated 2026-11-03

## Latest verified result (before this history-sync commit)

- PR #25 commit `40fd4e9ec48638761f6d9ba167929e913fa5f444`; [Actions coverage run 41479960938](https://github.com/aa0908601687-web/Commit-History/actions/runs/41479960938).
- 325 unit tests executed; **100.00% overall statement and branch coverage** for the repository coverage command: 1,867 statements, 0 missed; 542 branches, 0 missed.
- Some legacy ABE execution paths were covered via mock or inert test doubles. They are **not** live neural training, pretrained inference, or V6 integration.
- Existing V6 research workflow tested a Bayesian-style heuristic and local 8-qubit obsolete variational simulation. This does not establish a predictive advantage, deployed trained AI, or production E2E readiness.

## Historical context

Earlier commit `31cc3c09b9a5aaefd0b15d402392ac4f365875c9` and [coverage run 41472771972](https://github.com/aa0908601687-web/Commit-History/actions/runs/41472771972) reported 92%, 162 missed statements and 11 partial branches, with the global 100% threshold failing. Subsequent isolated tests closed these execution coverage gaps without modifying production selection.

## Current integration revision requires independent rerun

This PR branch now synchronizes the three existing history CSV files from main through draw **115000088** (first zone 04 06 16 19 21 37; second zone 07). Previous results must not be presented as a successful post-sync run until the new head's GitHub Actions complete and are checked.

The live official API probe remains read-only and does not persist predictions or history updates. Its latest previously verified run reported no new draws beyond 115000088; trained AI, full production E2E and predictive advantage remain **NOT VERIFIED**. PR #23 remains separate, ABE is excluded from V6, and real QPU is excluded. No evidence here licenses a main merge.

## Acceptance criteria

For the revised PR head, independently check: coverage, full suite, official API read-only probe, three-model cutoff/parity, local non-variational CUDA-Q comparison, and artifacts. Mark any missing/failed validation explicitly and do not claim production readiness from unit coverage alone.
