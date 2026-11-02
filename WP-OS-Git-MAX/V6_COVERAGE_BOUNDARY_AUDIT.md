# V6 coverage boundary audit — 2026-11-02

Source: PR #25 commit `31cc3c09b9a5aaefd0b15d402392ac4f365875c9`; GitHub Actions coverage run [41472771972](https://github.com/aa0908601687-web/Commit-History/actions/runs/41472771972).

- Tests executed: 283.
- Overall repository statement/branch coverage: **92%**; **162 missed statements**, **11 partial branches**; global `--fail-under=100` **FAILED**.
- **Every non-`abe_*` file displayed in the reported coverage table reached 100%** for that run.
- All uncovered statements and partial branches in that report were in legacy `abe_*` source files.
- This is **not** evidence that the entire repository reached 100% or that every possible behavior is correct.
- It does **not** authorize including any ABE code in the V6 runtime.
- V6 keeps ABE and hardware QPU excluded. The verified V6 bridge uses local QAOA simulation only.
- Live read-only official API probe [41472771891](https://github.com/aa0908601687-web/Commit-History/actions/runs/41472771891) succeeded, with observed cutoff `115000088`.
- For repository-wide 100%, the legacy ABE files also need legitimate coverage or an explicit separately reviewed repository scope decision. **Do not lower the global threshold, delete files, or silently exclude ABE from the existing report to claim success.**

Coverage is test execution evidence, not proof of predictive advantage.
