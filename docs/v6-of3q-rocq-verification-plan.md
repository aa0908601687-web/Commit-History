# V6 OF3Q — Rocq Formal Verification Plan

Status: specification only; no proof completion or end-to-end success claimed.

## Scope
1. OpenFE: feature schema, timestamp cutoffs, and no future-draw leakage.
2. LightGBM: input/output contract, finite scores and stable candidate identifiers.
3. CUDA-Q: 8-qubit bit order, shot decoding, and valid candidate indices.
4. QAOA: objective and constraints must match the exact-search reference.
5. Eight-candidate pool: unique first-zone values in 1..38, stable bijection from bit indices to candidate numbers.
6. Exact search: enumerate all 6-of-8 subsets (28 combinations) and certify argmax under the *same* scoring function and constraints.
7. Three-layer pipeline: typed interfaces, provenance and no stale V3 cache dependencies.
8. Final selection: six distinct first-zone numbers in 1..38 and one second-zone number in 1..8.

## Priority proof obligations
- Candidate uniqueness and range invariants.
- 8-bit mask popcount = 6; decoding preserves selected numbers.
- Exact search returns a feasible solution with score >= every feasible 6-of-8 combination.
- QAOA and exact search use the same objective encoding, coefficient scaling, penalty terms and endianness.
- Data feature timestamps are strictly earlier than the target draw.
- Output validation rejects malformed or missing second-zone predictions.

## Implementation and verification gates
- Inventory actual V6 code paths and pin commit SHA.
- Implement a pure functional reference specification in Rocq, then prove the invariants above.
- Connect production Python/CUDA-Q adapters to the verified reference with property-based and differential tests.
- Test all 28 6-of-8 subsets; test malformed input and bit-order reversals.
- Run CI with Rocq compiler, unit tests and full V6 E2E on actual dependencies.
- Publish logs, proof artifact hashes and a coverage matrix. Block merging on failed gates.

## Exclusions
No ABE, no real quantum QPU, no V3 cache restoration.

Formal proofs cover their stated specifications, not all runtime behavior, third-party library correctness, or any lottery predictive advantage.
