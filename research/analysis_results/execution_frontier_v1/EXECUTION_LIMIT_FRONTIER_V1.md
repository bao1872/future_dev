# E2 Limit Execution Frontier — Closure

**E2_CLOSED**. Strict-through is conservative; Touch is the upper view.

- `SINGLE_ATTEMPT`: Market > fixed Limit (`ROBUST_LIMIT_BEATS_MARKET=False`).
- `REASSESS_NEXT_H`: continuation is separate (`REASSESS_IMPROVES_SINGLE_ATTEMPT=True`, `REASSESS_BEATS_MARKET=True`).
- Full universe: 9,519 gid-region attempts / 9,015 eligible contacts.
- Scalar/vector parity: 16,000 rows, 0 mismatches.
- Synthetic: T1-T11 passed. No best RR selected. P1 and post-E2 axes untouched.
