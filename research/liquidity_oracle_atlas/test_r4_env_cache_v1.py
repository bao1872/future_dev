"""Formal cache contract for run_environment_m15 (U3 / U9-FIX1).

These are the tests the remote reviewer required:

A. fresh vs cached are identical
   - features equal, including the exact NaN mask
   - finite values exact (allclose)
   - touch_bits exact-equal
   - geom_by_decision exact semantic-equal (content + order)

B. cache hit
   - after a cache exists, monkeypatch the expensive compute path to raise
   - a second call must still succeed (proves it read the cache, not recomputed)

C. stale identity
   - changing the cache identity makes the old cache NOT silently reusable
   - (old identity -> load returns None; if compute is also patched to raise,
      the call raises, proving the stale cache was not used)
"""

import numpy as np
import pytest

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    _compute_r4_env,
    _load_r4_env_cache,
    _r4_env_identity,
    run_environment_m15,
)

SYM = "AU"
MAX_BARS = 2000  # small slice -> fast, still exercises the full env machinery


def _geom_equal(g1, g2) -> bool:
    assert len(g1) == len(g2)

    def _dump(g):
        # NaN-safe render: json.dumps emits NaN/Infinity as literals, so two
        # geometries compare equal iff their content (incl. NaN positions) matches.
        import json as _json

        import numpy as _np

        def _d(o):
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return str(o)

        return _json.dumps(g, default=_d, sort_keys=True)

    for a, b in zip(g1, g2):
        assert _dump(a) == _dump(b)
    return True


def test_fresh_vs_cached_equal():
    cached = run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=True)
    fresh = run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=False)

    # features: shape + columns
    fr, cr = fresh["features"], cached["features"]
    assert fr.shape == cr.shape
    assert list(fr.columns) == list(cr.columns)

    # features: NaN mask must be identical
    fa = fr.values.astype(float)
    ca = cr.values.astype(float)
    nan_f = np.isnan(fa)
    nan_c = np.isnan(ca)
    assert np.array_equal(nan_f, nan_c), "NaN mask mismatch between fresh and cached"

    # features: finite values
    nz = ~nan_f
    assert np.allclose(fa[nz], ca[nz], equal_nan=False), "finite feature values mismatch"

    # touch_bits: exact
    assert np.array_equal(fresh["touch_bits"], cached["touch_bits"])

    # geometry: exact semantic equal
    assert _geom_equal(fresh["geom_by_decision"], cached["geom_by_decision"])

    # exec_frame length matches
    assert len(fresh["exec_frame"]) == len(cached["exec_frame"])


def test_cache_hit_uses_cache(monkeypatch):
    run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=True)  # build cache

    def _boom(*a, **k):
        raise RuntimeError("expensive compute must NOT run on a cache hit")

    monkeypatch.setattr(
        "research.liquidity_oracle_atlas.build_execution_environment_m15_v1._compute_r4_env",
        _boom,
    )
    out = run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=True)
    assert out is not None
    assert "features" in out
    assert "touch_bits" in out


def test_stale_identity_rejected(monkeypatch):
    run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=True)  # build cache

    def _stale_ident(symbol, max_bars):
        return "STALE-IDENTITY-SHA"

    monkeypatch.setattr(
        "research.liquidity_oracle_atlas.build_execution_environment_m15_v1._r4_env_identity",
        _stale_ident,
    )

    # The stale identity must NOT load the (real) cached artifact.
    assert _load_r4_env_cache(SYM, "STALE-IDENTITY-SHA", MAX_BARS) is None

    # And if compute is also disabled, the call must fail (no silent reuse).
    def _boom(*a, **k):
        raise RuntimeError("stale cache must not be silently reused")

    monkeypatch.setattr(
        "research.liquidity_oracle_atlas.build_execution_environment_m15_v1._compute_r4_env",
        _boom,
    )
    with pytest.raises(RuntimeError):
        run_environment_m15(SYM, max_bars=MAX_BARS, _use_cache=True)
