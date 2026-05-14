import numpy as np
import pytest

# ── Shared helpers ───────────────────────────────────────────────────────────

class FakeEst:
    def __init__(self, imps):
        self.feature_importances_ = np.array(imps, dtype=float)

def make_estimators(n_features, n_estimators=3, seed=42):
    rng = np.random.default_rng(seed)
    return [FakeEst(rng.dirichlet(np.ones(n_features))) for _ in range(n_estimators)]

def top10_logic(f_cols, estimators, label_map):
    """Pure logic of write_feature_importance, no Firebase."""
    imps = np.mean([e.feature_importances_ for e in estimators], axis=0)
    pairs = sorted(zip(f_cols, imps), key=lambda x: -x[1])[:10]
    return [
        {'feature': f, 'importance': round(float(imp), 4), 'label': label_map.get(f, f)}
        for f, imp in pairs
    ]

# ── Feature importance tests ─────────────────────────────────────────────────

def test_top10_returns_at_most_10():
    cols = [f'f{i}' for i in range(50)]
    result = top10_logic(cols, make_estimators(50), {})
    assert len(result) == 10

def test_top10_sorted_descending():
    cols = [f'f{i}' for i in range(20)]
    result = top10_logic(cols, make_estimators(20), {})
    imps = [r['importance'] for r in result]
    assert imps == sorted(imps, reverse=True)

def test_top10_uses_label_map():
    cols = ['pm2_5_lag1', 'hour']
    labels = {'pm2_5_lag1': 'PM2.5 รอบก่อน', 'hour': 'ชั่วโมง'}
    result = top10_logic(cols, make_estimators(2), labels)
    by_feat = {r['feature']: r['label'] for r in result}
    assert by_feat['pm2_5_lag1'] == 'PM2.5 รอบก่อน'
    assert by_feat['hour'] == 'ชั่วโมง'

def test_unknown_feature_uses_raw_name():
    cols = ['mystery_feat']
    result = top10_logic(cols, make_estimators(1), {})
    assert result[0]['label'] == 'mystery_feat'

# ── Anomaly detection tests ──────────────────────────────────────────────────
import pandas as pd

_Z_THRESH  = 3.0
_MIN_STD   = 1.0
_ROLL_WIN  = 20

def detect_logic(pm_values: list, device: str = 'dev1') -> list:
    """Pure detect_anomalies logic — no Firebase, no timestamp."""
    arr = pd.Series(pm_values, dtype=float)
    if len(arr) < _ROLL_WIN + 1:
        return []
    window = arr.iloc[-(_ROLL_WIN + 1):-1]
    mu, sigma = window.mean(), window.std()
    latest = float(arr.iloc[-1])
    if sigma < _MIN_STD:
        return []
    z = (latest - mu) / sigma
    if abs(z) < _Z_THRESH:
        return []
    return [{'device': device, 'pm2_5': round(latest, 2), 'z_score': round(z, 2),
             'type': 'spike' if z > 0 else 'drop'}]

def test_stable_data_no_anomaly():
    assert detect_logic([25.0] * 25) == []

def test_spike_flagged():
    baseline = [18.0 if i % 2 == 0 else 22.0 for i in range(20)]
    result = detect_logic(baseline + [200.0])
    assert len(result) == 1 and result[0]['type'] == 'spike' and result[0]['z_score'] > 3

def test_drop_flagged():
    baseline = [78.0 if i % 2 == 0 else 82.0 for i in range(20)]
    result = detect_logic(baseline + [0.0])
    assert len(result) == 1 and result[0]['type'] == 'drop' and result[0]['z_score'] < -3

def test_short_data_skipped():
    assert detect_logic([50.0] * 5) == []

def test_low_variance_skipped():
    # std < 1.0, so nothing should be flagged
    assert detect_logic([10.0] * 20 + [10.05]) == []
