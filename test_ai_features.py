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
