# SmartAir v3.1 — Notifications, Feature Importance, Anomaly Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Feature Importance bar chart, Z-score Anomaly Detection, Line Messaging API alerts, and Browser Push Notifications to SmartAir KMUTT.

**Architecture:** `ai.py` (Railway server) runs every 10 min — it detects anomalies, extracts feature importance after retrain, and pushes notifications via Line API + pywebpush. `index.html` (single-file PWA) reads new Firebase paths and renders the UI. `sw.js` gains a push event handler for browser notifications.

**Tech Stack:** Python 3.10+, XGBoost, Firebase Admin SDK, `requests` (Line API), `pywebpush` (Web Push), Firebase JS SDK v9, vanilla JS, CSS-only bar chart

> **Security note on DOM rendering:** All data inserted via element.textContent or esc() sanitiser below comes from our own Firebase, written by ai.py. Still, every dynamic value is escaped before DOM insertion as defence-in-depth.

---

## Files Overview

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Create | Pin all Python deps including new `requests`, `pywebpush` |
| `generate_vapid.py` | Create | One-time VAPID key generation script |
| `ai.py` | Modify | `write_feature_importance()`, `detect_anomalies()`, `write_anomalies()`, `_line_push()`, `check_and_notify()`, `send_web_push()` |
| `index.html` | Modify | Feature importance card (Stats tab), anomaly section (Stats tab), anomaly badge (header), Line/Push settings (Settings tab), Firebase write imports |
| `sw.js` | Modify | push event handler + notificationclick handler |
| `test_ai_features.py` | Create | Pytest tests for `detect_anomalies()` and `write_feature_importance()` logic |

---

## Task 1: requirements.txt + VAPID Key Generation

**Files:**
- Create: `requirements.txt`
- Create: `generate_vapid.py`

- [ ] **Step 1: Create `requirements.txt`**

Create `C:\Users\ADMIN\Desktop\SMARTAIR\requirements.txt`:

```
firebase-admin==6.5.0
numpy>=1.24
pandas>=2.0
joblib>=1.3
xgboost>=2.0
scikit-learn>=1.3
requests>=2.31
pywebpush>=2.0.0
py-vapid>=1.9
```

- [ ] **Step 2: Install new dependencies**

```powershell
pip install requests pywebpush py-vapid
```

Expected output includes lines like: `Successfully installed requests-2.x pywebpush-2.x py-vapid-1.x`

- [ ] **Step 3: Create VAPID key generation script**

Create `C:\Users\ADMIN\Desktop\SMARTAIR\generate_vapid.py`:

```python
"""Run once to generate VAPID keys. Copy output into Railway environment variables."""
import base64
from py_vapid import Vapid
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

v = Vapid()
v.generate_keys()

private_pem = v.private_pem().decode().strip()
pub_bytes   = v.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
public_b64  = base64.urlsafe_b64encode(pub_bytes).decode().rstrip('=')

print("=== Copy these three values to Railway -> Variables ===")
print(f"VAPID_PRIVATE_KEY={private_pem}")
print(f"VAPID_PUBLIC_KEY={public_b64}")
print(f"VAPID_CLAIMS_EMAIL=mailto:blaster.teen15@gmail.com")
```

- [ ] **Step 4: Run the VAPID generator and save the keys**

```powershell
python generate_vapid.py
```

Copy the three output lines. Add them in Railway -> Your Service -> Variables:
- `VAPID_PRIVATE_KEY` — the full PEM string
- `VAPID_PUBLIC_KEY` — the base64url string (no `=` padding)
- `VAPID_CLAIMS_EMAIL` — `mailto:blaster.teen15@gmail.com`

Also paste `VAPID_PUBLIC_KEY` into a text file — you will need it in Task 8 Step 2.

- [ ] **Step 5: Commit**

```powershell
git add requirements.txt generate_vapid.py
git commit -m "chore: add requirements.txt and VAPID key generation script"
```

---

## Task 2: Feature Importance — ai.py

**Files:**
- Modify: `ai.py`
- Create: `test_ai_features.py`

- [ ] **Step 1: Write tests for feature importance extraction logic**

Create `C:\Users\ADMIN\Desktop\SMARTAIR\test_ai_features.py`:

```python
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
```

- [ ] **Step 2: Run the tests**

```powershell
pip install pytest -q
pytest test_ai_features.py -v
```

Expected: `4 passed`

- [ ] **Step 3: Add FEATURE_LABEL dict and FEATURE_IMP_PATH constant to ai.py**

In `ai.py`, after the `COL_ALIASES` dict (around line 78), insert:

```python
FEATURE_IMP_PATH = '/ai_analysis/feature_importance'

FEATURE_LABEL: dict[str, str] = {
    'pm1': 'PM1 ปัจจุบัน', 'pm2_5': 'PM2.5 ปัจจุบัน', 'pm10': 'PM10 ปัจจุบัน',
    'temperature': 'อุณหภูมิ', 'humidity': 'ความชื้น',
    'pm1_lag1': 'PM1 รอบก่อน',      'pm2_5_lag1': 'PM2.5 รอบก่อน',      'pm10_lag1': 'PM10 รอบก่อน',
    'pm1_lag2': 'PM1 2 รอบก่อน',    'pm2_5_lag2': 'PM2.5 2 รอบก่อน',    'pm10_lag2': 'PM10 2 รอบก่อน',
    'pm1_lag3': 'PM1 3 รอบก่อน',    'pm2_5_lag3': 'PM2.5 3 รอบก่อน',    'pm10_lag3': 'PM10 3 รอบก่อน',
    'temperature_lag1': 'อุณหภูมิรอบก่อน',  'humidity_lag1': 'ความชื้นรอบก่อน',
    'temperature_lag2': 'อุณหภูมิ 2 รอบก่อน', 'humidity_lag2': 'ความชื้น 2 รอบก่อน',
    'pm2_5_rmean3': 'ค่าเฉลี่ย PM2.5 (3)',  'pm2_5_rmean6': 'ค่าเฉลี่ย PM2.5 (6)',
    'pm2_5_rmean12': 'ค่าเฉลี่ย PM2.5 (12)',
    'pm10_rmean3': 'ค่าเฉลี่ย PM10 (3)',    'pm1_rmean3': 'ค่าเฉลี่ย PM1 (3)',
    'pm2_5_rstd3': 'ความผันผวน PM2.5 (3)', 'pm2_5_rmax3': 'PM2.5 สูงสุด (3)',
    'hour': 'ชั่วโมงของวัน',  'day_of_week': 'วันในสัปดาห์', 'month': 'เดือน',
    'is_rush': 'ชั่วโมงเร่งด่วน', 'is_night': 'กลางคืน',
    'sin_hour': 'รอบเวลา (sin)',  'cos_hour': 'รอบเวลา (cos)',
    'sin_dow': 'รอบสัปดาห์ (sin)', 'cos_dow': 'รอบสัปดาห์ (cos)',
    'pm_ratio': 'สัดส่วน PM2.5/PM10', 'pm1_pm25_r': 'สัดส่วน PM1/PM2.5',
    'aqi_approx': 'ค่า AQI ประมาณ',   'temp_x_hum': 'อุณหภูมิ × ความชื้น',
    'pm2_5_d1': 'การเปลี่ยน PM2.5 (1 รอบ)', 'pm10_d1': 'การเปลี่ยน PM10',
    'pm1_d1': 'การเปลี่ยน PM1',
}
```

- [ ] **Step 4: Add `write_feature_importance()` function to ai.py**

Add this function after `classify_aqi()` (after line 475):

```python
def write_feature_importance(device: str, model: MultiOutputRegressor, f_cols: list[str]) -> None:
    try:
        imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
        pairs = sorted(zip(f_cols, imps), key=lambda x: -x[1])[:10]
        top10 = [
            {'feature': f, 'importance': round(float(imp), 4), 'label': FEATURE_LABEL.get(f, f)}
            for f, imp in pairs
        ]
        _fb_set(f'{FEATURE_IMP_PATH}/{device}', {
            'features': top10,
            'updated_at': int(time.time()),
        })
        log.info(f"  {device}: wrote feature importance (top {len(top10)})")
    except Exception as exc:
        log.warning(f"write_feature_importance [{device}]: {exc}")
```

- [ ] **Step 5: Replace the old top-5 log with the new function call in `train_all_devices()`**

Find this block in `train_all_devices()` (around line 393):

```python
            if use_xgb:
                imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
                top5 = sorted(zip(f_cols, imps), key=lambda x: -x[1])[:5]
                log.info(f"  {device}: top-5 features = {[t[0] for t in top5]}")
```

Replace with:

```python
            if use_xgb:
                write_feature_importance(device, model, f_cols)
```

- [ ] **Step 6: Run tests again to confirm nothing broke**

```powershell
pytest test_ai_features.py -v
```

Expected: `4 passed`

- [ ] **Step 7: Commit**

```powershell
git add ai.py test_ai_features.py
git commit -m "feat: extract and persist XGBoost feature importance to Firebase"
```

---

## Task 3: Feature Importance — index.html

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add CSS for feature importance bar chart**

In `index.html`, find the closing `</style>` tag (line 503). Insert before it:

```css
/* ── Feature Importance ── */
.feat-imp-card { background:var(--glass); backdrop-filter:var(--glass-blur); -webkit-backdrop-filter:var(--glass-blur); border:1px solid var(--glass-border); border-radius:20px; padding:16px; margin-bottom:10px; box-shadow:var(--glass-shadow); }
.feat-imp-title { font-size:.76rem; font-weight:700; color:var(--sub); margin-bottom:12px; display:flex; align-items:center; gap:6px; }
.feat-imp-row { display:flex; align-items:center; gap:8px; margin-bottom:7px; }
.feat-imp-label { font-size:.68rem; color:var(--text); font-weight:500; min-width:140px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.feat-imp-bar-bg { flex:1; height:8px; background:var(--border); border-radius:99px; overflow:hidden; }
.feat-imp-bar { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--mint),var(--sky)); transition:width .6s cubic-bezier(.34,1.56,.64,1); }
.feat-imp-pct { font-family:'DM Mono',monospace; font-size:.65rem; color:var(--sub); min-width:36px; text-align:right; }
.feat-imp-updated { font-size:.6rem; color:var(--sub); margin-top:8px; text-align:right; }
```

- [ ] **Step 2: Add feature importance card HTML in Stats tab**

In `index.html`, find the closing `</section>` of `tab-stats` (line 842). Insert before it:

```html
    <!-- Feature Importance card -->
    <div class="feat-imp-card fade" id="feat-imp-card" style="animation-delay:.12s">
      <div class="feat-imp-title">
        <i class="ti ti-bulb" style="color:var(--amber)"></i>
        <span>ปัจจัยที่ AI ใช้พยากรณ์</span>
        <span style="margin-left:auto;font-size:.6rem;font-weight:400;color:var(--sub)" id="feat-imp-device-label"></span>
      </div>
      <div id="feat-imp-rows">
        <div style="text-align:center;color:var(--sub);font-size:.72rem;padding:12px 0">รอข้อมูล...</div>
      </div>
      <div class="feat-imp-updated" id="feat-imp-updated"></div>
    </div>
```

- [ ] **Step 3: Add JS helpers and feature importance listener**

In `index.html`, find the `const fbDb = getDatabase(initializeApp(FB));` line. After it, add:

```javascript
// Safe text escaping — prevents XSS for any string from Firebase
function esc(s) { const d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }
```

Then after the existing `onValue(ref(fbDb, '/ai_analysis/prediction'), ...)` listener block (around line 1450), add:

```javascript
// ── Feature Importance ───────────────────────────────────────
let _featImpUnsub = null;

function subscribeFeatImp(device) {
  if (_featImpUnsub) { _featImpUnsub(); _featImpUnsub = null; }
  _featImpUnsub = onValue(ref(fbDb, '/ai_analysis/feature_importance/' + device), snap => {
    const data      = snap.val();
    const container = document.getElementById('feat-imp-rows');
    const updEl     = document.getElementById('feat-imp-updated');
    const lblEl     = document.getElementById('feat-imp-device-label');

    if (!data || !Array.isArray(data.features) || data.features.length === 0) {
      container.textContent = 'ไม่มีข้อมูล (Ridge model หรือยังไม่ได้เทรน)';
      return;
    }
    lblEl.textContent = device;
    const maxImp = data.features[0].importance || 1;
    container.innerHTML = '';  // clear safely first
    data.features.forEach(f => {
      const pct    = ((f.importance / maxImp) * 100).toFixed(1);
      const valPct = (f.importance * 100).toFixed(1);

      const row   = document.createElement('div');
      row.className = 'feat-imp-row';

      const lbl   = document.createElement('div');
      lbl.className = 'feat-imp-label';
      lbl.title   = f.feature;
      lbl.textContent = f.label || f.feature;

      const barBg = document.createElement('div');
      barBg.className = 'feat-imp-bar-bg';
      const bar   = document.createElement('div');
      bar.className = 'feat-imp-bar';
      bar.style.width = pct + '%';
      barBg.appendChild(bar);

      const pctEl = document.createElement('div');
      pctEl.className = 'feat-imp-pct';
      pctEl.textContent = valPct + '%';

      row.appendChild(lbl);
      row.appendChild(barBg);
      row.appendChild(pctEl);
      container.appendChild(row);
    });

    if (data.updated_at) {
      updEl.textContent = 'อัปเดต: ' + new Date(data.updated_at * 1000).toLocaleString('th-TH');
    }
  });
}
```

- [ ] **Step 4: Call `subscribeFeatImp` on initial load and on device switch**

After `const fbDb = getDatabase(initializeApp(FB));`, add:

```javascript
subscribeFeatImp('device1');  // initial subscription
```

Find the `switchDevice` function. Inside it, before the closing brace `}`, add:

```javascript
  subscribeFeatImp(device);
```

- [ ] **Step 5: Manual verification**

1. Run `ai.py` for one retrain cycle (or wait for Railway)
2. Open `index.html` in browser → Stats tab → scroll to bottom
3. Confirm "ปัจจัยที่ AI ใช้พยากรณ์" card shows 10 coloured bars with Thai labels
4. Switch to another device tab — bars update to that device

- [ ] **Step 6: Commit**

```powershell
git add index.html
git commit -m "feat: add feature importance bar chart to Stats tab"
```

---

## Task 4: Anomaly Detection — ai.py

**Files:**
- Modify: `ai.py`
- Modify: `test_ai_features.py`

- [ ] **Step 1: Append anomaly detection tests to `test_ai_features.py`**

Append to the end of `test_ai_features.py`:

```python
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
    result = detect_logic([20.0] * 20 + [200.0])
    assert len(result) == 1 and result[0]['type'] == 'spike' and result[0]['z_score'] > 3

def test_drop_flagged():
    result = detect_logic([80.0] * 20 + [0.0])
    assert len(result) == 1 and result[0]['type'] == 'drop' and result[0]['z_score'] < -3

def test_short_data_skipped():
    assert detect_logic([50.0] * 5) == []

def test_low_variance_skipped():
    # std < 1.0, so nothing should be flagged
    assert detect_logic([10.0] * 20 + [10.05]) == []
```

- [ ] **Step 2: Run all tests**

```powershell
pytest test_ai_features.py -v
```

Expected: `9 passed`

- [ ] **Step 3: Add anomaly constants to ai.py**

In `ai.py`, after `FEATURE_IMP_PATH`, add:

```python
ANOMALY_PATH     = '/ai_analysis/anomalies'
ANOMALY_Z_THRESH = 3.0
ANOMALY_MIN_STD  = 1.0
ANOMALY_ROLL_WIN = 20
ANOMALY_KEEP     = 50
```

- [ ] **Step 4: Add `detect_anomalies()` and `write_anomalies()` to ai.py**

Add both functions after `write_feature_importance()`:

```python
def detect_anomalies(device_dfs: dict) -> list[dict]:
    anomalies = []
    now = int(time.time())
    for device, df in device_dfs.items():
        if len(df) < ANOMALY_ROLL_WIN + 1:
            continue
        pm     = df['pm2_5'].values
        window = pm[-(ANOMALY_ROLL_WIN + 1):-1]
        mu     = float(window.mean())
        sigma  = float(window.std())
        latest = float(pm[-1])
        if sigma < ANOMALY_MIN_STD:
            continue
        z = (latest - mu) / sigma
        if abs(z) < ANOMALY_Z_THRESH:
            continue
        anomalies.append({
            'device'      : device,
            'pm2_5'       : round(latest, 2),
            'z_score'     : round(z, 2),
            'type'        : 'spike' if z > 0 else 'drop',
            'timestamp'   : now,
            'datetime_str': datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S'),
        })
    return anomalies


def write_anomalies(anomalies: list[dict]) -> None:
    for evt in anomalies:
        key = f"{evt['timestamp']}_{evt['device']}"
        _fb_set(f'{ANOMALY_PATH}/{key}', evt)
        log.warning(
            f"ANOMALY [{evt['device']}] PM2.5={evt['pm2_5']} "
            f"z={evt['z_score']:+.2f} type={evt['type']}"
        )
    # Trim old entries, keep ANOMALY_KEEP most recent
    try:
        existing = db.reference(ANOMALY_PATH).order_by_key().get() or {}
        if isinstance(existing, dict) and len(existing) > ANOMALY_KEEP:
            to_del = sorted(existing.keys())[:len(existing) - ANOMALY_KEEP]
            for k in to_del:
                db.reference(f'{ANOMALY_PATH}/{k}').delete()
    except Exception as exc:
        log.warning(f"write_anomalies trim: {exc}")
```

- [ ] **Step 5: Call anomaly detection in `main()` loop**

In `main()`, find the line `device_dfs = fetch_all_devices()`. Directly after the `if not device_dfs:` guard block, insert:

```python
                # Detect and persist anomalies before training
                anomalies = detect_anomalies(device_dfs)
                if anomalies:
                    write_anomalies(anomalies)
```

Also add `anomalies = []` at the very start of the `try:` block so it is always defined even if `fetch_all_devices` fails early:

Find the opening `try:` (inside the `while True` loop) and add:

```python
            try:
                anomalies = []  # populated by detect_anomalies below
                device_dfs = fetch_all_devices()
```

- [ ] **Step 6: Run all tests**

```powershell
pytest test_ai_features.py -v
```

Expected: `9 passed`

- [ ] **Step 7: Commit**

```powershell
git add ai.py test_ai_features.py
git commit -m "feat: add Z-score anomaly detection with Firebase persistence"
```

---

## Task 5: Anomaly Detection — index.html

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add CSS for anomaly badge and event cards**

In `index.html`, before `</style>`, add:

```css
/* ── Anomaly badge ── */
.anomaly-badge { display:none; position:absolute; top:-4px; right:-4px; width:18px; height:18px; background:#dc2626; border-radius:50%; font-size:.6rem; font-weight:800; color:#fff; align-items:center; justify-content:center; border:2px solid var(--glass-strong,#fff); z-index:10; }
.anomaly-badge.show { display:flex; animation:badgePop .4s cubic-bezier(.34,1.56,.64,1); }
@keyframes badgePop { from{transform:scale(0)} to{transform:scale(1)} }

/* ── Anomaly events section ── */
.anomaly-section { background:var(--glass); backdrop-filter:var(--glass-blur); -webkit-backdrop-filter:var(--glass-blur); border:1px solid var(--glass-border); border-radius:20px; padding:14px 16px; margin-bottom:10px; box-shadow:var(--glass-shadow); }
.anomaly-title { font-size:.76rem; font-weight:700; color:var(--sub); margin-bottom:10px; display:flex; align-items:center; gap:6px; }
.anomaly-item { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid var(--border); }
.anomaly-item:last-child { border-bottom:none; }
.anomaly-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.anomaly-dot.spike { background:#dc2626; }
.anomaly-dot.drop  { background:#38bdf8; }
.anomaly-info { flex:1; min-width:0; }
.anomaly-device { font-size:.7rem; font-weight:700; color:var(--text); }
.anomaly-meta { font-size:.62rem; color:var(--sub); margin-top:1px; }
.anomaly-val { font-family:'DM Mono',monospace; font-size:.8rem; font-weight:600; color:var(--text); text-align:right; flex-shrink:0; }
```

- [ ] **Step 2: Add anomaly badge to the header logo**

Find the `.logo-icon` div (around line 524):

```html
          <div class="logo-icon">🌿</div>
```

Replace with:

```html
          <div class="logo-icon" style="position:relative">🌿
            <div class="anomaly-badge" id="anomaly-badge">!</div>
          </div>
```

- [ ] **Step 3: Add anomaly section HTML in Stats tab**

After the `feat-imp-card` div added in Task 3 (and before the `</section>` of `tab-stats`), insert:

```html
    <!-- Anomaly events -->
    <div class="anomaly-section fade" style="animation-delay:.14s">
      <div class="anomaly-title">
        <i class="ti ti-alert-triangle" style="color:#dc2626"></i>
        <span>เหตุผิดปกติล่าสุด</span>
        <span style="margin-left:auto;font-size:.6rem;font-weight:400" id="anomaly-count-label"></span>
      </div>
      <div id="anomaly-list"></div>
    </div>
```

- [ ] **Step 4: Add anomaly real-time listener JS**

After the `subscribeFeatImp` function (Task 3), add:

```javascript
// ── Anomaly listener ─────────────────────────────────────────
onValue(query(ref(fbDb, '/ai_analysis/anomalies'), limitToLast(50)), snap => {
  const raw       = snap.val();
  const listEl    = document.getElementById('anomaly-list');
  const badge     = document.getElementById('anomaly-badge');
  const countLbl  = document.getElementById('anomaly-count-label');

  listEl.textContent = '';  // clear without innerHTML

  if (!raw || typeof raw !== 'object' || Object.keys(raw).length === 0) {
    const empty = document.createElement('div');
    empty.style.cssText = 'text-align:center;color:var(--sub);font-size:.72rem;padding:12px 0';
    empty.textContent = 'ไม่มีเหตุผิดปกติล่าสุด';
    listEl.appendChild(empty);
    badge.classList.remove('show');
    countLbl.textContent = '';
    return;
  }

  const events = Object.values(raw)
    .filter(e => e && e.timestamp)
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 10);

  const nowS    = Date.now() / 1000;
  const recent  = events.filter(e => (nowS - e.timestamp) < 3600);

  // Badge
  if (recent.length > 0) {
    badge.textContent = recent.length > 9 ? '9+' : String(recent.length);
    badge.classList.add('show');
    // Toast for anomaly within last 5 min
    const newest = events[0];
    if (newest && (nowS - newest.timestamp) < 300) {
      const dir = newest.type === 'spike' ? 'พุ่งสูง' : 'ลดฮวบ';
      document.getElementById('alert-text').textContent =
        (newest.device || '?') + ': PM2.5 ' + dir + ' ' + newest.pm2_5 + ' µg/m³ (Z=' + newest.z_score + ')';
      document.getElementById('alert-toast').classList.add('show');
      if (typeof alertTimeout !== 'undefined' && alertTimeout) clearTimeout(alertTimeout);
      alertTimeout = setTimeout(dismissAlert, 8000);
    }
  } else {
    badge.classList.remove('show');
  }
  countLbl.textContent = events.length + ' รายการ';

  events.forEach(e => {
    const item = document.createElement('div');
    item.className = 'anomaly-item';

    const dot = document.createElement('div');
    dot.className = 'anomaly-dot ' + (e.type === 'spike' ? 'spike' : 'drop');

    const info = document.createElement('div');
    info.className = 'anomaly-info';

    const dev = document.createElement('div');
    dev.className = 'anomaly-device';
    dev.textContent = (e.device || '—') + ' · ' + (e.type === 'spike' ? 'สูงผิดปกติ' : 'ต่ำผิดปกติ');

    const meta = document.createElement('div');
    meta.className = 'anomaly-meta';
    const dt = e.datetime_str || new Date(e.timestamp * 1000).toLocaleString('th-TH');
    const sign = e.z_score > 0 ? '+' : '';
    meta.textContent = dt + ' · Z-score ' + sign + e.z_score;

    info.appendChild(dev);
    info.appendChild(meta);

    const val = document.createElement('div');
    val.className = 'anomaly-val';
    val.textContent = e.pm2_5 + ' µg/m³';

    item.appendChild(dot);
    item.appendChild(info);
    item.appendChild(val);
    listEl.appendChild(item);
  });
});
```

- [ ] **Step 5: Manual verification**

1. In Firebase console, manually write to `/ai_analysis/anomalies/test_spike`:
   `{"device":"device1","pm2_5":187.3,"z_score":4.2,"type":"spike","timestamp":1715666400,"datetime_str":"2026-05-14 08:00:00"}`
2. Open `index.html` — confirm red `!` badge on logo
3. Go to Stats tab — confirm anomaly item card shows device/value/type
4. Delete the test key from Firebase

- [ ] **Step 6: Commit**

```powershell
git add index.html
git commit -m "feat: add anomaly badge, event list, and toast to dashboard"
```

---

## Task 6: Notification Config Settings UI — index.html

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add `set` and `get` to the Firebase SDK import**

Find (around line 1012):

```javascript
import { getDatabase, ref, onValue, query, limitToLast }
  from "https://www.gstatic.com/firebasejs/9.22.0/firebase-database.js";
```

Replace with:

```javascript
import { getDatabase, ref, onValue, query, limitToLast, set, get }
  from "https://www.gstatic.com/firebasejs/9.22.0/firebase-database.js";
```

- [ ] **Step 2: Add CSS for settings input fields**

Before `</style>`, add:

```css
/* ── Settings input fields ── */
.setting-input { width:100%; border:1px solid var(--border); border-radius:10px; padding:9px 12px; font-size:.78rem; font-family:'Nunito','Sarabun',sans-serif; background:var(--raised); color:var(--text); margin-top:6px; outline:none; transition:border-color .2s; }
.setting-input:focus { border-color:var(--mint); }
.setting-input-label { font-size:.72rem; font-weight:600; color:var(--text); margin-top:10px; display:block; }
.setting-save-btn { margin-top:10px; width:100%; padding:10px; border-radius:12px; background:var(--mint); color:#fff; font-size:.82rem; font-weight:700; border:none; cursor:pointer; font-family:'Nunito','Sarabun',sans-serif; transition:background .2s; }
.setting-save-btn:hover { background:var(--mint-dark); }
.setting-save-ok { display:none; font-size:.7rem; color:var(--mint); text-align:center; margin-top:6px; font-weight:600; }
```

- [ ] **Step 3: Add Line Messaging settings card in Settings tab**

Find the first `</div></div>` closing the first `settings-section` card (after the threshold slider, around line 926). Insert a new section block after the closing `</div>` of that `settings-section`:

```html
    <div class="settings-section">
      <div class="settings-card">
        <div class="settings-title"><i class="ti ti-brand-line"></i> Line Messaging</div>
        <p style="font-size:.72rem;color:var(--sub);margin-bottom:10px;line-height:1.5">ต้องการ <b>Channel Access Token</b> และ <b>User ID</b> จาก Line Developers Console</p>
        <label class="setting-input-label" for="line-token-input">Channel Access Token</label>
        <input class="setting-input" type="password" id="line-token-input" placeholder="xxxxxxxxxx...">
        <label class="setting-input-label" for="line-uid-input">User ID ผู้รับ (Uxxxxxxx...)</label>
        <input class="setting-input" type="text" id="line-uid-input" placeholder="Uxxxxxxxxxxxxxxxxx">
        <label class="setting-input-label" for="line-threshold-input">แจ้งเตือนเมื่อ PM2.5 เกิน (µg/m³)</label>
        <input class="setting-input" type="number" id="line-threshold-input" min="10" max="300" value="50">
        <button class="setting-save-btn" onclick="saveLineConfig()">บันทึกการตั้งค่า Line</button>
        <div class="setting-save-ok" id="line-save-ok">✓ บันทึกแล้ว</div>
      </div>
    </div>
    <div class="settings-section">
      <div class="settings-card">
        <div class="settings-title"><i class="ti ti-bell-ringing-2"></i> Browser Push Notification</div>
        <div class="setting-row">
          <div class="setting-info">
            <div class="setting-name">เปิด Browser Push</div>
            <div class="setting-desc">แจ้งเตือนผ่าน browser แม้ปิดแอพอยู่ (ต้องอนุญาตใน browser)</div>
          </div>
          <label class="toggle"><input type="checkbox" id="push-toggle" onchange="togglePush(this.checked)"><span class="slider"></span></label>
        </div>
        <div style="font-size:.65rem;color:var(--sub);margin-top:6px" id="push-status">ยังไม่ได้เปิดใช้งาน</div>
      </div>
    </div>
```

- [ ] **Step 4: Add JS for saving and loading Line config**

After the existing alert toggle listener (around line 1132), add:

```javascript
// ── Line / Push config ────────────────────────────────────────
const _notifyRef = ref(fbDb, '/settings/notify_config');

window.saveLineConfig = async function () {
  const token     = document.getElementById('line-token-input').value.trim();
  const uid       = document.getElementById('line-uid-input').value.trim();
  const threshold = parseFloat(document.getElementById('line-threshold-input').value) || 50;
  if (!token || !uid) { alert('กรุณากรอก Token และ User ID'); return; }
  try {
    await set(ref(fbDb, '/settings/notify_config/line_channel_token'), token);
    await set(ref(fbDb, '/settings/notify_config/line_user_id'), uid);
    await set(ref(fbDb, '/settings/notify_config/threshold_pm25'), threshold);
    const okEl = document.getElementById('line-save-ok');
    okEl.style.display = 'block';
    setTimeout(() => { okEl.style.display = 'none'; }, 3000);
  } catch (e) { alert('บันทึกไม่สำเร็จ: ' + e.message); }
};

// Pre-fill saved values on load
get(_notifyRef).then(snap => {
  const cfg = snap.val() || {};
  if (cfg.line_channel_token) document.getElementById('line-token-input').value   = cfg.line_channel_token;
  if (cfg.line_user_id)       document.getElementById('line-uid-input').value     = cfg.line_user_id;
  if (cfg.threshold_pm25)     document.getElementById('line-threshold-input').value = cfg.threshold_pm25;
}).catch(() => {});
```

- [ ] **Step 5: Also sync the existing threshold slider to Firebase**

Find the existing slider change listener (around line 1126):

```javascript
slider.addEventListener('input', () => { updateSliderFill(); alertDismissed = false; });
```

Add after it:

```javascript
slider.addEventListener('change', () => {
  set(ref(fbDb, '/settings/notify_config/threshold_pm25'), parseInt(slider.value, 10)).catch(() => {});
});
```

- [ ] **Step 6: Manual verification**

1. Open `index.html` → Settings tab — confirm "Line Messaging" and "Browser Push" cards appear
2. Enter a dummy token and user ID → click บันทึก — confirm green "✓ บันทึกแล้ว"
3. Open Firebase console `/settings/notify_config` — confirm values written
4. Reload page — confirm inputs are pre-populated

- [ ] **Step 7: Commit**

```powershell
git add index.html
git commit -m "feat: add Line and Browser Push config settings UI"
```

---

## Task 7: Line Messaging API — ai.py

**Files:**
- Modify: `ai.py`

- [ ] **Step 1: Add `import requests` to ai.py**

After `import joblib`, add:

```python
import requests
```

- [ ] **Step 2: Add Line API constants and cooldown dict**

After `ANOMALY_KEEP`, add:

```python
LINE_PUSH_URL         = 'https://api.line.me/v2/bot/message/push'
NOTIFY_CONFIG_PATH    = '/settings/notify_config'
NOTIFY_COOLDOWN       = 1800   # 30 min between threshold alerts
ANOMALY_NOTIFY_CD     = 900    # 15 min between anomaly alerts per device

_last_notify_ts: dict[str, float] = {}
```

- [ ] **Step 3: Add `_line_push()` function**

Add after `write_anomalies()`:

```python
def _line_push(token: str, user_id: str, text: str) -> bool:
    try:
        resp = requests.post(
            LINE_PUSH_URL,
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
            json={'to': user_id, 'messages': [{'type': 'text', 'text': text}]},
            timeout=10,
        )
        if resp.status_code == 200:
            log.info("Line push OK")
            return True
        log.warning(f"Line push HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as exc:
        log.warning(f"Line push error: {exc}")
        return False
```

- [ ] **Step 4: Add `check_and_notify()` function**

Add after `_line_push()`:

```python
def check_and_notify(preds: np.ndarray, aqi: dict, anomalies: list[dict]) -> None:
    try:
        config = db.reference(NOTIFY_CONFIG_PATH).get() or {}
    except Exception as exc:
        log.warning(f"check_and_notify config read error: {exc}")
        return

    token     = str(config.get('line_channel_token', '')).strip()
    user_id   = str(config.get('line_user_id', '')).strip()
    threshold = float(config.get('threshold_pm25', 50))

    if not token or not user_id:
        return

    now  = time.time()
    pm25 = float(preds[1])

    if pm25 > threshold and (now - _last_notify_ts.get('threshold', 0)) >= NOTIFY_COOLDOWN:
        msg = (
            f"\U0001f6a8 SmartAir แจ้งเตือน\n"
            f"PM2.5 = {pm25:.1f} µg/m³\n"
            f"AQI: {aqi['label']}\n"
            f"เกินค่าที่ตั้งไว้ ({threshold:.0f} µg/m³)"
        )
        if _line_push(token, user_id, msg):
            _last_notify_ts['threshold'] = now

    for evt in anomalies:
        dev  = evt['device']
        key  = f'anomaly_{dev}'
        if (now - _last_notify_ts.get(key, 0)) >= ANOMALY_NOTIFY_CD:
            direction = '⬆️ พุ่งสูง' if evt['type'] == 'spike' else '⬇️ ลดฮวบ'
            msg = (
                f"⚠️ SmartAir ตรวจพบค่าผิดปกติ\n"
                f"{dev}: PM2.5 {direction}\n"
                f"ค่า: {evt['pm2_5']} µg/m³ (Z={evt['z_score']:+.1f})"
            )
            if _line_push(token, user_id, msg):
                _last_notify_ts[key] = now
```

Note: Thai strings are written as Unicode escapes to avoid encoding issues on Railway.
In plain Thai they read: "🚨 SmartAir แจ้งเตือน\nPM2.5 = X µg/m³\nAQI: Y\nเกินค่าที่ตั้งไว้ (Z µg/m³)" and "⚠️ SmartAir ตรวจพบค่าผิดปกติ\nDEVICE: PM2.5 DIRECTION\nค่า: X µg/m³ (Z=Y)"

If Railway handles UTF-8 correctly (check your existing Thai logs), you can write the strings in plain Thai instead.

- [ ] **Step 5: Call `check_and_notify()` in `main()` after `upload_result()`**

Find:

```python
                        upload_result(preds, metrics_dict, per_device)
```

Add after it:

```python
                        check_and_notify(preds, aqi, anomalies)
```

- [ ] **Step 6: Test with real Line credentials (optional, requires account)**

If you have credentials:
1. Write to Firebase `/settings/notify_config`: `{"line_channel_token":"xxx","line_user_id":"Uxxx","threshold_pm25":1}`
2. Run `python ai.py` for one cycle
3. Confirm Line message arrives
4. Restore `threshold_pm25` to 50

- [ ] **Step 7: Commit**

```powershell
git add ai.py
git commit -m "feat: add Line Messaging API notifications for PM2.5 threshold and anomalies"
```

---

## Task 8: Browser Push — sw.js + index.html + ai.py

**Files:**
- Modify: `sw.js`
- Modify: `index.html`
- Modify: `ai.py`

**Prerequisite:** VAPID keys from Task 1 are set in Railway and noted locally.

- [ ] **Step 1: Add push event handler to sw.js**

Append to the end of `sw.js`:

```javascript
self.addEventListener('push', e => {
  const data = e.data ? e.data.json() : {};
  e.waitUntil(
    self.registration.showNotification(data.title || 'SmartAir Alert', {
      body    : data.body || 'ตรวจพบค่าผิดปกติ',
      icon    : './icon-192.svg',
      badge   : './icon-192.svg',
      tag     : 'smartair-alert',
      renotify: true,
    })
  );
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.openWindow('./'));
});
```

- [ ] **Step 2: Add VAPID public key and push helpers to index.html**

After `const fbDb = getDatabase(initializeApp(FB));`, add:

```javascript
// Paste your VAPID_PUBLIC_KEY (base64url, no padding) from generate_vapid.py here:
const VAPID_PUBLIC_KEY = 'REPLACE_WITH_YOUR_VAPID_PUBLIC_KEY';

function _b64ToUint8(b64) {
  const pad = '='.repeat((4 - b64.length % 4) % 4);
  const raw = atob((b64 + pad).replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from([...raw].map(c => c.charCodeAt(0)));
}
```

Replace `REPLACE_WITH_YOUR_VAPID_PUBLIC_KEY` with the actual base64url key from Task 1 Step 4.

- [ ] **Step 3: Add `togglePush()` and `subscribePush()` to index.html**

After `_b64ToUint8`, add:

```javascript
window.togglePush = async function(enabled) {
  const statusEl = document.getElementById('push-status');
  if (!enabled) { statusEl.textContent = 'ปิด Browser Push แล้ว'; return; }
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    statusEl.textContent = 'Browser ไม่รองรับ Push Notification';
    document.getElementById('push-toggle').checked = false;
    return;
  }
  const perm = await Notification.requestPermission();
  if (perm !== 'granted') {
    statusEl.textContent = 'ไม่ได้รับอนุญาต — กรุณาอนุญาตใน browser settings';
    document.getElementById('push-toggle').checked = false;
    return;
  }
  try {
    const reg = await navigator.serviceWorker.ready;
    const sub = await reg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: _b64ToUint8(VAPID_PUBLIC_KEY),
    });
    const subJSON = JSON.parse(JSON.stringify(sub));
    const id = btoa(subJSON.endpoint).replace(/[^a-zA-Z0-9]/g, '').slice(0, 32);
    await set(ref(fbDb, '/settings/notify_config/push_subscriptions/' + id), subJSON);
    statusEl.textContent = 'เปิดใช้งาน Browser Push แล้ว';
  } catch(e) {
    statusEl.textContent = 'เกิดข้อผิดพลาด: ' + e.message;
    document.getElementById('push-toggle').checked = false;
  }
};
```

- [ ] **Step 4: Add `send_web_push()` to ai.py**

Add after `check_and_notify()`:

```python
def send_web_push(title: str, body: str, subscriptions: dict) -> None:
    priv_key = os.getenv('VAPID_PRIVATE_KEY', '')
    pub_key  = os.getenv('VAPID_PUBLIC_KEY', '')
    email    = os.getenv('VAPID_CLAIMS_EMAIL', 'mailto:admin@example.com')
    if not priv_key or not pub_key or not subscriptions:
        return
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        log.warning("pywebpush not installed — skipping Web Push")
        return

    import json
    payload  = json.dumps({'title': title, 'body': body})
    dead_ids = []

    for sub_id, sub_data in subscriptions.items():
        if not isinstance(sub_data, dict):
            continue
        try:
            webpush(
                subscription_info=sub_data,
                data=payload,
                vapid_private_key=priv_key,
                vapid_claims={'sub': email},
            )
            log.info(f"Web Push sent to {sub_id[:8]}…")
        except Exception as exc:
            if hasattr(exc, 'response') and exc.response and exc.response.status_code == 410:
                dead_ids.append(sub_id)
            else:
                log.warning(f"Web Push [{sub_id[:8]}]: {exc}")

    for sub_id in dead_ids:
        try:
            db.reference(f'{NOTIFY_CONFIG_PATH}/push_subscriptions/{sub_id}').delete()
            log.info(f"Removed expired push subscription {sub_id[:8]}")
        except Exception:
            pass
```

- [ ] **Step 5: Wire `send_web_push()` into `check_and_notify()`**

In `check_and_notify()`, after reading `config`, add:

```python
    subscriptions = config.get('push_subscriptions') or {}
```

In the threshold alert block, after `if _line_push(token, user_id, msg):`, add:

```python
                send_web_push('SmartAir Alert', f'PM2.5 = {pm25:.1f} ug/m3 (threshold {threshold:.0f})', subscriptions)
```

In the anomaly alert block, after `if _line_push(token, user_id, msg):`, add:

```python
                    send_web_push(f'SmartAir: {dev}', f'PM2.5 {evt["pm2_5"]} ug/m3 (Z={evt["z_score"]:+.1f})', subscriptions)
```

- [ ] **Step 6: Run all tests**

```powershell
pytest test_ai_features.py -v
```

Expected: `9 passed`

- [ ] **Step 7: End-to-end Browser Push test**

1. Serve `index.html` over HTTPS (Firebase Hosting or Railway — localhost won't work for Push)
2. Open Settings → enable "Browser Push" → allow browser permission
3. Confirm subscription entry appears in Firebase `/settings/notify_config/push_subscriptions/`
4. In Firebase, set `/settings/notify_config/threshold_pm25` to 1 (very low)
5. Run one cycle of `ai.py` on Railway
6. Confirm browser notification arrives

- [ ] **Step 8: Commit**

```powershell
git add sw.js index.html ai.py
git commit -m "feat: add Browser Push Notification via Web Push API and pywebpush"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Line Messaging API push | Task 7 |
| Browser Push Notification | Task 8 |
| Settings UI: token, uid, threshold, push toggle | Task 6 |
| Feature importance bar chart — ai.py | Task 2 |
| Feature importance bar chart — index.html | Task 3 |
| Device switch updates feat imp chart | Task 3 Step 4 |
| Z-score anomaly detection — ai.py | Task 4 |
| Anomaly events written to Firebase, trimmed to 50 | Task 4 |
| Anomaly badge on header | Task 5 |
| Anomaly events list in Stats tab | Task 5 |
| Anomaly toast alert | Task 5 |
| 30 min cooldown for threshold notify | Task 7 |
| 15 min cooldown for anomaly notify per device | Task 7 |
| Ridge model: no feature importance written | Task 2 (only XGBoost calls write_feature_importance) |
| Expired push subscriptions cleaned up | Task 8 |
| requirements.txt and VAPID setup | Task 1 |

**Placeholder scan:** No TBD or TODO present. All code blocks are complete.

**Type consistency:**
- `anomalies` list: created in Task 4, passed to `check_and_notify()` in Task 7 — same type `list[dict]`
- `NOTIFY_CONFIG_PATH` constant: defined in Task 7, used in `check_and_notify()` and `send_web_push()` — consistent
- Firebase path `/ai_analysis/feature_importance/{device}` — consistent between ai.py (Task 2) and index.html (Task 3)
- Firebase path `/ai_analysis/anomalies` — consistent between ai.py (Task 4) and index.html (Task 5)
- `esc()` helper: defined Task 3, used in anomaly listener Task 5 via DOM methods (not needed there)
- `_last_notify_ts` keys `'threshold'` and `'anomaly_{dev}'` — consistent between Task 7 definition and usage
