# SmartAir v3.1 — Line Bot Notifications, Feature Importance, Anomaly Detection

**Date:** 2026-05-14  
**Status:** Approved  
**Scope:** 3 new features added to existing SmartAir KMUTT air quality dashboard

---

## 1. Context

SmartAir is a real-time PM2.5 monitoring system for KMUTT Bangmod campus. It consists of:
- `ai.py` — Python server running continuously on Railway, fetches Firebase data, trains XGBoost/Ridge models, writes predictions back to Firebase every 10 minutes
- `index.html` — Single-page PWA (no build step), reads Firebase real-time, renders charts and AQI

Current notification: in-browser alert toast only (no external push).

---

## 2. Feature 1 — Line Messaging API + Browser Push Notification

### 2.1 Goal
Allow users to set a PM2.5 threshold in Settings. When the predicted PM2.5 exceeds that threshold, send:
- A Line message (via Line Messaging API push message) to the configured user
- A browser push notification (via Web Push API) for users who have the app open or installed as PWA

### 2.2 Firebase Config Schema
Path: `/settings/notify_config`
```json
{
  "threshold_pm25": 50,
  "line_channel_token": "Bearer xxxx...",
  "line_user_id": "Uxxxxxxxxxxxxxxx",
  "push_subscriptions": { "<sub_id>": { "endpoint": "...", "keys": {...} } }
}
```
The `push_subscriptions` map is managed by the frontend (register/unregister).

### 2.3 ai.py Changes
- On every prediction cycle, read `/settings/notify_config` from Firebase
- Compare `predicted PM2.5` against `threshold_pm25`
- If exceeded AND last notification was >30 minutes ago (in-memory cooldown per device):
  - POST to `https://api.line.me/v2/bot/message/push` with a Flex Message showing:
    - Current PM2.5 / PM10 / PM1 values
    - AQI label and color
    - Prediction horizon (10 min)
    - "ดูรายละเอียด" link to the SmartAir web app
- Cooldown state: `dict[device, last_notify_ts]` in memory, reset on restart
- Library: `requests` (already available via pip)

### 2.4 index.html Changes (Settings tab)
New "Line แจ้งเตือน" settings card:
- Input: Line Channel Access Token (masked input, saved to Firebase `/settings/notify_config/line_channel_token`)
- Input: Line User ID (saved to Firebase `/settings/notify_config/line_user_id`)
- Existing PM2.5 threshold slider — additionally writes value to Firebase

New "Browser Push" toggle:
- Calls `Notification.requestPermission()` on first enable
- Registers a push subscription via `PushManager.subscribe()` using VAPID public key
- Saves subscription object to Firebase `/settings/notify_config/push_subscriptions/{hash}`

### 2.5 Browser Push Architecture
- `sw.js` handles `push` event → shows notification with title "SmartAir Alert" and PM2.5 value
- VAPID key pair generated once and stored in environment variables on Railway
- `ai.py` sends Web Push payloads to all subscriptions in Firebase using `pywebpush` library
- Subscriptions that return 410 Gone are removed automatically

### 2.6 Constraints
- Line Notify API is shut down (March 2025). Use Line Messaging API (Push Message) instead.
- Line Channel Access Token must be a long-lived token from Line Developers console.
- Web Push requires HTTPS — SmartAir is served via Firebase Hosting/Railway which provides HTTPS.

---

## 3. Feature 2 — Feature Importance Explainability

### 3.1 Goal
After every XGBoost retrain, write the top-10 most important features per device to Firebase. The frontend reads this and renders a horizontal bar chart so users understand why the model makes its prediction.

### 3.2 Firebase Schema
Path: `/ai_analysis/feature_importance`
```json
{
  "device1": [
    {"feature": "pm2_5_lag1", "importance": 0.142, "label": "PM2.5 ชั่วโมงที่แล้ว"},
    {"feature": "pm2_5_rmean3", "importance": 0.098, "label": "PM2.5 เฉลี่ย 3 รอบ"},
    ...
  ],
  "updated_at": 1715666400
}
```

### 3.3 ai.py Changes
- After `model.fit(X, y)` for XGBoost: compute `np.mean([e.feature_importances_ for e in model.estimators_], axis=0)`
- Map each feature name to a Thai label (lookup dict in code)
- Sort descending, take top-10
- Write to `/ai_analysis/feature_importance/{device}`
- Ridge fallback: write empty list (Ridge has `coef_` not `feature_importances_`)

### 3.4 Feature Name → Thai Label Map (subset)
| Feature | Thai |
|---|---|
| `pm2_5_lag1` | PM2.5 รอบก่อน |
| `pm2_5_lag2` | PM2.5 2 รอบก่อน |
| `pm2_5_rmean3` | ค่าเฉลี่ย PM2.5 (3 รอบ) |
| `pm10_lag1` | PM10 รอบก่อน |
| `temperature_lag1` | อุณหภูมิรอบก่อน |
| `humidity_lag1` | ความชื้นรอบก่อน |
| `hour` | ชั่วโมงของวัน |
| `is_rush` | ชั่วโมงเร่งด่วน |
| `pm_ratio` | สัดส่วน PM2.5/PM10 |
| `aqi_approx` | ค่า AQI ประมาณ |

### 3.5 index.html Changes
New card in Stats tab: "ปัจจัยที่ AI ใช้พยากรณ์"
- Reads `/ai_analysis/feature_importance/{activeDevice}` via Firebase real-time listener
- Renders top-10 as CSS-only horizontal bars (no extra chart library):
  - Label (Thai name) | Bar (width = importance × 100%) | Value %
- Bar color: gradient from `--mint` to `--sky`
- Shows `updated_at` timestamp
- Fallback text if Ridge model or no data: "โมเดลนี้ไม่มีข้อมูล feature importance"

---

## 4. Feature 3 — Anomaly Detection (Z-score)

### 4.1 Goal
Detect statistically unusual PM2.5 readings per device (spikes, sudden drops) using a rolling Z-score. Flag anomalies in Firebase, show them in the dashboard, and include them in Line notifications.

### 4.2 Algorithm
For each device, after fetching history:
1. Compute `rolling_mean` and `rolling_std` with window=20 on PM2.5
2. `z_score = (current_pm25 - rolling_mean) / rolling_std`
3. If `|z_score| > 3.0` AND `rolling_std > 1.0` (avoid false positives when values are stable near 0): anomaly
4. Type: `"spike"` if z > 3, `"drop"` if z < -3

### 4.3 Firebase Schema
Path: `/ai_analysis/anomalies`
```json
{
  "1715666400": {
    "device": "device1",
    "pm2_5": 187.3,
    "z_score": 4.2,
    "type": "spike",
    "timestamp": 1715666400,
    "datetime_str": "2026-05-14 08:00:00"
  }
}
```
- Keep only the 50 most recent anomaly keys (trim on write)

### 4.4 ai.py Changes
- `detect_anomalies(df, device)` function: returns list of anomaly dicts for latest reading
- Called after `fetch_all_devices()`, before training
- Writes anomalies to Firebase
- If anomaly detected: trigger Line push (same pipeline as Feature 1, separate cooldown 15 min per device)

### 4.5 index.html Changes
- Header badge: red dot with count if any anomaly in last 1 hour
- Stats tab section "เหตุผิดปกติล่าสุด":
  - Card per anomaly event: timestamp, device name, PM2.5 value, Z-score, type (spike/drop)
  - Color: red for spike, blue for drop
  - Max 10 shown, "ดูทั้งหมด" expands to 50
- Real-time Firebase listener on `/ai_analysis/anomalies`:
  - When new anomaly arrives → show alert toast (reuse existing `.alert-toast`)

---

## 5. Data Flow Summary

```
Every 10-min cycle (ai.py):
  1. fetch_all_devices() from Firebase
  2. detect_anomalies() → write /ai_analysis/anomalies
  3. train if needed → write /ai_analysis/feature_importance
  4. predict → write /ai_analysis/prediction
  5. check_and_notify():
     - read /settings/notify_config
     - if PM2.5 > threshold or anomaly: send Line + Web Push
```

---

## 6. Error Handling

- **Line API failure:** Log warning, do not crash cycle. Retry next cycle if condition still met.
- **Invalid Line token:** Log error, skip notification until config updated.
- **Push subscription expired:** Remove from Firebase silently.
- **Insufficient data for Z-score:** Skip anomaly check if rolling window has <10 values.
- **Ridge model:** Feature importance section shows "ไม่มีข้อมูล (Ridge model)".

---

## 7. New Dependencies (ai.py)

| Package | Purpose |
|---|---|
| `requests` | Line Messaging API HTTP calls |
| `pywebpush` | Web Push payload encryption |

---

## 8. Environment Variables (Railway)

| Var | Description |
|---|---|
| `VAPID_PRIVATE_KEY` | VAPID private key for Web Push |
| `VAPID_PUBLIC_KEY` | VAPID public key (also hardcoded in sw.js) |
| `VAPID_CLAIMS_EMAIL` | mailto: for VAPID claims |

---

## 9. Files to Change

| File | Changes |
|---|---|
| `ai.py` | Add `detect_anomalies()`, `write_feature_importance()`, `check_and_notify()`, new deps |
| `index.html` | Settings inputs, feature importance card, anomaly section, push registration |
| `sw.js` | Add `push` event handler |
| `requirements.txt` (new) | `requests`, `pywebpush`, existing deps |
