"""
AI DUST PREDICTOR v3
- Per-device XGBoost models with R²-weighted average
- Model persistence (load on restart, retrain on schedule)
- Proper logging, retry logic, prediction horizon label
- Graceful shutdown
"""

import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_predictor.log', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_KEY = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')
DATABASE_URL        = 'https://test-70cc5-default-rtdb.asia-southeast1.firebasedatabase.app/'

SENSOR_BASE      = '/sensor'
PREDICTION_PATH  = '/ai_analysis/prediction'
HISTORY_PATH     = '/ai_analysis/forecast_history'
MODEL_SAVE_PATH  = 'dust_model_{device}.pkl'

FEATURE_COLS      = ['pm1', 'pm2_5', 'pm10', 'temperature', 'humidity']
TARGET_COLS       = ['pm1', 'pm2_5', 'pm10']
LAG_STEPS         = 6
ROLLING_WINDOWS   = [3, 6, 12]

RETRAIN_EVERY     = 10        # retrain every N cycles
LOOP_INTERVAL     = 600       # seconds between cycles
PREDICT_AHEAD_S   = 600       # label: prediction horizon in seconds
MIN_ROWS_TO_TRAIN = 50
MAX_RETRIES       = 3
RETRY_DELAY       = 5         # seconds between Firebase retries

# ── Firebase ──────────────────────────────────────────────────────────────────
def connect_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    log.info("Firebase connected")


def _fb_get(ref_path, limit=800):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return db.reference(ref_path).order_by_key().limit_to_last(limit).get()
        except Exception as exc:
            log.warning(f"Firebase GET attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def _fb_set(ref_path, data):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            db.reference(ref_path).set(data)
            return True
        except Exception as exc:
            log.warning(f"Firebase SET attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_all_devices(limit=800):
    """Return {device_name: DataFrame} for all /sensor/* paths."""
    try:
        sensor_ref = db.reference(SENSOR_BASE).get()
        if not sensor_ref:
            log.warning("No data at /sensor")
            return {}

        devices = list(sensor_ref.keys())
        log.info(f"Found {len(devices)} devices: {devices}")
        result = {}

        for device in devices:
            raw = _fb_get(f'{SENSOR_BASE}/{device}/history', limit=limit)
            if not raw:
                log.warning(f"{device}: no history")
                continue

            df = pd.DataFrame.from_dict(raw, orient='index')

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            elif 'datetime' in df.columns:
                df['timestamp'] = (
                    pd.to_datetime(df['datetime'], errors='coerce')
                    .astype('int64') // 10**9
                )
            else:
                df['timestamp'] = np.arange(len(df))

            for col in FEATURE_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    log.warning(f"{device}: missing '{col}', filling 0")
                    df[col] = 0.0

            df = df.sort_values('timestamp').reset_index(drop=True)

            for col in TARGET_COLS:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 3 * IQR) & (df[col] <= Q3 + 3 * IQR)]

            df[FEATURE_COLS] = df[FEATURE_COLS].interpolate('linear').ffill().bfill()
            df = df.reset_index(drop=True)

            result[device] = df[['timestamp'] + FEATURE_COLS]
            log.info(f"  {device}: {len(df)} rows (after cleaning)")

        return result

    except Exception as exc:
        log.error(f"fetch_all_devices: {exc}")
        return {}

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    out = df.copy().reset_index(drop=True)

    dt = pd.to_datetime(out['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Bangkok')
    out['hour']        = dt.dt.hour
    out['day_of_week'] = dt.dt.dayofweek
    out['month']       = dt.dt.month
    out['is_rush']     = out['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    out['is_night']    = ((out['hour'] >= 22) | (out['hour'] <= 6)).astype(int)
    out['sin_hour']    = np.sin(2 * np.pi * out['hour'] / 24)
    out['cos_hour']    = np.cos(2 * np.pi * out['hour'] / 24)

    feature_names = [
        'hour', 'day_of_week', 'month', 'is_rush', 'is_night',
        'sin_hour', 'cos_hour', 'temperature', 'humidity',
    ]

    for col in FEATURE_COLS:
        out[f'{col}_delta1'] = out[col].diff(1)
        out[f'{col}_delta2'] = out[col].diff(2)
        feature_names += [f'{col}_delta1', f'{col}_delta2']

    for col in FEATURE_COLS:
        for lag in range(1, LAG_STEPS + 1):
            name = f'{col}_lag{lag}'
            out[name] = out[col].shift(lag)
            feature_names.append(name)

    for col in TARGET_COLS:
        for w in ROLLING_WINDOWS:
            out[f'{col}_rmean{w}'] = out[col].shift(1).rolling(w).mean()
            out[f'{col}_rstd{w}']  = out[col].shift(1).rolling(w).std()
            out[f'{col}_rmax{w}']  = out[col].shift(1).rolling(w).max()
            feature_names += [f'{col}_rmean{w}', f'{col}_rstd{w}', f'{col}_rmax{w}']

    out['temp_x_hum'] = out['temperature'] * out['humidity']
    out['pm_ratio']   = out['pm2_5'] / (out['pm10'] + 1e-6)
    out['pm1_pm25_r'] = out['pm1'] / (out['pm2_5'] + 1e-6)
    feature_names += ['temp_x_hum', 'pm_ratio', 'pm1_pm25_r']

    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)

    valid = [c for c in feature_names if c in out.columns]
    return out, valid

# ── Model persistence ─────────────────────────────────────────────────────────
def load_saved_models():
    """Load all dust_model_*.pkl files from disk. Returns (models, feat_cols, metrics)."""
    models, feat_cols, metrics = {}, {}, {}
    for path in sorted(Path('.').glob('dust_model_*.pkl')):
        device = path.stem.replace('dust_model_', '')
        try:
            saved = joblib.load(path)
            models[device]    = saved['model']
            feat_cols[device] = saved['features']
            metrics[device]   = saved.get('metrics', (0.0, 0.0))
            log.info(f"Loaded model: {device}  R²={metrics[device][0]:.3f}")
        except Exception as exc:
            log.warning(f"Could not load {path}: {exc}")
    return models, feat_cols, metrics

# ── Training ──────────────────────────────────────────────────────────────────
def train_all_devices(device_dfs: dict):
    models, feat_cols, metrics_dict = {}, {}, {}

    for device, df in device_dfs.items():
        log.info(f"Training [{device}]  rows={len(df)}")
        df_feat, f_cols = build_features(df)

        if len(df_feat) < MIN_ROWS_TO_TRAIN:
            log.warning(f"  {device}: only {len(df_feat)} rows after feature eng — skip")
            continue

        X = df_feat[f_cols].values
        y = df_feat[TARGET_COLS].values

        xgb = XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
            random_state=42, verbosity=0,
        )
        model = MultiOutputRegressor(xgb)

        n_splits = min(5, len(X) // 20)
        if n_splits < 2:
            log.warning(f"  {device}: not enough data for CV — training without CV")
            model.fit(X, y)
            perf = (0.0, 0.0)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            r2_list, mae_list = [], []
            for tr, te in tscv.split(X):
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
                r2_list.append(r2_score(y[te], pred))
                mae_list.append(mean_absolute_error(y[te], pred))

            mean_r2  = float(np.mean(r2_list))
            mean_mae = float(np.mean(mae_list))
            log.info(f"  {device}: R²={mean_r2:.4f} ({mean_r2*100:.1f}%)  MAE={mean_mae:.4f}")
            log.info(f"  {device}: fold R²s = {[f'{v:.3f}' for v in r2_list]}")

            model.fit(X, y)
            imps = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
            top5 = sorted(zip(f_cols, imps), key=lambda x: -x[1])[:5]
            log.info(f"  {device}: top-5 features = {[t[0] for t in top5]}")
            perf = (mean_r2, mean_mae)

        path = MODEL_SAVE_PATH.format(device=device)
        joblib.dump({'model': model, 'features': f_cols, 'metrics': perf}, path)
        log.info(f"  Saved → {path}")

        models[device]       = model
        feat_cols[device]    = f_cols
        metrics_dict[device] = perf

    return models, feat_cols, metrics_dict

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_all(models, feat_cols, metrics_dict, device_dfs):
    """
    Predict next cycle for each device.
    Weighted average uses max(R², 0.01) so better-performing models get more weight.
    Returns: (weighted_avg_array, per_device_dict)
    """
    all_preds, weights = [], []
    per_device = {}

    for device, model in models.items():
        df = device_dfs.get(device)
        if df is None or len(df) == 0:
            continue
        try:
            df_feat, _ = build_features(df)
            if df_feat.empty:
                continue

            f_cols = feat_cols[device]
            latest = df_feat.sort_values('timestamp').iloc[[-1]]
            pred   = np.clip(model.predict(latest[f_cols].values)[0], 0, None)

            r2 = metrics_dict.get(device, (0.0, 0.0))[0]
            w  = max(r2, 0.01)

            per_device[device] = {
                'pm1':   round(float(pred[0]), 2),
                'pm2_5': round(float(pred[1]), 2),
                'pm10':  round(float(pred[2]), 2),
                'r2':    round(r2, 4),
            }
            all_preds.append(pred)
            weights.append(w)
            log.info(f"  {device}: PM1={pred[0]:.1f}  PM2.5={pred[1]:.1f}  PM10={pred[2]:.1f}  R²={r2:.3f}")

        except Exception as exc:
            log.error(f"predict [{device}]: {exc}")

    if not all_preds:
        return None, {}

    w = np.array(weights)
    w = w / w.sum()
    final = np.average(np.array(all_preds), axis=0, weights=w)
    return final, per_device

# ── AQI ───────────────────────────────────────────────────────────────────────
def classify_aqi(pm25):
    if   pm25 <= 12.0 : return {'level': 1, 'label': 'ดีมาก',          'color': 'green'}
    elif pm25 <= 35.4 : return {'level': 2, 'label': 'ดี',             'color': 'lightgreen'}
    elif pm25 <= 55.4 : return {'level': 3, 'label': 'ปานกลาง',        'color': 'yellow'}
    elif pm25 <= 150.4: return {'level': 4, 'label': 'มีผลต่อสุขภาพ',  'color': 'orange'}
    elif pm25 <= 250.4: return {'level': 5, 'label': 'อันตราย',         'color': 'red'}
    else              : return {'level': 6, 'label': 'อันตรายมาก',      'color': 'purple'}

# ── Upload ────────────────────────────────────────────────────────────────────
def upload_result(preds, metrics_dict, per_device):
    ts  = int(time.time())
    aqi = classify_aqi(float(preds[1]))

    r2_vals  = [v[0] for v in metrics_dict.values() if v[0] > 0]
    mae_vals = [v[1] for v in metrics_dict.values() if v[1] > 0]
    avg_r2   = float(np.mean(r2_vals))  if r2_vals  else 0.0
    avg_mae  = float(np.mean(mae_vals)) if mae_vals else 0.0

    data = {
        'pm1'              : round(float(preds[0]), 2),
        'pm2_5'            : round(float(preds[1]), 2),
        'pm10'             : round(float(preds[2]), 2),
        'r2_score'         : round(avg_r2, 4),
        'mae'              : round(avg_mae, 4),
        'confidence'       : round(avg_r2 * 100, 1),
        'aqi_level'        : aqi,
        'per_device'       : per_device,
        'predict_horizon_s': PREDICT_AHEAD_S,
        'updated_at'       : ts,
        'updated_str'      : datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
    }

    ok = _fb_set(PREDICTION_PATH, data)
    _fb_set(f'{HISTORY_PATH}/{ts}', data)
    if ok:
        log.info(f"Uploaded → {PREDICTION_PATH}")
    return data

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  AI DUST PREDICTOR v3  (per-device • R²-weighted • persistent)")
    log.info("=" * 60)

    connect_firebase()

    models, feat_cols, metrics_dict = load_saved_models()
    if models:
        log.info(f"Loaded {len(models)} saved model(s) — will retrain on cycle {RETRAIN_EVERY}")
    else:
        log.info("No saved models found — will train on first cycle")

    cycle = 0
    try:
        while True:
            cycle += 1
            log.info(f"──── Cycle {cycle}  [{datetime.now().strftime('%H:%M:%S')}] ────")

            device_dfs = fetch_all_devices()
            if not device_dfs:
                log.warning(f"No device data — sleeping {LOOP_INTERVAL}s")
                time.sleep(LOOP_INTERVAL)
                continue

            if not models or cycle % RETRAIN_EVERY == 0:
                models, feat_cols, metrics_dict = train_all_devices(device_dfs)

            if models:
                preds, per_device = predict_all(models, feat_cols, metrics_dict, device_dfs)

                if preds is not None:
                    aqi      = classify_aqi(float(preds[1]))
                    r2_vals  = [v[0] for v in metrics_dict.values() if v[0] > 0]
                    mae_vals = [v[1] for v in metrics_dict.values() if v[1] > 0]
                    avg_r2   = np.mean(r2_vals)  if r2_vals  else 0
                    avg_mae  = np.mean(mae_vals) if mae_vals else 0

                    log.info(f"Result → PM1={preds[0]:.2f}  PM2.5={preds[1]:.2f}  PM10={preds[2]:.2f}")
                    log.info(f"         AQI={aqi['label']}  R²={avg_r2*100:.1f}%  MAE={avg_mae:.2f} µg/m³")

                    upload_result(preds, metrics_dict, per_device)

            log.info(f"Sleeping {LOOP_INTERVAL}s…")
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log.info("Shutting down gracefully.")


if __name__ == '__main__':
    main()
