"""
AI DUST PREDICTOR v4
- XGBoost (primary) + Ridge regression (fallback for small datasets)
- Per-device models with R^2-weighted ensemble
- Column name normalization (handles various Firebase field naming)
- Single Firebase fetch per cycle (efficient)
- Adaptive retraining when model performance drops
- Robust error recovery for continuous operation
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# ── Logging ───────────────────────────────────────────────────────────────────
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.stream.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        _stream_handler,
        logging.FileHandler('ai_predictor.log', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT_KEY = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')
FIREBASE_CREDENTIALS_B64 = os.getenv('FIREBASE_CREDENTIALS_B64')  # Railway: base64-encoded JSON
DATABASE_URL        = 'https://test-70cc5-default-rtdb.asia-southeast1.firebasedatabase.app/'

SENSOR_BASE      = '/sensor'
PREDICTION_PATH  = '/ai_analysis/prediction'
HISTORY_PATH     = '/ai_analysis/forecast_history'
MODEL_SAVE_PATH  = 'dust_model_{device}.pkl'

TARGET_COLS    = ['pm1', 'pm2_5', 'pm10']
WEATHER_COLS   = ['temperature', 'humidity']
FEATURE_COLS   = TARGET_COLS + WEATHER_COLS

LAG_STEPS      = 6
ROLLING_WIN    = [3, 6, 12]

RETRAIN_EVERY        = 10      # retrain every N cycles
LOOP_INTERVAL        = 600     # seconds between prediction cycles
PREDICT_AHEAD_S      = 600     # prediction horizon label (seconds)
MIN_ROWS_TO_TRAIN    = 30      # minimum rows for Ridge fallback
MIN_ROWS_FOR_XGB     = 80      # minimum rows for XGBoost (use Ridge below this)
MAX_RETRIES          = 3
RETRY_DELAY          = 5
HISTORY_LIMIT        = 300     # recent regime only — old data has different distribution
RETRAIN_R2_THRESHOLD = 0.35    # force retrain if avg R^2 drops below this

# Column name aliases — maps canonical name -> possible Firebase field names
COL_ALIASES: dict[str, list[str]] = {
    'pm1'        : ['pm1', 'pm1_0', 'PM1', 'PM1_0', 'pm1.0'],
    'pm2_5'      : ['pm2_5', 'pm25', 'pm2.5', 'PM2_5', 'PM25', 'PM2.5'],
    'pm10'       : ['pm10', 'PM10'],
    'temperature': ['temperature', 'temp', 'Temperature', 'Temp', 'TEMP'],
    'humidity'   : ['humidity', 'humi', 'hum', 'Humidity', 'RH'],
    # timestamp/datetime are parsed separately — do NOT include here
}

FEATURE_IMP_PATH = '/ai_analysis/feature_importance'

FEATURE_LABEL: dict[str, str] = {
    # ── Raw base values (used in display, not direct feature_names entries for PM) ──
    'pm1'        : 'PM1 ปัจจุบัน',
    'pm2_5'      : 'PM2.5 ปัจจุบัน',
    'pm10'       : 'PM10 ปัจจุบัน',

    # ── Time features ────────────────────────────────────────────────────────────
    'hour'        : 'ชั่วโมงของวัน',
    'day_of_week' : 'วันในสัปดาห์',
    'month'       : 'เดือน',
    'is_rush'     : 'ชั่วโมงเร่งด่วน',
    'is_night'    : 'กลางคืน',
    'sin_hour'    : 'รอบเวลา (sin)',
    'cos_hour'    : 'รอบเวลา (cos)',
    'sin_dow'     : 'รอบสัปดาห์ (sin)',
    'cos_dow'     : 'รอบสัปดาห์ (cos)',

    # ── Base weather (appear directly in feature_names) ──────────────────────────
    'temperature' : 'อุณหภูมิ',
    'humidity'    : 'ความชื้น',

    # ── Delta features (เปลี่ยนแปลง) ─────────────────────────────────────────────
    'pm1_d1'         : 'PM1 เปลี่ยน 1 รอบ',
    'pm1_d2'         : 'PM1 เปลี่ยน 2 รอบ',
    'pm2_5_d1'       : 'PM2.5 เปลี่ยน 1 รอบ',
    'pm2_5_d2'       : 'PM2.5 เปลี่ยน 2 รอบ',
    'pm10_d1'        : 'PM10 เปลี่ยน 1 รอบ',
    'pm10_d2'        : 'PM10 เปลี่ยน 2 รอบ',
    'temperature_d1' : 'อุณหภูมิ เปลี่ยน 1 รอบ',
    'temperature_d2' : 'อุณหภูมิ เปลี่ยน 2 รอบ',
    'humidity_d1'    : 'ความชื้น เปลี่ยน 1 รอบ',
    'humidity_d2'    : 'ความชื้น เปลี่ยน 2 รอบ',

    # ── Lag features ─────────────────────────────────────────────────────────────
    'pm1_lag1'         : 'PM1 รอบก่อน',
    'pm1_lag2'         : 'PM1 2 รอบก่อน',
    'pm1_lag3'         : 'PM1 3 รอบก่อน',
    'pm1_lag4'         : 'PM1 4 รอบก่อน',
    'pm1_lag5'         : 'PM1 5 รอบก่อน',
    'pm1_lag6'         : 'PM1 6 รอบก่อน',
    'pm2_5_lag1'       : 'PM2.5 รอบก่อน',
    'pm2_5_lag2'       : 'PM2.5 2 รอบก่อน',
    'pm2_5_lag3'       : 'PM2.5 3 รอบก่อน',
    'pm2_5_lag4'       : 'PM2.5 4 รอบก่อน',
    'pm2_5_lag5'       : 'PM2.5 5 รอบก่อน',
    'pm2_5_lag6'       : 'PM2.5 6 รอบก่อน',
    'pm10_lag1'        : 'PM10 รอบก่อน',
    'pm10_lag2'        : 'PM10 2 รอบก่อน',
    'pm10_lag3'        : 'PM10 3 รอบก่อน',
    'pm10_lag4'        : 'PM10 4 รอบก่อน',
    'pm10_lag5'        : 'PM10 5 รอบก่อน',
    'pm10_lag6'        : 'PM10 6 รอบก่อน',
    'temperature_lag1' : 'อุณหภูมิ รอบก่อน',
    'temperature_lag2' : 'อุณหภูมิ 2 รอบก่อน',
    'temperature_lag3' : 'อุณหภูมิ 3 รอบก่อน',
    'temperature_lag4' : 'อุณหภูมิ 4 รอบก่อน',
    'temperature_lag5' : 'อุณหภูมิ 5 รอบก่อน',
    'temperature_lag6' : 'อุณหภูมิ 6 รอบก่อน',
    'humidity_lag1'    : 'ความชื้น รอบก่อน',
    'humidity_lag2'    : 'ความชื้น 2 รอบก่อน',
    'humidity_lag3'    : 'ความชื้น 3 รอบก่อน',
    'humidity_lag4'    : 'ความชื้น 4 รอบก่อน',
    'humidity_lag5'    : 'ความชื้น 5 รอบก่อน',
    'humidity_lag6'    : 'ความชื้น 6 รอบก่อน',

    # ── Rolling statistics — PM1 ─────────────────────────────────────────────────
    'pm1_rmean3'  : 'PM1 เฉลี่ย 3',
    'pm1_rmean6'  : 'PM1 เฉลี่ย 6',
    'pm1_rmean12' : 'PM1 เฉลี่ย 12',
    'pm1_rstd3'   : 'PM1 ผันผวน 3',
    'pm1_rstd6'   : 'PM1 ผันผวน 6',
    'pm1_rstd12'  : 'PM1 ผันผวน 12',
    'pm1_rmax3'   : 'PM1 สูงสุด 3',
    'pm1_rmax6'   : 'PM1 สูงสุด 6',
    'pm1_rmax12'  : 'PM1 สูงสุด 12',
    'pm1_rmin3'   : 'PM1 ต่ำสุด 3',
    'pm1_rmin6'   : 'PM1 ต่ำสุด 6',
    'pm1_rmin12'  : 'PM1 ต่ำสุด 12',

    # ── Rolling statistics — PM2.5 ───────────────────────────────────────────────
    'pm2_5_rmean3'  : 'PM2.5 เฉลี่ย 3',
    'pm2_5_rmean6'  : 'PM2.5 เฉลี่ย 6',
    'pm2_5_rmean12' : 'PM2.5 เฉลี่ย 12',
    'pm2_5_rstd3'   : 'PM2.5 ผันผวน 3',
    'pm2_5_rstd6'   : 'PM2.5 ผันผวน 6',
    'pm2_5_rstd12'  : 'PM2.5 ผันผวน 12',
    'pm2_5_rmax3'   : 'PM2.5 สูงสุด 3',
    'pm2_5_rmax6'   : 'PM2.5 สูงสุด 6',
    'pm2_5_rmax12'  : 'PM2.5 สูงสุด 12',
    'pm2_5_rmin3'   : 'PM2.5 ต่ำสุด 3',
    'pm2_5_rmin6'   : 'PM2.5 ต่ำสุด 6',
    'pm2_5_rmin12'  : 'PM2.5 ต่ำสุด 12',

    # ── Rolling statistics — PM10 ────────────────────────────────────────────────
    'pm10_rmean3'  : 'PM10 เฉลี่ย 3',
    'pm10_rmean6'  : 'PM10 เฉลี่ย 6',
    'pm10_rmean12' : 'PM10 เฉลี่ย 12',
    'pm10_rstd3'   : 'PM10 ผันผวน 3',
    'pm10_rstd6'   : 'PM10 ผันผวน 6',
    'pm10_rstd12'  : 'PM10 ผันผวน 12',
    'pm10_rmax3'   : 'PM10 สูงสุด 3',
    'pm10_rmax6'   : 'PM10 สูงสุด 6',
    'pm10_rmax12'  : 'PM10 สูงสุด 12',
    'pm10_rmin3'   : 'PM10 ต่ำสุด 3',
    'pm10_rmin6'   : 'PM10 ต่ำสุด 6',
    'pm10_rmin12'  : 'PM10 ต่ำสุด 12',

    # ── Interaction features ─────────────────────────────────────────────────────
    'temp_x_hum' : 'อุณหภูมิ×ความชื้น',
    'pm_ratio'   : 'สัดส่วน PM2.5/PM10',
    'pm1_pm25_r' : 'สัดส่วน PM1/PM2.5',
    'aqi_approx' : 'ค่า AQI ประมาณ',
}

# Possible names for the timestamp column in Firebase
TS_CANDIDATES = ['timestamp', 'time', 'ts', 'Timestamp', 'Time', 'datetime', 'Datetime']

# ── Firebase ──────────────────────────────────────────────────────────────────
def connect_firebase():
    if not firebase_admin._apps:
        if FIREBASE_CREDENTIALS_B64:
            import json, base64
            cred_dict = json.loads(base64.b64decode(FIREBASE_CREDENTIALS_B64))
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    log.info("Firebase connected")


def _fb_get(ref_path: str, limit: int = HISTORY_LIMIT):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return db.reference(ref_path).order_by_key().limit_to_last(limit).get()
        except Exception as exc:
            log.warning(f"Firebase GET attempt {attempt}/{MAX_RETRIES} [{ref_path}]: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def _fb_set(ref_path: str, data: dict) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            db.reference(ref_path).set(data)
            return True
        except Exception as exc:
            log.warning(f"Firebase SET attempt {attempt}/{MAX_RETRIES} [{ref_path}]: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return False

# ── Column normalization ───────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Firebase columns to canonical names using COL_ALIASES."""
    rename_map = {}
    for canonical, aliases in COL_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        log.info(f"  Column rename: {rename_map}")
        df = df.rename(columns=rename_map)
    return df

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_all_devices() -> dict[str, pd.DataFrame]:
    """
    Load sensor data from Firebase in a single pass.
    Returns {device_name: DataFrame} with canonical column names.
    """
    try:
        # Load entire /sensor tree once (gets device list + recent data)
        sensor_data = db.reference(SENSOR_BASE).get()
        if not sensor_data:
            log.warning("No data at /sensor")
            return {}

        devices = list(sensor_data.keys())
        log.info(f"Found {len(devices)} devices: {devices}")
        result = {}

        for device in devices:
            # Use already-loaded data; fall back to direct fetch for large histories
            device_data = sensor_data.get(device, {})
            raw = device_data.get('history')

            if not raw:
                # Try direct fetch (device history might exceed shallow load)
                raw = _fb_get(f'{SENSOR_BASE}/{device}/history', limit=HISTORY_LIMIT)

            if not raw:
                log.warning(f"{device}: no history data")
                continue

            # Keep only the last HISTORY_LIMIT records
            if isinstance(raw, dict):
                keys = sorted(raw.keys())[-HISTORY_LIMIT:]
                raw = {k: raw[k] for k in keys}
                df = pd.DataFrame.from_dict(raw, orient='index')
            else:
                log.warning(f"{device}: unexpected history format")
                continue

            # Parse timestamp BEFORE column normalization (avoids rename collision)
            ts_col = next((c for c in TS_CANDIDATES if c in df.columns), None)
            if ts_col:
                ts = pd.to_numeric(df[ts_col], errors='coerce')
                # If mostly NaN, the column likely holds datetime strings
                if ts.isna().mean() > 0.5:
                    ts = (
                        pd.to_datetime(df[ts_col], errors='coerce')
                        .astype('int64') // 10**9
                    )
                    ts = ts.where(ts > 0, other=np.nan)
                df['timestamp'] = ts
                # Handle millisecond timestamps (Unix ms > year 2100 in seconds)
                mask_ms = df['timestamp'] > 4_102_444_800
                df.loc[mask_ms, 'timestamp'] = df.loc[mask_ms, 'timestamp'] / 1000
            else:
                df['timestamp'] = np.arange(len(df)) * 60  # fallback: 1-min spacing

            # Now normalize sensor column names (temperature, humidity, pm2_5, etc.)
            df = normalize_columns(df)

            # Ensure all feature columns exist; fill missing weather with median
            for col in TARGET_COLS:
                if col not in df.columns:
                    log.warning(f"{device}: missing '{col}' — filling 0")
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors='coerce')

            for col in WEATHER_COLS:
                if col not in df.columns:
                    default = 25.0 if col == 'temperature' else 70.0
                    df[col] = default
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else (25.0 if col == 'temperature' else 70.0))

            df = df.sort_values('timestamp').reset_index(drop=True)

            # Clip outliers to 1st–99th percentile (preserve more data than IQR drop)
            for col in TARGET_COLS:
                lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
                df[col] = df[col].clip(lower=max(lo, 0), upper=hi)

            # Interpolate remaining NaNs
            df[FEATURE_COLS] = df[FEATURE_COLS].interpolate('linear').ffill().bfill()
            df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

            result[device] = df[['timestamp'] + FEATURE_COLS]
            log.info(f"  {device}: {len(df)} rows")

        return result

    except Exception as exc:
        log.error(f"fetch_all_devices error: {exc}")
        return {}

# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy().reset_index(drop=True)

    # Time features
    dt = pd.to_datetime(out['timestamp'], unit='s', utc=True).dt.tz_convert('Asia/Bangkok')
    out['hour']        = dt.dt.hour
    out['day_of_week'] = dt.dt.dayofweek
    out['month']       = dt.dt.month
    out['is_rush']     = out['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    out['is_night']    = ((out['hour'] >= 22) | (out['hour'] <= 6)).astype(int)
    out['sin_hour']    = np.sin(2 * np.pi * out['hour'] / 24)
    out['cos_hour']    = np.cos(2 * np.pi * out['hour'] / 24)
    out['sin_dow']     = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['cos_dow']     = np.cos(2 * np.pi * out['day_of_week'] / 7)

    feature_names = [
        'hour', 'day_of_week', 'month', 'is_rush', 'is_night',
        'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'temperature', 'humidity',
    ]

    # Delta features (rate of change)
    for col in FEATURE_COLS:
        out[f'{col}_d1'] = out[col].diff(1)
        out[f'{col}_d2'] = out[col].diff(2)
        feature_names += [f'{col}_d1', f'{col}_d2']

    # Lag features
    for col in FEATURE_COLS:
        for lag in range(1, LAG_STEPS + 1):
            name = f'{col}_lag{lag}'
            out[name] = out[col].shift(lag)
            feature_names.append(name)

    # Rolling statistics
    for col in TARGET_COLS:
        for w in ROLLING_WIN:
            shifted = out[col].shift(1)
            out[f'{col}_rmean{w}'] = shifted.rolling(w, min_periods=1).mean()
            out[f'{col}_rstd{w}']  = shifted.rolling(w, min_periods=2).std().fillna(0)
            out[f'{col}_rmax{w}']  = shifted.rolling(w, min_periods=1).max()
            out[f'{col}_rmin{w}']  = shifted.rolling(w, min_periods=1).min()
            feature_names += [f'{col}_rmean{w}', f'{col}_rstd{w}',
                               f'{col}_rmax{w}', f'{col}_rmin{w}']

    # Interaction features
    out['temp_x_hum']  = out['temperature'] * out['humidity']
    out['pm_ratio']    = out['pm2_5'] / (out['pm10'] + 1e-6)
    out['pm1_pm25_r']  = out['pm1'] / (out['pm2_5'] + 1e-6)
    out['aqi_approx']  = (out['pm2_5'] / 35.4).clip(0, 6)  # rough AQI scale
    feature_names += ['temp_x_hum', 'pm_ratio', 'pm1_pm25_r', 'aqi_approx']

    out.dropna(inplace=True)
    out.reset_index(drop=True, inplace=True)

    valid = [c for c in feature_names if c in out.columns]
    return out, valid

# ── Model selection ───────────────────────────────────────────────────────────
def _make_xgb_model() -> MultiOutputRegressor:
    """XGBoost model tuned for air quality time series (PM tabular prediction)."""
    xgb = XGBRegressor(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=0.05,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    return MultiOutputRegressor(xgb, n_jobs=1)


def _make_ridge_model() -> Pipeline:
    """Ridge regression fallback — fast, stable, works with small datasets."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0)),
    ])


# ── Model persistence ─────────────────────────────────────────────────────────
def load_saved_models() -> tuple[dict, dict, dict]:
    models, feat_cols, metrics = {}, {}, {}
    for path in sorted(Path('.').glob('dust_model_*.pkl')):
        device = path.stem.replace('dust_model_', '')
        try:
            saved = joblib.load(path)
            if not saved.get('delta', False):
                log.info(f"Skipping legacy model [{device}] (non-delta) — will retrain")
                continue
            models[device]    = saved['model']
            feat_cols[device] = saved['features']
            metrics[device]   = saved.get('metrics', (0.0, 0.0))
            kind              = saved.get('kind', 'xgb')
            log.info(f"Loaded model [{device}] type={kind}  R^2={metrics[device][0]:.3f}")
        except Exception as exc:
            log.warning(f"Could not load {path}: {exc}")
    return models, feat_cols, metrics

# ── Training ──────────────────────────────────────────────────────────────────
def train_all_devices(device_dfs: dict) -> tuple[dict, dict, dict]:
    models, feat_cols, metrics_dict = {}, {}, {}

    for device, df in device_dfs.items():
        log.info(f"Training [{device}]  rows={len(df)}")
        df_feat, f_cols = build_features(df)

        n_rows = len(df_feat)
        if n_rows < MIN_ROWS_TO_TRAIN:
            log.warning(f"  {device}: {n_rows} rows after feature eng — need ≥{MIN_ROWS_TO_TRAIN}, skip")
            continue

        # Predict delta (PM_{t+1} - PM_t) — removes trend, makes target stationary
        vals = df_feat[TARGET_COLS].values
        X = df_feat[f_cols].values[:-1]
        y = vals[1:] - vals[:-1]
        n_rows = len(X)

        if n_rows < MIN_ROWS_TO_TRAIN:
            log.warning(f"  {device}: {n_rows} rows after delta — skip")
            continue

        # Choose model based on available data
        use_xgb = n_rows >= MIN_ROWS_FOR_XGB
        model_kind = 'xgb' if use_xgb else 'ridge'
        model = _make_xgb_model() if use_xgb else _make_ridge_model()
        log.info(f"  {device}: using {'XGBoost' if use_xgb else 'Ridge'} ({n_rows} rows)")

        # Cross-validation for performance estimate
        n_splits = min(5, n_rows // 20)
        if n_splits < 2:
            model.fit(X, y)
            perf = (0.0, 0.0)
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            r2_list, mae_list = [], []
            for tr, te in tscv.split(X):
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
                r2_list.append(r2_score(y[te], pred, multioutput='uniform_average'))
                mae_list.append(mean_absolute_error(y[te], pred))

            mean_r2  = float(np.mean(r2_list))
            mean_mae = float(np.mean(mae_list))
            log.info(f"  {device}: R^2={mean_r2:.4f} ({mean_r2*100:.1f}%)  MAE={mean_mae:.4f}")
            log.info(f"  {device}: fold R^2s = {[f'{v:.3f}' for v in r2_list]}")

            # Final fit on all data
            model.fit(X, y)

            if use_xgb:
                write_feature_importance(device, model, f_cols)

            perf = (mean_r2, mean_mae)

        path = MODEL_SAVE_PATH.format(device=device)
        joblib.dump({'model': model, 'features': f_cols, 'metrics': perf, 'kind': model_kind, 'delta': True}, path)
        log.info(f"  Saved -> {path}")

        models[device]       = model
        feat_cols[device]    = f_cols
        metrics_dict[device] = perf

    return models, feat_cols, metrics_dict

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_all(
    models: dict, feat_cols: dict, metrics_dict: dict, device_dfs: dict
) -> tuple[np.ndarray | None, dict]:
    """
    Predict next cycle for each device.
    Weighted average uses max(R^2, 0.01) — better models get more weight.
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
            df_sorted = df_feat.sort_values('timestamp')
            latest = df_sorted.iloc[[-1]].copy()

            # Align feature columns — use assign() to avoid pandas CoW issues
            missing = {c: 0.0 for c in f_cols if c not in latest.columns}
            if missing:
                latest = latest.assign(**missing)

            raw_pred = model.predict(latest[f_cols].values)[0]

            # Model predicts delta; add to last known PM value
            last_pm = df_sorted[TARGET_COLS].iloc[-1].values
            delta = np.clip(raw_pred, -30, 30)
            pred = np.clip(last_pm + delta, 0, None)

            r2 = metrics_dict.get(device, (0.0, 0.0))[0]
            w  = max(r2, 0.01)

            per_device[device] = {
                'pm1'  : round(float(pred[0]), 2),
                'pm2_5': round(float(pred[1]), 2),
                'pm10' : round(float(pred[2]), 2),
                'r2'   : round(r2, 4),
            }
            all_preds.append(pred)
            weights.append(w)
            log.info(f"  {device}: PM1={pred[0]:.1f}  PM2.5={pred[1]:.1f}  PM10={pred[2]:.1f}  R^2={r2:.3f}")

        except Exception as exc:
            log.error(f"predict [{device}]: {exc}")

    if not all_preds:
        return None, {}

    w = np.array(weights) / np.sum(weights)
    final = np.average(np.array(all_preds), axis=0, weights=w)
    return final, per_device

# ── AQI classification ────────────────────────────────────────────────────────
def classify_aqi(pm25: float) -> dict:
    if   pm25 <= 12.0 : return {'level': 1, 'label': 'ดีมาก',         'color': '#16a34a'}
    elif pm25 <= 35.4 : return {'level': 2, 'label': 'ดี',            'color': '#65a30d'}
    elif pm25 <= 55.4 : return {'level': 3, 'label': 'ปานกลาง',       'color': '#d97706'}
    elif pm25 <= 150.4: return {'level': 4, 'label': 'มีผลต่อสุขภาพ', 'color': '#ea580c'}
    elif pm25 <= 250.4: return {'level': 5, 'label': 'อันตราย',        'color': '#dc2626'}
    else              : return {'level': 6, 'label': 'อันตรายมาก',     'color': '#7c3aed'}

# ── Feature importance ────────────────────────────────────────────────────────
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

# ── Upload to Firebase ────────────────────────────────────────────────────────
def upload_result(preds: np.ndarray, metrics_dict: dict, per_device: dict) -> dict:
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
        'confidence'       : round(max(avg_r2, 0) * 100, 1),
        'aqi_level'        : aqi,
        'per_device'       : per_device,
        'predict_horizon_s': PREDICT_AHEAD_S,
        'updated_at'       : ts,
        'updated_str'      : datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
    }

    ok = _fb_set(PREDICTION_PATH, data)
    _fb_set(f'{HISTORY_PATH}/{ts}', data)
    if ok:
        log.info(f"Uploaded -> {PREDICTION_PATH}")
    return data

# ── Should we retrain? ────────────────────────────────────────────────────────
def _should_retrain(cycle: int, metrics_dict: dict) -> bool:
    if cycle % RETRAIN_EVERY == 0:
        return True
    r2_vals = [v[0] for v in metrics_dict.values() if v[0] > 0]
    if r2_vals and np.mean(r2_vals) < RETRAIN_R2_THRESHOLD:
        log.info(f"R^2 dropped below {RETRAIN_R2_THRESHOLD:.2f} — forcing retrain")
        return True
    return False

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  AI DUST PREDICTOR v4  (XGBoost + Ridge • adaptive retrain)")
    log.info("=" * 60)

    connect_firebase()

    models, feat_cols, metrics_dict = load_saved_models()
    if models:
        log.info(f"Loaded {len(models)} saved model(s) — will retrain on cycle {RETRAIN_EVERY}")
    else:
        log.info("No saved models found — training on first cycle")

    cycle = 0
    consecutive_errors = 0

    try:
        while True:
            cycle += 1
            log.info(f"──── Cycle {cycle}  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ────")

            try:
                device_dfs = fetch_all_devices()
                if not device_dfs:
                    log.warning(f"No device data — retrying in {LOOP_INTERVAL}s")
                    consecutive_errors += 1
                    time.sleep(LOOP_INTERVAL)
                    continue

                # Train or retrain
                if not models or _should_retrain(cycle, metrics_dict):
                    new_models, new_feats, new_metrics = train_all_devices(device_dfs)
                    if new_models:
                        models, feat_cols, metrics_dict = new_models, new_feats, new_metrics
                    elif not models:
                        log.warning("Training produced no models — will retry next cycle")
                        time.sleep(LOOP_INTERVAL)
                        continue

                # Predict and upload
                if models:
                    preds, per_device = predict_all(models, feat_cols, metrics_dict, device_dfs)

                    if preds is not None:
                        aqi     = classify_aqi(float(preds[1]))
                        r2_vals = [v[0] for v in metrics_dict.values() if v[0] > 0]
                        avg_r2  = np.mean(r2_vals) if r2_vals else 0
                        avg_mae = np.mean([v[1] for v in metrics_dict.values() if v[1] > 0]) if r2_vals else 0

                        log.info(f"Result -> PM1={preds[0]:.2f}  PM2.5={preds[1]:.2f}  PM10={preds[2]:.2f}")
                        log.info(f"         AQI={aqi['label']}  R^2={avg_r2*100:.1f}%  MAE={avg_mae:.2f} ug/m3")

                        upload_result(preds, metrics_dict, per_device)
                    else:
                        # Models may be stale/incompatible — force retrain next cycle
                        log.warning("Prediction failed — forcing retrain on next cycle")
                        models = {}

                consecutive_errors = 0

            except Exception as exc:
                consecutive_errors += 1
                log.error(f"Cycle {cycle} error (#{consecutive_errors}): {exc}", exc_info=True)
                # Back off if repeated errors
                if consecutive_errors >= 5:
                    log.error("5 consecutive errors — sleeping 5 minutes before retry")
                    time.sleep(300)
                    consecutive_errors = 0

            log.info(f"Sleeping {LOOP_INTERVAL}s…")
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        log.info("Shutting down gracefully.")


if __name__ == '__main__':
    main()
