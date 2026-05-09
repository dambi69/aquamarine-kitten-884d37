"""
รันไฟล์นี้เพื่อเช็คชื่อ column จริงๆ ใน Firebase
"""
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd

SERVICE_ACCOUNT_KEY = 'serviceAccountKey.json'
DATABASE_URL = 'https://test-70cc5-default-rtdb.asia-southeast1.firebasedatabase.app/'

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

# ดึงข้อมูล device แรกมาดู
sensor_ref = db.reference('/sensor').get()
devices = list(sensor_ref.keys())
print(f"Devices: {devices}")

device = devices[0]
raw = db.reference(f'/sensor/{device}/history').order_by_key().limit_to_last(3).get()

df = pd.DataFrame.from_dict(raw, orient='index')
print(f"\n=== Columns ใน {device} ===")
print(df.columns.tolist())
print("\n=== ตัวอย่างข้อมูล ===")
print(df.to_string())