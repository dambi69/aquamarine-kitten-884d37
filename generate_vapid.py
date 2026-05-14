"""Run once to generate VAPID keys. Copy output into Railway environment variables."""
try:
    import base64
    from py_vapid import Vapid
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Fix: pip install -r requirements.txt")
    raise SystemExit(1)

v = Vapid()
v.generate_keys()

private_pem = v.private_pem().decode().strip()
pub_bytes   = v.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
public_b64  = base64.urlsafe_b64encode(pub_bytes).decode().rstrip('=')

print("=== Copy these three values to Railway -> Variables ===")
print(f"VAPID_PRIVATE_KEY={private_pem}")
print(f"VAPID_PUBLIC_KEY={public_b64}")
print(f"VAPID_CLAIMS_EMAIL=mailto:blaster.teen15@gmail.com")
