import os, json, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise ValueError(
        "SECRET_KEY not found in environment variables. "
        f"Please add it to {env_path}"
    )

SECRET_KEY_BYTES = base64.b64decode(SECRET_KEY)

def encrypt_payload(obj: dict) -> str:
    raw = json.dumps(obj).encode("utf-8")
    cipher = AES.new(SECRET_KEY_BYTES, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(raw, AES.block_size))
    payload = cipher.iv + ct_bytes  # IV in testa
    return base64.b64encode(payload).decode()

def decrypt_payload(encrypted_b64: str) -> dict:
    raw = base64.b64decode(encrypted_b64)
    iv, ct = raw[:16], raw[16:]
    cipher = AES.new(SECRET_KEY_BYTES, AES.MODE_CBC, iv)
    decrypted = unpad(cipher.decrypt(ct), AES.block_size)
    return json.loads(decrypted.decode("utf-8"))
