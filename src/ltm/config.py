"""Environment/config loader with no third-party dependencies."""
from pathlib import Path
import os

def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def setting(name, default):
    return os.getenv(name, default)

def db_path():
    return setting("LTM_DB_PATH", "data/memory.sqlite3")

def top_k():
    try:
        return max(1, int(setting("LTM_TOP_K", "5")))
    except ValueError:
        return 5
