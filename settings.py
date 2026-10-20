import json
from pathlib import Path

FILE = Path("settings.json")

def _load() -> dict:
    if FILE.exists():
        try:
            return json.loads(FILE.read_text())
        except Exception:
            pass
    return {"dry_run": True}

def _save(data: dict):
    FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def is_dry_run() -> bool:
    return _load().get("dry_run", True)

def set_dry_run(value: bool):
    data = _load()
    data["dry_run"] = bool(value)
    _save(data)