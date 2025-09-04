import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict
from flask import Flask, jsonify, request

# Config
BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIRS = ["train", "preprocess", "prediction", "utils"]  # dirs to scan (relative to backend/)
ALLOWED_EXT = ".py"
MAX_EXECUTION_SECONDS = int(os.getenv("MAX_EXECUTION_SECONDS", "600"))  # timeout for scripts
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

def slugify(name: str) -> str:
    # simple slugify: lower, replace non-alnum with '-', collapse dashes
    import re
    s = name.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-{2,}', '-', s)
    s = s.strip('-')
    return s or name

def discover_scripts() -> Dict[str, Dict]:
    """
    Scan SCRIPT_DIRS for .py files and build a mapping:
    {
      script_id: { "path": "<absolute path>", "name": "<display name>", "relpath": "<train/foo.py>" }
    }
    script_id is unique; if collision occurs it will use folder-prefix.
    """
    scripts = {}
    collisions = {}

    for subdir in SCRIPT_DIRS:
        dir_path = BASE_DIR / subdir
        if not dir_path.exists() or not dir_path.is_dir():
            continue

        for p in dir_path.rglob("*.py"):
            if p.name.startswith("__"):
                continue
            rel = p.relative_to(BASE_DIR)
            stem = p.stem  # filename without ext
            candidate_id = slugify(stem)
            # ensure uniqueness: if already exists, prefix with folder name
            if candidate_id in scripts:
                candidate_id = f"{slugify(subdir)}-{candidate_id}"
            scripts[candidate_id] = {
                "id": candidate_id,
                "name": stem,
                "relpath": str(rel).replace(os.sep, "/"),
                "path": str(p.resolve())
            }

    return scripts

@app.route("/api/scripts", methods=["GET"])
def list_scripts():
    """Return available scripts discovered on disk"""
    scripts = discover_scripts()
    return jsonify({"status": "ok", "scripts": list(scripts.values())}), 200

def run_script_process(script_path: str, args: list = None, timeout: int = MAX_EXECUTION_SECONDS):
    """
    Execute a script with subprocess using the same Python interpreter.
    Returns dict with stdout, stderr, returncode, duration_seconds.
    """
    args = args or []
    cmd = [sys.executable, script_path] + args

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        duration = time.time() - start
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "duration_seconds": round(duration, 3)
        }, None
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        return None, {
            "error": "timeout",
            "message": f"Script exceeded timeout of {timeout}s",
            "duration_seconds": round(duration, 3)
        }
    except Exception as e:
        duration = time.time() - start
        return None, {
            "error": "internal",
            "message": str(e),
            "duration_seconds": round(duration, 3)
        }

@app.route("/api/run/<script_id>", methods=["POST"])
def run_script(script_id):
    """
    Run a discovered script by its id.
    Request JSON (optional):
      {
        "args": ["--foo", "bar"]   // optional list of args passed to the script
      }
    Response:
      {
        "status": "success"|"error",
        "id": script_id,
        "stdout": "...",
        "stderr": "...",
        "returncode": 0,
        "duration_seconds": 1.23
      }
    """
    scripts = discover_scripts()
    script_entry = scripts.get(script_id)
    if not script_entry:
        return jsonify({"status": "error", "message": f"Script id '{script_id}' not found", "available": list(scripts.keys())}), 404

    payload = request.get_json(silent=True) or {}
    args = payload.get("args", [])
    if not isinstance(args, list):
        return jsonify({"status": "error", "message": "Field 'args' must be a list of strings"}), 400

    # Extra safety: ensure path is inside BASE_DIR
    script_path = Path(script_entry["path"])
    try:
        script_path.relative_to(BASE_DIR)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid script path"}), 500

    # Run
    result, err = run_script_process(str(script_path), args=args)
    if err:
        return jsonify({"status": "error", "message": err}), 500

    # If script returns non-zero, still return 200 but status "script_error". (You can change behavior.)
    resp = {
        "status": "success" if result["returncode"] == 0 else "script_error",
        "id": script_id,
        "relpath": script_entry["relpath"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "returncode": result["returncode"],
        "duration_seconds": result["duration_seconds"]
    }
    # Optionally mask very large outputs
    max_len = 200000  # 200KB of combined output
    combined = (resp.get("stdout") or "") + (resp.get("stderr") or "")
    if len(combined) > max_len:
        resp["note"] = "Output truncated due to size"
        resp["stdout"] = (resp["stdout"] or "")[:max_len//2]
        resp["stderr"] = (resp["stderr"] or "")[:max_len//2]

    return jsonify(resp), 200

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "backend healthy"}), 200

if __name__ == "__main__":
    # dev config
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=DEBUG)
