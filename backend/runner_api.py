from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, uuid, os, shlex, threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Runner API")

# Allow local dev and containerized frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scripts base: point to backend folder (this file's directory)
SCRIPTS_DIR = Path(__file__).resolve().parent

# Settings
DEFAULT_TIMEOUT_SEC = int(os.getenv("RUNNER_TIMEOUT_SEC", "1800"))
MAX_OUTPUT_CHARS = int(os.getenv("RUNNER_MAX_OUTPUT_CHARS", "200000"))

class RunRequest(BaseModel):
    script: str
    args: dict = None
    env: dict = None


class TrainSelection(BaseModel):
    # models can include: 'result', 'ah', 'goals'
    models: Optional[List[str]] = None

RUNS = {}

@app.get('/health')
def health():
    return {"status":"ok", "scripts_dir": str(SCRIPTS_DIR)}

def _safe_script_path(rel_script: str) -> Path:
    # Normalize and ensure the path stays within SCRIPTS_DIR
    p = (SCRIPTS_DIR / rel_script).resolve()
    try:
        p.relative_to(SCRIPTS_DIR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid script path")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"Script not found: {rel_script}")
    return p


def _build_cmd(script_path: Path, args: Optional[dict]) -> List[str]:
    cmd = ["python", str(script_path)]
    if args:
        for k, v in args.items():
            # simple --key value formatting
            cmd.append(f"--{k}")
            if v is not None and v != "":
                cmd.append(str(v))
    return cmd


def _run_once(rel_script: str, args: Optional[dict] = None, extra_env: Optional[dict] = None, timeout: Optional[int] = None):
    script_path = _safe_script_path(rel_script)
    # Run in the script's directory so its relative paths work
    workdir = script_path.parent
    cmd = _build_cmd(script_path, args)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    timeout = timeout or DEFAULT_TIMEOUT_SEC
    started = datetime.utcnow().isoformat() + "Z"
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout,
    )
    output = proc.stdout or ""
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
    return {
        "script": rel_script,
        "returncode": proc.returncode,
        "output": output,
        "started_at": started,
        "duration_sec": None,
    }


@app.post('/api/run_sync')
def run_sync(req: RunRequest):
    run_id = uuid.uuid4().hex[:12]
    try:
        result = _run_once(req.script, args=req.args, extra_env=req.env)
        RUNS[run_id] = result
        return {"run_id": run_id, **result}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Script timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/run/{run_id}')
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail='run not found')
    return RUNS[run_id]


# Workflow endpoints expected by the frontend
def _run_workflow(steps: List[str]) -> Dict:
    results = []
    overall_code = 0
    for idx, step in enumerate(steps, start=1):
        try:
            res = _run_once(step)
        except subprocess.TimeoutExpired:
            res = {"script": step, "returncode": -1, "output": "[timeout]", "duration_sec": None}
        except HTTPException as he:
            res = {"script": step, "returncode": -2, "output": f"[error] {he.detail}", "duration_sec": None}
        except Exception as e:
            res = {"script": step, "returncode": -3, "output": f"[error] {str(e)}", "duration_sec": None}
        results.append(res)
        if res.get("returncode", 1) != 0 and overall_code == 0:
            overall_code = res["returncode"]
    aggregated_output = "\n\n".join(
        [f"==> {r['script']} (code {r['returncode']})\n{r.get('output','')}" for r in results]
    )
    if len(aggregated_output) > MAX_OUTPUT_CHARS:
        aggregated_output = aggregated_output[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
    return {"returncode": overall_code, "steps": results, "output": aggregated_output}


@app.post('/api/run/data-process')
def run_data_process():
    steps = [
        'z_latest_performance.py',
        'z_latest_team_scr_ref.py',
    ]
    return _run_workflow(steps)


@app.post('/api/run/train-model')
def run_train_model(selection: Optional[TrainSelection] = None):
    # Always run preparatory steps, then selected model trainings
    base_steps = [
        'adjust_team_perform.py',
        'set_train_data.py',
        'ml_models/data-eng.py',
    ]
    model_map = {
        'result': 'ml_models/train_model_new.py',
        'ah': 'ml_models/train_AH_model.py',
        'goals': 'ml_models/train_goals_model.py',
    }
    if selection and selection.models:
        chosen = [model_map[m] for m in selection.models if m in model_map]
        if not chosen:
            chosen = list(model_map.values())
    else:
        chosen = list(model_map.values())
    steps = base_steps + chosen
    return _run_workflow(steps)


@app.post('/api/run/train-selected')
def run_train_selected(selection: Optional[TrainSelection] = None):
    # Run only selected training scripts (assumes prepared dataset exists)
    model_map = {
        'result': 'ml_models/train_model_new.py',
        'ah': 'ml_models/train_AH_model.py',
        'goals': 'ml_models/train_goals_model.py',
    }
    if selection and selection.models:
        steps = [model_map[m] for m in selection.models if m in model_map]
        if not steps:
            steps = list(model_map.values())
    else:
        steps = list(model_map.values())
    return _run_workflow(steps)


@app.post('/api/run/prepare-data')
def run_prepare_data():
    steps = [
        'adjust_team_perform.py',
        'set_train_data.py',
        'ml_models/data-eng.py',
    ]
    return _run_workflow(steps)


@app.post('/api/run/predict-res')
def run_predict_res():
    steps = [
        'ml_models/predict_fixtures.py',
        'ml_models/predict_AH_model.py',
        'ml_models/predict_goals.py',
    ]
    return _run_workflow(steps)


@app.post('/api/run/send-telegram')
def run_send_telegram():
    steps = [
        'send_tele.py',
    ]
    return _run_workflow(steps)
