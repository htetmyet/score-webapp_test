    # Deployable Score Webapp (FastAPI runner + React frontend)


    This package wraps your existing backend scripts with a small FastAPI runner (`backend/runner_api.py`) that executes single scripts or multi-step workflows.


    ## How it works


    - `POST /api/run_sync` with JSON body `{"script": "ml_models/train_model_new.py", "args": {"epochs": 3}}` runs a single script relative to the `backend/` folder and returns output.
    - Workflow endpoints (run multiple scripts in order):
        - `POST /api/run/data-process`
        - `POST /api/run/train-model`
        - `POST /api/run/predict-res`
        - `POST /api/run/send-telegram`


    ## Run locally (without Docker)


    1. Install deps: `pip install -r backend/requirements.txt` (if present)
    - Then: `pip install fastapi uvicorn`
2. Start server: `uvicorn backend.runner_api:app --reload --port 8000`
3. Start frontend (in another shell): `cd frontend && npm install && npm run dev` (or `npm run build` + serve)


    ## Run with Docker Compose


    ```bash
    docker compose build
    docker compose up
    ```

    Backend on http://localhost:8000 and frontend on http://localhost:3000.
    Frontend proxies `/api/*` to the backend via Nginx when using Docker Compose.
