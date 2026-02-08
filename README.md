# Local Viewshed Explorer

Web app for computing terrain viewsheds with a React + Vite + TypeScript frontend and a FastAPI backend.

## Structure
- `frontend/` React + Vite + TypeScript app
- `backend/` FastAPI service
- `types/` Shared JSON schema and TypeScript types

## Quick Start
Prereqs:
- Node.js (for the frontend)
- Python 3.12+ (for the backend)

Install dependencies (from repo root):
```bash
npm install
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

Run both frontend and backend:
```bash
npm run dev
```

You should see:
- Frontend at `http://localhost:5173/`
- Backend auto-starts on the first free port (usually `8000`). The dev script sets `VITE_API_BASE_URL` automatically.

## Starting Guide
1. Open the UI in your browser: `http://localhost:5173/`
2. Choose **Single** or **Complex** mode.
3. Click the map to place observer points (or use **Use My Location**).
4. (Optional) Draw a **Considered Area** to limit the compute region.
5. Choose **Mode** (Accurate/Fast), radius, and resolution.
6. Click **Compute**. Multi‑point jobs will show progress as `(completed/total)`.

Tips:
- Larger radius and finer resolution increase compute time.
- Use **Fast** mode for iteration, **Accurate** for final results.
- Prefetching DEM tiles improves performance and consistency.

## Dev Commands
Frontend only (from repo root):
```bash
npm run dev:frontend
```

Backend only (from repo root):
```bash
source backend/.venv/bin/activate
npm run dev:backend
```

## Prefetch DEM Tiles
You can predownload geography data (Terrarium DEM tiles) into the cache:
```bash
source backend/.venv/bin/activate
python backend/scripts/prefetch_dem.py --state utah --preset fast
```

Options:
```bash
python backend/scripts/prefetch_dem.py --help
```

## Notes
- The backend entry point is `backend/app/main.py`.
- Shared types live in `types/` and can be imported by the frontend via relative paths or set up as a workspace package.
