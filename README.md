# ribseg-viewer

A local web app for running rib segmentation on chest X-rays using the RibCXR model. Upload an image, run inference, and explore per-rib segmentation masks with an interactive overlay.

## Requirements

- Python 3.10+ with the packages listed in `backend/pyproject.toml`
- Node.js 18+
- A trained RibCXR checkpoint and its matching YAML config

The backend has been tested with the `unet++(b0)` checkpoint trained on VinDr-RibCXR.

## Quick start

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
cd backend
pip install -e ".[dev]"
```

This installs PyTorch, FastAPI, all medical imaging libraries, and the test dependencies into the project-local `.venv`. The training repository is not used as a Python environment. The app keeps only the supported runtime config files in `model/configs/`; checkpoint files stay outside git.

### 2. Configure model paths

The only required configuration is the path to your trained checkpoint. Create a `.env` file inside `backend/`:

```bash
cp backend/.env.example backend/.env
```

```
RIBCXR_CHECKPOINT=../model/multi_unetpp_b0_dice.pth
```

All other settings have sensible defaults. Override them as needed:

| Variable | Default | Description |
|---|---|---|
| `RIBCXR_CHECKPOINT` | _(required)_ | Path to the trained checkpoint file |
| `RIBCXR_CONFIG` | `model/configs/multi_unetpp_b0_dice.yaml` | YAML config matching the selected checkpoint |
| `RIBCXR_DEVICE` | `cuda` | `cuda` or `cpu`; falls back to `cpu` if CUDA is unavailable |
| `RIBCXR_THRESHOLD` | `0.5` | Default probability threshold for mask binarization |
| `RIBCXR_MAX_UPLOAD_MB` | `32` | Maximum accepted upload size |
| `RIBCXR_RESULTS_DIR` | `/tmp/ribseg_results` | Directory created at startup |

### 3. Start the backend

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 8110
```

The server starts on `http://localhost:8110`. The model is loaded at startup; the first request does not pay a cold-start penalty.

### 4. Install and start the frontend

```bash
cd frontend
npm install
npm run dev -- --port 5174
```

The Vite dev server starts on `http://localhost:5174` and proxies `/health`, `/model`, `/infer`, and `/results` to `http://localhost:8110`. By default, the frontend uses relative API paths so the proxy also works when opening the app through a remote hostname.

Open `http://localhost:5174` in a browser.

## Usage

1. Set the threshold in the header (default 0.5).
2. Drop a chest X-ray (PNG, JPEG, or WebP) onto the upload zone or click to select.
3. The backend runs inference, stores the result under `RIBCXR_RESULTS_DIR`, and returns masks for all 20 ribs (R1–R10, L1–L10).
4. Use the opacity slider to blend the segmentation overlay over the original image.
5. Toggle individual ribs or entire sides on and off using the sidebar controls.
6. Download the overlay, merged binary mask, or individual rib masks using the download buttons.
7. Use the history sidebar to reopen or delete saved inference results.

## API reference

All endpoints return JSON unless noted otherwise.

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model`

Returns model configuration and file existence flags.

```json
{
  "labels": ["R1", ..., "L10"],
  "threshold": 0.5,
  "device": "cuda",
  "config_path": "/path/to/config.yaml",
  "config_exists": true,
  "checkpoint_name": "multi_unetpp_b0_dice.pth",
  "checkpoint_exists": true
}
```

### `POST /infer`

Accepts `multipart/form-data`.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image/png, image/jpeg, image/webp | yes | Input chest X-ray |
| `threshold` | float | no | Overrides the server default for this request |

Response:

```json
{
  "id": "uuid",
  "filename": "xray.png",
  "created_at": "2026-04-30T12:00:00+00:00",
  "labels": ["R1", ..., "L10"],
  "threshold": 0.5,
  "original_height": 1024,
  "original_width": 1024,
  "areas": { "R1": 4812, "R2": 5031, ... },
  "inference_time_s": 0.335
}
```

Error codes: `413` file too large, `415` unsupported format, `422` unreadable image, `503` model unavailable or CUDA out of memory.

### `GET /results`

Returns saved inference history, newest first.

```json
{
  "results": [
    {
      "id": "uuid",
      "filename": "xray.png",
      "created_at": "2026-04-30T12:00:00+00:00",
      "labels": ["R1", "..."],
      "threshold": 0.5,
      "original_height": 1024,
      "original_width": 1024,
      "areas": { "R1": 4812 },
      "inference_time_s": 0.335
    }
  ]
}
```

### `GET /results/{id}/image`

Original grayscale image as PNG.

### `GET /results/{id}/overlay`

RGB overlay with all rib masks coloured using the TURBO colormap (alpha = 0.45), as PNG.

### `GET /results/{id}/overlay/{label}`

Single-rib transparent RGBA overlay for `label` (e.g. `R1`, `L10`), as PNG. The frontend uses these pre-colored overlays for fast per-rib toggling.

### `GET /results/{id}/mask`

Merged binary mask (union of all ribs), 0/255 grayscale PNG.

### `GET /results/{id}/mask/{label}`

Single-rib binary mask for `label` (e.g. `R1`, `L10`), 0/255 grayscale PNG.

### `DELETE /results/{id}`

Deletes the saved result directory from `RIBCXR_RESULTS_DIR`.

Results are stored on disk under `RIBCXR_RESULTS_DIR` as compressed PNG assets plus `meta.json`; the backend also keeps recently accessed results in process memory for faster repeated access.

## Command-line example

Run inference without the web UI using the original `infer.py` in the training repo:

```bash
cd /path/to/MIDL2021-VinDr-RibCXR
python infer.py \
  --config cvcore/config/multi_unet++_b0_diceloss.yaml \
  --checkpoint weights/multi_unetpp_b0_dice.pth \
  --image /path/to/xray.png \
  --output outputs/overlay.png \
  --mask-output outputs/mask.png \
  --threshold 0.5 \
  --device cuda
```

## Running tests

```bash
source .venv/bin/activate
cd backend
python -m pytest tests/ -v
```

20 tests cover preprocessing (grayscale and RGB input), checkpoint loading (plain dict, nested `state_dict`, `module.`-prefixed keys), postprocessing (mask shape, merged mask, overlay, area stats), and the API (all endpoints with a mocked segmenter).

## Repository layout

```
ribseg-viewer/
  backend/
    app/
      main.py              FastAPI entry point, all HTTP routes
      config.py            Pydantic settings, env-var overrides
      inference/
        ribcxr_model.py    RibCXRSegmenter: loads model, runs predict()
        preprocessing.py   load_image(): bytes -> (float32 array, model input)
        postprocessing.py  build_masks(), merged_mask(), make_overlay(), compute_areas()
        labels.py          RIB_LABELS constant
    tests/
      test_api.py
      test_checkpoint.py
      test_postprocessing.py
      test_preprocessing.py
    pyproject.toml         Dependency manifest
  frontend/
    src/
      api/client.ts        Typed fetch wrappers for all backend endpoints
      components/
        UploadPanel.tsx    Drag-and-drop upload zone
        LabelToggles.tsx   Per-rib checkbox controls with area display
        MetadataPanel.tsx  Result metadata display
      viewer/
        OverlayCanvas.tsx  Canvas-based image + overlay compositor
        ResultViewer.tsx   Full result layout: canvas, controls, sidebar
      App.tsx              Top-level state machine (idle / loading / result / error)
    vite.config.ts         Dev proxy to backend
  model/
    configs/
      README.md            Supported config policy
      multi_unetpp_b0_dice.yaml
    .gitkeep               Placeholder; place checkpoint files here (not committed)
```

## Model weights

Do not commit checkpoint files. Place weights under `model/` or the path set by `RIBCXR_CHECKPOINT`. The expected checkpoint for the default config is:

```
multi_unetpp_b0_dice.pth
```

Keep only supported YAML configs in `model/configs/`. Add another config only when there is a matching checkpoint and the app should expose that model variant.

Checkpoints may be a plain state dict or a dict containing a `state_dict` key. Keys prefixed with `module.` (from `DataParallel` training) are stripped automatically.

## Output format

The model produces 20 output channels in this order:

```
R1  R2  R3  R4  R5  R6  R7  R8  R9  R10
L1  L2  L3  L4  L5  L6  L7  L8  L9  L10
```

Each channel is a probability map. The server applies sigmoid, thresholds at the configured value, and resizes masks back to the original image dimensions using nearest-neighbour interpolation.
