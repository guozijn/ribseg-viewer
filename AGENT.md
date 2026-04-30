# Web Inference App Plan

This repository will become a web inference app for rib segmentation on chest X-rays using the RibCXR model code in `/home/zijianguo/Code/MIDL2021-VinDr-RibCXR`.

## Source Model Contract

The training/inference repository already provides the core inference behavior in `/home/zijianguo/Code/MIDL2021-VinDr-RibCXR/infer.py`.

Important implementation details to preserve:

- Model construction uses `cvcore.config.get_cfg_defaults()` plus a supported YAML config from `model/configs/`, such as `model/configs/multi_unetpp_b0_dice.yaml`.
- Model creation is handled by `cvcore.model.build_model(cfg)`.
- Supported model names include `unet(b0)`, `unet++(b0)`, `fpn(b0)`, and MONAI/SMP variants listed in `cvcore/model/model_zoo.py`.
- The expected RibCXR output has 20 channels:
  `R1`-`R10`, then `L1`-`L10`.
- Inference currently loads grayscale images, normalizes to `[0, 1]`, resizes to `512x512`, applies sigmoid, thresholds probabilities, and resizes masks back to the original image size for visualization.
- Checkpoints may be plain state dicts or dictionaries containing `state_dict`; keys may also be prefixed with `module.`.

## Target App Shape

Build a small full-stack app with:

- A Python inference backend.
- A web UI for uploading a chest X-ray and viewing model results.
- Local-first model configuration so the user can point the app at a config file and checkpoint.
- A clean path to later package with Docker or deploy behind a reverse proxy.

Recommended stack:

- Backend: FastAPI, Uvicorn, PyTorch, Albumentations, Pillow, OpenCV headless.
- Frontend: Vite + React + TypeScript.
- Visualization: browser canvas overlay with adjustable opacity and per-rib toggles.
- API format: multipart upload for input image, JSON metadata plus PNG result assets for output.

## Repository Layout

Proposed structure:

```text
ribseg-viewer/
  backend/
    app/
      main.py
      config.py
      inference/
        ribcxr_model.py
        preprocessing.py
        postprocessing.py
        labels.py
      static/
    pyproject.toml
    README.md
  frontend/
    src/
      api/
      components/
      viewer/
      App.tsx
    package.json
    vite.config.ts
  model/
    README.md
    .gitkeep
    configs/
      README.md
      <supported-ribcxr-config>.yaml
  AGENT.md
  CLAUDE.md
```

Do not commit trained checkpoints into git. Keep local weights under `model/` or another configured path, and document the expected filenames.

Keep only the necessary RibCXR YAML config files for checkpoints the app actually supports. Do not copy every training config by default. Each retained config should have a clear reason, such as matching the default checkpoint or enabling a deliberately supported model variant.

## Backend Implementation Plan

1. Create a backend package.
   - Add FastAPI app entry point in `backend/app/main.py`.
   - Add settings for `RIBCXR_CODE_DIR`, `RIBCXR_CONFIG`, `RIBCXR_CHECKPOINT`, `RIBCXR_DEVICE`, `RIBCXR_THRESHOLD`, upload size limits, and output retention.
   - Default `RIBCXR_CONFIG` should point to a checked-in config under `model/configs/` only when that config is known to match the selected checkpoint.
   - Use environment variables with sensible local defaults.

2. Vendor or import the training code deliberately.
   - First implementation can add `/home/zijianguo/Code/MIDL2021-VinDr-RibCXR` to `sys.path`.
   - Longer-term cleanup should package the RibCXR code as an installable local dependency.
   - Keep the app's inference wrapper isolated so this import strategy can change without touching API routes.
   - If vendoring later, include only minimal inference code plus the necessary supported config files, not the full training config set.

3. Extract reusable inference logic from `infer.py`.
   - Implement `RibCXRSegmenter` that loads config, builds the model, restores checkpoint, moves it to device, and keeps it warm in memory.
   - Reuse the checkpoint compatibility behavior from `load_state_dict`.
   - Disable gradients and call `model.eval()`.
   - Resolve `cuda` to `cpu` automatically when CUDA is unavailable.

4. Implement preprocessing.
   - Accept PNG, JPEG, and optionally DICOM later.
   - Convert to grayscale.
   - Preserve original dimensions.
   - Normalize pixel values to `[0, 1]`.
   - Resize model input to `512x512`, matching current `infer.py`.

5. Implement postprocessing.
   - Apply sigmoid probabilities.
   - Return per-rib masks at original image resolution.
   - Return merged binary mask.
   - Generate an RGB overlay equivalent to `infer.py`.
   - Include labels and simple area statistics per rib.

6. Expose API endpoints.
   - `GET /health`: reports service status and whether the model is loaded.
   - `GET /model`: returns labels, threshold, device, config path, and checkpoint basename.
   - `POST /infer`: accepts an uploaded image and optional threshold override.
   - `GET /results/{id}/image`: original normalized display image.
   - `GET /results/{id}/overlay`: overlay PNG.
   - `GET /results/{id}/mask`: merged binary mask PNG.
   - `GET /results/{id}/mask/{label}`: individual rib mask PNG.

7. Handle runtime concerns.
   - Validate file type and reject unsupported inputs.
   - Return structured errors for missing checkpoint/config, model load failure, invalid image, and CUDA out-of-memory.
   - Add request logging with inference time and image dimensions.
   - Avoid reloading the model per request.

## Frontend Implementation Plan

1. Scaffold a Vite React TypeScript app.
   - Keep the first screen as the actual inference workspace, not a marketing page.
   - Use a restrained clinical workstation layout: upload/control panel, image viewer, result controls, and metadata.

2. Build the upload workflow.
   - Drag-and-drop and file picker for one image at a time.
   - Show upload and inference progress states.
   - Display actionable error messages from the backend.

3. Build the viewer.
   - Show the original image and segmentation overlay.
   - Add opacity slider for overlay.
   - Add per-label toggles for `R1`-`R10` and `L1`-`L10`.
   - Add threshold control that can rerun inference.
   - Keep image aspect ratio stable across desktop and mobile.

4. Add result affordances.
   - Download overlay PNG.
   - Download merged mask PNG.
   - Download per-rib masks.
   - Show model metadata and basic per-rib area values.

5. Add frontend API layer.
   - Centralize backend calls under `frontend/src/api`.
   - Use typed response models for inference metadata.
   - Keep result URLs revocable and avoid leaking object URLs.

## Verification Plan

Backend:

- Unit test preprocessing with grayscale and RGB input images.
- Unit test checkpoint loading for normal and `module.`-prefixed state dicts.
- Unit test postprocessing shapes and label ordering.
- Add an integration test for `/infer` using a tiny generated image and a mocked model.

Frontend:

- Component tests for upload state, error state, label toggles, and opacity control.
- Browser smoke test that uploads a sample image and verifies the overlay renders.
- Check desktop and mobile viewports for non-overlapping controls.

Manual validation:

- Run the original `infer.py` and the backend endpoint on the same image, config, checkpoint, threshold, and device.
- Compare output mask dimensions, label count, merged mask, and overlay behavior.
- Record expected local command examples in the backend README.

## Milestones

1. Backend model wrapper and `/health`/`/model` endpoints.
2. `/infer` endpoint with overlay and mask outputs.
3. React upload flow and static result display.
4. Interactive overlay controls and per-rib toggles.
5. Download actions, metadata panel, and error handling.
6. Tests, local run documentation, and optional Docker packaging.

## Open Questions

- Which trained checkpoint should be the default for local development?
- Should the first release support DICOM, or only common image formats?
- Should per-rib masks be returned immediately in the `/infer` response, or generated on demand through result URLs?
- Should inference results be persisted on disk, kept in memory, or periodically cleaned from a temporary directory?
