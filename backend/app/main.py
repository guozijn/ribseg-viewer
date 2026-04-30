import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from .config import settings
from .inference import RibCXRSegmenter
from .inference.labels import RIB_LABELS
from .inference.postprocessing import make_label_overlay
from .inference.preprocessing import load_image

_log = logging.getLogger("ribseg")

_segmenter: RibCXRSegmenter | None = None
_results: dict[str, dict] = {}

ACCEPTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def _results_root() -> Path:
    return Path(settings.results_dir)


def _result_dir(result_id: str) -> Path:
    return _results_root() / result_id


def get_segmenter() -> RibCXRSegmenter:
    global _segmenter
    if _segmenter is None:
        if not settings.ribcxr_checkpoint:
            raise FileNotFoundError("RIBCXR_CHECKPOINT is not set")
        _segmenter = RibCXRSegmenter(
            config_path=settings.ribcxr_config,
            checkpoint_path=settings.ribcxr_checkpoint,
            device_str=settings.ribcxr_device,
            default_threshold=settings.ribcxr_threshold,
        )
    return _segmenter


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.results_dir, exist_ok=True)
    try:
        get_segmenter()
    except Exception as exc:
        _log.warning("Model load failed at startup: %s", exc)
    yield


app = FastAPI(title="RibSeg Viewer", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    loaded = _segmenter is not None
    return {"status": "ok", "model_loaded": loaded}


@app.get("/model")
def model_info() -> dict:
    config_path = settings.ribcxr_config
    checkpoint_path = settings.ribcxr_checkpoint
    return {
        "labels": RIB_LABELS,
        "threshold": settings.ribcxr_threshold,
        "device": settings.ribcxr_device,
        "config_path": config_path,
        "config_exists": Path(config_path).is_file(),
        "checkpoint_name": Path(checkpoint_path).name if checkpoint_path else "",
        "checkpoint_exists": Path(checkpoint_path).is_file() if checkpoint_path else False,
    }


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
) -> dict:
    if file.content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Accept: {sorted(ACCEPTED_CONTENT_TYPES)}",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb} MB)")

    try:
        image_np, model_input = load_image(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}") from exc

    try:
        seg = get_segmenter()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model load error: {exc}") from exc

    t0 = time.perf_counter()
    try:
        result = seg.predict(image_np, model_input, threshold=threshold)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(status_code=503, detail="CUDA out of memory") from exc
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    elapsed = time.perf_counter() - t0

    result_id = str(uuid.uuid4())
    meta = {
        "id": result_id,
        "filename": file.filename or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": result["labels"],
        "threshold": result["threshold"],
        "original_height": result["original_height"],
        "original_width": result["original_width"],
        "areas": result["areas"],
        "inference_time_s": round(elapsed, 3),
    }
    _save_result(result_id, image_np, result, meta)

    import logging
    logging.getLogger("ribseg").info(
        "infer id=%s elapsed=%.3fs h=%d w=%d",
        result_id,
        elapsed,
        result["original_height"],
        result["original_width"],
    )

    return meta


def _save_result(result_id: str, image_np: np.ndarray, result: dict, meta: dict) -> None:
    result_dir = _result_dir(result_id)
    result_dir.mkdir(parents=True, exist_ok=False)
    masks_dir = result_dir / "masks"
    masks_dir.mkdir()
    cv2.imwrite(str(result_dir / "image.png"), image_np.astype(np.uint8))
    cv2.imwrite(str(result_dir / "overlay.png"), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(result_dir / "merged_mask.png"), result["merged_mask"])
    for label, mask in zip(result["labels"], result["masks"]):
        cv2.imwrite(str(masks_dir / f"{label}.png"), mask.astype(np.uint8) * 255)
    (result_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _results[result_id] = {
        "image_np": image_np,
        "result": result,
        "meta": meta,
    }


def _png_bytes(arr: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", arr)
    if not success:
        raise RuntimeError("Failed to encode PNG")
    return bytes(buf)


def _rgba_png_bytes(arr: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
    if not success:
        raise RuntimeError("Failed to encode PNG")
    return bytes(buf)


def _get_result(result_id: str) -> dict:
    entry = _results.get(result_id)
    if entry is not None:
        return entry

    result_dir = _result_dir(result_id)
    meta_path = result_dir / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image_np = cv2.imread(str(result_dir / "image.png"), cv2.IMREAD_GRAYSCALE)
        overlay_bgr = cv2.imread(str(result_dir / "overlay.png"), cv2.IMREAD_COLOR)
        merged_mask = cv2.imread(str(result_dir / "merged_mask.png"), cv2.IMREAD_GRAYSCALE)
        if image_np is None or overlay_bgr is None or merged_mask is None:
            raise RuntimeError("Missing persisted PNG assets")
        masks = np.stack([
            (cv2.imread(str(result_dir / "masks" / f"{label}.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            for label in meta["labels"]
        ])
        result = {
            "masks": masks,
            "overlay": cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
            "merged_mask": merged_mask,
            "areas": meta["areas"],
            "labels": meta["labels"],
            "threshold": meta["threshold"],
            "original_height": meta["original_height"],
            "original_width": meta["original_width"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load result {result_id}: {exc}") from exc

    entry = {"image_np": image_np, "result": result, "meta": meta}
    _results[result_id] = entry
    return entry


@app.get("/results")
def list_results() -> dict:
    items = []
    if _results_root().is_dir():
        for meta_path in _results_root().glob("*/meta.json"):
            try:
                items.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                continue
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"results": items}


@app.delete("/results/{result_id}")
def delete_result(result_id: str) -> dict:
    _results.pop(result_id, None)
    result_dir = _result_dir(result_id)
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail=f"Result {result_id} not found")
    shutil.rmtree(result_dir)
    return {"deleted": result_id}


@app.get("/results/{result_id}/image")
def result_image(result_id: str) -> Response:
    _get_result(result_id)
    png_path = _result_dir(result_id) / "image.png"
    png = png_path.read_bytes()
    return Response(content=png, media_type="image/png")


@app.get("/results/{result_id}/overlay")
def result_overlay(result_id: str) -> Response:
    _get_result(result_id)
    png_path = _result_dir(result_id) / "overlay.png"
    png = png_path.read_bytes()
    return Response(content=png, media_type="image/png")


@app.get("/results/{result_id}/overlay/{label}")
def result_overlay_label(result_id: str, label: str) -> Response:
    entry = _get_result(result_id)
    if label not in RIB_LABELS:
        raise HTTPException(status_code=404, detail=f"Unknown label: {label}")
    idx = RIB_LABELS.index(label)
    rgba = make_label_overlay(entry["result"]["masks"][idx], idx)
    png = _rgba_png_bytes(rgba)
    return Response(content=png, media_type="image/png")


@app.get("/results/{result_id}/mask")
def result_mask(result_id: str) -> Response:
    _get_result(result_id)
    png_path = _result_dir(result_id) / "merged_mask.png"
    png = png_path.read_bytes()
    return Response(content=png, media_type="image/png")


@app.get("/results/{result_id}/mask/{label}")
def result_mask_label(result_id: str, label: str) -> Response:
    _get_result(result_id)
    if label not in RIB_LABELS:
        raise HTTPException(status_code=404, detail=f"Unknown label: {label}")
    png_path = _result_dir(result_id) / "masks" / f"{label}.png"
    png = png_path.read_bytes()
    return Response(content=png, media_type="image/png")
