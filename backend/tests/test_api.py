import io
import unittest.mock as mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, _results


def _make_png(size=(64, 48)) -> bytes:
    img = Image.new("L", size, color=100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_predict(image_np, model_input, threshold=None):
    h, w = image_np.shape[:2]
    thr = threshold if threshold is not None else 0.5
    masks = np.zeros((20, h, w), dtype=np.uint8)
    masks[0, 1:4, 1:4] = 1
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    combined = np.zeros((h, w), dtype=np.uint8)
    areas = {lbl: 0 for lbl in ["R1","R2","R3","R4","R5","R6","R7","R8","R9","R10",
                                  "L1","L2","L3","L4","L5","L6","L7","L8","L9","L10"]}
    return {
        "masks": masks,
        "overlay": overlay,
        "merged_mask": combined,
        "areas": areas,
        "labels": list(areas.keys()),
        "threshold": thr,
        "original_height": h,
        "original_width": w,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "results_dir", str(tmp_path))
    _results.clear()
    with TestClient(app) as c:
        yield c
    _results.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_model_endpoint(client):
    resp = client.get("/model")
    assert resp.status_code == 200
    data = resp.json()
    assert "labels" in data
    assert len(data["labels"]) == 20


def test_infer_unsupported_type(client):
    resp = client.post(
        "/infer",
        files={"file": ("test.bmp", b"\x00" * 10, "image/bmp")},
    )
    assert resp.status_code == 415


def test_infer_with_mock_model(client):
    from app import main as main_module
    seg_mock = mock.MagicMock()
    seg_mock.predict.side_effect = _fake_predict

    with mock.patch.object(main_module, "get_segmenter", return_value=seg_mock):
        png = _make_png((64, 48))
        resp = client.post(
            "/infer",
            files={"file": ("xray.png", png, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["original_height"] == 48
    assert data["original_width"] == 64
    assert len(data["areas"]) == 20


def test_result_image_endpoint(client):
    from app import main as main_module
    seg_mock = mock.MagicMock()
    seg_mock.predict.side_effect = _fake_predict

    with mock.patch.object(main_module, "get_segmenter", return_value=seg_mock):
        png = _make_png((32, 32))
        infer_resp = client.post(
            "/infer",
            files={"file": ("xray.png", png, "image/png")},
        )
    result_id = infer_resp.json()["id"]

    img_resp = client.get(f"/results/{result_id}/image")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"


def test_result_overlay_endpoint(client):
    from app import main as main_module
    seg_mock = mock.MagicMock()
    seg_mock.predict.side_effect = _fake_predict

    with mock.patch.object(main_module, "get_segmenter", return_value=seg_mock):
        png = _make_png((32, 32))
        infer_resp = client.post(
            "/infer",
            files={"file": ("xray.png", png, "image/png")},
        )
    result_id = infer_resp.json()["id"]

    overlay_resp = client.get(f"/results/{result_id}/overlay")
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"


def test_result_label_overlay_endpoint(client):
    from app import main as main_module
    seg_mock = mock.MagicMock()
    seg_mock.predict.side_effect = _fake_predict

    with mock.patch.object(main_module, "get_segmenter", return_value=seg_mock):
        png = _make_png((32, 32))
        infer_resp = client.post(
            "/infer",
            files={"file": ("xray.png", png, "image/png")},
        )
    result_id = infer_resp.json()["id"]

    overlay_resp = client.get(f"/results/{result_id}/overlay/R1")
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"


def test_results_history_and_delete(client):
    from app import main as main_module
    seg_mock = mock.MagicMock()
    seg_mock.predict.side_effect = _fake_predict

    with mock.patch.object(main_module, "get_segmenter", return_value=seg_mock):
        png = _make_png((32, 32))
        infer_resp = client.post(
            "/infer",
            files={"file": ("history.png", png, "image/png")},
        )
    result_id = infer_resp.json()["id"]

    history_resp = client.get("/results")
    assert history_resp.status_code == 200
    items = history_resp.json()["results"]
    assert len(items) == 1
    assert items[0]["id"] == result_id
    assert items[0]["filename"] == "history.png"

    _results.clear()
    img_resp = client.get(f"/results/{result_id}/image")
    assert img_resp.status_code == 200

    delete_resp = client.delete(f"/results/{result_id}")
    assert delete_resp.status_code == 200
    assert client.get(f"/results/{result_id}/image").status_code == 404


def test_result_not_found(client):
    resp = client.get("/results/nonexistent-id/image")
    assert resp.status_code == 404
