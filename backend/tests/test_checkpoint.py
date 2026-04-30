import sys
import types
import torch
import pytest


def _make_tiny_state_dict():
    return {"weight": torch.zeros(2, 2)}


def _load_state_dict_fn(model, state_dict):
    from app.inference.ribcxr_model import _load_state_dict
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(state_dict, f.name)
        path = f.name
    try:
        _load_state_dict(model, path, torch.device("cpu"))
    finally:
        os.unlink(path)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    def load_state_dict(self, state_dict, strict=True):
        self.weight.data.copy_(state_dict["weight"])


def test_plain_state_dict():
    model = _TinyModel()
    sd = {"weight": torch.ones(2, 2)}
    _load_state_dict_fn(model, sd)
    assert model.weight.data.sum().item() == pytest.approx(4.0)


def test_nested_state_dict():
    model = _TinyModel()
    sd = {"state_dict": {"weight": torch.ones(2, 2) * 3}}
    _load_state_dict_fn(model, sd)
    assert model.weight.data.sum().item() == pytest.approx(12.0)


def test_module_prefix_stripped():
    class _StrictModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))

    model = _StrictModel()
    sd = {"module.weight": torch.ones(2, 2) * 5}

    import tempfile, os
    from app.inference.ribcxr_model import _load_state_dict
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(sd, f.name)
        path = f.name
    try:
        _load_state_dict(model, path, torch.device("cpu"))
    finally:
        os.unlink(path)

    assert model.weight.data.sum().item() == pytest.approx(20.0)
