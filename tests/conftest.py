import io
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Model / data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_model():
    """A tiny stand-in model — an nn.Linear that mimics a loaded checkpoint."""
    model = nn.Linear(512, 3)
    model._loaded_from = "fake.pth"
    return model


@pytest.fixture(scope="session")
def fake_class_names():
    return ["cat", "dog", "bird"]


@pytest.fixture(scope="session")
def sample_image_bytes():
    """64x64 solid-red PNG image as raw bytes."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# API mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def mock_classifier(fake_model, fake_class_names):
    """
    Patches api.load_model and api.clf_predict.

    Yields the mocked clf_predict so individual tests can override its
    return value or side_effect.
    """
    fake_result = {
        "predicted": "cat",
        "scores": {"cat": 0.9, "dog": 0.08, "bird": 0.02},
    }
    with patch("api.load_model", return_value=(fake_model, fake_class_names)) as _mock_load, \
         patch("api.clf_predict", return_value=fake_result) as mock_predict:
        yield mock_predict


@pytest.fixture(scope="function")
def api_client(mock_classifier, fake_model, fake_class_names):
    """
    FastAPI TestClient with model state pre-populated so /health and
    /model/info work without touching the filesystem.
    """
    import api as api_module
    from fastapi.testclient import TestClient

    # Pre-populate the module-level cache so get_model() skips file I/O.
    original_model = api_module._model
    original_class_names = api_module._class_names
    original_cache = api_module._model_path_cache

    api_module._model = fake_model
    api_module._class_names = fake_class_names
    api_module._model_path_cache = api_module.MODEL_PATH

    client = TestClient(api_module.app)
    yield client

    # Restore original state after the test.
    api_module._model = original_model
    api_module._class_names = original_class_names
    api_module._model_path_cache = original_cache
