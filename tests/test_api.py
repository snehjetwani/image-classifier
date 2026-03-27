import io


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_includes_model_path(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert "model" in response.json()


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------

def test_model_info_returns_architecture(api_client):
    response = api_client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["architecture"] == "ResNet18"


def test_model_info_returns_classes(api_client):
    response = api_client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["classes"] == ["cat", "dog", "bird"]


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

def test_predict_returns_label(api_client, sample_image_bytes):
    response = api_client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["predicted"] == "cat"


def test_predict_returns_confidence(api_client, sample_image_bytes):
    response = api_client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
    )
    assert response.status_code == 200
    confidence = response.json()["confidence"]
    assert 0 <= confidence <= 1


def test_predict_returns_all_scores(api_client, sample_image_bytes):
    response = api_client.post(
        "/predict",
        files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
    )
    assert response.status_code == 200
    assert set(response.json()["scores"].keys()) == {"cat", "dog", "bird"}


def test_predict_rejects_missing_file(api_client):
    response = api_client.post("/predict")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------

def test_batch_returns_list(api_client, sample_image_bytes):
    files = [
        ("files", ("img1.png", io.BytesIO(sample_image_bytes), "image/png")),
        ("files", ("img2.png", io.BytesIO(sample_image_bytes), "image/png")),
    ]
    response = api_client.post("/batch", files=files)
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_batch_each_item_has_filename(api_client, sample_image_bytes):
    files = [
        ("files", ("single.png", io.BytesIO(sample_image_bytes), "image/png")),
    ]
    response = api_client.post("/batch", files=files)
    assert response.status_code == 200
    assert response.json()[0]["filename"] is not None


def test_batch_captures_errors_per_file(api_client, sample_image_bytes, mock_classifier):
    """When clf_predict raises, the error is stored per-item rather than propagated."""
    mock_classifier.side_effect = Exception("boom")

    files = [
        ("files", ("bad.png", io.BytesIO(sample_image_bytes), "image/png")),
    ]
    response = api_client.post("/batch", files=files)
    assert response.status_code == 200
    item = response.json()[0]
    assert item.get("error") is not None
    assert "boom" in item["error"]
