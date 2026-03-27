# image-classifier

A custom image classifier built on ResNet18 transfer learning. Train on any labeled image dataset, then run predictions through a Gradio web app, a REST API, or the CLI.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset layout

```
data/
    train/
        cat/
        dog/
        ...
    val/
        cat/
        dog/
        ...
```

## Train

```bash
python train.py --data_dir data --epochs 10 --output model.pth
```

Options:
- `--batch_size` — default `32`
- `--lr` — learning rate, default `1e-3`
- `--unfreeze` — fine-tune the full backbone (not just the head)

The best checkpoint by val accuracy is saved automatically.

---

## Web App (Gradio)

Start the interactive web UI:

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860).

**Three tabs:**

| Tab | What it does |
|---|---|
| **Classify** | Drag-and-drop single image → top-K predictions with confidence bars. Low-confidence results are flagged with ⚠. Keeps a running history of the last 20 predictions. |
| **Batch** | Upload multiple images at once → results table with per-image scores. Download as CSV. |
| **Model Info** | Shows architecture, total/trainable parameter counts, class list, and file size. |

Options:
- `--model model.pth` — model to load on startup (default: `model.pth`)
- `--top 5` — number of top predictions to display (default: `3`)
- `--port 8080` — port to serve on (default: `7860`)
- `--share` — generate a public Gradio link

---

## REST API (FastAPI)

```bash
uvicorn api:app --reload
```

Docs auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

Set the model path via env var:

```bash
MODEL_PATH=model.pth uvicorn api:app --reload
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + model status |
| `GET` | `/model/info` | Architecture, class list, parameter counts |
| `POST` | `/predict` | Classify a single uploaded image |
| `POST` | `/batch` | Classify multiple images; errors captured per-file |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cat.jpg" | python -m json.tool
```

---

## CLI Prediction

```bash
# Single image
python predict.py --model model.pth --images cat.jpg

# Folder of images, show top-5
python predict.py --model model.pth --images /path/to/folder --top 5
```

---

## Tests

```bash
# API + unit tests (no model file needed — uses mocks)
pytest tests/test_api.py -v

# UI tests (requires the app to be running on port 7861)
python app.py --port 7861 &
pytest tests/test_ui.py -v
```

Install Playwright browsers once:

```bash
playwright install chromium
```

---

## Project structure

| File | Purpose |
|---|---|
| `classifier.py` | Model definition, transforms, save/load, single-image predict |
| `train.py` | Training loop — validation, LR scheduling, best-model checkpointing |
| `predict.py` | CLI batch inference |
| `app.py` | Gradio web UI (Classify / Batch / Model Info tabs) |
| `api.py` | FastAPI REST API (`/predict`, `/batch`, `/health`, `/model/info`) |
| `tests/conftest.py` | Shared pytest fixtures — mocked model, sample image, test client |
| `tests/test_api.py` | API endpoint tests (11 tests, no model file required) |
| `tests/test_ui.py` | Playwright browser tests for the Gradio UI |
