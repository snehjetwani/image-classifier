# image-classifier

A custom image classifier built on ResNet18 transfer learning. Train on your own dataset, then run predictions via CLI or a Gradio web app.

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
- `--unfreeze` — fine-tune the full backbone instead of just the head

## Web App (Gradio)

Start the interactive web UI — drag and drop an image and see top-K predictions with confidence scores.

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

Options:
- `--model model.pth` — model to load on startup (default: `model.pth`)
- `--top 5` — number of top predictions to display (default: `3`)
- `--port 8080` — port to serve on (default: `7860`)
- `--share` — generate a public Gradio link (useful for demos)

## CLI Prediction

```bash
# Single image
python predict.py --model model.pth --images cat.jpg

# Folder of images, show top-5
python predict.py --model model.pth --images /path/to/folder --top 5
```

## Files

| File | Purpose |
|---|---|
| `classifier.py` | Model definition, transforms, save/load, predict |
| `train.py` | Training loop with validation and checkpointing |
| `predict.py` | CLI batch inference |
| `app.py` | Gradio web UI |
