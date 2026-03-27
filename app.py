"""
Gradio web app for the image classifier.

Usage:
    python app.py                        # uses model.pth by default
    python app.py --model my_model.pth   # specify a model path
    python app.py --top 5                # show top-5 predictions
    python app.py --share                # create a public Gradio link
"""
import argparse
import os

import gradio as gr

from classifier import load_model, predict

# ---------------------------------------------------------------------------
# Globals (set once at startup)
# ---------------------------------------------------------------------------
_model = None
_class_names = []
_top_k = 3


def _load(model_path: str) -> tuple:
    """Load model from path, return (model, class_names) or raise."""
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path!r}")
    return load_model(model_path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def classify(image, model_path: str, top_k: int):
    """
    Called by Gradio on every submission.

    Parameters
    ----------
    image      : PIL.Image coming from the Gradio Image component
    model_path : str path to .pth file (from the textbox)
    top_k      : how many top classes to show

    Returns
    -------
    label_html : str  — large prediction label (HTML)
    bar_chart  : dict — Gradio Label component data  {"label": conf, ...}
    status     : str  — status / error message
    """
    global _model, _class_names

    if image is None:
        return "", {}, "Upload an image to get started."

    # (Re-)load model if the path changed or model not yet loaded
    try:
        if _model is None or model_path != getattr(_model, "_loaded_from", None):
            _model, _class_names = _load(model_path)
            _model._loaded_from = model_path  # tag so we know what's cached
    except FileNotFoundError as exc:
        return "", {}, f"Model error: {exc}"

    # Save image to a temp file so predict() can open it
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        result = predict(_model, _class_names, tmp_path)
    except Exception as exc:
        return "", {}, f"Inference error: {exc}"
    finally:
        os.remove(tmp_path)

    # Build top-k confidence dict for the Label component
    top_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)[:top_k]
    confidence_dict = {cls: round(score, 4) for cls, score in top_scores}

    predicted = result["predicted"]
    top_conf = result["scores"][predicted]
    label_html = (
        f"<div style='text-align:center; padding: 12px;'>"
        f"<span style='font-size: 2rem; font-weight: 700;'>{predicted}</span>"
        f"<br><span style='font-size: 1.1rem; color: #888;'>{top_conf:.1%} confidence</span>"
        f"</div>"
    )

    classes_str = ", ".join(_class_names)
    status = f"Model loaded — {len(_class_names)} classes: {classes_str}"

    return label_html, confidence_dict, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(default_model: str, top_k: int) -> gr.Blocks:
    with gr.Blocks(title="Image Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Image Classifier\nDrag in an image to classify it with your trained model.")

        with gr.Row():
            # Left column — inputs
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload image")
                model_path_box = gr.Textbox(
                    value=default_model,
                    label="Model path (.pth)",
                    placeholder="model.pth",
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=top_k,
                    label="Top-K predictions",
                )
                submit_btn = gr.Button("Classify", variant="primary")

            # Right column — outputs
            with gr.Column(scale=1):
                label_out = gr.HTML(label="Prediction")
                chart_out = gr.Label(label="Confidence scores", num_top_classes=top_k)
                status_out = gr.Textbox(label="Status", interactive=False, lines=1)

        # Wire up
        inputs = [image_input, model_path_box, top_k_slider]
        outputs = [label_out, chart_out, status_out]
        submit_btn.click(classify, inputs=inputs, outputs=outputs)
        # Also classify automatically when an image is uploaded
        image_input.change(classify, inputs=inputs, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio web UI for the image classifier")
    parser.add_argument("--model", default="model.pth", help="Default model path (.pth)")
    parser.add_argument("--top", type=int, default=3, help="Default top-K predictions to show")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    demo = build_ui(default_model=args.model, top_k=args.top)
    demo.launch(server_port=args.port, share=args.share)
