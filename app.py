"""
Gradio web app for the image classifier — multi-tab edition.

Usage:
    python app.py                          # uses model.pth by default
    python app.py --model my_model.pth
    python app.py --top 5 --port 7860 --share
"""
import argparse
import os
import tempfile
from datetime import datetime

import gradio as gr
import pandas as pd

from classifier import load_model, predict

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model = None
_class_names: list[str] = []
_model_path_cache: str | None = None
_history: list[dict] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_model(model_path: str) -> None:
    """Load the model if the path has changed since the last call."""
    global _model, _class_names, _model_path_cache

    if _model is not None and model_path == _model_path_cache:
        return

    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path!r}. "
            "Please provide a valid path to a .pth checkpoint."
        )

    _model, _class_names = load_model(model_path)
    _model._loaded_from = model_path
    _model_path_cache = model_path


def _device_name() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Tab 1 — Classify
# ---------------------------------------------------------------------------
def classify_single(image, model_path: str, top_k: int, confidence_threshold: float):
    """
    Classify a single image.

    Returns
    -------
    label_html       : str
    confidence_dict  : dict[str, float]
    status           : str
    history_df       : pd.DataFrame
    """
    global _history

    if image is None:
        empty_df = pd.DataFrame(columns=["Time", "Prediction", "Confidence", "Flag"])
        return "", {}, "Upload an image to get started.", empty_df

    try:
        _ensure_model(model_path)
    except FileNotFoundError as exc:
        empty_df = pd.DataFrame(columns=["Time", "Prediction", "Confidence", "Flag"])
        return "", {}, f"Model error: {exc}", empty_df

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        result = predict(_model, _class_names, tmp_path)
    except Exception as exc:
        empty_df = pd.DataFrame(columns=["Time", "Prediction", "Confidence", "Flag"])
        return "", {}, f"Inference error: {exc}", empty_df
    finally:
        os.remove(tmp_path)

    predicted = result["predicted"]
    top_confidence = result["scores"][predicted]

    # Build top-k confidence dict for the Label component
    sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
    confidence_dict = {cls: round(score, 4) for cls, score in sorted_scores[:top_k]}

    # Build label HTML
    low_conf = top_confidence < confidence_threshold
    warning_html = (
        "<br><span style='font-size:0.95rem; color:#e67e22;'>&#9888; Low confidence</span>"
        if low_conf
        else ""
    )
    label_html = (
        "<div style='text-align:center; padding:12px;'>"
        f"<span style='font-size:2rem; font-weight:700;'>{predicted}</span>"
        f"<br><span style='font-size:1.1rem; color:#888;'>{top_confidence:.1%} confidence</span>"
        f"{warning_html}"
        "</div>"
    )

    device = _device_name()
    status = f"{os.path.basename(model_path)} · {len(_class_names)} classes · {device}"

    # Update history
    flag = "⚠" if low_conf else ""
    _history.insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Prediction": predicted,
        "Confidence": f"{top_confidence:.1%}",
        "Flag": flag,
    })
    _history = _history[:20]
    history_df = pd.DataFrame(_history, columns=["Time", "Prediction", "Confidence", "Flag"])

    return label_html, confidence_dict, status, history_df


# ---------------------------------------------------------------------------
# Tab 2 — Batch
# ---------------------------------------------------------------------------
def classify_batch(files, model_path: str, top_k: int):
    """
    Classify a list of uploaded files.

    Returns
    -------
    results_df : pd.DataFrame
    csv_path   : str  (path to a temporary CSV file)
    status     : str
    """
    if not files:
        empty_df = pd.DataFrame()
        return empty_df, None, "No files uploaded."

    try:
        _ensure_model(model_path)
    except FileNotFoundError as exc:
        return pd.DataFrame(), None, f"Model error: {exc}"

    rows = []
    for file_obj in files:
        file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        filename = os.path.basename(file_path)
        try:
            result = predict(_model, _class_names, file_path)
            predicted = result["predicted"]
            top_conf = result["scores"][predicted]
            sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
            row = {"filename": filename, "prediction": predicted, "confidence": round(top_conf, 4)}
            for i, (cls, score) in enumerate(sorted_scores[:top_k], start=1):
                row[f"top{i}_class"] = cls
                row[f"top{i}_score"] = round(score, 4)
            rows.append(row)
        except Exception as exc:
            rows.append({"filename": filename, "prediction": f"ERROR: {exc}", "confidence": None})

    results_df = pd.DataFrame(rows)

    # Write CSV to a temp file so Gradio can serve it as a download
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as csv_file:
        csv_path = csv_file.name
    results_df.to_csv(csv_path, index=False)

    device = _device_name()
    status = (
        f"Processed {len(files)} file(s) · "
        f"{os.path.basename(model_path)} · {len(_class_names)} classes · {device}"
    )
    return results_df, csv_path, status


# ---------------------------------------------------------------------------
# Tab 3 — Model Info
# ---------------------------------------------------------------------------
def get_model_info(model_path: str) -> str:
    """Return a markdown string describing the loaded model."""
    try:
        _ensure_model(model_path)
    except FileNotFoundError as exc:
        return f"**Error:** {exc}"

    total_params = sum(p.numel() for p in _model.parameters())
    trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    file_size_mb = (
        round(os.path.getsize(model_path) / (1024 * 1024), 2)
        if os.path.isfile(model_path)
        else "N/A"
    )
    class_list_md = "\n".join(f"  - {cls}" for cls in _class_names)

    return (
        f"## Model Information\n\n"
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| **Architecture** | ResNet18 |\n"
        f"| **Total parameters** | {total_params:,} |\n"
        f"| **Trainable parameters** | {trainable_params:,} |\n"
        f"| **Number of classes** | {len(_class_names)} |\n"
        f"| **File size** | {file_size_mb} MB |\n\n"
        f"### Classes\n\n"
        f"{class_list_md}\n"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(default_model: str, top_k: int) -> gr.Blocks:
    with gr.Blocks(title="Image Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Image Classifier")

        # ---- Tab 1: Classify ------------------------------------------------
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Drop an image here")
                    model_path_box = gr.Textbox(
                        value=default_model,
                        label="Model path (.pth)",
                        placeholder="model.pth",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, step=1,
                        value=top_k, label="Top-K predictions",
                    )
                    conf_threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=0.5, label="Confidence threshold",
                    )
                    submit_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=1):
                    label_out = gr.HTML(label="Prediction")
                    chart_out = gr.Label(label="Confidence scores", num_top_classes=top_k)
                    status_out = gr.Textbox(label="Status", interactive=False, lines=1)

            history_df_out = gr.Dataframe(
                headers=["Time", "Prediction", "Confidence", "Flag"],
                label="Last 20 predictions",
                interactive=False,
            )

            classify_inputs = [image_input, model_path_box, top_k_slider, conf_threshold_slider]
            classify_outputs = [label_out, chart_out, status_out, history_df_out]
            submit_btn.click(classify_single, inputs=classify_inputs, outputs=classify_outputs)
            image_input.change(classify_single, inputs=classify_inputs, outputs=classify_outputs)

        # ---- Tab 2: Batch ---------------------------------------------------
        with gr.Tab("Batch"):
            batch_files_input = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload images",
            )
            batch_model_path_box = gr.Textbox(
                value=default_model,
                label="Model path (.pth)",
                placeholder="model.pth",
            )
            batch_top_k_slider = gr.Slider(
                minimum=1, maximum=10, step=1,
                value=top_k, label="Top-K predictions",
            )
            batch_run_btn = gr.Button("Run Batch", variant="primary")
            batch_results_df = gr.Dataframe(label="Batch results", interactive=False)
            batch_csv_download = gr.File(label="Download CSV")
            batch_status_out = gr.Textbox(label="Status", interactive=False, lines=1)

            batch_run_btn.click(
                classify_batch,
                inputs=[batch_files_input, batch_model_path_box, batch_top_k_slider],
                outputs=[batch_results_df, batch_csv_download, batch_status_out],
            )

        # ---- Tab 3: Model Info ----------------------------------------------
        with gr.Tab("Model Info"):
            info_model_path_box = gr.Textbox(
                value=default_model,
                label="Model path (.pth)",
                placeholder="model.pth",
            )
            info_load_btn = gr.Button("Load Info", variant="primary")
            info_markdown_out = gr.Markdown()

            info_load_btn.click(
                get_model_info,
                inputs=[info_model_path_box],
                outputs=[info_markdown_out],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio web UI for the image classifier")
    parser.add_argument("--model", default="model.pth", help="Default model path (.pth)")
    parser.add_argument("--top", type=int, default=3, help="Default top-K predictions")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    demo = build_ui(default_model=args.model, top_k=args.top)
    demo.launch(server_port=args.port, share=args.share)
