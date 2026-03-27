"""
Run inference on one or more images using a saved model.

Usage:
    python predict.py --model model.pth --images cat.jpg dog.png
    python predict.py --model model.pth --images /path/to/folder/
"""
import argparse
import os

from classifier import load_model, predict


def collect_image_paths(inputs: list[str]) -> list[str]:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            for fname in sorted(os.listdir(item)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(item, fname))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            print(f"Warning: '{item}' not found, skipping.")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Classify images with a trained model")
    parser.add_argument("--model", default="model.pth", help="Path to saved model (.pth)")
    parser.add_argument("--images", nargs="+", required=True, help="Image file(s) or folder(s)")
    parser.add_argument("--top", type=int, default=3, help="Show top-N class scores")
    args = parser.parse_args()

    model, class_names = load_model(args.model)
    image_paths = collect_image_paths(args.images)

    if not image_paths:
        print("No images found.")
        return

    for path in image_paths:
        try:
            result = predict(model, class_names, path)
        except Exception as e:
            print(f"{path}: ERROR — {e}")
            continue

        top_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)[: args.top]
        scores_str = "  ".join(f"{cls}: {score:.2%}" for cls, score in top_scores)
        print(f"{os.path.basename(path):<30} -> {result['predicted']:<20} | {scores_str}")


if __name__ == "__main__":
    main()
