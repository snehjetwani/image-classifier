"""
Image classifier using transfer learning with a pre-trained ResNet18 model.
"""
import json
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Return a ResNet18 with the final layer replaced for `num_classes`."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_model(model: nn.Module, class_names: list[str], path: str) -> None:
    """Save model weights and class names to `path`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "class_names": class_names},
        path,
    )
    print(f"Model saved to {path}")


def load_model(path: str) -> tuple[nn.Module, list[str]]:
    """Load model and class names from `path`."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names


def predict(model: nn.Module, class_names: list[str], image_path: str, device: str = "cpu") -> dict:
    """Classify a single image. Returns a dict with label and confidence scores."""
    image = Image.open(image_path).convert("RGB")
    tensor = EVAL_TRANSFORMS(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    if isinstance(probs, float):
        probs = [probs]
    scores = {cls: round(prob, 4) for cls, prob in zip(class_names, probs)}
    predicted_label = max(scores, key=scores.get)
    return {"predicted": predicted_label, "scores": scores}
