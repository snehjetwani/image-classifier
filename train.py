"""
Train the image classifier on a custom dataset.

Expected dataset layout:
    data/
        train/
            cat/  dog/  ...
        val/
            cat/  dog/  ...

Usage:
    python train.py --data_dir data --epochs 10 --output model.pth
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from classifier import TRAIN_TRANSFORMS, EVAL_TRANSFORMS, build_model, save_model


def train(data_dir: str, epochs: int, batch_size: int, lr: float, output: str, unfreeze: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=TRAIN_TRANSFORMS)
    val_dataset = ImageFolder(root=f"{data_dir}/val", transform=EVAL_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names), freeze_backbone=not unfreeze).to(device)

    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, class_names, output)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--data_dir", default="data", help="Root data directory with train/ and val/ subdirs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="model.pth", help="Path to save best model")
    parser.add_argument("--unfreeze", action="store_true", help="Fine-tune entire backbone (slower, may need lower lr)")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output, args.unfreeze)
