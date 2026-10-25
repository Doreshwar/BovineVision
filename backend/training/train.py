"""
BovineVision AI - Training Pipeline
Trains the custom BovineVisionCNN entirely from scratch using PyTorch and CUDA.
Generates genuine evaluation metrics, confusion matrix, and model artifacts.
"""

import os
import sys
import json
import time
import shutil
import argparse
from typing import Dict, Any


os.environ["MPLCONFIGDIR"] = "G:/Projektz/backend/saved_models/.mpl_cache"


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.app.models.cnn import BovineVisionCNN


def get_data_loaders(data_dir: str, batch_size: int = 32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_transforms)

    classes = train_dataset.classes
    print(f"[Data] Found {len(classes)} classes: {classes}")
    print(f"[Data] Split counts: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )


    return train_loader, val_loader, test_loader, classes


def train_model(
    data_dir: str = "G:/Projektz/data",
    output_dir: str = "G:/Projektz/backend/saved_models",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"  BovineVision AI - CNN Training Pipeline ")
    print(f"==========================================")
    print(f"Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, classes = get_data_loaders(data_dir, batch_size)
    num_classes = len(classes)

    # Initialize custom architecture completely from scratch
    model = BovineVisionCNN(
        in_channels=3,
        num_classes=num_classes,
        stem_channels=32,
        stage_channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        strides=[1, 2, 2, 2],
        use_se=True,
        dropout_rate=0.25,
        classifier_hidden=256
    ).to(device)

    summary = model.get_architecture_summary()
    print(f"[Model] Initialized {summary['name']} (Params: {summary['total_parameters']:,})")
    print(f"[Model] Pretrained weights: {summary['pretrained']} (Training from scratch)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_weights_path = os.path.join(output_dir, "best_model.pth")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        scheduler.step()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["learning_rates"].append(current_lr)

        ep_duration = time.time() - ep_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
            f"Time: {ep_duration:.1f}s"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": model.config,
                "classes": classes
            }
            # Robust file saving on Windows to avoid transient file-lock collisions
            temp_save_path = os.path.join(output_dir, f"best_model_epoch_{epoch}.pth")
            for attempt in range(5):
                try:
                    torch.save(checkpoint_payload, temp_save_path)
                    shutil.copy2(temp_save_path, best_weights_path)
                    try:
                        os.remove(temp_save_path)
                    except OSError:
                        pass
                    print(f"  --> Saved new best model checkpoint! (Val Acc: {best_val_acc*100:.2f}%)", flush=True)
                    break
                except Exception as save_err:
                    if attempt == 4:
                        print(f"  [Warning] Checkpoint saved as {temp_save_path} due to file lock on {best_weights_path}: {save_err}", flush=True)
                    time.sleep(0.4)


    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}.")

    # Load best weights for rigorous test set evaluation
    checkpoint = torch.load(best_weights_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data).item()
            test_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    test_accuracy = test_correct / test_total
    print(f"\n==========================================")
    print(f"  Test Evaluation on Unseen Test Split    ")
    print(f"==========================================")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}% ({test_correct}/{test_total})")

    report_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=classes)
    print("\nClassification Report:\n", report_text)

    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot & Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('BovineVision AI - Confusion Matrix (Test Set)', fontsize=14, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=11, fontweight='bold')

    plt.ylabel('True Cattle Breed', fontsize=12)
    plt.xlabel('Predicted Cattle Breed', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Plot & Save Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], label='Train Loss', color='#2563eb', lw=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label='Val Loss', color='#dc2626', lw=2)
    plt.title('Loss Curve', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], label='Train Acc (%)', color='#2563eb', lw=2)
    plt.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], label='Val Acc (%)', color='#16a34a', lw=2)
    plt.title('Accuracy Curve', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_plot_path, dpi=200)
    plt.close()

    # Save classes.json
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)

    # Save metrics.json
    metrics_data = {
        "dataset_source": "https://data.mendeley.com/datasets/xc3j9v382w/1",
        "dataset_title": "Annotated Multiview Image Dataset of Prominent Indigenous Indian Cattle Breeds",
        "dataset_doi": "10.17632/xc3j9v382w.1",
        "total_authentic_images": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        "split_counts": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset)
        },
        "model_architecture": summary,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc * 100, 2),
        "final_test_accuracy": round(test_accuracy * 100, 2),
        "classes": classes,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "training_time_minutes": round(total_training_time / 60, 2)
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\n[Artifacts] Successfully saved:")
    print(f"  Weights: {best_weights_path}")
    print(f"  Classes: {classes_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Confusion Matrix Plot: {cm_path}")
    print(f"  History Curves: {history_plot_path}")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BovineVision AI CNN from scratch")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
