
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import numpy as np
from tqdm import tqdm

from dataset_loader import get_dataloaders, NUM_CLASSES
from qcnn_model import QCNN, ClassicalCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(model, model_name, train_loader, val_loader, epochs, lr, device):
    """
    Full training loop.

    Features:
        - AdamW with decoupled weight decay
        - Linear warmup (5 epochs) + Cosine annealing with warm restarts
        - Label smoothing (0.1) for better generalization
        - Gradient clipping at 1.0
        - Early stopping with patience=15
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Learning rate: {lr} | Epochs: {epochs}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / f"{model_name}_best.pth")
            print(f"  >>> New best model saved (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s | Best Val Acc: {best_val_acc:.2f}%")

    history["best_val_acc"] = best_val_acc
    history["training_time_s"] = elapsed
    history["total_params"] = sum(p.numel() for p in model.parameters())

    with open(RESULTS_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(RESULTS_DIR / f"{model_name}_best.pth", weights_only=True))
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train QCNN / Classical CNN on the emotion dataset")
    parser.add_argument("--model", choices=["qcnn", "classical", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for classical CNN")
    parser.add_argument("--qcnn_lr", type=float, default=0.002,
                        help="Learning rate for QCNN")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        batch_size=args.batch_size
    )

    results = {}

    if args.model in ("classical", "both"):
        model = ClassicalCNN(num_classes=num_classes)
        model, history = train_model(
            model, "classical_cnn", train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=DEVICE
        )
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        print(f"\n[Classical CNN] Test Accuracy: {test_acc:.2f}%")
        results["classical_cnn"] = {"test_acc": test_acc, "test_loss": test_loss}
        np.savez(RESULTS_DIR / "classical_cnn_preds.npz", preds=preds, labels=labels)

    if args.model in ("qcnn", "both"):
        model = QCNN(num_classes=num_classes)
        model, history = train_model(
            model, "qcnn", train_loader, val_loader,
            epochs=args.epochs, lr=args.qcnn_lr, device=DEVICE
        )
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
        print(f"\n[QCNN] Test Accuracy: {test_acc:.2f}%")
        results["qcnn"] = {"test_acc": test_acc, "test_loss": test_loss}
        np.savez(RESULTS_DIR / "qcnn_preds.npz", preds=preds, labels=labels)

    if args.model == "both" and len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        classical_acc = results["classical_cnn"]["test_acc"]
        qcnn_acc = results["qcnn"]["test_acc"]
        improvement = qcnn_acc - classical_acc
        print(f"  Classical CNN Test Accuracy: {classical_acc:.2f}%")
        print(f"  QCNN Test Accuracy:          {qcnn_acc:.2f}%")
        print(f"  Improvement:                 {improvement:+.2f}%")
        print("=" * 60)

    with open(RESULTS_DIR / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


# """
# Training script for QCNN and Classical CNN models.

# Designed to achieve >80% test accuracy with:
#     - AdamW optimizer with weight decay
#     - Cosine annealing with warm restarts
#     - Label smoothing cross-entropy loss
#     - Gradient clipping
#     - Extended patience for early stopping

# Usage:
#     python train.py --model qcnn --epochs 50 --batch_size 32
#     python train.py --model classical --epochs 50 --batch_size 64
#     python train.py --model both --epochs 50
# """

# import argparse
# import time
# import json
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
# import numpy as np
# from tqdm import tqdm

# from dataset_loader import get_dataloaders, NUM_CLASSES
# from qcnn_model import QCNN, ClassicalCNN


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RESULTS_DIR = Path("results")
# RESULTS_DIR.mkdir(exist_ok=True)


# def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     pbar = tqdm(loader, desc="Training", leave=False)
#     for images, labels in pbar:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#         pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

#     epoch_loss = running_loss / total
#     epoch_acc = 100.0 * correct / total
#     return epoch_loss, epoch_acc


# @torch.no_grad()
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []

#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

#     epoch_loss = running_loss / total
#     epoch_acc = 100.0 * correct / total
#     return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


# def train_model(model, model_name, train_loader, val_loader, epochs, lr, device):
#     """
#     Full training loop optimized for >80% accuracy.

#     Features:
#         - AdamW with decoupled weight decay
#         - Linear warmup (5 epochs) + Cosine annealing with warm restarts
#         - Label smoothing (0.1) for better generalization
#         - Gradient clipping at 1.0
#         - Early stopping with patience=15
#     """
#     print(f"\n{'='*60}")
#     print(f"Training {model_name}")
#     print(f"{'='*60}")
#     print(f"Device: {device}")
#     print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
#     print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     print(f"Learning rate: {lr} | Epochs: {epochs}")

#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

#     warmup_epochs = 5
#     warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
#     cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01)
#     scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

#     history = {
#         "train_loss": [], "train_acc": [],
#         "val_loss": [], "val_acc": [],
#     }
#     best_val_acc = 0.0
#     patience = 15
#     patience_counter = 0
#     start_time = time.time()

#     for epoch in range(1, epochs + 1):
#         print(f"\nEpoch {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f})")

#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, criterion, optimizer, device
#         )
#         val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
#         scheduler.step()

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["val_loss"].append(val_loss)
#         history["val_acc"].append(val_acc)

#         print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
#         print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             patience_counter = 0
#             torch.save(model.state_dict(), RESULTS_DIR / f"{model_name}_best.pth")
#             print(f"  >>> New best model saved (val_acc: {val_acc:.2f}%)")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"  Early stopping at epoch {epoch}")
#                 break

#     elapsed = time.time() - start_time
#     print(f"\nTraining complete in {elapsed:.1f}s | Best Val Acc: {best_val_acc:.2f}%")

#     history["best_val_acc"] = best_val_acc
#     history["training_time_s"] = elapsed
#     history["total_params"] = sum(p.numel() for p in model.parameters())

#     with open(RESULTS_DIR / f"{model_name}_history.json", "w") as f:
#         json.dump(history, f, indent=2)

#     model.load_state_dict(torch.load(RESULTS_DIR / f"{model_name}_best.pth", weights_only=True))
#     return model, history


# def main():
#     parser = argparse.ArgumentParser(description="Train QCNN / Classical CNN (>80% target)")
#     parser.add_argument("--model", choices=["qcnn", "classical", "both"], default="both")
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=0.001,
#                         help="Learning rate for classical CNN")
#     parser.add_argument("--qcnn_lr", type=float, default=0.002,
#                         help="Learning rate for QCNN")
#     args = parser.parse_args()

#     train_loader, val_loader, test_loader, num_classes = get_dataloaders(
#         batch_size=args.batch_size
#     )

#     results = {}

#     if args.model in ("classical", "both"):
#         model = ClassicalCNN(num_classes=num_classes)
#         model, history = train_model(
#             model, "classical_cnn", train_loader, val_loader,
#             epochs=args.epochs, lr=args.lr, device=DEVICE
#         )
#         criterion = nn.CrossEntropyLoss()
#         test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
#         print(f"\n[Classical CNN] Test Accuracy: {test_acc:.2f}%")
#         results["classical_cnn"] = {"test_acc": test_acc, "test_loss": test_loss}
#         np.savez(RESULTS_DIR / "classical_cnn_preds.npz", preds=preds, labels=labels)

#     if args.model in ("qcnn", "both"):
#         model = QCNN(num_classes=num_classes)
#         model, history = train_model(
#             model, "qcnn", train_loader, val_loader,
#             epochs=args.epochs, lr=args.qcnn_lr, device=DEVICE
#         )
#         criterion = nn.CrossEntropyLoss()
#         test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
#         print(f"\n[QCNN] Test Accuracy: {test_acc:.2f}%")
#         results["qcnn"] = {"test_acc": test_acc, "test_loss": test_loss}
#         np.savez(RESULTS_DIR / "qcnn_preds.npz", preds=preds, labels=labels)

#     if args.model == "both" and len(results) == 2:
#         print("\n" + "=" * 60)
#         print("COMPARISON SUMMARY")
#         print("=" * 60)
#         classical_acc = results["classical_cnn"]["test_acc"]
#         qcnn_acc = results["qcnn"]["test_acc"]
#         improvement = qcnn_acc - classical_acc
#         print(f"  Classical CNN Test Accuracy: {classical_acc:.2f}%")
#         print(f"  QCNN Test Accuracy:          {qcnn_acc:.2f}%")
#         print(f"  Improvement:                 {improvement:+.2f}%")
#         if qcnn_acc >= 80:
#             print(f"  TARGET ACHIEVED: QCNN accuracy >= 80%")
#         print("=" * 60)

#     with open(RESULTS_DIR / "final_results.json", "w") as f:
#         json.dump(results, f, indent=2)


# if __name__ == "__main__":
#     main()
