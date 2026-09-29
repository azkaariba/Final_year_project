"""
Evaluation and visualization script.

Generates:
    - Training curves comparison plot
    - Confusion matrices
    - Per-class accuracy comparison
    - Classification report
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from dataset_loader import CLASS_NAMES

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves():
    """Plot loss and accuracy curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = {}
    for name in ["classical_cnn", "qcnn"]:
        path = RESULTS_DIR / f"{name}_history.json"
        if path.exists():
            with open(path) as f:
                models[name] = json.load(f)

    if not models:
        print("No training history found. Run train.py first.")
        return

    colors = {"classical_cnn": "#2196F3", "qcnn": "#E91E63"}
    labels = {"classical_cnn": "Classical CNN", "qcnn": "QCNN"}

    for name, hist in models.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], '-', color=colors[name],
                     label=f'{labels[name]} (train)', alpha=0.7)
        axes[0].plot(epochs, hist["val_loss"], '--', color=colors[name],
                     label=f'{labels[name]} (val)')

        axes[1].plot(epochs, hist["train_acc"], '-', color=colors[name],
                     label=f'{labels[name]} (train)', alpha=0.7)
        axes[1].plot(epochs, hist["val_acc"], '--', color=colors[name],
                     label=f'{labels[name]} (val)')

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'training_curves.png'}")


def plot_confusion_matrix(preds, labels, model_name, display_name):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_normalized, annot=False, cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, vmin=0, vmax=1
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {display_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / f'confusion_matrix_{model_name}.png'}")


def plot_per_class_accuracy():
    """Bar chart comparing per-class accuracy between models."""
    model_data = {}
    for name in ["classical_cnn", "qcnn"]:
        path = RESULTS_DIR / f"{name}_preds.npz"
        if path.exists():
            data = np.load(path)
            model_data[name] = (data["preds"], data["labels"])

    if len(model_data) < 2:
        print("Need both model predictions for comparison. Run train.py --model both first.")
        return

    fig, ax = plt.subplots(figsize=(20, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    for i, (name, (preds, labels)) in enumerate(model_data.items()):
        per_class_acc = []
        for cls_idx in range(len(CLASS_NAMES)):
            mask = labels == cls_idx
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).mean() * 100
            else:
                acc = 0
            per_class_acc.append(acc)

        offset = -width / 2 + i * width
        color = "#2196F3" if name == "classical_cnn" else "#E91E63"
        label = "Classical CNN" if name == "classical_cnn" else "QCNN"
        ax.bar(x + offset, per_class_acc, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy Comparison: Classical CNN vs QCNN")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'per_class_accuracy.png'}")


def generate_classification_reports():
    """Print and save detailed classification reports."""
    display_names = {"classical_cnn": "Classical CNN", "qcnn": "QCNN"}

    for name in ["classical_cnn", "qcnn"]:
        path = RESULTS_DIR / f"{name}_preds.npz"
        if not path.exists():
            continue

        data = np.load(path)
        preds, labels = data["preds"], data["labels"]

        print(f"\n{'='*60}")
        print(f"Classification Report: {display_names[name]}")
        print(f"{'='*60}")

        acc = accuracy_score(labels, preds) * 100
        f1_macro = f1_score(labels, preds, average='macro') * 100
        f1_weighted = f1_score(labels, preds, average='weighted') * 100

        print(f"  Overall Accuracy:    {acc:.2f}%")
        print(f"  Macro F1-Score:      {f1_macro:.2f}%")
        print(f"  Weighted F1-Score:   {f1_weighted:.2f}%")

        report = classification_report(
            labels, preds,
            target_names=CLASS_NAMES,
            output_dict=True
        )
        with open(RESULTS_DIR / f"{name}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n  Detailed report saved to: {RESULTS_DIR / f'{name}_report.json'}")

        plot_confusion_matrix(preds, labels, name, display_names[name])


def print_summary():
    """Print final comparison summary."""
    results_path = RESULTS_DIR / "final_results.json"
    if not results_path.exists():
        return

    with open(results_path) as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for name, data in results.items():
        display = "Classical CNN" if "classical" in name else "QCNN"
        print(f"\n  {display}:")
        print(f"    Test Accuracy: {data['test_acc']:.2f}%")
        print(f"    Test Loss:     {data['test_loss']:.4f}")

    if "classical_cnn" in results and "qcnn" in results:
        improvement = results["qcnn"]["test_acc"] - results["classical_cnn"]["test_acc"]
        print(f"\n  QCNN Improvement: {improvement:+.2f}%")
    print("=" * 60)


def main():
    print("Generating evaluation plots and reports...\n")
    plot_training_curves()
    generate_classification_reports()
    plot_per_class_accuracy()
    print_summary()
    print(f"\nAll plots saved in: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
