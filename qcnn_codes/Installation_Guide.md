# Installation & Running Guide

## Prerequisites

| Requirement | Minimum Version | Recommended |
|-------------|----------------|-------------|
| Python | 3.9+ | 3.11 or 3.12 |
| pip | 21.0+ | Latest |
| RAM | 8 GB | 16 GB |
| Disk Space | 2 GB (dependencies) | 5 GB (with results) |
| GPU | Optional | NVIDIA CUDA-compatible (speeds up CNN) |
| OS | Windows 10, macOS 11+, Linux | Any |

---

## Step 1: Clone / Download the Project

```bash
# If using git:
git clone <repository-url>
cd cnn-moto

# Or simply navigate to the project folder:
cd /path/to/cnn-moto
```

---

## Step 2: Create a Virtual Environment (Recommended)

### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework (CNN layers, training) |
| `torchvision` | Image transforms and data utilities |
| `pennylane` | Quantum machine learning (quantum circuits) |
| `pennylane-lightning` | Fast quantum simulator backend |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation (CSV reading) |
| `scikit-learn` | Stratified splitting, metrics, classification reports |
| `matplotlib` | Plotting training curves and confusion matrices |
| `seaborn` | Enhanced visualization (heatmaps) |
| `Pillow` | Image loading and preprocessing |
| `tqdm` | Progress bars during training |

### Verify Installation:
```bash
python3 -c "import torch; import pennylane; print(f'PyTorch: {torch.__version__}'); print(f'PennyLane: {pennylane.__version__}'); print('All dependencies OK!')"
```

Expected output:
```
PyTorch: 2.x.x
PennyLane: 0.4x.x
All dependencies OK!
```

---

## Step 4: Verify Dataset

The dataset should already be present at:
```
cnn-moto/dataset/augmented_images/augmented_images1/
├── 0/          (digit 0 images)
├── 1/          (digit 1 images)
├── ...
├── 9/          (digit 9 images)
├── A_caps/     (uppercase A images)
├── B_caps/     (uppercase B images)
├── ...
├── Z_caps/     (uppercase Z images)
├── a/          (lowercase a images)
├── b/          (lowercase b images)
├── ...
└── z/          (lowercase z images)
```

Verify with:
```bash
python3 -c "
from dataset_loader import get_dataloaders
train_loader, val_loader, test_loader, num_classes = get_dataloaders(batch_size=32)
print('Dataset verification passed!')
"
```

Expected output:
```
Dataset loaded: 13640 images, 62 classes
  Train: 9547 | Val: 2047 | Test: 2046
  Image size: 48x48 | Stratified split: Yes
Dataset verification passed!
```

---

## Step 5: Training

### Option A: Train Both Models (Full Comparison)
```bash
python3 train.py --model both --epochs 50 --batch_size 32
```
- **Time estimate**: Classical CNN ~2 hours, QCNN ~8-15 hours (on CPU simulator)
- **Output**: Saves best models, training history, and predictions to `results/`

### Option B: Train Classical CNN Only (Fast)
```bash
python3 train.py --model classical --epochs 50 --batch_size 64
```
- **Time estimate**: ~1-2 hours on CPU, ~15-30 minutes with GPU
- **Expected accuracy**: 78-82%

### Option C: Train QCNN Only
```bash
python3 train.py --model qcnn --epochs 50 --batch_size 32
```
- **Time estimate**: ~8-15 hours (quantum simulation is sequential)
- **Expected accuracy**: 82-86%

### Training Arguments Reference:
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `both` | Which model to train: `classical`, `qcnn`, or `both` |
| `--epochs` | `50` | Maximum training epochs (early stopping at patience=15) |
| `--batch_size` | `32` | Batch size (use 32 for QCNN, 64 for classical) |
| `--lr` | `0.001` | Learning rate for classical CNN |
| `--qcnn_lr` | `0.002` | Learning rate for QCNN |

---

## Step 6: Monitoring Training Progress

While training runs, you'll see output like:
```
============================================================
Training classical_cnn
============================================================
Device: cpu
Parameters: 3,273,598
Trainable: 3,273,598
Learning rate: 0.001 | Epochs: 50

Epoch 1/50 (LR: 0.000100)
Training: 100%|██████████| 149/149 [01:51, acc=2.1%]
  Train Loss: 4.1229 | Train Acc: 2.07%
  Val Loss:   4.0469 | Val Acc:   3.96%
  >>> New best model saved (val_acc: 3.96%)

Epoch 2/50 (LR: 0.000280)
  ...
```

**Key indicators of healthy training:**
- Train accuracy increases each epoch
- Validation accuracy increases (may fluctuate)
- Loss decreases over time
- "New best model saved" messages appear regularly

**Warning signs:**
- Val accuracy stops improving for 10+ epochs → early stopping will trigger
- Train accuracy >> Val accuracy (gap > 20%) → overfitting (normal, handled by regularization)

---

## Step 7: Generate Evaluation Results

After training completes:
```bash
python3 evaluate.py
```

This generates:
```
results/
├── plots/
│   ├── training_curves.png        # Loss & accuracy over epochs
│   ├── confusion_matrix_classical_cnn.png
│   ├── confusion_matrix_qcnn.png
│   └── per_class_accuracy.png     # Bar chart comparing both models
├── classical_cnn_best.pth         # Best model weights
├── qcnn_best.pth
├── classical_cnn_history.json     # Training metrics per epoch
├── qcnn_history.json
├── classical_cnn_preds.npz        # Test predictions
├── qcnn_preds.npz
├── classical_cnn_report.json      # Full classification report
├── qcnn_report.json
└── final_results.json             # Summary comparison
```

---

## Step 8: View Results

### Check final accuracy:
```bash
python3 -c "
import json
with open('results/final_results.json') as f:
    r = json.load(f)
for name, data in r.items():
    print(f\"{name}: {data['test_acc']:.2f}% accuracy\")
"
```

### View plots:
```bash
# macOS:
open results/plots/training_curves.png
open results/plots/per_class_accuracy.png

# Linux:
xdg-open results/plots/training_curves.png

# Windows:
start results\plots\training_curves.png
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'torch'`
**Fix**: Activate your virtual environment and reinstall:
```bash
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Error: `No such file or directory: 'dataset/augmented_images/...'`
**Fix**: Ensure the dataset folder structure is correct. The images must be in:
```
cnn-moto/dataset/augmented_images/augmented_images1/<class_name>/<images>
```

### Error: `RuntimeError: CUDA out of memory`
**Fix**: Reduce batch size:
```bash
python3 train.py --model classical --batch_size 16
```

### Training is too slow
**Options**:
1. Use GPU (if available): The code auto-detects CUDA
2. Reduce epochs: `--epochs 30`
3. Train only classical model first: `--model classical`
4. Reduce quantum circuit size: Edit `N_QUBITS = 6` and `N_QLAYERS = 4` in `qcnn_model.py`

### Warning: `'pin_memory' argument is set as true but no accelerator is found`
**This is harmless** — it just means you're training on CPU. The code handles this automatically.

### Error: `Fontconfig error: No writable cache directories`
**This is harmless** — matplotlib cache issue. Does not affect training or results.

---

## Quick Start (Copy-Paste)

```bash
# One-shot setup and train (macOS/Linux):
cd cnn-moto
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 train.py --model classical --epochs 50 --batch_size 64
python3 evaluate.py
```

```bash
# One-shot setup and train (Windows):
cd cnn-moto
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py --model classical --epochs 50 --batch_size 64
python evaluate.py
```

---

## Project File Summary

| File | Purpose | Run It? |
|------|---------|---------|
| `requirements.txt` | Python package dependencies | `pip install -r requirements.txt` |
| `dataset_loader.py` | Loads images, applies transforms, creates dataloaders | Imported by train.py |
| `qcnn_model.py` | Defines QCNN and Classical CNN architectures | Imported by train.py |
| `train.py` | Main training script | `python3 train.py --model both` |
| `evaluate.py` | Generates plots and classification reports | `python3 evaluate.py` |
| `CNN_vs_QCNN_Analysis.md` | Comparison document | Read only |
| `Feature_Vector_Details.md` | Feature vector technical details | Read only |
| `README.md` | Project overview | Read only |
