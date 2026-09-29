"""
Dataset loader for the AffectNet-style emotion dataset:
AFFECTNET_FE_DATASET_SPLIT/train and AFFECTNET_FE_DATASET_SPLIT/test,
8 classes: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise.

The train folder is further split into train/val here (80/20), since the
training script expects a separate validation set for early stopping.
The test folder is kept fully separate for final evaluation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

TRAIN_DIR = "AFFECTNET_FE_DATASET_SPLIT/train"
TEST_DIR = "AFFECTNET_FE_DATASET_SPLIT/test"

IMAGE_SIZE = 64      # unchanged from the Keras version
VAL_SPLIT = 0.2      # carve 20% of the train folder out for validation
SEED = 42

NUM_CLASSES = 8


def get_dataloaders(batch_size=32, image_size=IMAGE_SIZE, val_split=VAL_SPLIT, seed=SEED):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Two ImageFolder instances over the same directory, different transforms.
    # ImageFolder lists files in a deterministic sorted order, so splitting
    # both with the same seed produces matching indices - the augmented
    # version is used for training, the plain version for validation.
    train_source = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_source = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)

    val_size = int(len(train_source) * val_split)
    train_size = len(train_source) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(train_source)), [train_size, val_size], generator=generator
    )

    train_dataset = torch.utils.data.Subset(train_source, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_source, val_indices.indices)

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_source.classes)

    print("Classes found:", train_source.class_to_idx)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, num_classes

# """
# Dataset loader for handwritten character images.

# Key improvements for >80% accuracy:
#     - Larger image size (48x48) preserves fine details for similar chars
#     - Stronger but character-safe augmentation pipeline
#     - Proper train/val/test split with separate transforms
#     - Stratified sampling to ensure balanced class representation
# """

# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from sklearn.model_selection import StratifiedShuffleSplit


# IMG_SIZE = 48
# DATASET_ROOT = Path(__file__).parent / "dataset" / "augmented_images" / "augmented_images1"

# CLASS_NAMES = (
#     [str(i) for i in range(10)]
#     + [f"{chr(c)}_caps" for c in range(ord('A'), ord('Z') + 1)]
#     + [chr(c) for c in range(ord('a'), ord('z') + 1)]
# )

# LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
# NUM_CLASSES = len(CLASS_NAMES)


# class HandwrittenCharDataset(Dataset):
#     """Custom dataset that loads images from class-specific subdirectories."""

#     def __init__(self, root_dir=None, transform=None, samples=None):
#         self.root_dir = Path(root_dir) if root_dir else DATASET_ROOT
#         self.transform = transform

#         if samples is not None:
#             self.samples = samples
#         else:
#             self.samples = []
#             for class_name in CLASS_NAMES:
#                 class_dir = self.root_dir / class_name
#                 if not class_dir.exists():
#                     continue
#                 label = LABEL_MAP[class_name]
#                 for img_file in sorted(class_dir.iterdir()):
#                     if img_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
#                         self.samples.append((str(img_file), label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         image = Image.open(img_path).convert('L')

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# def get_train_transforms():
#     """Aggressive but character-safe augmentation for training."""
#     return transforms.Compose([
#         transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
#         transforms.RandomCrop(IMG_SIZE),
#         transforms.RandomRotation(15),
#         transforms.RandomAffine(
#             degrees=0,
#             translate=(0.08, 0.08),
#             scale=(0.9, 1.1),
#             shear=5,
#         ),
#         transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
#         transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#         transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
#     ])


# def get_eval_transforms():
#     """Clean transform for validation/test (no augmentation)."""
#     return transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ])


# def get_dataloaders(batch_size=64, val_split=0.15, test_split=0.15, num_workers=0):
#     """
#     Create stratified train, validation, and test dataloaders.

#     Uses stratified splitting to ensure each class is proportionally
#     represented in all splits — critical for 62-class classification.

#     Returns:
#         train_loader, val_loader, test_loader, num_classes
#     """
#     full_dataset = HandwrittenCharDataset()
#     all_samples = full_dataset.samples
#     all_labels = [label for _, label in all_samples]

#     splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
#     train_val_idx, test_idx = next(splitter.split(all_samples, all_labels))

#     train_val_labels = [all_labels[i] for i in train_val_idx]
#     val_ratio = val_split / (1 - test_split)
#     splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
#     train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, train_val_labels))
#     train_idx = train_val_idx[train_idx_rel]
#     val_idx = train_val_idx[val_idx_rel]

#     train_samples = [all_samples[i] for i in train_idx]
#     val_samples = [all_samples[i] for i in val_idx]
#     test_samples = [all_samples[i] for i in test_idx]

#     train_dataset = HandwrittenCharDataset(
#         transform=get_train_transforms(), samples=train_samples
#     )
#     val_dataset = HandwrittenCharDataset(
#         transform=get_eval_transforms(), samples=val_samples
#     )
#     test_dataset = HandwrittenCharDataset(
#         transform=get_eval_transforms(), samples=test_samples
#     )

#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=True, drop_last=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )

#     print(f"Dataset loaded: {len(all_samples)} images, {NUM_CLASSES} classes")
#     print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
#     print(f"  Image size: {IMG_SIZE}x{IMG_SIZE} | Stratified split: Yes")

#     return train_loader, val_loader, test_loader, NUM_CLASSES
