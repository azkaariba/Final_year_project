# Emotion Recognition using CNN

A deep learning project that trains a **Convolutional Neural Network (CNN)** to recognize handwritten characters from images. Built with **TensorFlow / Keras** and **OpenCV**.

---

## 📁 Project Structure

```
char/
├── train.py            # Script to train the CNN model
├── predict.py          # Script to predict characters from images
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── dataset/            # (You provide this)
│   ├── train/          # Training images organized by class folders
│   └── test/           # Testing images organized by class folders
└── model/              # (Auto-created after training)
    ├── cnn_model.keras  # Saved trained model
    └── classes.txt      # List of class labels
```

---

## ⚙️ Installation Guide

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** (Python package manager)

### Step 1 — Clone or Download the Project

If you have this as a Git repository:

```bash
git clone <your-repo-url>
cd char
```

Or simply navigate to the project folder:

```bash
cd /path/to/char
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package          | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `tensorflow`     | Deep learning framework for building the CNN |
| `numpy`          | Numerical operations on image arrays         |
| `opencv-python`  | Image reading, resizing, and preprocessing   |

### Step 4 — Prepare Your Dataset

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── A/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   ├── B/
│   ├── C/
│   └── ...
└── test/
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

Each subfolder name becomes a **class label**. Images should be grayscale handwritten characters.

---

## 🚀 How to Run

### Train the Model

```bash
python train.py
```

This will:
1. Load and augment training images from `dataset/train/`
2. Build and compile a 3-layer CNN
3. Train for up to 20 epochs (with early stopping)
4. Save the best model to `model/cnn_model.keras`
5. Save class labels to `model/classes.txt`

### Predict a Character

**Single image:**

```bash
python predict.py path/to/image.png
```

**Entire folder of images:**

```bash
python predict.py path/to/folder/
```

---

## 🧠 Code Explanation

### `train.py` — Model Training

#### 1. Imports & Parameters

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMAGE_SIZE = (64, 64)   # All images are resized to 64×64 pixels
BATCH_SIZE = 32          # Number of images processed per training step
EPOCHS = 20              # Maximum training iterations over the full dataset
```

- **`Sequential`** — A linear stack of layers; we add layers one after another.
- **`IMAGE_SIZE`** — Every input image is resized to 64×64 for uniform input to the network.
- **`BATCH_SIZE`** — Controls memory usage and training speed.
- **`EPOCHS`** — The model sees the entire dataset 20 times (unless early stopping triggers).

---

#### 2. Data Augmentation & Loading

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values from [0,255] → [0,1]
    rotation_range=10,         # Randomly rotate images by ±10°
    width_shift_range=0.1,     # Randomly shift images horizontally by 10%
    height_shift_range=0.1,    # Randomly shift images vertically by 10%
    zoom_range=0.1,            # Randomly zoom in/out by 10%
    shear_range=0.1            # Randomly apply shearing transformation
)
```

- **`rescale=1./255`** — Neural networks work better with small numbers. This converts pixel values (0–255) to the range (0–1).
- **Augmentation** (`rotation`, `shift`, `zoom`, `shear`) — Creates slightly modified versions of each image during training. This helps the model generalize and prevents **overfitting** (memorizing training data instead of learning patterns).

```python
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",    # Load images as single-channel (black & white)
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # One-hot encode labels (e.g., [0,0,1,0,...])
    shuffle=True               # Randomize order each epoch
)
```

- **`flow_from_directory`** — Automatically reads images from subfolders, using folder names as class labels.
- **`color_mode="grayscale"`** — Handwritten characters don't need color, so we use 1 channel instead of 3 (RGB).
- **`class_mode="categorical"`** — Converts labels into one-hot vectors for multi-class classification.

---

#### 3. CNN Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])
```

**Layer-by-layer breakdown:**

| Layer                | What It Does                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Conv2D(32, (3,3))`  | Applies 32 filters of size 3×3 to detect low-level features (edges, curves)                                  |
| `MaxPooling2D(2,2)`  | Reduces spatial dimensions by half (64→32), keeping the strongest features                                    |
| `Conv2D(64, (3,3))`  | Applies 64 filters to detect mid-level features (strokes, corners)                                           |
| `MaxPooling2D(2,2)`  | Reduces dimensions again (32→16→...)                                                                         |
| `Conv2D(128, (3,3))` | Applies 128 filters to detect high-level features (character shapes)                                         |
| `MaxPooling2D(2,2)`  | Final spatial reduction                                                                                       |
| `Flatten()`          | Converts the 2D feature maps into a 1D vector so Dense layers can process them                               |
| `Dense(256)`         | A fully connected layer with 256 neurons — learns complex combinations of detected features                   |
| `Dropout(0.5)`       | Randomly turns off 50% of neurons during training to prevent overfitting                                      |
| `Dense(num_classes)` | Output layer with one neuron per class; **softmax** converts outputs to probabilities that sum to 1           |

---

#### 4. Compilation

```python
model.compile(
    optimizer="adam",                    # Adaptive learning rate optimizer
    loss="categorical_crossentropy",     # Standard loss for multi-class classification
    metrics=["accuracy"]                 # Track accuracy during training
)
```

- **Adam optimizer** — Automatically adjusts learning rates per parameter; widely used and effective.
- **Categorical crossentropy** — Measures how far the predicted probability distribution is from the true label.

---

#### 5. Callbacks & Training

```python
checkpoint = ModelCheckpoint(
    "model/cnn_model.keras",
    monitor="val_accuracy",      # Watch validation accuracy
    save_best_only=True,         # Only save when accuracy improves
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",          # Watch validation loss
    patience=5,                  # Stop if no improvement for 5 epochs
    restore_best_weights=True    # Roll back to the best weights
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)
```

- **ModelCheckpoint** — Saves the model only when validation accuracy improves, so you always keep the best version.
- **EarlyStopping** — Stops training if the model stops improving for 5 consecutive epochs, saving time and avoiding overfitting.
- **`model.fit()`** — The actual training loop. The model learns by adjusting its weights to minimize the loss function.

---

#### 6. Evaluation & Saving

```python
loss, accuracy = model.evaluate(test_generator)
model.save("model/cnn_model.keras")

with open("model/classes.txt", "w") as f:
    for cls in train_generator.class_indices:
        f.write(cls + "\n")
```

- **`model.evaluate()`** — Tests the trained model on unseen test data and reports accuracy and loss.
- **`model.save()`** — Saves the complete model (architecture + weights) in Keras format.
- **`classes.txt`** — Stores the class names so `predict.py` knows what each output index means.

---

### `predict.py` — Character Prediction

#### 1. Image Path Handling

```python
def get_image_paths(image_path):
    if os.path.isfile(image_path):
        return [image_path]               # Single image
    if os.path.isdir(image_path):
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        # Collect all image files from the folder
        ...
    return []
```

- Accepts either a **single image file** or an **entire directory**.
- Filters files by common image extensions so non-image files are ignored.

---

#### 2. Loading the Model

```python
def load_model_and_classes():
    model = tf.keras.models.load_model("model/cnn_model.keras")
    with open("model/classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names
```

- Loads the trained CNN model and the class labels from the `model/` folder.

---

#### 3. Preprocessing & Prediction

```python
def predict_image(model, class_names, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # Read as grayscale
    image = cv2.resize(image, (64, 64))                     # Resize to 64×64
    image = image.astype("float32") / 255.0                 # Normalize to [0,1]
    image = np.expand_dims(image, axis=-1)                  # Add channel dim → (64,64,1)
    image = np.expand_dims(image, axis=0)                   # Add batch dim → (1,64,64,1)

    prediction = model.predict(image, verbose=0)
    predicted_index = int(np.argmax(prediction))            # Index of highest probability
    predicted_class = class_names[predicted_index]          # Map index to class name
    confidence = float(prediction[0][predicted_index] * 100)
    return predicted_class, confidence
```

**Step-by-step:**

1. **Read** the image in grayscale (matching the training format).
2. **Resize** to 64×64 pixels (same as training input size).
3. **Normalize** pixel values to 0–1 (same as training preprocessing).
4. **Expand dimensions** — The model expects input shape `(batch_size, 64, 64, 1)`, so we add the batch and channel dimensions.
5. **Predict** — The model outputs a probability array. `argmax` finds the class with the highest confidence.

---

#### 4. CLI Entry Point

```python
parser = argparse.ArgumentParser(description="Predict handwritten characters from images")
parser.add_argument("image_path", nargs="?", default="image_path",
                    help="Path to an image or folder of images")
```

- Uses Python's `argparse` to accept the image path from the command line.
- `nargs="?"` makes the argument optional.

---

## 📊 Expected Output

### Training

```
Epoch 1/20
100/100 [==============================] - 12s - loss: 2.1045 - accuracy: 0.3521 - val_loss: 1.2034 - val_accuracy: 0.6123
...
Test Accuracy : 92.45 %
Test Loss     : 0.2531
CNN Model Saved Successfully!
```

### Prediction

```
==============================
Image           : sample_A.png
Predicted Class : A
Confidence      : 97.83%
==============================
```

---

## 📝 Notes

- The model uses **grayscale 64×64** images. Make sure your input images are clear handwritten characters.
- The commented-out code in `train.py` (lines 167–345) is an alternative **transfer learning** approach using **MobileNetV2**. It uses a pretrained model on ImageNet and fine-tunes it for character recognition. This can yield higher accuracy but requires more computational resources.
- Increase `EPOCHS` or add more training data for better accuracy.
- Use a GPU for significantly faster training (`pip install tensorflow[and-cuda]` for NVIDIA GPUs).
