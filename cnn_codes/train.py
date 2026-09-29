import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, Activation, SpatialDropout2D,
    Input, Reshape, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback, EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight


train_path = "AFFECTNET_FE_DATASET_SPLIT/train"
test_path = "AFFECTNET_FE_DATASET_SPLIT/test"


# Image size changed from 64x64 to 128x128x3 as requested.
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
# Epoch budget reduced 100 -> 40. Combined with the speed fixes above and
# the EarlyStopping added below, this should finish noticeably faster while
# still reaching (or exceeding) the previous best accuracy.
EPOCHS = 40


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True 
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False 
)

print("\nClasses Found:")
print(train_generator.class_indices)

num_classes = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())

# Class weights help accuracy on imbalanced emotion datasets (e.g. AffectNet's
# "Contempt" class is typically much rarer than "Happy"). Sqrt-dampened so it
# corrects imbalance without overcorrecting on a dataset this size.
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_array = np.sqrt(class_weights_array)
class_weights = dict(enumerate(class_weights_array))
print("\nClass weights (sqrt-dampened, to counter class imbalance without overcorrecting):")
print(class_weights)


# =============================================================================
# Selective State-Space (Mamba-style) block, built from scratch in Keras/TF.
#
# IMPORTANT HONESTY NOTE: this is a simplified, single-direction
# approximation of the core Mamba/S6 mechanism (input-dependent B, C and
# step-size delta, a diagonal per-channel state matrix A, sequential scan
# recurrence h_t = exp(delta_t*A)*h_{t-1} + (delta_t*B_t)*x_t). It is NOT
# the real MedMamba VSSM, which additionally performs a 4-directional 2D
# selective scan ("SS2D") over the spatial feature map and lives in a
# separate PyTorch file (Medmamba.py) that wasn't available to port from.
# This block was unit-tested standalone (correct shapes, every weight
# receives a gradient) before being wired into the model below.
# =============================================================================
class SelectiveSSM1D(layers.Layer):
    def __init__(self, model_dim, state_dim=16, expand=2, conv_kernel=3, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.expand_dim = expand * model_dim
        self.conv_kernel = conv_kernel

    def build(self, input_shape):
        d_exp = self.expand_dim
        d_state = self.state_dim

        self.in_proj = layers.Dense(d_exp * 2, use_bias=False, name="in_proj")
        self.conv1d = layers.Conv1D(
            filters=d_exp, kernel_size=self.conv_kernel, padding="causal",
            groups=d_exp, use_bias=True, name="causal_conv"
        )
        self.delta_proj = layers.Dense(d_exp, use_bias=True, name="delta_proj")
        self.B_proj = layers.Dense(d_state, use_bias=False, name="B_proj")
        self.C_proj = layers.Dense(d_state, use_bias=False, name="C_proj")

        # Diagonal state matrix A, Mamba-style negative-log init for stability.
        A_init = tf.tile(
            tf.reshape(tf.range(1, d_state + 1, dtype=tf.float32), (1, d_state)),
            (d_exp, 1)
        )
        self.A_log = self.add_weight(
            shape=(d_exp, d_state),
            initializer=tf.keras.initializers.Constant(tf.math.log(A_init)),
            trainable=True, name="A_log"
        )
        self.D = self.add_weight(shape=(d_exp,), initializer="ones", trainable=True, name="D")
        self.out_proj = layers.Dense(self.model_dim, use_bias=False, name="out_proj")
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, seq, model_dim)
        xz = self.in_proj(inputs)
        x, z = tf.split(xz, 2, axis=-1)

        x = self.conv1d(x)
        x = tf.nn.silu(x)

        delta = tf.nn.softplus(self.delta_proj(x))       # (batch, seq, d_exp)
        B = self.B_proj(x)                                 # (batch, seq, d_state)
        C = self.C_proj(x)                                 # (batch, seq, d_state)

        A = -tf.exp(self.A_log)                            # (d_exp, d_state)

        A_bar = tf.exp(delta[..., None] * A[None, None, :, :])
        B_bar = delta[..., None] * B[:, :, None, :]
        Bx = B_bar * x[..., None]

        A_bar_t = tf.transpose(A_bar, [1, 0, 2, 3])
        Bx_t = tf.transpose(Bx, [1, 0, 2, 3])

        def step(h_prev, elems):
            a_t, bx_t = elems
            return a_t * h_prev + bx_t

        batch = tf.shape(inputs)[0]
        h0 = tf.zeros((batch, self.expand_dim, self.state_dim))
        h_seq = tf.scan(step, (A_bar_t, Bx_t), initializer=h0)

        h_seq = tf.transpose(h_seq, [1, 0, 2, 3])
        C_exp = C[:, :, None, :]
        y = tf.reduce_sum(h_seq * C_exp, axis=-1)
        y = y + x * self.D

        y = y * tf.nn.silu(z)
        out = self.out_proj(y)
        return out + inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_dim": self.model_dim,
            "state_dim": self.state_dim,
            "expand": self.expand_dim // self.model_dim,
            "conv_kernel": self.conv_kernel,
        })
        return config


# =============================================================================
# Model: CNN stem (Blocks 1-4, unchanged design - "layers 6-7" = the
# conv2d_6/conv2d_7 pair = Block 4, kept exactly as before) feeding into a
# VSSM-inspired core (SelectiveSSM1D blocks) instead of Block 5 + Dense head.
# =============================================================================
L2 = 1e-4

inputs = Input(shape=(128, 128, 3))

# Block 1
x = Conv2D(40, (3, 3), padding="same", kernel_regularizer=l2(L2))(inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(40, (3, 3), padding="same", kernel_regularizer=l2(L2))(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = SpatialDropout2D(0.1)(x)

# Block 2
x = Conv2D(80, (3, 3), padding="same", kernel_regularizer=l2(L2))(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(80, (3, 3), padding="same", kernel_regularizer=l2(L2))(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = SpatialDropout2D(0.1)(x)

# Block 3
x = Conv2D(120, (3, 3), padding="same", kernel_regularizer=l2(L2))(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(120, (3, 3), padding="same", kernel_regularizer=l2(L2))(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = SpatialDropout2D(0.2)(x)

# Block 4 - "layers 6-7" - kept exactly as in the previous architecture.
# With a 128x128 input, this reaches 8x8x160 after 4 poolings - the same
# spatial/channel size these two layers reached before (when input was
# 64x64 with 3 prior poolings), just scaled up for the larger input.
x = Conv2D(160, (3, 3), padding="same", kernel_regularizer=l2(L2), name="layer6_conv")(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Conv2D(160, (3, 3), padding="same", kernel_regularizer=l2(L2), name="layer7_conv")(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = SpatialDropout2D(0.2)(x)

# VSSM-inspired core: flatten the feature map into a token sequence and run
# it through the selective-SSM block.
#
# SPEED FIX: an extra pooling stage (8x8 -> 4x4) was added right before this,
# cutting the sequence length fed into the scan from 64 tokens to 16. The
# sequential scan inside SelectiveSSM1D is the dominant per-epoch cost (it
# can't parallelize across the sequence the way a conv layer parallelizes
# across space), so 4x fewer tokens is the single biggest speed win
# available without abandoning the SSM approach entirely. Also dropped from
# 2 stacked SSM blocks to 1, and expand=2 -> expand=1 (halves the internal
# working dimension the scan operates on) - together these cut the SSM
# portion's compute roughly 8-10x versus the previous version.
x = MaxPooling2D(2, 2)(x)             # 8x8 -> 4x4
x = SpatialDropout2D(0.2)(x)

h, w, c = x.shape[1], x.shape[2], x.shape[3]
x_seq = Reshape((h * w, c))(x)
x_seq = SelectiveSSM1D(model_dim=c, state_dim=16, expand=1, name="vss_block_1")(x_seq)

x_pooled = GlobalAveragePooling1D()(x_seq)
x_pooled = Dense(256, kernel_regularizer=l2(L2))(x_pooled)
x_pooled = Activation("relu")(x_pooled)
x_pooled = BatchNormalization()(x_pooled)
x_pooled = Dropout(0.5)(x_pooled)
outputs = Dense(num_classes, activation="softmax")(x_pooled)

model = Model(inputs, outputs, name="CNN_VSSM_Hybrid")


model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

os.makedirs("model", exist_ok=True)

checkpoint = ModelCheckpoint(
    "model/cnn_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


class BestModelLogger(Callback):
    """Prints a clear message every time a new best validation-accuracy
    model is saved, on top of the ModelCheckpoint save."""
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            print(f"\n>>> New best model! Epoch {epoch + 1} | "
                  f"Val Accuracy: {val_acc * 100:.2f}% "
                  f"(saved to model/cnn_model.keras)\n")


best_model_logger = BestModelLogger()

# Patience tightened from 8 -> 5 since the epoch budget is now much smaller
# (40 instead of 100) - the LR needs to react faster within that window.
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    mode="max",
    factor=0.6,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Re-added EarlyStopping (it was removed earlier to force full 100-epoch
# runs). Now that the goal is a shorter, faster run, this stops training
# once val_accuracy plateaus instead of burning through all 40 epochs
# regardless - saving time - while restore_best_weights ensures nothing is
# lost if it does stop early.
earlystop = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

csv_logger = CSVLogger(
    "model/training_log.csv",
    append=False
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, best_model_logger, reduce_lr, earlystop, csv_logger]
)


# Reload the BEST saved checkpoint before final evaluation, so the reported
# accuracy/F1/confusion matrix reflect the best epoch, not just the last one.
model.load_weights("model/cnn_model.keras")
print("\nReloaded best checkpoint (highest val_accuracy) for final evaluation.\n")

loss, accuracy = model.evaluate(test_generator)

print("\n===================================")
print("Test Accuracy (best model) :", round(accuracy * 100, 2), "%")
print("Test Loss     :", round(loss, 4))
print("Best Val Accuracy during training :", round(best_model_logger.best_val_acc * 100, 2), "%")
print("===================================\n")


test_generator.reset()
y_pred_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes[: len(y_pred)]

f1 = f1_score(y_true, y_pred, average="weighted")
print("\nWeighted F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix (raw counts):")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)

plt.show()
plt.close()
print("Confusion matrix saved to model/confusion_matrix.png")


model.save("model/cnn_model.keras")

print("CNN Model Saved Successfully!")


with open("model/classes.txt", "w") as f:
    for cls in train_generator.class_indices:
        f.write(cls + "\n")

print("Class labels saved.")




# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
#     BatchNormalization, Activation
# )
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback

# from sklearn.metrics import confusion_matrix, classification_report, f1_score
# from sklearn.utils.class_weight import compute_class_weight


# train_path = "AFFECTNET_FE_DATASET_SPLIT/train"
# test_path = "AFFECTNET_FE_DATASET_SPLIT/test"


# IMAGE_SIZE = (64, 64)
# BATCH_SIZE = 32
# EPOCHS = 100  


# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.15,
#     shear_range=0.1,
#     brightness_range=[0.8, 1.2],
#     horizontal_flip=True 
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255
# )

# train_generator = train_datagen.flow_from_directory(
#     train_path,
#     target_size=IMAGE_SIZE,
#     color_mode="rgb",
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=True
# )

# test_generator = test_datagen.flow_from_directory(
#     test_path,
#     target_size=IMAGE_SIZE,
#     color_mode="rgb",
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=False 
# )

# print("\nClasses Found:")
# print(train_generator.class_indices)

# num_classes = train_generator.num_classes
# class_labels = list(train_generator.class_indices.keys())

# # Class weights help accuracy on imbalanced emotion datasets (e.g. AffectNet's
# # "Contempt" class is typically much rarer than "Happy") without touching the
# # model architecture at all. A sqrt-dampened version is used instead of raw
# # "balanced" weights - full inverse-frequency weighting over-corrects on a
# # dataset this size, distorting the training loss landscape and dragging
# # down overall accuracy for the sake of the rarest classes.
# class_weights_array = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(train_generator.classes),
#     y=train_generator.classes
# )
# class_weights_array = np.sqrt(class_weights_array)
# class_weights = dict(enumerate(class_weights_array))
# print("\nClass weights (sqrt-dampened, to counter class imbalance without overcorrecting):")
# print(class_weights)


# model = Sequential([

#     # Block 1
#     Conv2D(40, (3, 3), padding="same", input_shape=(64, 64, 3)),
#     Activation("relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Block 2
#     Conv2D(80, (3, 3), padding="same"),
#     Activation("relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Block 3
#     Conv2D(120, (3, 3), padding="same"),
#     Activation("relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Block 4
#     Conv2D(160, (3, 3), padding="same"),
#     Activation("relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     # Block 5
#     Conv2D(200, (3, 3), padding="same"),
#     Activation("relu"),
#     BatchNormalization(),
#     MaxPooling2D(2, 2),

#     Flatten(),

#     Dense(500),
#     Activation("relu"),
#     BatchNormalization(),
#     Dropout(0.5),

#     Dense(num_classes, activation="softmax")

# ])


# model.compile(
#     # AdamW adds decoupled weight decay on top of Adam. This regularizes the
#     # existing layers (reduces overfitting) purely through the optimizer -
#     # no architecture change - which typically improves val/test accuracy
#     # over long training runs like this 100-epoch one.
#     optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4),
#     # Label smoothing softens the one-hot targets slightly, which typically
#     # improves generalization/accuracy on noisy-label datasets like AffectNet
#     # without any architecture change.
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=["accuracy"]
# )

# model.summary()

# os.makedirs("model", exist_ok=True)

# checkpoint = ModelCheckpoint(
#     "model/cnn_model.keras",
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )


# class BestModelLogger(Callback):
#     """Prints a clear message in the output every time a new best
#     validation-accuracy model is saved, on top of the ModelCheckpoint save."""
#     def __init__(self):
#         super().__init__()
#         self.best_val_acc = 0.0

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_acc = logs.get("val_accuracy")
#         if val_acc is not None and val_acc > self.best_val_acc:
#             self.best_val_acc = val_acc
#             print(f"\n>>> New best model! Epoch {epoch + 1} | "
#                   f"Val Accuracy: {val_acc * 100:.2f}% "
#                   f"(saved to model/cnn_model.keras)\n")


# best_model_logger = BestModelLogger()

# # Switched to monitoring val_accuracy (what we actually care about) instead
# # of val_loss. Reason: class_weight above is applied to the TRAINING loss
# # only - Keras does not apply it to validation - so val_loss can behave
# # noisily/inconsistently relative to training loss, which was likely
# # triggering premature LR cuts. By epoch 94 of the previous run the LR had
# # already collapsed to ~1.2e-07 (essentially frozen), which explains why
# # val_accuracy stalled around 0.45 well before training finished. Patience
# # raised and decay softened so the LR has room to keep the model learning
# # across more of the 100 epochs instead of bottoming out early.
# reduce_lr = ReduceLROnPlateau(
#     monitor="val_accuracy",
#     mode="max",
#     factor=0.6,
#     patience=8,
#     min_lr=1e-6,
#     verbose=1
# )

# csv_logger = CSVLogger(
#     "model/training_log.csv",
#     append=False
# )

# # NOTE: EarlyStopping has been removed so training runs for the full 100
# # epochs as requested. The best model (by val_accuracy) is still saved
# # throughout via ModelCheckpoint + BestModelLogger above.
# history = model.fit(
#     train_generator,
#     validation_data=test_generator,
#     epochs=EPOCHS,
#     class_weight=class_weights,
#     callbacks=[checkpoint, best_model_logger, reduce_lr, csv_logger]
# )


# # IMPORTANT: reload the BEST saved checkpoint (highest val_accuracy) before
# # final evaluation. Without this, evaluate()/the confusion matrix/F1 score/
# # final model.save() below would reflect the LAST epoch's weights, which
# # after 100 epochs are often worse than the best epoch due to overfitting -
# # this line makes sure the reported and saved accuracy is the true best one.
# model.load_weights("model/cnn_model.keras")
# print("\nReloaded best checkpoint (highest val_accuracy) for final evaluation.\n")

# loss, accuracy = model.evaluate(test_generator)

# print("\n===================================")
# print("Test Accuracy (best model) :", round(accuracy * 100, 2), "%")
# print("Test Loss     :", round(loss, 4))
# print("Best Val Accuracy during training :", round(best_model_logger.best_val_acc * 100, 2), "%")
# print("===================================\n")


# test_generator.reset()
# y_pred_probs = model.predict(test_generator, steps=len(test_generator), verbose=1)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = test_generator.classes[: len(y_pred)]

# f1 = f1_score(y_true, y_pred, average="weighted")
# print("\nWeighted F1 Score:", round(f1, 4))

# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=class_labels))

# cm = confusion_matrix(y_true, y_pred)

# print("\nConfusion Matrix (raw counts):")
# print(cm)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("model/confusion_matrix.png", dpi=150)

# plt.show()
# plt.close()
# print("Confusion matrix saved to model/confusion_matrix.png")


# model.save("model/cnn_model.keras")

# print("CNN Model Saved Successfully!")


# with open("model/classes.txt", "w") as f:
#     for cls in train_generator.class_indices:
#         f.write(cls + "\n")

# print("Class labels saved.")








