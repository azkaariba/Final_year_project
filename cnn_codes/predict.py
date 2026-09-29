# Predict a single image

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf


def get_image_paths(image_path):
    image_path = os.path.abspath(image_path)

    if os.path.isfile(image_path):
        return [image_path]

    if os.path.isdir(image_path):
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_files = []

        for filename in os.listdir(image_path):
            full_path = os.path.join(image_path, filename)
            if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in allowed_exts:
                image_files.append(full_path)

        return sorted(image_files)

    return []


def load_model_and_classes():
    model = tf.keras.models.load_model("model/cnn_model.keras")

    with open("model/classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names


def predict_image(model, class_names, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (64, 64))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[0][predicted_index] * 100)

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten characters from images")
    parser.add_argument("image_path", nargs="?", default="image_path", help="Path to an image or folder of images")
    args = parser.parse_args()

    image_paths = get_image_paths(args.image_path)

    if not image_paths:
        print(f"No image files found in: {args.image_path}")
        return

    model, class_names = load_model_and_classes()

    for image_path in image_paths:
        try:
            predicted_class, confidence = predict_image(model, class_names, image_path)
        except ValueError as exc:
            print(exc)
            continue

        print("\n==============================")
        print("Image           :", os.path.basename(image_path))
        print("Predicted Class :", predicted_class)
        print("Confidence      : {:.2f}%".format(confidence))
        print("==============================")


if __name__ == "__main__":
    main()
