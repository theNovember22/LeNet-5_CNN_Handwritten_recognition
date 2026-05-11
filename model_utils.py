from pathlib import Path
from io import BytesIO
import base64

import numpy as np
from PIL import Image, ImageChops, ImageOps


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lenet5_mnist.keras"


def build_lenet5(input_shape=(32, 32, 1)):
    from tensorflow.keras import layers, losses, models

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(6, 5, activation="tanh"),
            layers.AveragePooling2D(2),
            layers.Activation("sigmoid"),
            layers.Conv2D(16, 5, activation="tanh"),
            layers.AveragePooling2D(2),
            layers.Activation("sigmoid"),
            layers.Conv2D(120, 5, activation="tanh"),
            layers.Flatten(),
            layers.Dense(84, activation="tanh"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def load_mnist_data():
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.cast(tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]), tf.float32) / 255.0
    x_test = tf.cast(tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]), tf.float32) / 255.0

    x_train = tf.expand_dims(x_train, axis=3)
    x_test = tf.expand_dims(x_test, axis=3)

    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    x_train = x_train[:-2000, :, :, :]
    y_train = y_train[:-2000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_trained_model(model_path=MODEL_PATH):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def _crop_digit(image, threshold=30):
    mask = image.point(lambda pixel: 255 if pixel > threshold else 0)
    background = Image.new("L", mask.size, 0)
    bbox = ImageChops.difference(mask, background).getbbox()
    return image.crop(bbox) if bbox else image


def _has_light_background(image):
    array = np.asarray(image, dtype=np.float32)
    border = np.concatenate(
        [
            array[0, :],
            array[-1, :],
            array[:, 0],
            array[:, -1],
        ]
    )
    return float(border.mean()) > 127.0


def _center_by_mass(image):
    array = np.asarray(image, dtype=np.float32)
    total = array.sum()
    if total <= 0:
        return image

    y_indices, x_indices = np.indices(array.shape)
    center_x = float((x_indices * array).sum() / total)
    center_y = float((y_indices * array).sum() / total)
    shift_x = int(round((image.width - 1) / 2.0 - center_x))
    shift_y = int(round((image.height - 1) / 2.0 - center_y))

    centered = Image.new("L", image.size, 0)
    centered.paste(image, (shift_x, shift_y))
    return centered


def _image_to_data_url(image):
    buffer = BytesIO()
    image.resize((140, 140), Image.Resampling.NEAREST).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preprocess_digit_image(file_obj, return_preview=False):
    image = Image.open(file_obj).convert("L")
    image = ImageOps.autocontrast(image)

    if _has_light_background(image):
        image = ImageOps.invert(image)

    is_mnist_sized = image.size in ((28, 28), (32, 32))
    if not is_mnist_sized:
        image = _crop_digit(image)
        image.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        left = (28 - image.width) // 2
        top = (28 - image.height) // 2
        canvas.paste(image, (left, top))
        canvas = _center_by_mass(canvas)
    elif image.size == (32, 32):
        canvas = image.crop((2, 2, 30, 30))
    else:
        canvas = image

    padded = ImageOps.expand(canvas, border=2, fill=0)
    array = np.asarray(padded, dtype=np.float32) / 255.0
    model_input = array.reshape(1, 32, 32, 1)

    if return_preview:
        return model_input, _image_to_data_url(canvas)
    return model_input


def predict_digit(model, file_obj):
    image_array, processed_image = preprocess_digit_image(file_obj, return_preview=True)
    probabilities = model.predict(image_array, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return digit, confidence, probabilities.tolist(), processed_image
