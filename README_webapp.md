# MNIST Digit Recognition Web App

This app uses the LeNet-5 TensorFlow CNN to predict a handwritten digit from an uploaded image.

## 1. Train and save the model

```powershell
cd "E:\mtech1st year 2025 - 2026\SEM2\ANN\CNN_as"
python train_model.py --epochs 10
```

This creates:

```text
CNN_as/models/lenet5_mnist.keras
```

## 2. Start the web app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

- The uploaded image should contain one handwritten digit only.
- MNIST-style images work best: dark background with a light digit, or white paper with a dark digit.
- The app shows a **Model Input** preview after prediction. If that preview does not look like a centered white digit on a black background, the model will probably predict incorrectly.
- Avoid images with multiple digits, shadows, ruled-paper lines, colored backgrounds, or very small digits.
- If the app says the model file is missing, run the training command first.
