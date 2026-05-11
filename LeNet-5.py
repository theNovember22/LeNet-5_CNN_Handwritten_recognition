import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import MODEL_PATH, build_lenet5, load_mnist_data


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

    model = build_lenet5(input_shape=x_train.shape[1:])
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_val, y_val),
    )

    model.evaluate(x_test, y_test)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].title.set_text("Training Loss vs Validation Loss")
    axs[0].legend(["Train", "Val"])

    axs[1].plot(history.history["accuracy"])
    axs[1].plot(history.history["val_accuracy"])
    axs[1].title.set_text("Training Accuracy vs Validation Accuracy")
    axs[1].legend(["Train", "Val"])

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.arange(10),
        yticklabels=np.arange(10),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    class_report = classification_report(y_test, y_pred_classes)
    print("Classification Report:\n", class_report)

    plt.show()


if __name__ == "__main__":
    main()
