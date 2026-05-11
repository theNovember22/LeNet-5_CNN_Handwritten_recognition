import argparse

from model_utils import MODEL_PATH, build_lenet5, load_mnist_data


def train(epochs=10, batch_size=64):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

    model = build_lenet5(input_shape=x_train.shape[1:])
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the LeNet-5 MNIST model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
