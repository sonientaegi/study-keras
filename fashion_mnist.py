import os.path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

_model = 'fashion_mnist_4a.h5'


def study(X_train, y_train, X_valid, y_valid):
    import time
    log_path = os.path.join(os.curdir, "logs", time.strftime("run_%Y%m%d%H%M%S"))
    callback_tensorboard = keras.callbacks.TensorBoard(log_path)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(200, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[callback_tensorboard])
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    model.save(_model)


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.0

    if not os.path.exists(_model):
        study(X_train, y_train, X_valid, y_valid)
    model = keras.models.load_model(_model)
    model.summary()
    model.evaluate(X_test, y_test)
