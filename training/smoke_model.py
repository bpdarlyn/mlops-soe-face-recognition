import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim=20, width=32):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(width, activation="relu")(inp)
    x = layers.Dense(width, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def make_data(n=4000, d=20, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    y = (X[:, :3].sum(axis=1) + 0.5 * X[:, 3] > 0).astype("float32")
    return X, y

def train_smoke(epochs=5, batch=64):
    X, y = make_data()
    Xtr, ytr = X[:3200], y[:3200]
    Xva, yva = X[3200:], y[3200:]
    model = build_model(input_dim=X.shape[1])
    cb = [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=epochs, batch_size=batch, verbose=0, callbacks=cb)
    return model, hist.history
