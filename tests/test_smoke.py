import numpy as np
from training.smoke_model import build_model, make_data, train_smoke

def test_build_model():
    m = build_model(input_dim=20, width=16)
    assert m.input_shape == (None, 20)
    assert m.output_shape == (None, 1)

def test_train_one_epoch():
    m, hist = train_smoke(epochs=1, batch=64)
    # Debe existir val_accuracy y estar en [0,1]
    assert "val_accuracy" in hist
    v = float(hist["val_accuracy"][-1])
    assert 0.0 <= v <= 1.0

def test_predict_shape():
    m, _ = train_smoke(epochs=1, batch=64)
    X, _ = make_data(n=10, d=20, seed=123)
    yhat = m.predict(X, verbose=0)
    assert yhat.shape == (10, 1)
    # Probabilidades en (0,1)
    assert np.all((yhat > 0) & (yhat < 1))
