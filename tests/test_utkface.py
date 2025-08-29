import os, pytest, tensorflow as tf
from training.datasets.utkface import make_datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "UTKFace", "labels.csv")),
                    reason="No hay labels.csv (ejecuta scripts/prepare_utkface.py y coloca imágenes)")
def test_dataset_shapes():
    train_ds, val_ds = make_datasets(ROOT, img_size=(160,160), batch=8, val_split=0.1)
    x, y = next(iter(train_ds))
    assert x.shape[1:] == (160,160,3)
    assert "age" in y and "gender" in y
    assert y["age"].shape[0] == 8 and y["gender"].shape[0] == 8
