import os, mlflow, mlflow.keras, tf2onnx, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from training.datasets.utkface import make_datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("AgeGender-UTKFace")

IMG = (160,160,3)

def build_model():
    base = keras.applications.MobileNetV2(input_shape=IMG, include_top=False, weights="imagenet")
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    age    = layers.Dense(1, name="age")(x)                         # regresión
    gender = layers.Dense(1, activation="sigmoid", name="gender")(x)  # binario
    m = keras.Model(base.input, [age, gender])
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"age":"mae", "gender":"binary_crossentropy"},
        metrics={"age":["mae"], "gender":["accuracy"]}
    )
    return m, base

if __name__ == "__main__":
    train_ds, val_ds = make_datasets(ROOT, img_size=(160,160), batch=32, val_split=0.1)
    with mlflow.start_run(run_name="mbv2_freeze") as run:
        params = {"img": IMG, "epochs": 8, "batch": 32, "lr": 1e-3}
        mlflow.log_params(params)
        model, base = build_model()

        cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=params["epochs"], callbacks=cb)

        # log métricas claves
        mlflow.log_metric("val_age_mae", float(hist.history["val_age_mae"][-1]))
        mlflow.log_metric("val_gender_acc", float(hist.history["val_gender_accuracy"][-1]))

        # guardar SavedModel
        os.makedirs("artifacts", exist_ok=True)
        saved_dir = "artifacts/age_gender_savedmodel"
        model.save(saved_dir)
        mlflow.log_artifacts(saved_dir, artifact_path="keras_model")

        # exportar ONNX
        spec = (tf.TensorSpec((None, *IMG), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open("artifacts/age_gender.onnx","wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact("artifacts/age_gender.onnx", artifact_path="onnx")
