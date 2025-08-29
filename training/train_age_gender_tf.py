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
    # 1) Augmentation (ligero, seguro)
    augmenter = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1, 0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.2),
            layers.Lambda(lambda x: tf.clip_by_value(
                tf.image.adjust_gamma(x, gamma=tf.random.uniform([], 0.8, 1.2)), 0.0, 1.0
            )),
        ],
        name="augment",
    )

    inputs = keras.Input(shape=IMG)
    x = augmenter(inputs)

    base = keras.applications.MobileNetV2(input_shape=IMG, include_top=False, weights="imagenet")
    base.trainable = False

    x = base(x, training=False)               # usar BN en modo inferencia mientras está congelado
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    age    = layers.Dense(1, name="age")(x)                         # regresión
    gender = layers.Dense(1, activation="sigmoid", name="gender")(x)  # binario

    model = keras.Model(inputs, [age, gender])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"age": "mae", "gender": "binary_crossentropy"},
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )
    return model, base

if __name__ == "__main__":
    train_ds, val_ds = make_datasets(ROOT, img_size=(160,160), batch=32, val_split=0.1)
    with mlflow.start_run(run_name="mbv2_freeze") as run:
        params = {"img": IMG, "epochs": 8, "batch": 32, "lr": 1e-3}
        mlflow.log_params(params)
        model, base = build_model()

        cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=params["epochs"], callbacks=cb)

        # Fine-tuning parcial: descongela las últimas ~20 capas
        for L in base.layers[-20:]:
            L.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),  # LR más baja
            loss={"age": "mae", "gender": "binary_crossentropy"},
            metrics={"age": ["mae"], "gender": ["accuracy"]}
        )

        hist2 = model.fit(train_ds, validation_data=val_ds, epochs=8,
                          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

        # log métricas claves
        mlflow.log_metric("val_age_mae", float(hist.history["val_age_mae"][-1]))
        mlflow.log_metric("val_gender_acc", float(hist.history["val_gender_accuracy"][-1]))

        # guardar SavedModel
        os.makedirs("artifacts", exist_ok=True)
        saved_dir = "artifacts/age_gender_savedmodel"
        model.save(saved_dir)
        mlflow.log_artifacts(saved_dir, artifact_path="keras_model")

        # === Curvas de entrenamiento como artefacto ===
        import matplotlib.pyplot as plt
        import numpy as np
        import os, mlflow

        os.makedirs("artifacts", exist_ok=True)
        plt.figure()
        plt.plot(hist.history["gender_accuracy"], label="gender_acc")
        plt.plot(hist.history["val_gender_accuracy"], label="val_gender_acc")
        plt.title("Accuracy (genero) - entrenamiento")
        plt.xlabel("Época");
        plt.legend();
        plt.tight_layout()
        plt.savefig("artifacts/curve_gender_acc.png");
        plt.close()
        mlflow.log_artifact("artifacts/curve_gender_acc.png", artifact_path="plots")

        # === Matriz de confusión de 'género' en validación ===
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        y_true = []
        y_pred = []
        for xb, yb in val_ds:
            age_hat, gen_hat = model.predict(xb, verbose=0)
            y_true.append(yb["gender"].numpy().astype(int))
            y_pred.append((gen_hat >= 0.5).astype(int))
        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0=male, 1=female (UTKFace convención)
        disp = ConfusionMatrixDisplay(cm, display_labels=["male(0)", "female(1)"])
        disp.plot(values_format="d")
        plt.title("Matriz de confusión (validación)")
        plt.tight_layout()
        plt.savefig("artifacts/cm_gender.png");
        plt.close()
        mlflow.log_artifact("artifacts/cm_gender.png", artifact_path="plots")

        # exportar ONNX
        spec = (tf.TensorSpec((None, *IMG), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open("artifacts/age_gender.onnx","wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact("artifacts/age_gender.onnx", artifact_path="onnx")


