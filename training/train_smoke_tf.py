import os, mlflow, mlflow.keras, tf2onnx, tensorflow as tf
from training.smoke_model import train_smoke, build_model

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("SMOKE_TF")

if __name__ == "__main__":
    with mlflow.start_run(run_name="smoke_dense") as run:
        params = {"epochs": 8, "batch": 64, "lr": 1e-3}
        mlflow.log_params(params)

        model, hist = train_smoke(epochs=params["epochs"], batch=params["batch"])
        mlflow.log_metric("val_acc", float(hist["val_accuracy"][-1]))
        mlflow.log_metric("val_loss", float(hist["val_loss"][-1]))

        # Guardar SavedModel
        saved_dir = "artifacts/smoke_savedmodel"
        os.makedirs("artifacts", exist_ok=True)
        model.save(saved_dir)
        mlflow.log_artifacts(saved_dir, artifact_path="keras_model")

        # Exportar ONNX
        spec = (tf.TensorSpec((None, 20), tf.float32, name="input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open("artifacts/smoke.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact("artifacts/smoke.onnx", artifact_path="onnx")

        print("RUN_ID:", run.info.run_id)
