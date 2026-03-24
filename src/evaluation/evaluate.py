"""
evaluate.py
-----------
Evaluation step para el SageMaker Pipeline BYOC.

Lee el modelo entrenado y los datos de validación, calcula el RMSE
y escribe el resultado en /opt/ml/processing/output/evaluation/evaluation.json.

El ConditionStep del pipeline compara este RMSE contra rmse_threshold
para decidir si registrar el modelo o terminar en estado fallido.
"""

import json
import os
import pathlib
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# ── Constantes ────────────────────────────────────────────────────────────────
FEATURE_COLS = ["date_block_num", "shop_id", "item_id", "item_category_id",
                "lag_1", "lag_3", "lag_6", "lag_12"]
CLIP         = 20


if __name__ == "__main__":

    # ── Rutas SageMaker ───────────────────────────────────────────────────────
    model_dir      = "/opt/ml/processing/input/model"
    test_dir       = "/opt/ml/processing/input/test"
    output_dir     = "/opt/ml/processing/output/evaluation"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Carga el modelo ────────────────────────────────────────────────────
    print("Loading model...")
    model_path = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/tmp/model")

    # Busca el archivo .pkl dentro del tar
    model_file = None
    for fname in os.listdir("/tmp/model"):
        if fname.endswith(".pkl"):
            model_file = os.path.join("/tmp/model", fname)
            break

    if model_file is None:
        raise FileNotFoundError("No .pkl file found in model.tar.gz")

    model = joblib.load(model_file)
    print("Model loaded: {}".format(model_file))

    # ── 2. Carga los datos de validación ──────────────────────────────────────
    print("Loading validation data...")
    features_path = os.path.join(test_dir, "validation_features.csv")
    labels_path   = os.path.join(test_dir, "validation_labels.csv")

    X_val = pd.read_csv(features_path, header=None, names=FEATURE_COLS)
    y_val = pd.read_csv(labels_path,   header=None, names=["item_cnt_month"])

    y_val_clipped = y_val["item_cnt_month"].clip(0, CLIP)

    print("Validation set: {} records".format(len(X_val)))

    # ── 3. Genera predicciones y calcula RMSE ────────────────────────────────
    print("Generating predictions...")
    predictions = model.predict(X_val)
    predictions_clipped = np.clip(predictions, 0, CLIP)

    rmse = float(np.sqrt(mean_squared_error(y_val_clipped, predictions_clipped)))
    std  = float(np.std(y_val_clipped.values - predictions_clipped))

    print("Validation RMSE: {:.4f}".format(rmse))
    print("Std deviation:   {:.4f}".format(std))

    # ── 4. Escribe evaluation.json ────────────────────────────────────────────
    # Formato que espera el ConditionStep del pipeline
    report = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": std,
            }
        }
    }

    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(report, f)

    print("Evaluation report saved to: {}".format(evaluation_path))
    print("Done.")
