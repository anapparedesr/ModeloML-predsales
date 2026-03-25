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
FEATURE_COLS = ["lag_1", "lag_3", "lag_6", "lag_12"]
CLIP = 20

if __name__ == "__main__":

    # ── Rutas SageMaker ───────────────────────────────────────────────────────
    model_dir  = "/opt/ml/processing/input/model"
    test_dir   = "/opt/ml/processing/input/test"

    print("Archivos en test_dir:", os.listdir(test_dir))
    output_dir = "/opt/ml/processing/output/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Carga el modelo ────────────────────────────────────────────────────
    print("Loading model...")
    model_path = os.path.join(model_dir, "model.tar.gz")
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/tmp/model")  # cuidado con Python 3.14 y el warning de metadata

    # Busca el archivo .pkl dentro del tar
    model_file = None
    for fname in os.listdir("/tmp/model"):
        if fname.endswith(".pkl"):
            model_file = os.path.join("/tmp/model", fname)
            break
    if model_file is None:
        raise FileNotFoundError("No .pkl file found in model.tar.gz")

    model = joblib.load(model_file)
    print(f"Model loaded: {model_file}")

    # ── 2. Carga los datos de validación ─────────────────────────────────────
    print("Loading validation data...")
    features_path = os.path.join(test_dir, "grid_model.csv")
    ALL_COLS = ["date_block_num", "shop_id", "item_id", "item_category_id",
                "lag_1", "lag_3", "lag_6", "lag_12", "item_cnt_month"]
    df_val = pd.read_csv(features_path, names=ALL_COLS, header=0)
    X_val = df_val[["lag_1", "lag_3", "lag_6", "lag_12"]]
    y_val_clipped = df_val["item_cnt_month"].clip(0, CLIP)

    # ── 3. Genera predicciones y calcula RMSE ────────────────────────────────
    print("Generating predictions...")
    predictions = model.predict(X_val)
    predictions_clipped = np.clip(predictions, 0, CLIP)

    rmse = float(np.sqrt(mean_squared_error(y_val_clipped, predictions_clipped)))
    std = float(np.std(y_val_clipped.values-predictions_clipped))

    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Std deviation:   {std:.4f}")

    # ── 4. Escribe evaluation.json ────────────────────────────────────────────
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

    print(f"Evaluation report saved to: {evaluation_path}")
    print("Done.")