# librerías 
import pandas as pd
from pathlib import Path
import joblib


def run_inference(inference_dir: Path, artifacts_dir: Path, predictions_dir: Path):

    # cargar modelo
    model = joblib.load(artifacts_dir / "random_forest_lags.pkl")

    # cargar datos
    data = pd.read_csv(inference_dir / "test_with_lags.csv")

    # features
    lag_cols = ['lag_1', 'lag_3', 'lag_6', 'lag_12']
    X = data[lag_cols]

    # predicciones
    data['prediction'] = model.predict(X)

    # guardar
    predictions_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(predictions_dir / "predictions.csv", index=False)

if __name__ == "__main__":
    print("Running inference.py")
    BASE_DIR = Path(__file__).resolve().parent.parent

    run_inference(
        inference_dir=BASE_DIR / "data" / "inference",
        artifacts_dir=BASE_DIR / "artifacts",
        predictions_dir=BASE_DIR / "data" / "predictions"
    )
