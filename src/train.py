# librerías 
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

def train_data(prep_dir: Path, artifacts_dir: Path):

    # cargar datos 
    grid_model = pd.read_csv(prep_dir / "grid_model.csv")
    
    # split 
    train_grid = grid_model[grid_model['date_block_num'] <= 32].copy()
    val_grid   = grid_model[grid_model['date_block_num'] == 33].copy()

    # definir X e y
    lag_cols = ['lag_1', 'lag_3', 'lag_6', 'lag_12']
    x_train = train_grid[lag_cols]
    y_train = train_grid['item_cnt_month'].clip(0,17)

    x_val = val_grid[lag_cols]
    y_val   = val_grid['item_cnt_month'].clip(0,17)

    # modelo 
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=10,
        n_jobs=-1
    )
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_val)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    print(f"Validation RMSE (RF): {rmse_rf:.4f}")


        # Guardar output ---
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "random_forest_lags.pkl"
    print("Model will be saved to:", model_path.resolve())
    joblib.dump(rf, model_path)

    return rmse_rf

# Punto de entrada
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    prep_dir = BASE_DIR / "data" / "prep"
    artifacts_dir = BASE_DIR / "artifacts"

    train_data(prep_dir, artifacts_dir)