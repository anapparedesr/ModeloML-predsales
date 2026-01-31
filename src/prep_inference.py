# librerías 
import pandas as pd
from pathlib import Path

# Datos para inferencia 
def inference_data(raw_dir: Path, inference_dir: Path, prep_dir: Path):

    # cargar datos 
    test = pd.read_csv(raw_dir / "test.csv")
    grid_model = pd.read_csv(prep_dir / "grid_model.csv")

    lags = grid_model[
        grid_model['date_block_num'].isin([33,31,28,21]) 
    ][['shop_id','item_id', 'date_block_num', 'item_cnt_month']].copy()

    lag_1  = lags[lags['date_block_num'] == 33].rename(columns={'item_cnt_month': 'lag_1'})
    lag_3  = lags[lags['date_block_num'] == 31].rename(columns={'item_cnt_month': 'lag_3'})
    lag_6  = lags[lags['date_block_num'] == 28].rename(columns={'item_cnt_month': 'lag_6'})
    lag_12 = lags[lags['date_block_num'] == 21].rename(columns={'item_cnt_month': 'lag_12'})
    test = test.merge(lag_1[['shop_id','item_id','lag_1']], on=['shop_id','item_id'], how='left')
    test = test.merge(lag_3[['shop_id','item_id','lag_3']], on=['shop_id','item_id'], how='left')
    test = test.merge(lag_6[['shop_id','item_id','lag_6']], on=['shop_id','item_id'], how='left')
    test = test.merge(lag_12[['shop_id','item_id','lag_12']], on=['shop_id','item_id'], how='left')

    for col in ['lag_1', 'lag_3', 'lag_6', 'lag_12']:
        test[col] = test[col].fillna(0)

    # Guardar output ---
    inference_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(inference_dir / "test_with_lags.csv", index=False)

# Punto de entrada
if __name__ == "__main__":
    raw_dir = Path("data/raw")
    inference_dir = Path("data/inference")
    prep_dir = Path("data/prep")

    inference_data(raw_dir, inference_dir, prep_dir)