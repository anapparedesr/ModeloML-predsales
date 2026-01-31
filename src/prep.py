# librerías 
import pandas as pd
from pathlib import Path

 # Datos para modelaje
def prepare_data(raw_dir: Path, prep_dir: Path):

    # cargar datos 
    sales_train = pd.read_csv(raw_dir / "sales_train.csv")
    items = pd.read_csv(raw_dir / "items.csv")

    #Limpieza básica 
    sales_train_cleaned = sales_train[sales_train['item_price'] > 0]
    sales_train_cleaned = sales_train_cleaned[sales_train_cleaned['item_cnt_day'] >= 0]

    # agregar datos a nivel mensual
    monthly = (
        sales_train_cleaned
        .groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
        .agg({'item_cnt_day': 'sum'})
        .rename(columns={'item_cnt_day': 'item_cnt_month'})
    )

    # clipping
    p99 = monthly['item_cnt_month'].quantile(0.99)
    monthly['item_cnt_month'] = monthly['item_cnt_month'].clip(0, p99)

    # merge con items para obtener item_category_id
    monthly = monthly.merge(
        items[['item_id', 'item_category_id']],
        on='item_id',
        how='left'
    )

    # lags 
    monthly_fe = monthly.sort_values(
        ['shop_id', 'item_id', 'date_block_num']
    )
    meses_total = sorted(monthly_fe['date_block_num'].unique())
    comb_activas = monthly_fe[['shop_id', 'item_id']].drop_duplicates()
    
    grid_list = []
    for month in meses_total:
        month_grid = comb_activas.copy()
        month_grid['date_block_num'] = month
        grid_list.append(month_grid)

    grid = pd.concat(grid_list, ignore_index=True)
    del grid_list
    grid = grid[['date_block_num', 'shop_id', 'item_id']]

    grid = grid.merge(
        monthly_fe[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']],
        on=['date_block_num', 'shop_id', 'item_id'], #estoy uniendo por estas columnas
        how='left' #mantenemos todo el grid completo
    )
    grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)

    grid = grid.sort_values(['shop_id', 'item_id', 'date_block_num'])

    for lag in [1, 3, 6, 12]:
        grid[f'lag_{lag}'] = (
            grid
            .groupby(['shop_id', 'item_id'])['item_cnt_month']
            .shift(lag)
        )
    
    grid_model = grid.copy()
    features = ['lag_1', 'lag_3', 'lag_6', 'lag_12']
    grid_model = grid_model.dropna(subset=features)

    # Guardar output ---
    prep_dir.mkdir(parents=True, exist_ok=True)
    grid_model.to_csv(prep_dir / "grid_model.csv", index=False)

# Punto de entrada
if __name__ == "__main__":
    raw_dir = Path("data/raw")
    prep_dir = Path("data/prep")

    prepare_data(raw_dir, prep_dir)

