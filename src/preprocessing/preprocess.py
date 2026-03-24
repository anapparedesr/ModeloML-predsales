from __future__ import print_function, unicode_literals
import argparse
import os

import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────
COLUMN_UNITS       = "item_cnt_day"
COLUMN_UNITS_MONTH = "item_cnt_month"
COLUMN_PRICE       = "item_price"
CLIP               = 20
LAGS               = [1, 3, 6, 12]

MONTH_TRAIN_END    = 31   # meses 0–31 → train
MONTH_VAL          = 32   # mes 32     → validation
MONTH_TEST_BLOCK   = 33   # mes 33     → usado para generar features del test de Kaggle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()

    # ── Rutas SageMaker ───────────────────────────────────────────────────────
    input_dir      = "/opt/ml/processing/input"
    train_dir      = "/opt/ml/processing/output/train"
    validation_dir = "/opt/ml/processing/output/validation"
    test_dir       = "/opt/ml/processing/output/test"

    os.makedirs(train_dir,      exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir,       exist_ok=True)

    # ── 1. Carga ──────────────────────────────────────────────────────────────
    print("Loading sales data...")
    sales = pd.read_csv(os.path.join(input_dir, "sales_train.csv"))
    print("Sales loaded: {} records".format(len(sales)))

    print("Loading items data...")
    items = pd.read_csv(os.path.join(input_dir, "items.csv"))
    print("Items loaded: {} records".format(len(items)))

    print("Loading Kaggle test data...")
    kaggle_test = pd.read_csv(os.path.join(input_dir, "test.csv"))
    print("Kaggle test loaded: {} records".format(len(kaggle_test)))

    # ── 2. Limpieza ───────────────────────────────────────────────────────────
    print("Cleaning sales data...")
    sales = sales[(sales[COLUMN_PRICE] > 0) & (sales[COLUMN_UNITS] >= 0)].copy()
    print("After cleaning: {} records".format(len(sales)))

    # ── 3. Agregación mensual ─────────────────────────────────────────────────
    print("Aggregating to monthly level...")
    monthly = (
        sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg({COLUMN_UNITS: "sum"})
        .rename(columns={COLUMN_UNITS: COLUMN_UNITS_MONTH})
    )
    monthly[COLUMN_UNITS_MONTH] = monthly[COLUMN_UNITS_MONTH].clip(0, CLIP)
    print("Monthly aggregation: {} records".format(len(monthly)))

    # ── 4. Agrega item_category_id ────────────────────────────────────────────
    print("Adding item category information...")
    monthly = monthly.merge(
        items[["item_id", "item_category_id"]], on="item_id", how="left"
    )

    # ── 5. Grid completo (meses 0–33) ─────────────────────────────────────────
    print("Building complete shop-item-month grid...")
    monthly_sorted = monthly.sort_values(["shop_id", "item_id", "date_block_num"])
    total_months   = sorted(monthly_sorted["date_block_num"].unique())
    comb_active    = monthly_sorted[["shop_id", "item_id"]].drop_duplicates()

    grid = pd.concat(
        [comb_active.assign(date_block_num=m) for m in total_months],
        ignore_index=True
    )[["date_block_num", "shop_id", "item_id"]]

    grid = grid.merge(
        monthly_sorted[["date_block_num", "shop_id", "item_id",
                         COLUMN_UNITS_MONTH, "item_category_id"]],
        on=["date_block_num", "shop_id", "item_id"],
        how="left",
    )
    grid[COLUMN_UNITS_MONTH] = grid[COLUMN_UNITS_MONTH].fillna(0)
    grid["item_category_id"] = grid["item_category_id"].fillna(
        grid["item_category_id"].mode()[0]
    )
    grid = grid.sort_values(["shop_id", "item_id", "date_block_num"])
    print("Grid built: {} records".format(len(grid)))

    # ── 6. Lag features ───────────────────────────────────────────────────────
    print("Generating lag features: {}...".format(LAGS))
    for lag in LAGS:
        grid["lag_{}".format(lag)] = (
            grid.groupby(["shop_id", "item_id"])[COLUMN_UNITS_MONTH].shift(lag)
        )

    # ── 7. Filtra casos completos ─────────────────────────────────────────────
    lag_cols = ["lag_{}".format(l) for l in LAGS]
    grid = grid.dropna(subset=lag_cols)
    print("After filtering incomplete cases: {} records".format(len(grid)))

    # ── 8. Splits temporales: train / validation ──────────────────────────────
    feature_cols = ["date_block_num", "shop_id", "item_id", "item_category_id"] + lag_cols

    X_train = grid[grid["date_block_num"] <= MONTH_TRAIN_END][feature_cols]
    y_train = grid[grid["date_block_num"] <= MONTH_TRAIN_END][COLUMN_UNITS_MONTH]

    X_val   = grid[grid["date_block_num"] == MONTH_VAL][feature_cols]
    y_val   = grid[grid["date_block_num"] == MONTH_VAL][COLUMN_UNITS_MONTH]

    print("Train shape:      {}".format(X_train.shape))
    print("Validation shape: {}".format(X_val.shape))

    # ── 9. Prepara test de Kaggle (mes 34) ────────────────────────────────────
    # El test.csv de Kaggle tiene shop_id e item_id para el mes 34.
    # Generamos sus lag features usando el grid de meses anteriores (hasta mes 33).
    print("Preparing Kaggle test features (month 34)...")

    # Tomamos el último estado del grid para calcular lags del mes 34
    last_grid = grid[grid["date_block_num"] == MONTH_TEST_BLOCK][
        ["shop_id", "item_id", "item_category_id"] + lag_cols
    ].copy()

    # Shift manual: lag_1 del mes 34 = item_cnt_month del mes 33
    # Para eso necesitamos recalcular con el grid extendido al mes 34
    # Construimos un mini-grid del mes 34 a partir del test.csv de Kaggle
    kaggle_test["date_block_num"] = 34
    kaggle_test = kaggle_test.merge(
        items[["item_id", "item_category_id"]], on="item_id", how="left"
    )
    kaggle_test["item_category_id"] = kaggle_test["item_category_id"].fillna(
        kaggle_test["item_category_id"].mode()[0]
    )

    # Extendemos el grid con el mes 34 para calcular lags
    grid_extended = pd.concat(
        [grid[["date_block_num", "shop_id", "item_id", "item_category_id", COLUMN_UNITS_MONTH]],
         kaggle_test[["date_block_num", "shop_id", "item_id", "item_category_id"]].assign(
             **{COLUMN_UNITS_MONTH: 0}
         )],
        ignore_index=True
    ).sort_values(["shop_id", "item_id", "date_block_num"])

    for lag in LAGS:
        grid_extended["lag_{}".format(lag)] = (
            grid_extended.groupby(["shop_id", "item_id"])[COLUMN_UNITS_MONTH].shift(lag)
        )

    # Filtramos solo el mes 34 y las columnas necesarias
    test_month34 = grid_extended[grid_extended["date_block_num"] == 34].copy()
    test_month34 = test_month34.dropna(subset=lag_cols)

    # Pegamos el ID de Kaggle para poder hacer el submit después
    X_test = test_month34[feature_cols].copy()

    print("Test (Kaggle) shape: {}".format(X_test.shape))

    # ── 10. Guarda los outputs ─────────────────────────────────────────────────
    print("Saving outputs...")

    X_train.to_csv(os.path.join(train_dir,      "train_features.csv"),      header=False, index=False)
    y_train.to_csv(os.path.join(train_dir,      "train_labels.csv"),        header=False, index=False)

    X_val.to_csv(  os.path.join(validation_dir, "validation_features.csv"), header=False, index=False)
    y_val.to_csv(  os.path.join(validation_dir, "validation_labels.csv"),   header=False, index=False)

    X_test.to_csv( os.path.join(test_dir,       "test_features.csv"),       header=False, index=False)

    print("Preprocessing complete.")
    print("  train:      {} records".format(len(X_train)))
    print("  validation: {} records".format(len(X_val)))
    print("  test:       {} records".format(len(X_test)))
