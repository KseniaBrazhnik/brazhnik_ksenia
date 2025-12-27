"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMRegressor


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    submission = pd.DataFrame({
        'row_id': range(len(predictions['price_p05'])),
        'price_p05': predictions['price_p05'],
        'price_p95': predictions['price_p95']
    })

    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def add_features(df, fit_transform=True, encoders=None,
                 category_width_map=None, global_price_range_median=None):
    df = df.copy()
    df['product_id'] = df['product_id'].astype(str)

    if fit_transform:
        df['price_mid'] = (df['price_p05'] + df['price_p95']) / 2
        df['price_range'] = df['price_p95'] - df['price_p05']
        df['rel_price_range'] = df['price_range'] / (df['price_mid'] + 1e-6)
        df['category_median_width'] = df.groupby('third_category_id')['price_range'].transform('median')

        df['price_p05_lag1'] = df.groupby('product_id')['price_p05'].shift(1)
        df['price_p95_lag1'] = df.groupby('product_id')['price_p95'].shift(1)
        df['price_p05_7d'] = (
            df.groupby('product_id')['price_p05'].shift(1)
            .groupby(df['product_id']).rolling(7, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df['price_p95_7d'] = (
            df.groupby('product_id')['price_p95'].shift(1)
            .groupby(df['product_id']).rolling(7, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Признак редкости
        product_counts = df['product_id'].value_counts()
        rare_products = set(product_counts[product_counts <= 5].index)
        df['is_rare_product'] = df['product_id'].isin(rare_products).astype(int)

        anomaly_features = ['price_p05', 'price_p95', 'price_range', 'n_stores', 'avg_temperature']
        iso = IsolationForest(contamination=0.01, random_state=322)
        df['is_anomaly'] = (iso.fit_predict(df[anomaly_features].fillna(0)) == -1).astype(int)

        df = df.dropna(subset=['price_p05_lag1', 'price_p95_lag1']).reset_index(drop=True)

        le_dict = {}
        cat_cols = ['management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        return df, le_dict

    else:
        df['price_mid'] = (df['price_p05_last'] + df['price_p95_last']) / 2
        df['price_range'] = global_price_range_median
        df['rel_price_range'] = global_price_range_median / (df['price_mid'] + 1e-6)
        df['category_median_width'] = df['third_category_id'].map(category_width_map).fillna(global_price_range_median)

        df['price_p05_lag1'] = df['price_p05_last']
        df['price_p95_lag1'] = df['price_p95_last']
        df['price_p05_7d'] = df['price_p05_last']
        df['price_p95_7d'] = df['price_p95_last']

        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['is_rare_product'] = 0
        df['is_anomaly'] = 0

        cat_cols = ['management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str)
            unknown = ~df[col].isin(le.classes_)
            df.loc[unknown, col] = le.classes_[0]
            df[col] = le.transform(df[col])
        return df


def compute_iou_batch(y_true_low, y_true_high, y_pred_low, y_pred_high, eps=1e-6):
    y_true_low = np.array(y_true_low, dtype=np.float64)
    y_true_high = np.array(y_true_high, dtype=np.float64)
    y_pred_low = np.array(y_pred_low, dtype=np.float64)
    y_pred_high = np.array(y_pred_high, dtype=np.float64)

    y_true_low_adj = y_true_low - eps
    y_true_high_adj = y_true_high + eps
    y_pred_low_adj = y_pred_low - eps
    y_pred_high_adj = y_pred_high + eps

    intersection = np.maximum(0.0, np.minimum(y_true_high_adj, y_pred_high_adj) - np.maximum(y_true_low_adj, y_pred_low_adj))
    union = (y_true_high_adj - y_true_low_adj) + (y_pred_high_adj - y_pred_low_adj) - intersection
    return np.mean(intersection / np.maximum(union, eps))


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    np.random.seed(322)
    random.seed(322)

    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)

    # =============================================================================
    # ЗАГРУЗКА
    # =============================================================================
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    df_train['product_id'] = df_train['product_id'].astype(str)
    df_test['product_id'] = df_test['product_id'].astype(str)
    df_train['dt'] = pd.to_datetime(df_train['dt'])
    df_test['dt'] = pd.to_datetime(df_test['dt'])

    for df in [df_train, df_test]:
        df['dow'] = df['dt'].dt.dayofweek
        df['day_of_month'] = df['dt'].dt.day
        df['month'] = df['dt'].dt.month

    df_train = df_train.sort_values(['product_id', 'dt']).reset_index(drop=True)

    cat_cols = ['management_group_id', 'first_category_id', 'second_category_id', 'third_category_id']

    # =============================================================================
    # РАЗБИЕНИЕ НА TRAIN (70%) / VAL (30%)
    # =============================================================================
    val_start = '2024-05-09'

    train_mask = df_train['dt'] < val_start
    val_mask = df_train['dt'] >= val_start

    df_train_split = df_train[train_mask].copy().reset_index(drop=True)
    df_val_split = df_train[val_mask].copy().reset_index(drop=True)

    print(f"Train: {df_train_split['dt'].min()} → {df_train_split['dt'].max()} | {len(df_train_split)}")
    print(f"Val:   {df_val_split['dt'].min()} → {df_val_split['dt'].max()} | {len(df_val_split)}")

    # =============================================================================
    # ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ TRAIN_SPLIT
    # =============================================================================
    df_train_feat, le_dict = add_features(df_train_split, fit_transform=True)

    category_width_map = df_train_feat.groupby('third_category_id')['price_range'].median().to_dict()
    global_price_range_median = df_train_feat['price_range'].median()

    feature_cols = [
        'price_p05_lag1', 'price_p95_lag1',
        'price_p05_7d', 'price_p95_7d',
        'category_median_width', 'rel_price_range',
        'n_stores', 'activity_flag', 'holiday_flag',
        'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
        'is_anomaly', 'is_rare_product'
    ] + cat_cols

    # =============================================================================
    # ОБУЧЕНИЕ НА TRAIN_SPLIT
    # =============================================================================
    model_low = LGBMRegressor(objective='quantile', alpha=0.05, random_state=322,
                              n_estimators=1000, learning_rate=0.03, num_leaves=63,
                              min_data_in_leaf=20, feature_fraction=0.9, verbose=-1)
    model_high = LGBMRegressor(objective='quantile', alpha=0.95, random_state=322,
                               n_estimators=1000, learning_rate=0.03, num_leaves=63,
                               min_data_in_leaf=20, feature_fraction=0.9, verbose=-1)

    model_low.fit(df_train_feat[feature_cols], df_train_feat['price_p05'])
    model_high.fit(df_train_feat[feature_cols], df_train_feat['price_p95'])

    # =============================================================================
    # ВАЛИДАЦИЯ — ROBUST КАЛИБРОВКА
    # =============================================================================
    last_prices_train = df_train_feat.groupby('product_id').tail(1)[
        ['product_id', 'price_p05', 'price_p95']
    ].rename(columns={'price_p05': 'price_p05_last', 'price_p95': 'price_p95_last'})

    df_val_with_lags = df_val_split.merge(last_prices_train, on='product_id', how='left')
    med_p05 = df_train_feat['price_p05'].median()
    med_p95 = df_train_feat['price_p95'].median()
    df_val_with_lags['price_p05_last'] = df_val_with_lags['price_p05_last'].fillna(med_p05)
    df_val_with_lags['price_p95_last'] = df_val_with_lags['price_p95_last'].fillna(med_p95)

    df_val_feat = add_features(
        df_val_with_lags,
        fit_transform=False,
        encoders=le_dict,
        category_width_map=category_width_map,
        global_price_range_median=global_price_range_median
    )

    p05_raw_val = model_low.predict(df_val_feat[feature_cols])
    p95_raw_val = model_high.predict(df_val_feat[feature_cols])

    p05_raw_val = np.clip(np.minimum(p05_raw_val, p95_raw_val), 0.01, None)
    p95_raw_val = np.clip(np.maximum(p95_raw_val, p05_raw_val + 0.01), 0.02, None)

    mid_val = (p05_raw_val + p95_raw_val) / 2
    width_val = p95_raw_val - p05_raw_val

    # === СБОР ВСЕХ КАНДИДАТОВ ===
    candidates = []
    mid_shifts = np.arange(-0.05, 0.051, 0.01)
    width_mults = np.arange(0.6, 1.41, 0.02)

    for mid_shift in mid_shifts:
        for width_mult in width_mults:
            p05_cand = mid_val * (1 + mid_shift) - (width_val * width_mult) / 2
            p95_cand = mid_val * (1 + mid_shift) + (width_val * width_mult) / 2
            p05_cand = np.clip(p05_cand, 0.01, None)
            p95_cand = np.clip(p95_cand, p05_cand + 0.01, None)

            iou = compute_iou_batch(df_val_split['price_p05'], df_val_split['price_p95'],
                                    p05_cand, p95_cand)
            candidates.append((iou, mid_shift, width_mult))

    # Сортируем по IoU
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_iou, best_mid_shift, best_width_mult = candidates[0]

    # ROBUST SELECTION
    threshold = best_iou - 0.001
    robust_candidates = [c for c in candidates if c[0] >= threshold]

    best_dist = float('inf')
    robust_mid_shift = 0.0
    robust_width_mult = 1.0
    robust_iou = best_iou

    for iou, ms, wm in robust_candidates:
        dist = abs(ms - 0.0) + abs(wm - 1.0)
        if dist < best_dist:
            best_dist = dist
            robust_mid_shift = ms
            robust_width_mult = wm
            robust_iou = iou

    # =============================================================================
    # ФИНАЛЬНАЯ МОДЕЛЬ НА ВСЕХ ДАННЫХ
    # =============================================================================
    df_full_train, le_dict_full = add_features(df_train, fit_transform=True)

    model_low_full = LGBMRegressor(objective='quantile', alpha=0.05, random_state=322,
                                   n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                   min_data_in_leaf=20, feature_fraction=0.9, verbose=-1)
    model_high_full = LGBMRegressor(objective='quantile', alpha=0.95, random_state=322,
                                    n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                    min_data_in_leaf=20, feature_fraction=0.9, verbose=-1)

    model_low_full.fit(df_full_train[feature_cols], df_full_train['price_p05'])
    model_high_full.fit(df_full_train[feature_cols], df_full_train['price_p95'])

    # =============================================================================
    # ТЕСТ
    # =============================================================================
    last_prices_full = df_full_train.groupby('product_id').tail(1)[['product_id', 'price_p05', 'price_p95']].rename(
        columns={'price_p05': 'price_p05_last', 'price_p95': 'price_p95_last'}
    )

    df_test_lags = df_test.merge(last_prices_full, on='product_id', how='left')
    med_p05 = df_full_train['price_p05'].median()
    med_p95 = df_full_train['price_p95'].median()
    df_test_lags['price_p05_last'] = df_test_lags['price_p05_last'].fillna(med_p05)
    df_test_lags['price_p95_last'] = df_test_lags['price_p95_last'].fillna(med_p95)

    df_test_feat = add_features(
        df_test_lags,
        fit_transform=False,
        encoders=le_dict_full,
        category_width_map=category_width_map,
        global_price_range_median=global_price_range_median
    )

    p05_raw = model_low_full.predict(df_test_feat[feature_cols])
    p95_raw = model_high_full.predict(df_test_feat[feature_cols])

    p05_raw = np.clip(np.minimum(p05_raw, p95_raw), 0.01, None)
    p95_raw = np.clip(np.maximum(p95_raw, p05_raw + 0.01), 0.02, None)

    mid = (p05_raw + p95_raw) / 2
    width = p95_raw - p05_raw

    # Применяем ROBUST параметры
    p05_final = mid * (1 + robust_mid_shift) - (width * robust_width_mult) / 2
    p95_final = mid * (1 + robust_mid_shift) + (width * robust_width_mult) / 2

    p05_final = np.clip(p05_final, 0.01, None)
    p95_final = np.clip(p95_final, p05_final + 0.01, None)

    # =============================================================================
    # СОЗДАНИЕ PREDICTIONS
    # =============================================================================
    predictions = {
        'price_p05': p05_final,
        'price_p95': p95_final
    }

    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(predictions)

    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()