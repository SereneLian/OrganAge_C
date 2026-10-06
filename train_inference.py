import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def _make_strat_bins_codes(y, q=10):
    """
    Create stratification bins for regression via quantiles, and return integer codes.
    This avoids sklearn 'unknown' target type issues with pandas Interval/Categorical.
    """
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)
    # if any NaN remains (shouldn't if label cleaned upstream), fill with median defensively
    if y.isna().any():
        y = y.fillna(y.median())

    bins = pd.qcut(y, q=q, duplicates="drop")   # categorical intervals
    return bins.cat.codes.values                # int codes: 0..(n_bins-1)

def run_lgbm_regression_with_search(
    data,
    label,
    features,
    param_grid=None,
    n_splits=10,
    test_size=0.2,
    random_state=42,
    scoring='neg_root_mean_squared_error',
    plot_importance=True):
    """
    1) Split into stratified train/test
    2) Run GridSearchCV over LGBMRegressor
    3) Fit best estimator on full train set
    4) Evaluate on test set
    5) Show feature importances
    
    Returns:
    - best_model: the GridSearchCV.best_estimator_
    - best_params: dict of best hyperparameters
    - test_results: dict of evaluation metrics on test
    - importance_df: DataFrame of feature importances
    - train_preds, test_preds: numpy arrays
    """
    # 1) train/test split with stratification on label-deciles
    label_bins = pd.qcut(data[label], q=10, duplicates='drop')
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=label_bins
    )
    X_train = train_df[features].values
    y_train = train_df[label].values
    X_test  = test_df[features].values
    y_test  = test_df[label].values

    # 2) define default param grid if not provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'max_depth': [-1, 10, 20],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

    # 3) set up GridSearchCV
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_model = lgb.LGBMRegressor(random_state=random_state)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    print("🔍 Starting hyperparameter search with GridSearchCV...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(f"✅ Best params: {best_params}")
    print(f"Best CV score ({scoring}): {grid.best_score_:.4f}")

    # 4) Predict & evaluate
    train_preds = best_model.predict(X_train)
    test_preds  = best_model.predict(X_test)

    rmse = mean_squared_error(y_test, test_preds, squared=False)
    mae  = mean_absolute_error(y_test, test_preds)
    r2   = r2_score(y_test, test_preds)
    r, pval = pearsonr(y_test, test_preds)

    test_results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson_r': r,
        'pearson_p': pval
    }

    print("\n📈 Test Set Evaluation:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson R: {r:.4f} (p={pval:.2e})")

    # 5) Feature importances
    imp = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': imp
    }).sort_values('importance', ascending=False)
    
    print("\n🔥 Top 20 Feature Importances:")
    print(importance_df.head(20))

    if plot_importance:
        plt.figure(figsize=(8,6))
        top20 = importance_df.head(20)[::-1]
        plt.barh(top20['feature'], top20['importance'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.show()

    # attach predictions
    data_with_preds = data.copy()
    data_with_preds['prediction'] = np.nan
    data_with_preds.loc[train_df.index, 'prediction'] = train_preds
    data_with_preds.loc[test_df.index,  'prediction'] = test_preds

    return best_model, best_params, test_results, importance_df, data_with_preds


def run_lgbm_nested_cv_no_search(
    data_df,
    label,
    features,
    test_size=0.2,
    random_state=42,
    n_splits=10,
    early_stopping_rounds=200,
    learning_rate=0.03,
    n_estimators=5000,
    lgbm_params=None,
    plot_importance=True,
):
    """
    One-dataset, paper-friendly design:
      1) External hold-out test split (stratified by label quantile bins)
      2) 10-fold stratified CV on training set -> OOF predictions + best_iteration per fold (early stopping)
      3) Choose robust best_iteration (median across folds), refit on full training set
      4) Infer on hold-out test
      5) Provide OOF / test / full predictions (full for downstream analysis, NOT for performance reporting)

    Returns:
      best_model, cv_df, cv_summary, test_results, importance_df, data_pred, (idx_train, idx_test)
    """

    # -------------------------
    # 0) Copy + basic cleaning
    # -------------------------
    data = data_df.copy()

    # Ensure label is numeric and clean inf
    data[label] = pd.to_numeric(data[label], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[label])

    # Keep X with feature names (pandas DF) for cleaner downstream use
    X = data[features]
    y = data[label].values.astype(float)

    # -------------------------
    # 1) External test split
    # -------------------------
    strat_bins = _make_strat_bins_codes(y, q=10)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data.index.values,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_bins
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # -------------------------
    # 2) 10-fold CV on training
    # -------------------------
    train_bins = _make_strat_bins_codes(y_train, q=10)  # IMPORTANT: int codes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Base params (you can extend via lgbm_params)
    base_params = dict(
        objective="regression",
        # random_state=random_state,
        # n_estimators=n_estimators,      # large; early stopping picks best_iteration_
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
        # LightGBM handles NaN natively
    )
    if lgbm_params is not None:
        base_params.update(lgbm_params)

    X_train_np = X_train.values
    y_train_np = np.asarray(y_train, dtype=float)

    oof_pred = np.zeros(len(X_train_np), dtype=float)
    fold_metrics = []
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_np, train_bins), 1):
        X_tr, X_va = X_train_np[tr_idx], X_train_np[va_idx]
        y_tr, y_va = y_train_np[tr_idx], y_train_np[va_idx]

        m = lgb.LGBMRegressor(**base_params, random_state=random_state + fold)

        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
        )

        pred_va = m.predict(X_va, num_iteration=m.best_iteration_)
        oof_pred[va_idx] = pred_va
        best_iters.append(int(m.best_iteration_))

        rmse = mean_squared_error(y_va, pred_va, squared=False)
        mae = mean_absolute_error(y_va, pred_va)
        r2 = r2_score(y_va, pred_va)
        r, _ = pearsonr(y_va, pred_va)

        fold_metrics.append({
            "fold": fold,
            "best_iter": int(m.best_iteration_),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson_r": r
        })

    cv_df = pd.DataFrame(fold_metrics)
    cv_summary = cv_df[["rmse", "mae", "r2", "pearson_r"]].agg(["mean", "std"])

    print("\n10-fold CV summary (OOF):")
    print(cv_summary)

    best_iter = int(np.median(best_iters))
    print(f"\nChosen best_iteration (median across folds): {best_iter}")

    # -------------------------
    # 3) Refit best model on full training set
    # -------------------------
    best_model = lgb.LGBMRegressor(**base_params, n_estimators=best_iter, random_state=random_state)
    best_model.fit(X_train, y_train)

    # -------------------------
    # 4) Infer on held-out test
    # -------------------------
    test_pred = best_model.predict(X_test)

    test_results = {
        "rmse": mean_squared_error(y_test, test_pred, squared=False),
        "mae": mean_absolute_error(y_test, test_pred),
        "r2": r2_score(y_test, test_pred),
        "pearson_r": pearsonr(y_test, test_pred)[0],
        "pearson_p": pearsonr(y_test, test_pred)[1],
    }

    print("\n📈 Held-out Test Evaluation:")
    print(
        f"RMSE: {test_results['rmse']:.4f}, "
        f"MAE: {test_results['mae']:.4f}, "
        f"R²: {test_results['r2']:.4f}, "
        f"Pearson R: {test_results['pearson_r']:.4f} (p={test_results['pearson_p']:.2e})"
    )

    # -------------------------
    # 5) Feature importance
    # -------------------------
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    if plot_importance:
        import matplotlib.pyplot as plt
        top = importance_df.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top["Feature"][::-1], top["Importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importances (LightGBM)")
        plt.tight_layout()
        plt.show()

    # -------------------------
    # 6) Attach predictions
    # -------------------------
    data_pred = data_df.copy()

    # OOF predictions only exist for the training split indices
    data_pred["pred_oof"] = np.nan
    data_pred.loc[idx_train, "pred_oof"] = oof_pred

    # Test predictions only for held-out test indices
    data_pred["pred_test"] = np.nan
    data_pred.loc[idx_test, "pred_test"] = test_pred

    # Full-data prediction (for downstream analysis; DO NOT use to report model performance)
    data_pred["pred_full"] = np.nan
    # only predict for rows that survived label cleaning in `data`
    data_pred.loc[data.index, "pred_full"] = best_model.predict(data[features])

    return best_model, cv_df, cv_summary, test_results, importance_df, data_pred, (idx_train, idx_test)

data_m = pd.read_csv('') # load instance 2 male data
data_m_3= pd.read_csv('') # load instance 3 male data
numeric_features= pd.read_csv('') #load input features name
label = 'label'
data_df = data_m #male dataset
best_model_m_c, cv_df_m_c, cv_summary_m_c, test_results_m_c, importance_df_m_c, data_pred_m_c, (_, _) =     run_lgbm_nested_cv_no_search(
        data_df=data_df, label=label, features=numeric_features,
        test_size=0.2, n_splits=10,
        random_state=42, plot_importance=True)

missing_cols = [c for c in numeric_features if c not in data_m_3.columns]
print("Missing in instance 3:", len(missing_cols))
X_m3 = data_m_3.reindex(columns=numeric_features)
preds_m3 = best_model_m_c.predict(X_m3)
