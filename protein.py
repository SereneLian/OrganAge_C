import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List

def _impute_series(s: pd.Series):
    """Median for numeric; mode for non-numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median())
    mode = s.mode(dropna=True)
    fillv = mode.iloc[0] if not mode.empty else (s.dropna().iloc[0] if s.dropna().size else np.nan)
    return s.fillna(fillv)

def gwas_like(
    df: pd.DataFrame,
    features: List[str],
    covariates: List[str],
    outcome: str = "prediction",
    drop_na_outcome: bool = True,
    impute_covariates: bool = True,     # <- NEW
    impute_feature: bool = False,       # <- NEW (protein default: NO impute)
) -> pd.DataFrame:
    """
    For each feature (protein), fit: outcome ~ feature + covariates (OLS).
    - Feature (protein) is NOT imputed by default: only participants with non-missing protein are used.
    - Covariates can be imputed (default) or complete-case (set impute_covariates=False).
    - One-hot encodes non-numeric covariates (drop_first=True).
    - Bonferroni correction across tested proteins.
    """

    needed = list(dict.fromkeys([outcome] + features + covariates))
    data = df.loc[:, [c for c in needed if c in df.columns]].copy()

    if drop_na_outcome and outcome in data.columns:
        data = data.loc[data[outcome].notna()].copy()

    # split covariates into numeric / categorical
    cov_num = [c for c in covariates if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
    cov_cat = [c for c in covariates if c in data.columns and not pd.api.types.is_numeric_dtype(data[c])]

    # covariate missing handling
    if impute_covariates:
        for c in covariates:
            if c in data.columns:
                data[c] = _impute_series(data[c])
    else:
        # strict complete-case on covariates (but not on proteins yet)
        data = data.dropna(subset=[c for c in covariates if c in data.columns]).copy()

    # one-hot encode categorical covariates
    if cov_cat:
        data = pd.get_dummies(data, columns=cov_cat, drop_first=True)

    # covariate design columns
    covar_cols = [c for c in data.columns if c != outcome and c in cov_num] + \
                 [c for c in data.columns if cov_cat and c.startswith(tuple([f"{c}_" for c in cov_cat]))]

    results = []
    for feat in features:
        if feat not in data.columns:
            continue

        d = data[[outcome] + covar_cols + [feat]].copy()

        # ---- FEATURE missing handling (protein) ----
        if impute_feature:
            d[feat] = _impute_series(d[feat])
        else:
            d = d.dropna(subset=[feat]).copy()   # <- KEY CHANGE

        # coerce feature to numeric (protein should be numeric)
        if not pd.api.types.is_numeric_dtype(d[feat]):
            s_feat = d[feat]
            if isinstance(s_feat, pd.DataFrame):
                s_feat = s_feat.iloc[:, 0]
            s_feat = pd.to_numeric(s_feat, errors="coerce").astype(float)
            # if not imputing feature, drop remaining NaN after coercion
            if impute_feature:
                s_feat = s_feat.fillna(s_feat.median())
            else:
                d = d.loc[s_feat.notna()].copy()
                s_feat = s_feat.loc[d.index]
            d[feat] = s_feat

        # if too few rows, skip
        if d.shape[0] < 50:
            results.append({"feature": feat, "n": int(d.shape[0]), "beta": np.nan, "se": np.nan,
                            "t": np.nan, "p": np.nan, "r2": np.nan})
            continue

        X = sm.add_constant(pd.concat([d[[feat]], d[covar_cols]], axis=1), has_constant="add")
        y = d[outcome].astype(float)

        try:
            model = sm.OLS(y, X, missing="drop").fit()
            beta = model.params.get(feat, np.nan)
            se   = model.bse.get(feat, np.nan)
            tval = model.tvalues.get(feat, np.nan)
            pval = model.pvalues.get(feat, np.nan)

            results.append({
                "feature": feat,
                "n": int(model.nobs),
                "beta": float(beta) if beta==beta else np.nan,
                "se": float(se) if se==se else np.nan,
                "t": float(tval) if tval==tval else np.nan,
                "p": float(pval) if pval==pval else np.nan,
                "r2": float(model.rsquared),
            })
        except Exception as e:
            results.append({"feature": feat, "n": int(d.shape[0]), "beta": np.nan, "se": np.nan,
                            "t": np.nan, "p": np.nan, "r2": np.nan, "error": str(e)})

    out = pd.DataFrame(results)

    # Bonferroni over tested proteins with valid p
    if not out.empty and "p" in out.columns:
        m = out["p"].notna().sum()
        out["p_bonf"] = (out["p"] * m).clip(upper=1.0)

    out["significant"] = out.get("p_bonf", np.nan) < 0.05
    out = out.sort_values("p", na_position="last").reset_index(drop=True)
    return out

thr = 0.2  # keep columns with <20% missing
# Filter columns with <20% missing across all participants
data_bc = pd.read_csv('')
data_heart = pd.read_csv('')
data_brain = pd.read_csv('')
bc_filtered_cols = data_bc.columns[data_bc.isnull().mean() < thr].tolist()
heart_filtered_cols = data_heart.columns[data_heart.isnull().mean() < thr].tolist()
brain_filtered_cols = data_brain.columns[data_brain.isnull().mean() < thr].tolist()

print("BC filtered columns:", len(bc_filtered_cols))
print("Heart filtered columns:", len(heart_filtered_cols))
print("Brain filtered columns:", len(brain_filtered_cols))

# Find common columns among the three datasets
common_cols = list(
    set(bc_filtered_cols)
    .intersection(heart_filtered_cols)
    .intersection(brain_filtered_cols)
)
print("Common columns across all organs:", len(common_cols))

bodycom_df =  gwas_like(df = data_bc, features= common_cols, 
                        covariates = ['baseline_age', 'Sex', 'BMI'],
                        outcome = "prediction", drop_na_outcome=True, ) 
