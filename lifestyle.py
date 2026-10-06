from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pandas.api.types import is_object_dtype

# Preprocessors
num_cov_pre = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
num_feat_pre = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_feat_pre = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", drop=None))
])

def lr_test(ll_full, ll_red, df_full, df_red):
    LR = 2*(ll_full - ll_red)
    df_diff = df_full - df_red
    p = chi2.sf(LR, df_diff)
    return LR, df_diff, p

def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _coerce_categorical_to_str_keep_na(s: pd.Series) -> pd.Series:
    """
    Make categorical input safe for OneHotEncoder:
    - keep NaN as NaN
    - convert everything else to string
    """
    s2 = s.copy()
    mask = s2.isna()
    s2 = s2.astype(str)
    s2[mask] = np.nan
    return s2

def fit_one_factor_lrt(df, label, feature, covariates):
    needed = [label] + covariates + [feature]
    d = df[needed].dropna(subset=[label]).copy()

    # Decide feature type
    feat_is_numeric = pd.api.types.is_numeric_dtype(d[feature])

    if not feat_is_numeric:
        d[feature] = _coerce_categorical_to_str_keep_na(d[feature])

    full_trans = [("cov_num", num_cov_pre, covariates)]
    if feat_is_numeric:
        full_trans.append(("feat_num", num_feat_pre, [feature]))
    else:
        full_trans.append(("feat_cat", cat_feat_pre, [feature]))

    full_ct = ColumnTransformer(full_trans, remainder="drop")
    red_ct  = ColumnTransformer([("cov_num", num_cov_pre, covariates)], remainder="drop")

    y = d[label].values

    X_full = _to_dense(full_ct.fit_transform(d))
    X_red  = _to_dense(red_ct.fit_transform(d))

    X_full = sm.add_constant(X_full, has_constant="add")
    X_red  = sm.add_constant(X_red,  has_constant="add")

    fam = families.Gaussian()
    full_model = GLM(y, X_full, family=fam).fit()
    red_model  = GLM(y, X_red,  family=fam).fit()

    LR, df_diff, p_global = lr_test(full_model.llf, red_model.llf,
                                    full_model.df_model, red_model.df_model)

    return {
        "feature": feature,
        "n": int(len(d)),
        "p_global": float(p_global),
        "q_global": np.nan,
        "LR": float(LR),
        "df_diff": int(df_diff),
        "feat_is_numeric": bool(feat_is_numeric)
    }

def run_factor_screen(df, label, features, covariates):
    out = [fit_one_factor_lrt(df, label, f, covariates) for f in features]
    out = pd.DataFrame(out)
    rej, qvals, _, _ = multipletests(out["p_global"].values, method="fdr_bh")
    out["q_global"] = qvals
    out["significant_FDR_5pct"] = rej
    return out


df = pd.read_csv('')
filter_ids = pd.read_csv('') # env factors id list
all_based_col = filter_ids.loc[filter_ids.Sex == 'All', 'Name'].tolist()
if 'BMI' not in all_based_col:
    all_based_col = all_based_col + ['BMI']

replace_dict = {
    'Do not know': np.nan,
    'Prefer not to answer': np.nan,
    'Do not remember': np.nan,
    'Less than one': 0.5,
}
for col in df.columns:
    if is_object_dtype(df[col]):
        df[col] = df[col].replace(replace_dict)

cols = [c for c in all_based_col if c in df.columns]
features = df[cols].columns[df[cols].isnull().mean() < 0.2].tolist()
results = run_factor_screen(df, 'prediction', features, ['Sex', 'baseline_age'])