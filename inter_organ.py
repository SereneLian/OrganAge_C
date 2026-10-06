import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import statsmodels.api as sm

# instance 2
def partial_corr(df, x, y, covars):
    """
    Partial correlation between x and y adjusting for covars.
    Returns r, p, n_used.
    """
    d = df[[x, y] + covars].replace([np.inf, -np.inf], np.nan).dropna()
    n = d.shape[0]
    if n < 5:
        return np.nan, np.nan, n

    X = sm.add_constant(d[covars])
    rx = sm.OLS(d[x].astype(float), X).fit().resid
    ry = sm.OLS(d[y].astype(float), X).fit().resid
    r, p = pearsonr(rx, ry)
    return r, p, n

ins2_df = pd.read_csv('')

pairs = [
    ('Brain_Age_ins2', 'Heart_Age_ins2'),
    ('Brain_Age_ins2', 'BodyCom_Age_ins2'),
    ('Heart_Age_ins2', 'BodyCom_Age_ins2'),
]

rows = []
for x, y in pairs:
    r, p, n = partial_corr(ins2_df, x, y, covars=['BA_ins2', 'Sex'])
    rows.append({
        'X': x,
        'Y': y,
        'Partial_r': r,
        'P_value': p,
        'N': n
    })

ins2_assoc_table = pd.DataFrame(rows)

# instance 2->3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

inter_df = pd.read_csv('') #load the patients with repated aging

systems = {
    "Brain":   ("Brain_Age_ins2",   "Brain_Age_ins3"),
    "Heart":   ("Heart_Age_ins2",   "Heart_Age_ins3"),
    "BodyCom": ("BodyCom_Age_ins2", "BodyCom_Age_ins3"),
    # "ClinicalAge": ("ClinicalAge_ins2", "ClinicalAge_ins3")  #
}

# 1) compute deltas
for sys, (c2, c3) in systems.items():
    inter_df[f"Delta_{sys}"] = inter_df[c3] - inter_df[c2]

def residualize(series, cov_df):
    d = pd.concat([series, cov_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y = d.iloc[:, 0].astype(float)
    X = sm.add_constant(d.iloc[:, 1:])
    fit = sm.OLS(y, X).fit()
    r = y - fit.predict(X)
    r.name = series.name
    return r

# covariates for longitudinal coupling
covars = ["BA_ins2", "BA_diff"] + (["Sex"] if "Sex" in inter_df.columns else [])
cov_df = inter_df[covars].copy()

# 2) residualize each delta
delta_resid = {}
for sys in systems.keys():
    col = f"Delta_{sys}"
    delta_resid[sys] = residualize(inter_df[col], cov_df)

# 3) pairwise correlations of residuals
rows = []
sys_list = list(systems.keys())
for i, A in enumerate(sys_list):
    for j, B in enumerate(sys_list):
        if j <= i:
            continue
        r = pd.concat([delta_resid[A], delta_resid[B]], axis=1).dropna()
        if r.shape[0] < 50:
            continue
        corr = r.corr().iloc[0,1]
        # quick p-value via t approximation
        n = r.shape[0]
        t = corr * np.sqrt((n - 2) / max(1e-12, (1 - corr**2)))
        p = 2 * (1 - stats.t.cdf(np.abs(t), df=n-2))
        rows.append({"A": A, "B": B, "N": n, "Corr": corr, "P": p})

delta_coupling_corr = pd.DataFrame(rows)

