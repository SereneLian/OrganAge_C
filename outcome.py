import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import os

# survival
def hr_regression(ints_cols, ehr_age_map_all, category_to_dummy=None, death_label='event', penalizer=0):
    # 1) Copy and (later) drop rows with missing predictor data
    df = ehr_age_map_all.copy()
    # 2) One-hot–encode any specified categorical column(s)
    if category_to_dummy is not None:
        df[category_to_dummy] = df[category_to_dummy].astype('int')
        # allow either a single col or list
        cats = [category_to_dummy] if isinstance(category_to_dummy, str) else list(category_to_dummy)
        df = pd.get_dummies(df, columns=cats, drop_first=True)
        # adjust ints_cols: remove originals, add new dummies
        dummy_cols = [c for c in df.columns
                      for orig in cats
                      if c.startswith(f"{orig}_")]
        ints_cols = [c for c in ints_cols if c not in cats] + dummy_cols
    print(f'ints_cols: {ints_cols}')
    # 3) Drop rows with NaNs in any of the predictor columns
    print(f'before: {len(df)} rows')
    df = df.dropna(subset=ints_cols)
    print(f' after: {len(df)} rows')

    # 4) Build dataframe for CoxPHFitter
    df_pred = df[ints_cols + [death_label, 'time2death']]

    # 5) Fit the model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_pred, duration_col='time2death', event_col=death_label)

    # 6) Extract and format results
    summary_df = cph.summary[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']
    ].copy()

    # Round numeric columns
    for col in ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']:
        summary_df[col] = summary_df[col].round(3)

    # Scientific notation for p-values
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")

    # Add model-level c-index
    summary_df['c-index'] = round(cph.concordance_index_, 3)

    return summary_df

df = pd.read_csv('') #load prediction results
organ_df = hr_regression(ints_cols = ['prediction'], ehr_age_map_all = df)
adj_col = ['prediction','baseline_age','BMI', 'Systolic Blood Pressure', 'Smoking', 'IMD']
organ_df_adj = hr_regression(ints_cols = adj_col, ehr_age_map_all = df)

# disease

import pandas as pd
import warnings
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceWarning
from lifelines.exceptions import ConvergenceError

def hr_regression_d(ints_cols, ehr_age_map_all, category_to_dummy=None, disease='', penalizer_sequence=None):
    # set up
    if penalizer_sequence is None:
        penalizer_sequence = [0, 0.01, 0.1, 0.5, 1.0]

    eve  = f"{disease}_event"
    time = f"time2_{disease}"

    # copy & dummy‐encode if needed
    df = ehr_age_map_all.copy()
    if category_to_dummy is not None:
        cats = [category_to_dummy] if isinstance(category_to_dummy, str) else list(category_to_dummy)
        df[cats] = df[cats].astype(int)
        df = pd.get_dummies(df, columns=cats, drop_first=True)
        # rebuild ints_cols
        dummy_cols = [c for c in df.columns for orig in cats if c.startswith(f"{orig}_")]
        ints_cols = [c for c in ints_cols if c not in cats] + dummy_cols

    # drop missing
    df = df.dropna(subset=ints_cols)

    # prepare data slice
    df_pred = df[ints_cols + [eve, time]]

    last_exc = None
    for pen in penalizer_sequence:
        cph = CoxPHFitter(penalizer=pen)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                cph.fit(df_pred, duration_col=time, event_col=eve)
                # if we got any convergence warnings, treat as failure
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    print(f"⚠️ ConvergenceWarning with penalizer={pen}; retrying.")
                    continue
            # success: format and return
            summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']].copy()
            # rounding
            for col in ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']:
                summary_df[col] = summary_df[col].round(3)
            summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")
            summary_df['c-index'] = round(cph.concordance_index_, 3)
            return summary_df, pen

        except ZeroDivisionError:
            print(f"🚫 ZeroDivisionError with penalizer={pen}; retrying.")
            last_exc = "ZeroDivisionError"
        except ConvergenceError:
            print(f"🚫 ConvergenceError with penalizer={pen}; retrying.")
            last_exc = "ConvergenceError"
        except Exception as e:
            print(f"❗ Unexpected error with penalizer={pen}: {e}")
            last_exc = e
            break

    # if we exhaust sequence
    print(f"❌ All penalizers failed (last error: {last_exc}).")
    # return empty table + NaN so downstream code can still assemble rows
    empty = pd.DataFrame(np.nan, index=ints_cols,
        columns=['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', 'c-index'])
    return empty, np.nan


heart_all    = []
diseases = pd.read_csv('')['disease'].to_list() # load disease names
columns = ['HR', 'HR lower 95%', 'HR upper 95%', 'z', 'p', 'c-index', 'disease']

for d in diseases:
    print(f"\nProcessing disease: {d}")

    # load
    df_female = pd.read_csv('')
    df_female['Sex'] = 0
    df_male  = pd.read_csv('')
    df_male['Sex'] = 1

    df_all = pd.concat([df_female, df_male], ignore_index=True)

    adj_col = ['prediction','baseline_age', 'Sex']
    for c in adj_col:
        if c not in ['Sex','Smoking']:
            df_all[c] = (df_all[c] - df_all[c].mean()) / df_all[c].std()

# all
    summary_all, pen_all = hr_regression_d(ints_cols=adj_col, ehr_age_map_all=df_all, disease=d)
    if 'prediction' in summary_all.index:
        row_all = summary_all.loc['prediction'].to_list()
    else:
        row_all = [np.nan]*6
    heart_all.append(row_all + [d])

df_heart_all = pd.DataFrame(heart_all, columns=columns)
