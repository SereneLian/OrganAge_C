from pathlib import Path
import os
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

BASE_DIR = Path("")
RESULT_DIR = BASE_DIR / "results"
DISEASE_ROOT = BASE_DIR / "data/disease"
INS = "2"
SUMMARY_FILE = DISEASE_ROOT / f"ins{INS}_summary.csv"
OUTPUT_DIR = Path("final_results")
SPEC_DIR = OUTPUT_DIR / "disease_age_specifity"


MORANDI = {"brain": "#6E6A9F", "vascular": "#B65C5C", "metabolic": "#C9A227"}
disease_union = [
    "Cerebrovascular diseases", "Stroke", "Diabetes", "Chronic kidney disease",
    "Heart failure", "Atrial fibrillation", "Ischaemic heart disease",
    "Dyslipidaemia", "Chronic liver diseases", "Osteoarthritis", "Rheumatoid arthritis",
]


def _read_organ_age(ins, organ, sex, result_path):
    """
    Read OrganAge residual files.

    If sex == "f" or "m":
        read one sex-specific file.
    If sex == "all":
        read both female and male files, then merge.
    """

    def _read_one(sex_code):
        fp = os.path.join(result_path, f"{organ}_ins{ins}_{sex_code}_residual.csv")
        df = pd.read_csv(fp)

        keep = ["eid", "prediction", "baseline_age", "Sex", "BMI"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        df = df.rename(columns={
            "eid": "patid",
            "prediction": "organage_gap"
        })

        df["patid"] = df["patid"].astype(int)

        # Ensure Sex column exists and is usable when merging f/m files
        if "Sex" not in df.columns:
            df["Sex"] = sex_code

        # Optional: keep original file sex label for debugging
        df["sex_file"] = sex_code

        return df

    if sex in ["f", "m"]:
        return _read_one(sex)

    elif sex == "all":
        df_f = _read_one("f")
        df_m = _read_one("m")

        df_all = pd.concat([df_f, df_m], axis=0, ignore_index=True)

        # Safety check: remove duplicated patid if any
        # Normally there should be no overlap between f and m residual files.
        if df_all["patid"].duplicated().any():
            dup_n = df_all["patid"].duplicated().sum()
            print(f"Warning: {dup_n} duplicated patid values found after merging f and m files.")
            df_all = df_all.drop_duplicates(subset=["patid"], keep="first")

        return df_all

    else:
        raise ValueError("sex must be one of: 'f', 'm', or 'all'")

def _read_disease_ids(ins, disease_name, disease_path_root):
    fp = os.path.join(disease_path_root, f"ins{ins}", f"{disease_name}.csv")
    d = pd.read_csv(fp)
    col = "patid" if "patid" in d.columns else "eid"
    ids = pd.Series(d[col]).dropna().astype(int).unique()
    return set(ids)

def _make_no_disease_mask(df_org, diseases_31, disease_path_root, ins):
    """
    no_disease = free of 31 diseases + obesity (BMI>=30)
    """
    all_ids = set()
    for dis in diseases_31:
        all_ids |= _read_disease_ids(ins, dis, disease_path_root)

    has_any_31 = df_org["patid"].isin(all_ids)
    has_obesity = (df_org["BMI"] >= 30) if "BMI" in df_org.columns else False

    no_disease = (~has_any_31) & (~has_obesity)
    return no_disease

def _fit_adjusted_contrast(df, group_col, covariates, robust=True):
    """
    OLS: organage_gap ~ group + covariates
    Return beta, CI, p, n
    """
    cov_part = " + ".join(covariates) if covariates else "1"
    formula = f"organage_gap ~ {group_col} + {cov_part}"

    m = smf.ols(formula, data=df).fit(cov_type="HC3" if robust else "nonrobust")
    beta = float(m.params[group_col])
    se = float(m.bse[group_col])
    p = float(m.pvalues[group_col])
    lo, hi = beta - 1.96*se, beta + 1.96*se
    return beta, lo, hi, p, int(m.nobs)

def run_baseline_burden_one_organ(
    ins, organ, sex, diseases_31, result_path,
    disease_path_root=" ",
    min_case=10, min_ctrl=200
):
    """
    For each disease (31) + obesity:
      compare disease group vs strict no_disease reference
      adjusted for baseline_age (and Sex if sex=='all')
    """
    df_org = _read_organ_age(ins, organ, sex, result_path)

    # strict reference pool
    no_dis_mask = _make_no_disease_mask(df_org, diseases_31, disease_path_root, ins)
    ref_ids = set(df_org.loc[no_dis_mask, "patid"].values.tolist())

    rows = []

    # helper: run one group vs reference
    def _run_group(group_name, case_ids):
        case_ids = set(case_ids)
        # disease group can overlap with other diseases; but reference is strict
        d = df_org.loc[df_org["patid"].isin(case_ids | ref_ids)].copy()
        d["group"] = d["patid"].isin(case_ids).astype(int)

        n_case = int(d["group"].sum())
        n_ctrl = int((d["group"] == 0).sum())
        if n_case < min_case or n_ctrl < min_ctrl:
            return

        if sex == "all":
            covs = ["baseline_age", "Sex"] if "Sex" in d.columns else ["baseline_age"]
        else:
            covs = ["baseline_age"]

        beta, lo, hi, p, n = _fit_adjusted_contrast(d, "group", covs, robust=True)

        rows.append({
            "organ": organ,
            "sex": sex,
            "disease": group_name,
            "beta": beta,
            "ci_low": lo,
            "ci_high": hi,
            "p": p,
            "n": n,
            "n_case": n_case,
            "n_ctrl": n_ctrl
        })

    # 31 diseases
    for dis in diseases_31:
        ids = _read_disease_ids(ins, dis, disease_path_root)
        _run_group(dis, ids)

    # obesity group (BMI>=30)
    if "BMI" in df_org.columns:
        obesity_ids = df_org.loc[df_org["BMI"] >= 30, "patid"].astype(int).unique()
        _run_group("obesity", obesity_ids)

    out = pd.DataFrame(rows, columns=["organ", "sex", "disease", "beta", "ci_low", "ci_high", "p", "n", "n_case", "n_ctrl"])
    if not out.empty:
        out = out.sort_values("p").reset_index(drop=True)
    return out

def plot_union_diseases_one_sex(df_all, sex_label, diseases_order, savepath=None):
    organs = ["brain","vascular","metabolic"]

    d = df_all.copy()
    d = d[d["organ_label"].isin(organs)].copy()

    d = d[d["disease_clean"].isin(diseases_order)].copy()

    idx = pd.MultiIndex.from_product([diseases_order, organs], names=["disease_clean","organ_label"])
    g = (d.set_index(["disease_clean","organ_label"])
           .reindex(idx)
           .reset_index())

    missing_rows = g["beta"].isna() | g["ci_low"].isna() | g["ci_high"].isna()
    if missing_rows.any():
        miss = g.loc[missing_rows, ["disease_clean","organ_label"]]
        print("WARNING: missing beta/CI for the following disease×organ rows:")
        print(miss.to_string(index=False))

    base_y = {dis:i for i, dis in enumerate(diseases_order)}
    offsets = {"brain":-0.22, "vascular":0.0, "metabolic":0.22}
    g["y"] = g["disease_clean"].map(base_y).astype(float) + g["organ_label"].map(offsets).astype(float)

    fig_h = max(6.0, 0.48*len(diseases_order) + 1.8)
    fig, ax = plt.subplots(figsize=(9.6, fig_h))

    for organ in organs:
        sub = g[(g["organ_label"] == organ) & (~g["beta"].isna()) & (~g["ci_low"].isna()) & (~g["ci_high"].isna())].copy()
        if sub.empty:
            continue
        ax.errorbar(
            sub["beta"], sub["y"],
            xerr=[sub["beta"]-sub["ci_low"], sub["ci_high"]-sub["beta"]],
            fmt="o", color=MORANDI[organ], ecolor="0.78",
            elinewidth=1.2, capsize=3, markersize=5.8,
            label=organ.capitalize()
        )

    ax.axvline(0, color="0.6", linestyle="--", lw=1)

    y_ticks = [base_y[dis] for dis in diseases_order]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(diseases_order, fontsize=11)

    ax.set_xlabel("Adjusted difference in OrganAge gap vs disease-free (years)")
    ax.set_title(f"{sex_label}: disease burden contrasts across organs", loc="left", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()



