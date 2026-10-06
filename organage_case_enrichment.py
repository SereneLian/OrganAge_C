import numpy as np
import pandas as pd



STRATEGIES = {
    "Chronological age": "baseline_age",
    "BMI": "BMI",
    "Brain gap": "brain_gap",
    "Cardiovascular gap": "heart_gap",
    "Metabolic gap": "metabolic_gap",
}


def _check_unique_id(df, name):
    if df["eid"].duplicated().any():
        raise ValueError(f"{name}: duplicate eid values")


def prepare_disease_cohort(cohort, outcome, prevalent, disease):
    """Construct the common disease-specific population used by all five strategies."""
    event_col = f"{disease}_event"
    time_col = f"time2_{disease}"

    required_cohort = ["eid"] + list(STRATEGIES.values())
    required_outcome = ["eid", event_col, time_col]

    missing = set(required_cohort) - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort missing columns: {sorted(missing)}")
    missing = set(required_outcome) - set(outcome.columns)
    if missing:
        raise ValueError(f"{disease} outcome missing columns: {sorted(missing)}")

    c = cohort[required_cohort].copy()
    o = outcome[required_outcome].copy()
    _check_unique_id(c, "cohort")
    _check_unique_id(o, f"{disease} outcome")

    for col in STRATEGIES.values():
        c[col] = pd.to_numeric(c[col], errors="raise")
    for col in [event_col, time_col]:
        o[col] = pd.to_numeric(o[col], errors="raise")

    c = c.replace([np.inf, -np.inf], np.nan)
    o = o.replace([np.inf, -np.inf], np.nan)

    if not o[event_col].dropna().isin([0, 1]).all():
        raise ValueError(f"{disease}: event indicator must be 0/1")

    # Non-positive BMI is invalid, matching the original analysis.
    c.loc[c["BMI"].le(0), "BMI"] = np.nan

    # Exclude prevalent disease before imaging baseline.
    prevalent = set(pd.to_numeric(pd.Series(list(prevalent)), errors="raise").dropna().astype("int64"))
    d = c.loc[~c["eid"].isin(prevalent)].copy()

    # All five strategies are evaluated in exactly the same complete-case population.
    d = d.dropna(subset=list(STRATEGIES.values()))

    d = d.merge(o, on="eid", how="left", validate="one_to_one")
    d = d.dropna(subset=[event_col, time_col])
    d = d.loc[d[time_col] > 0].copy()
    d[event_col] = d[event_col].astype(int)

    return d.sort_values("eid").reset_index(drop=True)


def top_k_indices(scores, k, tie_break):
    """Select exactly k highest scores; resolve threshold ties randomly."""
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)
    needed = k - len(above)

    if needed == len(tied):
        chosen_ties = tied
    else:
        order = np.argsort(tie_break[tied], kind="stable")
        chosen_ties = tied[order[:needed]]

    return np.concatenate([above, chosen_ties])


def analyse_top10(data, disease, analysis="Main", capacity=0.10,
                  n_boot=2000, seed=2026):
    """Point estimates and paired participant-level bootstrap CIs."""
    event_col = f"{disease}_event"
    time_col = f"time2_{disease}"

    if data.empty:
        raise ValueError(f"{disease}, {analysis}: empty analysis population")
    if not 0 < capacity < 1:
        raise ValueError("capacity must be between 0 and 1")

    d = data.sort_values("eid").reset_index(drop=True)
    strategy_names = list(STRATEGIES)
    score_cols = list(STRATEGIES.values())

    scores = d[score_cols].to_numpy(dtype=float)
    events = d[event_col].to_numpy(dtype=int)
    times = d[time_col].to_numpy(dtype=float)

    n = len(d)
    k = int(np.ceil(capacity * n))
    total_cases = int(events.sum())
    n_strategies = len(strategy_names)

    point_seed, boot_seed = np.random.SeedSequence(seed).spawn(2)
    point_rng = np.random.default_rng(point_seed)
    boot_rng = np.random.default_rng(boot_seed)

    # Point estimates: one outcome-independent random tie-break vector
    # is shared across strategies.
    tie_break = point_rng.random(n)
    observed_cases = np.zeros(n_strategies, dtype=int)
    median_times = np.zeros(n_strategies)

    for j in range(n_strategies):
        selected = top_k_indices(scores[:, j], k, tie_break)
        observed_cases[j] = events[selected].sum()
        median_times[j] = np.median(times[selected])

    point_yields = 1000 * observed_cases / k

    # Paired bootstrap: the same resampled participants are used for all
    # five strategies within each bootstrap iteration.
    boot_yields = np.empty((n_boot, n_strategies))

    for b in range(n_boot):
        sampled = boot_rng.integers(0, n, size=n)
        sample_events = events[sampled]
        sample_ties = boot_rng.random(n)

        for j in range(n_strategies):
            selected = top_k_indices(scores[sampled, j], k, sample_ties)
            boot_yields[b, j] = 1000 * sample_events[selected].sum() / k

    yield_ci = np.percentile(boot_yields, [2.5, 97.5], axis=0)

    age_idx = strategy_names.index("Chronological age")
    bmi_idx = strategy_names.index("BMI")

    boot_diff_age = boot_yields - boot_yields[:, [age_idx]]
    boot_diff_bmi = boot_yields - boot_yields[:, [bmi_idx]]
    age_diff_ci = np.percentile(boot_diff_age, [2.5, 97.5], axis=0)
    bmi_diff_ci = np.percentile(boot_diff_bmi, [2.5, 97.5], axis=0)

    rows = []
    for j, strategy in enumerate(strategy_names):
        rows.append({
            "disease": disease,
            "analysis": analysis,
            "strategy": strategy,
            "n_eligible": n,
            "total_observed_cases": total_cases,
            "n_selected": k,
            "selected_pct": 100 * k / n,
            "observed_cases_selected": observed_cases[j],
            "cases_per_1000": point_yields[j],
            "cases_per_1000_ci_low": yield_ci[0, j],
            "cases_per_1000_ci_high": yield_ci[1, j],
            "diff_vs_age": point_yields[j] - point_yields[age_idx],
            "diff_vs_age_ci_low": age_diff_ci[0, j],
            "diff_vs_age_ci_high": age_diff_ci[1, j],
            "diff_vs_bmi": point_yields[j] - point_yields[bmi_idx],
            "diff_vs_bmi_ci_low": bmi_diff_ci[0, j],
            "diff_vs_bmi_ci_high": bmi_diff_ci[1, j],
            "observed_cases_captured_pct": (
                100 * observed_cases[j] / total_cases if total_cases > 0 else np.nan
            ),
            "enrichment_vs_age": (
                point_yields[j] / point_yields[age_idx]
                if point_yields[age_idx] > 0 else np.nan
            ),
            "enrichment_vs_bmi": (
                point_yields[j] / point_yields[bmi_idx]
                if point_yields[bmi_idx] > 0 else np.nan
            ),
            "median_observed_time_months": median_times[j],
            "n_boot": n_boot,
        })

    return pd.DataFrame(rows)


def run_analysis(cohort, outcomes, prevalent_ids, diseases,
                 capacity=0.10, n_boot=2000, seed=2026):
    """
    Run the main and 24-month landmark analyses.

    Parameters
    ----------
    cohort : DataFrame
        One row per participant; both sexes pooled. Required columns:
        eid, baseline_age, BMI, brain_gap, heart_gap, metabolic_gap.
    outcomes : dict[str, DataFrame]
        Disease-specific event/time tables.
    prevalent_ids : dict[str, iterable]
        IDs with each disease before imaging baseline.
    diseases : iterable[str]
        Disease names matching outcome column prefixes.
    """
    diseases = list(dict.fromkeys(diseases))
    results = []
    disease_seeds = np.random.SeedSequence(seed).spawn(len(diseases))

    for disease, disease_seed in zip(diseases, disease_seeds):
        d = prepare_disease_cohort(
            cohort=cohort,
            outcome=outcomes[disease],
            prevalent=prevalent_ids[disease],
            disease=disease,
        )

        time_col = f"time2_{disease}"
        subsets = {
            "Main": d,
            # Excludes both events and censoring occurring within 24 months.
            "Landmark24": d.loc[d[time_col] > 24].copy(),
        }

        analysis_seeds = disease_seed.spawn(2)
        for (analysis, subset), analysis_seed in zip(subsets.items(), analysis_seeds):
            results.append(
                analyse_top10(
                    data=subset,
                    disease=disease,
                    analysis=analysis,
                    capacity=capacity,
                    n_boot=n_boot,
                    seed=analysis_seed.generate_state(1).item(),
                )
            )

    return pd.concat(results, ignore_index=True)


#
# results = run_analysis(
#     cohort=cohort,
#     outcomes=outcomes,
#     prevalent_ids=prevalent_ids,
#     diseases=diseases_31,
#     capacity=0.10,
#     n_boot=2000,
#     seed=2026,
# )
