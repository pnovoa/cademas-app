"""Perturbation-based local ML attributions for MOJO models."""

from __future__ import annotations

from typing import Callable

import h2o
import numpy as np
import pandas as pd

from explain_utils import aggregate_weighted_attributions, resolve_feature_column


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def compute_feature_baseline(series: pd.Series) -> Any:
    """Cohort baseline: median for numeric columns, mode for categorical."""
    clean = series.dropna()
    if clean.empty:
        return np.nan
    if _is_numeric(clean):
        return float(clean.median())
    modes = clean.mode()
    return modes.iloc[0] if not modes.empty else clean.iloc[0]


def compute_baselines(model_df: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    baselines = {}
    for feature in feature_names:
        col = resolve_feature_column(feature, model_df.columns)
        if col is None:
            continue
        baselines[feature] = compute_feature_baseline(model_df[col])
    return baselines


def extract_positive_probability(preds: pd.DataFrame, p_col_hint: str | None = None) -> np.ndarray:
    if p_col_hint and p_col_hint in preds.columns:
        return preds[p_col_hint].astype(float).values
    if "p1" in preds.columns:
        return preds["p1"].astype(float).values
    return preds.iloc[:, -1].astype(float).values


def predict_mojo_probabilities(mojo, model_df: pd.DataFrame, p_col_hint: str | None = None) -> np.ndarray:
    hf = h2o.H2OFrame(model_df)
    try:
        preds = mojo.predict(hf).as_data_frame()
        return extract_positive_probability(preds, p_col_hint)
    finally:
        h2o.remove(hf)


def compute_model_attributions(
    mojo,
    model_df: pd.DataFrame,
    feature_names: list[str],
    p_col_hint: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    One-at-a-time perturbation attributions for a single MOJO model.

    Δ_{f}(i) = p(x_i) - p(x_i with feature f set to cohort baseline)
    """
    warnings: list[str] = []
    n_rows = len(model_df)
    resolved_features: list[str] = []
    column_map: dict[str, str] = {}

    for feature in feature_names:
        col = resolve_feature_column(feature, model_df.columns)
        if col is None:
            warnings.append(f"Feature '{feature}' not found in dataset; skipped.")
            continue
        resolved_features.append(feature)
        column_map[feature] = col

    if not resolved_features:
        return pd.DataFrame(index=range(n_rows)), warnings

    baselines = compute_baselines(model_df, resolved_features)
    base_probs = predict_mojo_probabilities(mojo, model_df, p_col_hint)

    contributions = pd.DataFrame(0.0, index=range(n_rows), columns=resolved_features)

    for feature in resolved_features:
        col = column_map[feature]
        baseline_val = baselines.get(feature)
        if pd.isna(baseline_val):
            warnings.append(f"Could not compute baseline for '{feature}'; skipped.")
            continue

        perturbed_df = model_df.copy()
        perturbed_df[col] = baseline_val
        try:
            perturbed_probs = predict_mojo_probabilities(mojo, perturbed_df, p_col_hint)
        except Exception as exc:
            warnings.append(f"Perturbation failed for '{feature}': {exc}")
            continue

        contributions[feature] = base_probs - perturbed_probs

    return contributions, warnings


def compute_ml_attributions(
    model_df: pd.DataFrame,
    feature_config: dict,
    weights: dict[str, float],
    model_mojo_map: dict[str, object],
    p_col_hint: str | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """
    Compute per-model and aggregated perturbation attributions.

    Parameters
    ----------
    model_mojo_map:
        Mapping of model filename -> imported H2O MOJO object.
    """
    by_model: dict[str, pd.DataFrame] = {}
    all_warnings: list[str] = []
    features_by_model: dict[str, list[str]] = {}

    model_items = [name for name in model_mojo_map if name in weights]
    total_steps = max(len(model_items), 1)

    for step_idx, model_name in enumerate(model_items):
        mojo = model_mojo_map[model_name]
        features = feature_config.get(model_name, {}).get("features", [])
        features_by_model[model_name] = list(features)

        frame, warnings = compute_model_attributions(
            mojo,
            model_df,
            features,
            p_col_hint=p_col_hint,
        )
        by_model[model_name] = frame
        all_warnings.extend(warnings)

        if progress_callback is not None:
            progress_callback((step_idx + 1) / total_steps)

    aggregated = aggregate_weighted_attributions(by_model, weights)

    return {
        "by_model": by_model,
        "aggregated": aggregated,
        "meta": {
            "method": "perturbation_one_at_a_time",
            "baseline": "cohort_median_or_mode",
            "features_by_model": features_by_model,
            "warnings": all_warnings,
        },
    }
