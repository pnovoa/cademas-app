"""Unit tests for Explain tab helpers (no H2O required)."""

import pandas as pd
import pytest

from explain_utils import (
    aggregate_weighted_attributions,
    bottleneck_rule,
    build_nl_summary,
    classify_balance_stance,
    format_priority_verdict,
    humanize_rule,
    resolve_feature_column,
    split_context_rules,
    top_contributions,
)


def test_resolve_feature_column_alias():
    cols = ["age", "over18", "department"]
    assert resolve_feature_column("over_18", cols) == "over18"
    assert resolve_feature_column("age", cols) == "age"
    assert resolve_feature_column("missing", cols) is None


def test_aggregate_weighted_attributions():
    by_model = {
        "m1.zip": pd.DataFrame({"age": [0.2, 0.0], "gender": [0.1, 0.05]}),
        "m2.zip": pd.DataFrame({"age": [0.0, 0.3], "income": [0.4, 0.1]}),
    }
    weights = {"m1.zip": 0.6, "m2.zip": 0.4}
    aggregated = aggregate_weighted_attributions(by_model, weights)

    assert pytest.approx(aggregated.loc[0, "age"]) == 0.12
    assert pytest.approx(aggregated.loc[0, "gender"]) == 0.06
    assert pytest.approx(aggregated.loc[0, "income"]) == 0.16
    assert pytest.approx(aggregated.loc[1, "age"]) == 0.12


def test_top_contributions_preserves_sign():
    series = pd.Series({"a": 0.01, "b": -0.9, "c": 0.5})
    top = top_contributions(series, k=2)
    assert list(top.index) == ["b", "c"]
    assert top["b"] == -0.9


def test_bottleneck_rule_minimum():
    fuzzy_row = pd.Series(
        {
            "mu_salary_ok": 0.8,
            "mu_tenure_ok": 0.2,
            "mu_commitment": 0.6,
        }
    )
    result = bottleneck_rule(fuzzy_row, "minimum (strict)")
    assert result == ("tenure_ok", 0.2)


def test_split_context_rules():
    fuzzy_row = pd.Series(
        {
            "mu_salary_ok": 0.85,
            "mu_tenure_ok": 0.15,
            "mu_commitment": 0.55,
        }
    )
    drivers, bottlenecks = split_context_rules(fuzzy_row)
    assert drivers == [("salary_ok", 0.85)]
    assert bottlenecks == [("tenure_ok", 0.15)]


def test_humanize_rule_prefers_full_name():
    context_config = {
        "rules": [
            {"name": "tenure_relevant", "full_name": "Tenure relevant"},
        ]
    }
    assert humanize_rule("tenure_relevant", context_config) == "Tenure relevant"
    assert humanize_rule("missing_rule", context_config) == "Missing rule"


def test_build_nl_summary_uses_bottlenecks_when_context_is_low():
    fuzzy_row = pd.Series({"mu_salary_ok": 0.0, "mu_tenure_ok": 0.1, "mu_commitment": 0.55})
    summary = build_nl_summary(
        case_id="Emma Smith",
        priority=0.62,
        ri=0.82,
        ci=0.35,
        lambda_val=0.5,
        top_ml=pd.Series({"over_time": 0.18, "monthly_income": 0.11}),
        master_row=pd.Series({"over_time": "Yes", "monthly_income": 5484}),
        fuzzy_row=fuzzy_row,
    )
    assert "**Emma Smith**" in summary
    assert "**62%**" in summary
    assert "does not meet the policy criteria for" in summary
    assert "**Salary ok** (μ=0.00)" in summary
    assert "supported by" not in summary
    assert "Tier 2 - Medium Priority" in summary
    assert "retention" not in summary.lower()
    assert "employee" not in summary.lower()


def test_build_nl_summary_uses_drivers_when_context_is_high():
    fuzzy_row = pd.Series({"mu_salary_ok": 0.9, "mu_tenure_ok": 0.2, "mu_commitment": 0.85})
    summary = build_nl_summary(
        case_id="Emma Smith",
        priority=0.78,
        ri=0.55,
        ci=0.72,
        lambda_val=0.55,
        top_ml=pd.Series({"over_time": 0.12}),
        master_row=pd.Series({"over_time": "Yes"}),
        fuzzy_row=fuzzy_row,
    )
    assert "aligns strongly with the policy rules" in summary
    assert "Tier 1 - High Priority" in summary
    assert classify_balance_stance(0.55) == "predictive-risk-led"


def test_format_priority_verdict_tiers():
    assert "Tier 1 - High Priority" in format_priority_verdict(0.75)
    assert "intervention or resource allocation" in format_priority_verdict(0.90).lower()
    assert "Tier 2 - Medium Priority" in format_priority_verdict(0.50)
    assert "Tier 2 - Medium Priority" in format_priority_verdict(0.74)
    assert "Tier 3 - Low Priority" in format_priority_verdict(0.49)
    assert "Tier 3 - Low Priority" in format_priority_verdict(0.10)


def test_build_nl_summary_accepts_custom_vocabulary():
    summary = build_nl_summary(
        case_id="TX-001",
        priority=0.80,
        ri=0.70,
        ci=0.60,
        lambda_val=0.5,
        top_ml=pd.Series({"amount": 0.2}),
        master_row=pd.Series({"amount": 1000}),
        fuzzy_row=pd.Series({"mu_rule_a": 0.8}),
        vocabulary={
            "target_case": "the transaction",
            "target_action": "fraud review",
        },
    )
    assert "In context, the transaction" in summary
    assert "fraud review" in summary.lower()
