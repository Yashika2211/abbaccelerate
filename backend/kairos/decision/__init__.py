"""The Decision Layer: what makes Kairos more than a leaderboard.

Pure post-hoc economics over cached predictions. Nothing in this package imports
a modelling library, which is what lets the what-if simulator recompute the entire
decision surface in milliseconds without retraining.
"""

from kairos.decision.cost import (
    CostConfig,
    ClassificationCosts,
    classification_cost_matrix,
    confusion_counts,
    expected_cost,
    cost_breakdown,
    annualize,
    rul_policy_cost,
    RulPolicyResult,
)

__all__ = [
    "CostConfig",
    "ClassificationCosts",
    "classification_cost_matrix",
    "confusion_counts",
    "expected_cost",
    "cost_breakdown",
    "annualize",
    "rul_policy_cost",
    "RulPolicyResult",
]
