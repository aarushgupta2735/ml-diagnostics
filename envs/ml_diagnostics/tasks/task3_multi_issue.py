# envs/ml_diagnostics/tasks/task3_multi_issue.py

from typing import Dict, Any
import random

TASK3_SCENARIOS = [
    {
        "issues": [
            {
                "issue": "class_imbalance",
                "severity": "critical",
                "priority": 1,
                "fix": "use weighted loss or oversample minority classes",
                "keywords": ["imbalance", "class", "distribution", "weighted", "minority"],
            },
            {
                "issue": "wrong_loss_function",
                "severity": "critical",
                "priority": 2,
                "fix": "replace MSE with CrossEntropyLoss",
                "keywords": ["mse", "cross entropy", "classification", "loss function", "crossentropy"],
            },
            {
                "issue": "gradient_explosion",
                "severity": "high",
                "priority": 3,
                "fix": "add batch normalization or gradient clipping",
                "keywords": ["gradient", "explod", "nan", "clip", "batch norm", "unstable"],
            },
        ],
        "config": {
            "learning_rate": 0.01,
            "loss_function": "mse",
            "task_type": "multi_class_classification",
            "num_classes": 10,
            "batch_norm": False,
            "model_depth": 12,
            "optimizer": "adam",
        },
        "data_sample": {
            "label_distribution": {
                "0": 0.91, "1": 0.01, "2": 0.01,
                "3": 0.01, "4": 0.01, "5": 0.01,
                "6": 0.01, "7": 0.01, "8": 0.01, "9": 0.01,
            },
            "feature_stats": {
                "mean": 847.3, "std": 1203.1,
                "min": -50, "max": 9500,
            },
        },
        "loss_curve": [2.3, 2.3, 2.3, 2.3, "nan", "nan"],
        "grad_norm_curve": [0.8, 1.2, 4.3, 18.7, "nan", "nan"],
        "train_acc_curve": [0.91, 0.91, 0.91, 0.91, "nan", "nan"],
    },
    {
        "issues": [
            {
                "issue": "data_leakage",
                "severity": "critical",
                "priority": 1,
                "fix": "recompute normalization stats on training set only",
                "keywords": ["leakage", "leak", "normalization", "test set", "val set", "statistics"],
            },
            {
                "issue": "vanishing_gradients",
                "severity": "high",
                "priority": 2,
                "fix": "use residual connections or replace sigmoid with relu",
                "keywords": ["vanish", "gradient", "sigmoid", "relu", "residual", "dead"],
            },
            {
                "issue": "incorrect_weight_initialization",
                "severity": "medium",
                "priority": 3,
                "fix": "use xavier or kaiming initialization",
                "keywords": ["init", "weight", "xavier", "kaiming", "random", "zero"],
            },
        ],
        "config": {
            "learning_rate": 0.001,
            "loss_function": "cross_entropy",
            "task_type": "binary_classification",
            "activation": "sigmoid",
            "model_depth": 15,
            "weight_init": "zeros",
            "normalization": "computed_on_full_dataset",
        },
        "data_sample": {
            "label_distribution": {"0": 0.51, "1": 0.49},
            "feature_stats": {
                "mean": 0.0, "std": 1.0,
                "min": -3.1, "max": 3.2,
            },
        },
        "loss_curve": [0.693, 0.692, 0.691, 0.691, 0.691, 0.691],
        "grad_norm_curve": [0.001, 0.0008, 0.0005, 0.0003, 0.0002, 0.0001],
        "train_acc_curve": [0.51, 0.51, 0.51, 0.51, 0.51, 0.51],
    },
]

VALID_ISSUES = {
    "class_imbalance",
    "wrong_loss_function",
    "gradient_explosion",
    "vanishing_gradients",
    "data_leakage",
    "incorrect_weight_initialization",
    "learning_rate_too_high",
    "learning_rate_too_low",
    "wrong_optimizer",
}

CORRECT_PRIORITY_ORDER = ["critical", "high", "medium", "low"]


def get_task3_scenario(seed: int = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    scenario = rng.choice(TASK3_SCENARIOS)
    return {
        "task_id": 3,
        "difficulty": "hard",
        "description": (
            "Multiple things are wrong with this training run simultaneously. "
            "Identify ALL issues, their severity, and the correct order to fix them. "
            "Explain how the issues interact with each other."
        ),
        "config": scenario["config"],
        "data_sample": scenario["data_sample"],
        "loss_curve": scenario["loss_curve"],
        "grad_norm_curve": scenario["grad_norm_curve"],
        "train_acc_curve": scenario["train_acc_curve"],
        "_ground_truth": {
            "issues": scenario["issues"],
            "issue_names": {i["issue"] for i in scenario["issues"]},
            "priority_map": {i["issue"]: i["priority"] for i in scenario["issues"]},
            "severity_map": {i["issue"]: i["severity"] for i in scenario["issues"]},
            "keyword_map": {i["issue"]: i["keywords"] for i in scenario["issues"]},
        },
    }


def grade_task3(scenario: Dict[str, Any], actions: list) -> float:
    truth = scenario["_ground_truth"]
    score = 0.0

    # find last submit_diagnosis action
    diagnosis_action = None
    for action in actions:
        if action["action_type"] == "submit_diagnosis":
            diagnosis_action = action["payload"]

    if diagnosis_action is None:
        return 0.0

    submitted_issues = diagnosis_action.get("issues", [])
    submitted_issue_names = {i.get("issue") for i in submitted_issues}
    correct_issue_names = truth["issue_names"]
    hallucinated = submitted_issue_names - VALID_ISSUES

    # +0.2 per correctly identified issue (max 0.6)
    correct_hits = submitted_issue_names & correct_issue_names
    score += len(correct_hits) * 0.2

    # -0.1 per hallucinated issue
    score -= len(hallucinated) * 0.1

    # -0.2 if NaN/explosion present but missed entirely
    has_nan_issue = any(
        i["issue"] in {"gradient_explosion", "vanishing_gradients"}
        for i in truth["issues"]
    )
    agent_caught_nan = any(
        i.get("issue") in {"gradient_explosion", "vanishing_gradients"}
        for i in submitted_issues
    )
    if has_nan_issue and not agent_caught_nan:
        score -= 0.2

    # +0.1 per correct fix with right keywords (max 0.3)
    for submitted in submitted_issues:
        issue_name = submitted.get("issue")
        if issue_name not in correct_issue_names:
            continue
        fix_text = submitted.get("fix", "").lower()
        keywords = truth["keyword_map"].get(issue_name, [])
        if any(kw in fix_text for kw in keywords):
            score += 0.1

    # +0.1 priority order is correct (critical before high before medium)
    submitted_order = [
        i.get("issue") for i in submitted_issues
        if i.get("issue") in correct_issue_names
    ]
    expected_order = sorted(
        correct_hits,
        key=lambda x: truth["priority_map"].get(x, 99)
    )
    if submitted_order == expected_order:
        score += 0.1

    # +0.05 interaction note present and non-empty
    interaction = diagnosis_action.get("interaction_note", "")
    if len(interaction) > 20:
        score += 0.05

    return round(max(min(score, 1.0), 0.0), 3)