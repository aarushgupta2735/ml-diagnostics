# envs/ml_diagnostics/tasks/task1_hyperparams.py

from dataclasses import dataclass
from typing import Dict, Any
import random

# All possible hyperparameter issues and their signatures
TASK1_SCENARIOS = [
    {
        "issue": "learning_rate_too_high",
        "config": {
            "learning_rate": 0.9,
            "batch_size": 32,
            "optimizer": "adam",
            "epochs": 50,
        },
        "loss_curve": [2.3, 2.8, 3.1, 3.6, 4.2, 5.1, 6.3, 8.1],
        "val_loss_curve": [2.4, 2.9, 3.3, 3.8, 4.5, 5.4, 6.8, 8.9],
        "final_train_acc": 0.11,
        "final_val_acc": 0.10,
        "acceptable_fix_range": (0.0001, 0.01),
        "fix_key": "learning_rate",
    },
    {
        "issue": "learning_rate_too_low",
        "config": {
            "learning_rate": 0.000001,
            "batch_size": 32,
            "optimizer": "adam",
            "epochs": 50,
        },
        "loss_curve": [2.3, 2.29, 2.28, 2.27, 2.26, 2.25, 2.24, 2.23],
        "val_loss_curve": [2.31, 2.30, 2.29, 2.28, 2.27, 2.26, 2.25, 2.24],
        "final_train_acc": 0.13,
        "final_val_acc": 0.12,
        "acceptable_fix_range": (0.0001, 0.01),
        "fix_key": "learning_rate",
    },
    {
        "issue": "batch_size_too_large",
        "config": {
            "learning_rate": 0.001,
            "batch_size": 2048,
            "optimizer": "adam",
            "epochs": 50,
        },
        "loss_curve": [2.3, 2.1, 1.95, 1.92, 1.91, 1.91, 1.91, 1.91],
        "val_loss_curve": [2.4, 2.2, 2.1, 2.09, 2.09, 2.09, 2.09, 2.09],
        "final_train_acc": 0.42,
        "final_val_acc": 0.38,
        "acceptable_fix_range": (16, 256),
        "fix_key": "batch_size",
    },
    {
        "issue": "wrong_optimizer",
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "sgd",
            "epochs": 50,
            "task_note": "transformer model"
        },
        "loss_curve": [2.3, 2.1, 1.9, 1.85, 1.82, 1.80, 1.79, 1.79],
        "val_loss_curve": [2.4, 2.2, 2.05, 2.01, 1.99, 1.98, 1.98, 1.98],
        "final_train_acc": 0.45,
        "final_val_acc": 0.40,
        "acceptable_fix_range": None,  # not a numeric fix
        "fix_key": "optimizer",
        "acceptable_fix_values": ["adam", "adamw"],
    },
]


def get_task1_scenario(seed: int = None) -> Dict[str, Any]:
    """Return a random task 1 scenario, seeded for reproducibility."""
    rng = random.Random(seed)
    scenario = rng.choice(TASK1_SCENARIOS)
    return {
        "task_id": 1,
        "difficulty": "easy",
        "description": (
            "A model is training but something is wrong with the hyperparameters. "
            "Identify the single hyperparameter issue and suggest a fix."
        ),
        "config": scenario["config"],
        "loss_curve": scenario["loss_curve"],
        "val_loss_curve": scenario["val_loss_curve"],
        "final_train_acc": scenario["final_train_acc"],
        "final_val_acc": scenario["final_val_acc"],
        # hidden from agent — used by grader only
        "_ground_truth": {
            "issue": scenario["issue"],
            "fix_key": scenario["fix_key"],
            "acceptable_fix_range": scenario.get("acceptable_fix_range"),
            "acceptable_fix_values": scenario.get("acceptable_fix_values"),
        },
    }


def grade_task1(scenario: Dict[str, Any], actions: list) -> float:
    """
    Grade all actions taken during task 1.
    Returns a score between 0.0 and 1.0.
    """
    truth = scenario["_ground_truth"]
    score = 0.0
    diagnosis_action = None
    fix_action = None

    # find the last submit_diagnosis and submit_fix actions
    for action in actions:
        if action["action_type"] == "submit_diagnosis":
            diagnosis_action = action["payload"]
        if action["action_type"] == "submit_fix":
            fix_action = action["payload"]

    if diagnosis_action is None:
        return 0.0

    # +0.4 correct issue identified
    if diagnosis_action.get("issue") == truth["issue"]:
        score += 0.4

        # +0.3 fix value in acceptable range (only if diagnosis correct)
        if fix_action is not None:
            suggested = fix_action.get("suggested_value")
            if truth["acceptable_fix_range"] is not None and suggested is not None:
                low, high = truth["acceptable_fix_range"]
                if low <= float(suggested) <= high:
                    score += 0.3
            elif truth.get("acceptable_fix_values") and suggested is not None:
                if str(suggested).lower() in truth["acceptable_fix_values"]:
                    score += 0.3

    # +0.2 reasoning mentions correct symptom
    reasoning = diagnosis_action.get("reasoning", "").lower()
    symptom_keywords = {
        "learning_rate_too_high": ["diverge", "explod", "increas", "unstable"],
        "learning_rate_too_low": ["slow", "barely", "not moving", "stall"],
        "batch_size_too_large": ["plateau", "sharp", "generaliz", "flat"],
        "wrong_optimizer": ["sgd", "adam", "transformer", "momentum"],
    }
    keywords = symptom_keywords.get(truth["issue"], [])
    if any(kw in reasoning for kw in keywords):
        score += 0.2

    # +0.1 no hallucinated secondary issues
    if not diagnosis_action.get("secondary_issues"):
        score += 0.1

    return round(min(score, 1.0), 3)