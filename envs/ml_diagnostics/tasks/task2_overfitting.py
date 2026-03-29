# envs/ml_diagnostics/tasks/task2_overfitting.py

from typing import Dict, Any
import random

TASK2_SCENARIOS = [
    {
        "root_causes": ["model_too_large", "no_regularization", "insufficient_data"],
        "config": {
            "model_layers": [512, 512, 512, 256],
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset_size": 800,
            "augmentation": False,
            "epochs": 100,
        },
        "train_acc_curve": [0.51, 0.63, 0.74, 0.85, 0.93, 0.97, 0.98, 0.99],
        "val_acc_curve":   [0.50, 0.61, 0.67, 0.68, 0.65, 0.61, 0.58, 0.55],
        "train_loss_curve": [1.8, 1.4, 1.1, 0.7, 0.4, 0.2, 0.1, 0.05],
        "val_loss_curve":   [1.9, 1.5, 1.3, 1.4, 1.6, 1.9, 2.1, 2.4],
        "valid_fixes": [
            {"action": "add_dropout", "acceptable_range": (0.2, 0.5)},
            {"action": "reduce_layers", "acceptable_values": [[256, 128], [128, 64], [256]]},
            {"action": "add_weight_decay", "acceptable_range": (0.0001, 0.1)},
            {"action": "add_augmentation", "acceptable_values": [True]},
        ],
        "fix_priority": ["add_dropout", "reduce_layers", "add_weight_decay"],
    },
    {
        "root_causes": ["no_regularization", "insufficient_data"],
        "config": {
            "model_layers": [128, 64],
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset_size": 200,
            "augmentation": False,
            "epochs": 200,
        },
        "train_acc_curve": [0.52, 0.68, 0.79, 0.88, 0.94, 0.98, 0.99, 1.0],
        "val_acc_curve":   [0.51, 0.65, 0.70, 0.69, 0.66, 0.62, 0.59, 0.56],
        "train_loss_curve": [1.7, 1.2, 0.9, 0.6, 0.3, 0.1, 0.05, 0.01],
        "val_loss_curve":   [1.8, 1.3, 1.2, 1.3, 1.5, 1.8, 2.0, 2.3],
        "valid_fixes": [
            {"action": "add_dropout", "acceptable_range": (0.2, 0.5)},
            {"action": "add_weight_decay", "acceptable_range": (0.0001, 0.1)},
            {"action": "add_augmentation", "acceptable_values": [True]},
            {"action": "collect_more_data", "acceptable_values": [True]},
        ],
        "fix_priority": ["add_dropout", "add_weight_decay", "collect_more_data"],
    },
]

# all valid root cause labels the agent can use
VALID_ROOT_CAUSES = {
    "model_too_large",
    "no_regularization",
    "insufficient_data",
    "no_augmentation",
    "too_many_epochs",
}

# all valid fix actions the agent can use
VALID_FIX_ACTIONS = {
    "add_dropout",
    "reduce_layers",
    "add_weight_decay",
    "add_augmentation",
    "collect_more_data",
    "early_stopping",
}


def get_task2_scenario(seed: int = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    scenario = rng.choice(TASK2_SCENARIOS)
    return {
        "task_id": 2,
        "difficulty": "medium",
        "description": (
            "A model is overfitting. Identify all root causes "
            "and prescribe a prioritized fix plan."
        ),
        "config": scenario["config"],
        "train_acc_curve": scenario["train_acc_curve"],
        "val_acc_curve": scenario["val_acc_curve"],
        "train_loss_curve": scenario["train_loss_curve"],
        "val_loss_curve": scenario["val_loss_curve"],
        "_ground_truth": {
            "root_causes": scenario["root_causes"],
            "valid_fixes": scenario["valid_fixes"],
            "fix_priority": scenario["fix_priority"],
        },
    }


def grade_task2(scenario: Dict[str, Any], actions: list) -> float:
    truth = scenario["_ground_truth"]
    score = 0.0

    # find last submit_diagnosis and submit_fix
    diagnosis_action = None
    fix_action = None
    for action in actions:
        if action["action_type"] == "submit_diagnosis":
            diagnosis_action = action["payload"]
        if action["action_type"] == "submit_fix":
            fix_action = action["payload"]

    if diagnosis_action is None:
        return 0.0

    # +0.2 top level diagnosis is overfitting
    if diagnosis_action.get("diagnosis") == "overfitting":
        score += 0.2

    # +0.1 per correct root cause identified (max 0.3)
    submitted_causes = set(diagnosis_action.get("root_causes", []))
    correct_causes = set(truth["root_causes"])
    hallucinated_causes = submitted_causes - VALID_ROOT_CAUSES
    correct_hits = submitted_causes & correct_causes

    score += min(len(correct_hits) * 0.1, 0.3)

    # -0.05 per hallucinated cause
    score -= len(hallucinated_causes) * 0.05

    # grade fixes
    if fix_action is not None:
        submitted_fixes = fix_action.get("fixes", [])
        valid_fix_actions = {f["action"] for f in truth["valid_fixes"]}
        fix_lookup = {f["action"]: f for f in truth["valid_fixes"]}

        correct_fix_count = 0
        for fix in submitted_fixes:
            action_name = fix.get("action")
            if action_name not in VALID_FIX_ACTIONS:
                score -= 0.05  # hallucinated fix action
                continue
            if action_name in valid_fix_actions:
                truth_fix = fix_lookup[action_name]
                # check value is in acceptable range
                value = fix.get("value")
                if "acceptable_range" in truth_fix and value is not None:
                    low, high = truth_fix["acceptable_range"]
                    if low <= float(value) <= high:
                        score += 0.1
                        correct_fix_count += 1
                elif "acceptable_values" in truth_fix:
                    score += 0.1
                    correct_fix_count += 1
                else:
                    score += 0.05  # fix action correct but no value needed

        # +0.1 priority order correct
        submitted_order = [f["action"] for f in submitted_fixes
                          if f.get("action") in valid_fix_actions]
        expected_order = [f for f in truth["fix_priority"]
                         if f in submitted_order]
        if submitted_order[:len(expected_order)] == expected_order:
            score += 0.1

        # -0.1 contradictory fixes (e.g. increase and decrease model size)
        fix_names = {f.get("action") for f in submitted_fixes}
        if "reduce_layers" in fix_names and "add_layers" in fix_names:
            score -= 0.1

    return round(max(min(score, 1.0), 0.0), 3)