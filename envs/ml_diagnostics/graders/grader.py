# envs/ml_diagnostics/graders/grader.py

from typing import Dict, Any, List
from envs.ml_diagnostics.tasks.task1_hyperparams import get_task1_scenario, grade_task1
from envs.ml_diagnostics.tasks.task2_overfitting import get_task2_scenario, grade_task2
from envs.ml_diagnostics.tasks.task3_multi_issue import get_task3_scenario, grade_task3


STRICT_SCORE_EPS = 1e-6


TASK_REGISTRY = {
    1: {
        "get_scenario": get_task1_scenario,
        "grade": grade_task1,
        "max_steps": 5,
        "description": "Hyperparameter Diagnosis (Easy)",
    },
    2: {
        "get_scenario": get_task2_scenario,
        "grade": grade_task2,
        "max_steps": 7,
        "description": "Overfitting Detection and Fix (Medium)",
    },
    3: {
        "get_scenario": get_task3_scenario,
        "grade": grade_task3,
        "max_steps": 10,
        "description": "Multi-Issue Training Failure (Hard)",
    },
}


class MLDiagnosticsGrader:
    """
    Central grader for the ML Diagnostics environment.
    Manages scenario loading, step-level reward, and final scoring.
    """

    def __init__(self, task_id: int, seed: int = None):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Invalid task_id {task_id}. Must be 1, 2, or 3.")

        self.task_id = task_id
        self.seed = seed
        self._registry = TASK_REGISTRY[task_id]
        self.scenario = self._registry["get_scenario"](seed=seed)
        self.max_steps = self._registry["max_steps"]
        self.actions: List[Dict[str, Any]] = []
        self.step_rewards: List[float] = []
        self.total_score: float = 0.0
        self.issues_found: List[str] = []

    def get_initial_context(self) -> Dict[str, Any]:
        """
        Return the initial training context shown to the agent.
        Ground truth is always stripped out.
        """
        return {
            k: v for k, v in self.scenario.items()
            if not k.startswith("_")
        }

    def get_additional_data(self, data_type: str) -> Dict[str, Any]:
        """
        Return additional data when agent requests it.
        Rewards information-seeking behavior.
        """
        available = {
            "grad_norms": self.scenario.get("grad_norm_curve"),
            "data_distribution": self.scenario.get("data_sample", {}).get("label_distribution"),
            "feature_stats": self.scenario.get("data_sample", {}).get("feature_stats"),
            "full_config": self.scenario.get("config"),
        }
        if data_type not in available:
            return {"error": f"Unknown data type: {data_type}. Available: {list(available.keys())}"}
        if available[data_type] is None:
            return {"error": f"{data_type} not available for this task."}
        return {data_type: available[data_type]}

    def process_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one agent action and return step reward + feedback.
        This is called by the environment on every step().
        """
        action_type = action.get("action_type")
        payload = action.get("payload", {})
        step_reward = 0.0
        feedback = ""

        if action_type == "request_data":
            data_type = payload.get("data_type", "")
            result = self.get_additional_data(data_type)
            if "error" not in result:
                step_reward = 0.05  # reward information seeking
                feedback = f"Data retrieved: {data_type}"
            else:
                step_reward = -0.02  # penalty for requesting invalid data
                feedback = result["error"]
            self.actions.append(action)
            self.step_rewards.append(step_reward)
            return {
                "step_reward": step_reward,
                "feedback": feedback,
                "data": result,
            }

        elif action_type == "submit_diagnosis":
            # partial scoring — check how many issues found so far
            partial_score = self._partial_diagnosis_score(payload)
            new_issues = partial_score["new_issues_found"]

            if new_issues:
                step_reward = len(new_issues) * 0.1
                self.issues_found.extend(new_issues)
                feedback = f"Good — identified: {', '.join(new_issues)}"
            else:
                step_reward = -0.05  # wrong or duplicate diagnosis
                feedback = "No new issues identified. Try requesting more data or revising."

            self.actions.append(action)
            self.step_rewards.append(step_reward)
            return {
                "step_reward": step_reward,
                "feedback": feedback,
                "issues_found_so_far": self.issues_found,
            }

        elif action_type == "submit_fix":
            # small reward for submitting a fix — full scoring at episode end
            step_reward = 0.05
            feedback = "Fix recorded. Submit your final diagnosis to complete the episode."
            self.actions.append(action)
            self.step_rewards.append(step_reward)
            return {
                "step_reward": step_reward,
                "feedback": feedback,
            }

        else:
            step_reward = -0.05
            feedback = f"Unknown action type: {action_type}"
            self.step_rewards.append(step_reward)
            return {
                "step_reward": step_reward,
                "feedback": feedback,
            }

    def compute_final_score(self) -> float:
        """
        Run the task-specific grader on all accumulated actions.
        Called when episode ends (done=True).
        """
        grade_fn = self._registry["grade"]
        raw_score = float(grade_fn(self.scenario, self.actions))
        # Submission requires strict bounds, so never return exact 0 or 1.
        self.total_score = min(max(raw_score, STRICT_SCORE_EPS), 1.0 - STRICT_SCORE_EPS)
        return self.total_score

    def _partial_diagnosis_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check how many correct issues the agent has found so far.
        Used for step-level reward during submit_diagnosis.
        """
        if self.task_id == 1:
            correct_issue = self.scenario["_ground_truth"]["issue"]
            submitted = payload.get("issue", "")
            new_issues = [submitted] if submitted == correct_issue and submitted not in self.issues_found else []

        elif self.task_id == 2:
            correct_causes = set(self.scenario["_ground_truth"]["root_causes"])
            submitted_causes = set(payload.get("root_causes", []))
            new_issues = list((submitted_causes & correct_causes) - set(self.issues_found))

        elif self.task_id == 3:
            correct_issues = self.scenario["_ground_truth"]["issue_names"]
            submitted = {i.get("issue") for i in payload.get("issues", [])}
            new_issues = list((submitted & correct_issues) - set(self.issues_found))

        else:
            new_issues = []

        return {"new_issues_found": new_issues}


def run_all_tasks(seed: int = 42) -> Dict[str, Any]:
    """
    Utility to run and score all 3 tasks with empty actions.
    Used for testing the grader pipeline end to end.
    """
    results = {}
    for task_id in [1, 2, 3]:
        grader = MLDiagnosticsGrader(task_id=task_id, seed=seed)
        results[f"task_{task_id}"] = {
            "description": TASK_REGISTRY[task_id]["description"],
            "scenario_context": grader.get_initial_context(),
            "score_with_no_actions": grader.compute_final_score(),
        }
    return results