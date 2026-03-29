# envs/ml_diagnostics/client.py

from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from envs.ml_diagnostics.models import MLAction, MLObservation, MLState


class MLDiagnosticsEnv(EnvClient[MLAction, MLObservation, MLState]):

    def _step_payload(self, action: MLAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "payload": action.payload,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MLObservation]:
        obs_data = payload["observation"]
        obs = MLObservation(
            task_id=obs_data["task_id"],
            step=obs_data["step"],
            training_context=obs_data["training_context"],
            feedback=obs_data["feedback"],
            score_so_far=obs_data["score_so_far"],
            available_actions=obs_data["available_actions"],
            message=obs_data["message"],
            done=payload["done"],
            reward=payload["reward"],
        )
        return StepResult(observation=obs, done=payload["done"])

    def _parse_state(self, payload: Dict[str, Any]) -> MLState:
        return MLState(
            task_id=payload.get("task_id", 1),
            issues_found=payload.get("issues_found", []),
            fixes_submitted=payload.get("fixes_submitted", []),
            data_requested=payload.get("data_requested", []),
            score=payload.get("score", 0.0),
            done=payload.get("done", False),
            step_count=payload.get("step_count", 0),
        )


if __name__ == "__main__":
    with MLDiagnosticsEnv(base_url="http://localhost:8003").sync() as env:
        for task_id in [1, 2, 3]:
            print(f"\n{'='*50}")
            print(f"TASK {task_id}")
            print('='*50)

            obs = env.reset(task_id=task_id, seed=42)
            print(f"Context: {obs.observation.training_context}")
            print(f"Feedback: {obs.observation.feedback}")

            # dummy actions just to test the pipeline
            obs = env.step(MLAction(
                action_type="request_data",
                payload={"data_type": "grad_norms"}
            ))
            print(f"\nStep 1 | Reward: {obs.observation.reward} | {obs.observation.feedback}")

            obs = env.step(MLAction(
                action_type="submit_diagnosis",
                payload={
                    "issue": "learning_rate_too_high",
                    "reasoning": "loss is diverging which indicates lr too high",
                }
            ))
            print(f"Step 2 | Reward: {obs.observation.reward} | {obs.observation.feedback}")

            obs = env.step(MLAction(
                action_type="submit_fix",
                payload={"suggested_value": 0.001, "fix_key": "learning_rate"}
            ))
            print(f"Step 3 | Reward: {obs.observation.reward} | {obs.observation.feedback}")
            print(f"Done: {obs.done} | Final Score: {obs.observation.score_so_far}")