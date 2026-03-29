# envs/ml_diagnostics/server/ml_diagnostics.py

from typing import Optional, Any
from openenv.core.env_server import Environment
from envs.ml_diagnostics.models import MLAction, MLObservation, MLState
from envs.ml_diagnostics.graders.grader import MLDiagnosticsGrader, TASK_REGISTRY


class MLDiagnosticsEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state = MLState()
        self._grader: Optional[MLDiagnosticsGrader] = None
        self._seed: Optional[int] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs: Any,
    ) -> MLObservation:

        self._seed = seed
        self._grader = MLDiagnosticsGrader(task_id=task_id, seed=seed)

        self._state = MLState(
            task_id=task_id,
            step_count=0,
            done=False,
        )

        context = self._grader.get_initial_context()

        return MLObservation(
            task_id=task_id,
            step=0,
            training_context=context,
            feedback=f"Task {task_id} started: {TASK_REGISTRY[task_id]['description']}. "
                     f"You have {self._grader.max_steps} steps.",
            score_so_far=0.0,
            available_actions=["request_data", "submit_diagnosis", "submit_fix"],
            message="Analyze the training context and diagnose the issue(s).",
        )

    def step(
        self,
        action: MLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MLObservation:

        if self._state.done:
            return self._make_observation(
                feedback="Episode already done. Call reset().",
                extra_context={},
                step_reward=0.0,
            )

        if self._grader is None:
            return self._make_observation(
                feedback="Environment not initialized. Call reset() first.",
                extra_context={},
                step_reward=0.0,
            )

        # process the action
        result = self._grader.process_step({
            "action_type": action.action_type,
            "payload": action.payload,
        })

        self._state.step_count += 1
        step_reward = result.get("step_reward", 0.0)
        feedback = result.get("feedback", "")
        extra_context = {k: v for k, v in result.items()
                        if k not in {"step_reward", "feedback"}}

        # update state tracking
        if action.action_type == "submit_diagnosis":
            self._state.issues_found = self._grader.issues_found.copy()

        # check if episode should end
        done = False
        final_score = None

        episode_complete = (
            action.action_type == "submit_fix"
            and len(self._grader.actions) >= 2  # at least one diagnosis + one fix
        )
        max_steps_reached = self._state.step_count >= self._grader.max_steps

        if episode_complete or max_steps_reached:
            done = True
            final_score = self._grader.compute_final_score()
            self._state.done = True
            self._state.score = final_score

            if max_steps_reached and not episode_complete:
                feedback += f" Max steps reached. Final score: {final_score:.3f}"
            else:
                feedback += f" Episode complete. Final score: {final_score:.3f}"

        # set reward on observation
        reward = final_score if done else step_reward

        return self._make_observation(
            feedback=feedback,
            extra_context=extra_context,
            step_reward=reward,
            done=done,
        )

    @property
    def state(self) -> MLState:
        return self._state

    # --- helpers ---

    def _make_observation(
        self,
        feedback: str,
        extra_context: dict,
        step_reward: float,
        done: bool = False,
    ) -> MLObservation:

        # merge extra context (e.g. requested data) into training context
        context = {}
        if self._grader is not None:
            context = self._grader.get_initial_context()
        context.update(extra_context)

        available_actions = ["request_data", "submit_diagnosis", "submit_fix"]
        if done:
            available_actions = []

        return MLObservation(
            task_id=self._state.task_id,
            step=self._state.step_count,
            training_context=context,
            feedback=feedback,
            score_so_far=self._state.score,
            available_actions=available_actions,
            message="",
            done=done,
            reward=step_reward,
        )