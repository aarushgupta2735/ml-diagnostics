#!/usr/bin/env python3
"""Hackathon submission inference runner for ml_diagnostics.

This script emits strict stdout markers:
- [START] once per task episode
- [STEP] once per env.step result
- [END] once per task episode (always, even on errors)
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from typing import Any, Dict, List

from envs.ml_diagnostics.client import MLDiagnosticsEnv
from envs.ml_diagnostics.models import MLAction


def _load_local_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into process environment."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            if line.startswith("export "):
                line = line[len("export ") :]

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env_file()


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_IDS = [1, 2, 3]
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
SEED = int(os.getenv("SEED", "42"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

TASK_NAMES = {
    1: "hyperparameter-diagnosis",
    2: "overfitting-detection-and-fix",
    3: "multi-issue-training-failure",
}
BENCHMARK = os.getenv("BENCHMARK", "ml-diagnostics")

SYSTEM_PROMPT = (
    "You are an expert ML engineer diagnosing training failures. "
    "Respond ONLY with a JSON object: "
    '{"action_type": "...", "payload": {...}}. '
    "Valid action_type values: request_data, submit_diagnosis, submit_fix."
)


def validate_runtime_config() -> None:
    """Fail fast when required runtime configuration is missing."""
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY must be set")

    if not API_BASE_URL.strip():
        raise RuntimeError("API_BASE_URL must be a non-empty string")

    if not MODEL_NAME.strip():
        raise RuntimeError("MODEL_NAME must be a non-empty string")


def _compact(value: Any) -> str:
    """Return a single-line representation suitable for stdout contract fields."""
    if isinstance(value, str):
        return value.replace("\n", " ").strip()
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = _compact(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={_compact(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _parse_model_action(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed


def get_model_action(client: Any, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
        temperature=0.0,
        stream=False,
    )
    message = completion.choices[0].message.content or ""
    parsed = _parse_model_action(message)

    action_type = parsed.get("action_type", "request_data")
    payload = parsed.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    return {"action_type": str(action_type), "payload": payload}


def _build_sync_env():
    """Build and return a connected sync environment client."""
    if LOCAL_IMAGE_NAME:
        async_client = asyncio.run(MLDiagnosticsEnv.from_docker_image(LOCAL_IMAGE_NAME))
        return async_client.sync()

    return MLDiagnosticsEnv(base_url=ENV_BASE_URL).sync()


def _create_openai_client() -> Any:
    """Create an OpenAI-compatible client via runtime import."""
    openai_module = importlib.import_module("openai")
    openai_client_cls = getattr(openai_module, "OpenAI")
    return openai_client_cls(base_url=API_BASE_URL, api_key=API_KEY)


def run_task(client: Any, task_id: int, seed: int) -> float:
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None
    conversation: List[Dict[str, str]] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = _build_sync_env()
        env.connect()
        step_result = env.reset(task_id=task_id, seed=seed)
        observation = step_result.observation
        done = bool(step_result.done)

        conversation = [
            {
                "role": "user",
                "content": (
                    f"Task {task_id} started. "
                    f"Context: {json.dumps(observation.training_context, separators=(',', ':'))}. "
                    f"Feedback: {observation.feedback}."
                ),
            }
        ]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_error = None
            try:
                action_dict = get_model_action(client, conversation)
            except Exception as exc:
                action_error = str(exc)
                action_dict = {
                    "action_type": "request_data",
                    "payload": {"data_type": "full_config"},
                }

            action = MLAction(
                action_type=action_dict.get("action_type", "request_data"),
                payload=action_dict.get("payload", {}),
            )

            try:
                step_result = env.step(action)
                reward = float(
                    step_result.reward
                    if step_result.reward is not None
                    else (step_result.observation.reward or 0.0)
                )
                done = bool(step_result.done)
                observation = step_result.observation
            except Exception as exc:
                reward = 0.0
                done = True
                observation = None
                action_error = str(exc)

            steps_taken = step
            rewards.append(reward)
            log_step(
                step=step,
                action=_compact(action_dict),
                reward=reward,
                done=done,
                error=action_error,
            )

            if observation is not None:
                conversation.append({"role": "assistant", "content": _compact(action_dict)})
                conversation.append(
                    {
                        "role": "user",
                        "content": (
                            f"Feedback: {observation.feedback}. "
                            f"Score so far: {observation.score_so_far}."
                        ),
                    }
                )
                score = float(observation.score_so_far)

        score = min(max(score, 0.0), 1.0)
        success = bool(score >= SUCCESS_SCORE_THRESHOLD)
        return score
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    validate_runtime_config()

    client = _create_openai_client()
    for task_id in TASK_IDS:
        run_task(client=client, task_id=task_id, seed=SEED)


if __name__ == "__main__":
    main()