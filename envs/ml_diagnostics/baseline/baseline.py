# envs/ml_diagnostics/baseline/baseline.py

import os
import json
from envs.ml_diagnostics.client import MLDiagnosticsEnv
from envs.ml_diagnostics.models import MLAction

# replace imports
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)


SYSTEM_PROMPT = """You are an expert ML engineer diagnosing training failures.
You will be shown training configs, loss curves, and other diagnostics.
You must identify issues and suggest fixes.

You interact with the environment through these action types:

1. request_data — ask for more diagnostic information
   payload: {"data_type": "grad_norms" | "data_distribution" | "feature_stats" | "full_config"}

2. submit_diagnosis — identify the issue(s)
   For task 1: payload: {"issue": "<issue_name>", "reasoning": "<why>"}
   For task 2: payload: {"diagnosis": "overfitting", "root_causes": ["<cause1>", ...]}
   For task 3: payload: {"issues": [{"issue": "<name>", "severity": "critical|high|medium", "fix": "<fix>", "priority": 1}], "interaction_note": "<how issues interact>"}

3. submit_fix — provide the fix
   For task 1: payload: {"fix_key": "<param>", "suggested_value": <value>}
   For task 2: payload: {"fixes": [{"action": "<fix_action>", "value": <value>, "priority": 1}]}

Valid issue names for task 1: learning_rate_too_high, learning_rate_too_low, batch_size_too_large, wrong_optimizer
Valid root causes for task 2: model_too_large, no_regularization, insufficient_data, no_augmentation, too_many_epochs
Valid fix actions for task 2: add_dropout, reduce_layers, add_weight_decay, add_augmentation, collect_more_data, early_stopping
Valid issues for task 3: class_imbalance, wrong_loss_function, gradient_explosion, vanishing_gradients, data_leakage, incorrect_weight_initialization

Respond ONLY with a JSON object with keys: action_type, payload. Nothing else."""


def get_llm_action(conversation: list) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def run_task(task_id: int, seed: int = 42) -> float:
    """Run one full task episode with the LLM agent."""
    print(f"\n{'='*50}")
    print(f"TASK {task_id}")
    print('='*50)

    with MLDiagnosticsEnv(base_url="http://localhost:8003").sync() as env:
        obs = env.reset(task_id=task_id, seed=seed)
        context = obs.observation.training_context
        feedback = obs.observation.feedback

        print(f"Context: {json.dumps(context, indent=2)}")

        # build conversation history
        conversation = [
            {"role": "user", "content": (
                f"Task {task_id} started.\n"
                f"Training context:\n{json.dumps(context, indent=2)}\n\n"
                f"Available steps: {obs.observation.available_actions}\n"
                f"Feedback: {feedback}"
            )},
        ]

        step = 0
        while not obs.done:
            # get LLM action
            try:
                action_dict = get_llm_action(conversation)
                print(f"\nStep {step+1} | LLM action: {json.dumps(action_dict)}")
            except Exception as e:
                print(f"LLM parse error: {e}")
                break

            # execute action
            action = MLAction(
                action_type=action_dict.get("action_type", "submit_diagnosis"),
                payload=action_dict.get("payload", {}),
            )
            obs = env.step(action)

            print(f"Reward: {obs.observation.reward:.3f} | Feedback: {obs.observation.feedback}")

            # add to conversation history
            conversation.append({
                "role": "assistant",
                "content": json.dumps(action_dict),
            })
            conversation.append({
                "role": "user",
                "content": (
                    f"Feedback: {obs.observation.feedback}\n"
                    f"Score so far: {obs.observation.score_so_far}\n"
                    f"Step: {obs.observation.step}/{obs.observation.available_actions}\n"
                    f"Additional data: {json.dumps(obs.observation.training_context)}"
                    if not obs.done else
                    f"Episode complete. Final score: {obs.observation.score_so_far}"
                ),
            })

            step += 1

        final_score = obs.observation.score_so_far
        print(f"\nFinal score: {final_score}")
        return final_score


def main():
    scores = {}
    for task_id in [1, 2, 3]:
        scores[f"task_{task_id}"] = run_task(task_id=task_id, seed=42)

    print(f"\n{'='*50}")
    print("BASELINE RESULTS")
    print('='*50)
    for task, score in scores.items():
        print(f"{task}: {score:.3f}")
    print(f"Average: {sum(scores.values()) / len(scores):.3f}")


if __name__ == "__main__":
    main()