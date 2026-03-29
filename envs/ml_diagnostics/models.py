# envs/ml_diagnostics/models.py

from typing import List, Optional, Dict, Any
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


# --- Action ---

class MLAction(Action):
    action_type: str  # "submit_diagnosis" | "request_data" | "submit_fix"
    payload: Dict[str, Any] = Field(default_factory=dict)
    # Examples:
    # action_type="request_data", payload={"data_type": "grad_norms"}
    # action_type="submit_diagnosis", payload={"issue": "learning_rate_too_high", ...}
    # action_type="submit_fix", payload={"fixes": [...]}


# --- Observation ---

class MLObservation(Observation):
    task_id: int                          # 1, 2, or 3
    step: int                             # current step number
    training_context: Dict[str, Any]      # config, curves, data stats visible so far
    feedback: str                         # natural language feedback on last action
    score_so_far: float                   # cumulative score this episode
    available_actions: List[str]          # what action_types are valid right now
    message: str


# --- State ---

class MLState(State):
    task_id: int = 1
    issues_found: List[str] = Field(default_factory=list)
    issues_missed: List[str] = Field(default_factory=list)
    fixes_submitted: List[Dict[str, Any]] = Field(default_factory=list)
    data_requested: List[str] = Field(default_factory=list)
    score: float = 0.0
    done: bool = False
    max_steps: int = 10