# envs/ml_diagnostics/server/app.py

from openenv.core.env_server import create_app
from envs.ml_diagnostics.server.ml_diagnostics import MLDiagnosticsEnvironment
from envs.ml_diagnostics.models import MLAction, MLObservation

app = create_app(MLDiagnosticsEnvironment, MLAction, MLObservation)