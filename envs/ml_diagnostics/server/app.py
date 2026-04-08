# envs/ml_diagnostics/server/app.py

from openenv.core.env_server import create_app

try:
	from envs.ml_diagnostics.models import MLAction, MLObservation
	from envs.ml_diagnostics.server.ml_diagnostics import MLDiagnosticsEnvironment
except ModuleNotFoundError:
	# Support running from env-local layout where imports are package-relative.
	from ..models import MLAction, MLObservation
	from .ml_diagnostics import MLDiagnosticsEnvironment


app = create_app(MLDiagnosticsEnvironment, MLAction, MLObservation)


def main() -> None:
	"""Run the environment server for local validation/deployment."""
	import uvicorn

	uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
	main()