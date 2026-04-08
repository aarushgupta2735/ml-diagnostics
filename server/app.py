"""Root server shim for validator compatibility.

Delegates to the actual ml_diagnostics environment app implementation.
"""

from envs.ml_diagnostics.server.app import app


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()