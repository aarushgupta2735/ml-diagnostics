---
title: ML Diagnostics
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
short_description: 'RL environment for diagnosing ML training failures'
---

## Hackathon Submission

### Required Environment Variables

- API_BASE_URL: OpenAI-compatible API endpoint.
- MODEL_NAME: Model identifier used for chat completions.
- HF_TOKEN: API token (or API_KEY as fallback).
- LOCAL_IMAGE_NAME: Optional local Docker image name when using from_docker_image mode.
- ENV_BASE_URL: Optional running env URL for local server mode (default: http://localhost:7860).

For local development, copy `.env.example` to `.env` and fill your values.
The `.env` file is gitignored and should never be committed.

If a token was ever pasted in a terminal or remote URL, revoke it in Hugging Face settings and create a fresh token.

### Inference Entrypoint

Run the mandatory root script:

```bash
python inference.py
```

The script emits strict stdout records for every task episode:

- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

### Pre-Submission Validation

Use the validator wrapper before submitting:

```bash
./scripts/validate-submission.sh <space_url> .
```

It performs three fail-fast checks:

1. Space URL responds 200 on POST /reset.
2. Docker build succeeds.
3. openenv validate succeeds.

Validated command used for this project:

```bash
DOCKER_BUILD_TIMEOUT=1800 ./scripts/validate-submission.sh https://aarushgupta2735-ml-diagnostics.hf.space .
```

### Docker Permission Note (Linux)

If Docker fails with permission denied on `/var/run/docker.sock`, refresh shell group context:

```bash
newgrp docker
docker ps
```

### GitHub + Hugging Face Submission

This hackathon requires both links:

1. GitHub repository URL
2. Hugging Face Space URL

Recommended remote setup:

```bash
git remote set-url origin https://github.com/aarushgupta2735/ml-diagnostics.git
git remote set-url hf https://huggingface.co/spaces/aarushgupta2735/ml_diagnostics
git branch -M main
git push -u origin main
git push hf main
```

### If huggingface-cli Is Not Found

Install Hugging Face Hub in your environment and log in from Python:

```bash
/home/aarush/miniconda3/envs/openenv_course/bin/python -m pip install -U huggingface_hub
/home/aarush/miniconda3/envs/openenv_course/bin/python -c "from huggingface_hub import login; login()"
```

Then retry:

```bash
git push hf main
```