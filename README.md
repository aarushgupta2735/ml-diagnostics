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