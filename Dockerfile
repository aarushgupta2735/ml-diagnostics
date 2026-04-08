FROM python:3.11-slim
WORKDIR /app
COPY . .
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m pip install --upgrade pip && \
	pip install --no-cache-dir --retries 10 --timeout 180 -e .
EXPOSE 7860
CMD ["uvicorn", "envs.ml_diagnostics.server.app:app", "--host", "0.0.0.0", "--port", "7860"]