FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Simulate downloading a model artifact from MLflow for this run.
RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["sh", "-c", "echo Serving model for Run ID: ${RUN_ID}"]
