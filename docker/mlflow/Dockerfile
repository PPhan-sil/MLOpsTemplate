ARG PYTHON_VERSION
ARG DEBIAN_VERSION
ARG TAG=${PYTHON_VERSION}-${DEBIAN_VERSION}
FROM python:${TAG}

# Set the labels
LABEL maintainer="ipeterphan@gmail.com"
LABEL version="1.0"
LABEL description="MLFlow Docker Image for Development"

USER root

# MLflow version
ARG MLFLOW_VERSION

# Set the MLflow server port
ARG MLFLOW_SERVER_PORT
ENV MLFLOW_SERVER_PORT=${MLFLOW_SERVER_PORT}

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        curl \
        git \
        vim \
        telnet \
        wget && \
    apt-get clean \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip
RUN pip install mlflow==$MLFLOW_VERSION

# Expose the MLflow port
EXPOSE $MLFLOW_SERVER_PORT

CMD mlflow server \
  --backend-store-uri "$MLFLOW_BACKEND_STORE" \
  --default-artifact-root "$MLFLOW_ARTIFACT_STORE" \
  --host 0.0.0.0 \
  --port "$MLFLOW_SERVER_PORT"
