ARG PYTHON_VERSION
ARG DEBIAN_VERSION
ARG JUPYTER_TOKEN
ARG TAG=${PYTHON_VERSION}-${DEBIAN_VERSION}
FROM python:${TAG}

# Set the labels
LABEL maintainer="ipeterphan@gmail.com"
LABEL version="1.0"
LABEL description="Docker Image for PyTorch Development."

USER root

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    acl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Copy dependencies
COPY ./requirements.txt /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8888

CMD jupyter notebook --port=8888 --allow-root --ip=* --NotebookApp.token=${JUPYTER_TOKEN}
