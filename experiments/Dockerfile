ARG BASE_IMAGE=choice-api:dev
FROM ${BASE_IMAGE}
USER root
RUN /usr/local/bin/python -m pip install --no-cache-dir --no-deps \
    --target=/opt/probe-deps bitsandbytes==0.50.2
ENV PYTHONPATH=/opt/probe-deps:/app
