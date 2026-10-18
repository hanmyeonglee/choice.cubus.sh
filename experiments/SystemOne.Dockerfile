ARG BASE_IMAGE=choice-api:dev
FROM ${BASE_IMAGE}
USER root
RUN /usr/local/bin/python -m pip install --no-cache-dir --no-deps \
    --target=/opt/system-one-deps \
    requests==2.34.2 urllib3==2.8.0 charset-normalizer==3.5.2
ENV PYTHONPATH=/opt/system-one-deps:/probe/models/system-one/sources/laya:/probe/models/system-one/sources/gliner2:/app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 LAYA_CUDA_AMP=fp16
