FROM nvcr.io/nvidia/vllm:25.09-py3 AS base

ENV HF_HOME=/data/shared/VibeVoice/models

RUN mkdir -p /data/build

WORKDIR /data/build

COPY ./vibevoice ./vibevoice
COPY ./pyproject.toml ./pyproject.toml

RUN pip install .

RUN mkdir -p /data/shared/VibeVoice
WORKDIR /data/shared/VibeVoice
RUN rm -rf /data/build

CMD ["bash"]
