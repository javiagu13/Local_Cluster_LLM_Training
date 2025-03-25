FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv git wget vim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Install DeepSpeed separately to ensure correct CUDA compatibility
RUN pip install deepspeed==0.15.4

COPY . /app

# Ensure datasets are mounted or copied under /app/data before running
CMD ["bash", "run_training.sh"]
