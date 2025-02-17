FROM python:3.10-slim
# FROM nvidia/cuda:11.8.0-base-ubuntu20.04 
# ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies (Worker Template)

RUN pip install transformers -U

RUN pip install git+https://github.com/huggingface/diffusers.git
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# RUN pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu118

RUN pip install protobuf sentencepiece bitsandbytes==0.45.2 accelerate==1.3.0 safetensors runpod
# Cache Models
COPY cache.py /cache.py
RUN python /cache.py && \
    rm /cache.py

# Add src files (Worker Template)
ADD . .

CMD python -u /main.py