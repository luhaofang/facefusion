# 使用带有 CUDA 和 cuDNN 的官方基础镜像
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置环境变量
ARG FACEFUSION_VERSION=3.1.2
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/dist-packages/tensorrt_libs

# 创建工作目录
WORKDIR /facefusion

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y \
    python3.12 \
    python-is-python3 \
    pip \
    git \
    curl \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 安装 TensorRT
RUN pip install tensorrt==10.8.0.43 --extra-index-url https://pypi.nvidia.com

# 克隆 FaceFusion 仓库
RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch .

# 安装 FaceFusion
RUN python install.py --onnxruntime cuda --skip-conda

# 创建必要的卷目录
RUN mkdir -p /facefusion/.assets && \
    mkdir -p /facefusion/.caches && \
    mkdir -p /facefusion/.jobs

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "facefusion.py", "run", "--execution-providers", "tensorrt"]
