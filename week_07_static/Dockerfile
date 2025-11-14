FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 安装 system Python，让 Modal 内置的 `python -m pip ...` 步骤能跑通
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip openmpi-bin libopenmpi-dev make && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# 这里的工作目录只是个占位，真正代码是通过挂载进来的 /root/project
WORKDIR /root/project