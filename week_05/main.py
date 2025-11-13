from modal import App, Image

app = App("week5-flashattn2")

# CUDA 开发镜像：带 nvcc 和 CUDA headers
cuda_image = (
    Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",          # 加一个带 pip 的 Python 进去
    )
    # 先从默认 PyPI 安装编译工具
    .pip_install(
        "ninja",
        "setuptools",
    )
    # 再从 PyTorch CUDA 源安装 torch / torchvision / torchaudio
    .pip_install(
        "torch==2.3.0+cu121",
        "torchvision==0.18.0+cu121",
        "torchaudio==2.3.0+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # 把当前目录 week_05 打包进镜像 /root/app 里
    .add_local_dir(".", remote_path="/root/app")
)

@app.function(
    image=cuda_image,
    gpu="H100",        # 你已经在用 H100，可以继续用
    timeout=600,
    env={
        # 保险起见，指明 CUDA 路径（某些场景可省略）
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda",
    },
)
def run_test():
    import os, sys

    # 确保 Python 能找到我们的代码
    sys.path.insert(0, "/root/app")
    os.chdir("/root/app")

    print("Current working directory:", os.getcwd())
    print("Python sys.path[0..4]:", sys.path[:5])
    print("CUDA_HOME:", os.environ.get("CUDA_HOME"))

    import test_flash_attn2

    print("=== Running forward check ===")
    test_flash_attn2.check_forward()

    print("=== Running backward check ===")
    test_flash_attn2.check_backward()

    print("=== Running benchmark ===")
    test_flash_attn2.bench()


if __name__ == "__main__":
    run_test.remote()