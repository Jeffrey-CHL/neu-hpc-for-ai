import modal
import subprocess
import os

# Define the base image with CUDA and add local week_02 directory
image = (
    modal.Image.debian_slim()
    .apt_install("python3", "python3-pip", "build-essential", "wget", "gnupg")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-2 cuda-compiler-12-2",
        "rm cuda-keyring_1.1-1_all.deb",
    )
    # ✅ 挂载 week_02 文件夹
    .add_local_dir(".", "/root/week_02")
)

# Define the Modal app
app = modal.App(name="benchmark-app")

@app.function(image=image, gpu="any")
def benchmark():
    # Add nvcc to PATH
    os.environ["PATH"] += ":/usr/local/cuda-12.2/bin"

    # Switch to mounted folder
    os.chdir("/root/week_02")

    print(">>> Compiling gemm.cu with nvcc...")
    subprocess.run(["nvcc", "gemm.cu", "-o", "gemm"], check=True)

    print(">>> Running benchmark...")
    result = subprocess.run(
        ["./gemm"], capture_output=True, text=True, check=True
    )

    print(result.stdout)
