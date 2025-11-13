import os
import subprocess

import modal

app = modal.App("week4-flashattention")

# Use an NVIDIA CUDA image that already has nvcc (devel variant)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("build-essential")
    # 👇 VERY IMPORTANT: include the whole current folder into the image
    .add_local_dir(".", "/root/project")
)


@app.function(image=image, gpu="a10g", timeout=600)
def run():
    # Switch into the project directory inside the container
    os.chdir("/root/project")
    print("CWD inside container:", os.getcwd())
    print("Files in CWD:", os.listdir("."))

    # Compile the CUDA source file
    print("Compiling with nvcc...")
    subprocess.check_call(
        ["nvcc", "-O3", "-std=c++17", "flash_attention.cu", "-o", "flash_attention"]
    )

    # Run the compiled binary
    print("Running FlashAttention...")
    subprocess.check_call(["./flash_attention"])