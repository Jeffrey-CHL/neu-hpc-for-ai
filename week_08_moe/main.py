# main.py
# Modal entrypoint to run the Week 8 DeepSeek-style MoE CUDA assignment
# on a GPU-enabled container.

import modal

app = modal.App("week08-deepseek-moe")

# Use NVIDIA's official CUDA image as base and ask Modal to add Python 3.11.
# This way:
#   - CUDA toolkit (nvcc) comes from the base image.
#   - Python is injected by Modal (so "python -m pip ..." works).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",   # IMPORTANT: string version, NOT True
    )
    .apt_install(
        "build-essential",
        "make",
    )
    # Copy the local Week 8 directory into the image
    .add_local_dir(".", remote_path="/root/week_08_moe")
)


@app.function(
    image=image,
    gpu="A10G",   # or any GPU type your course allows
    timeout=600,
)
def run_moe():
    """Build and run the CUDA MoE example inside a GPU container."""
    import os
    import subprocess

    workdir = "/root/week_08_moe"
    os.chdir(workdir)

    print("[Modal] Current working directory:", os.getcwd())
    print("[Modal] Listing files:")
    subprocess.check_call(["ls", "-la"])

    print("[Modal] Building deepseek_moe with make...")
    subprocess.check_call(["make"])

    print("[Modal] Running ./deepseek_moe ...")
    subprocess.check_call(["./deepseek_moe"])


# Run with:
#   modal run main.py::run_moe
if __name__ == "__main__":
    app.cli()