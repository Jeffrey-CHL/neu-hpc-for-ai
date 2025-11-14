import os
import pathlib
import subprocess

import modal

# 这里用 Stub 还是可以的，只是有 deprecation warning，不影响作业
stub = modal.Stub("week07-dist-flashattn")

# 把本地整个 week_07_static 目录挂载到容器里的 /root/project
project_mount = modal.Mount.from_local_dir(
    ".",  # 本地当前目录
    remote_path="/root/project",
)

# 用我们自己写的 Dockerfile（已经安装了 python3 + openmpi + nvcc）
base_image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_mount=project_mount,
)


@stub.function(
    image=base_image,
    mounts=[project_mount],
    gpu="A100",
    timeout=600,
)
def run():
    base_dir = pathlib.Path("/root/project")
    os.chdir(base_dir)

    print("== base_dir ==", base_dir)
    print("== Files in project ==", os.listdir())

    # 先清理旧的可执行文件
    subprocess.call(["rm", "-rf", "bin"])
    print("🔧 Building...")

    # 编译 CUDA + MPI 程序（用 Makefile）
    subprocess.check_call(["make"])

    print("🚀 Running distributed FlashAttention...")

    # 允许以 root 身份跑 mpirun（Modal 容器默认是 root）
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"

    subprocess.check_call(
        [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            "4",
            "./bin/flash_attn",
        ],
        env=env,
    )


if __name__ == "__main__":
    with stub.run():
        run.remote()