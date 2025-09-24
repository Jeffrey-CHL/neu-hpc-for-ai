import modal

# Build an image with CUDA and include gemm.cu inside
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install("build-essential")
    .add_local_file("gemm.cu", "/root/gemm.cu")   # ✅ 新写法：把 gemm.cu 打进镜像
)

app = modal.App("week03-gemm")

@app.function(image=image, gpu="A10G", timeout=60*30)
def run():
    import os, subprocess

    os.chdir("/root")

    # Compile
    print("== Compiling gemm.cu with nvcc ==")
    subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "gemm.cu", "-o", "gemm"],
        check=True,
    )

    # Benchmark configs
    M = N = K = 1024
    alpha, beta = "1.0", "1.0"
    tiles = [16, 32]
    modes = ["naive", "tiled"]

    results = []

    for mode in modes:
        for tile in tiles:
            print(f"\n== Running mode={mode}, TILE={tile} ==")
            proc = subprocess.run(
                [
                    "./gemm",
                    str(M), str(N), str(K),
                    "--mode", mode,
                    "--ta", "N", "--tb", "N",
                    "--alpha", alpha, "--beta", beta,
                    "--tile", str(tile),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            out = proc.stdout
            print(out)

            maxerr, rmserr = "NA", "NA"
            for line in out.splitlines():
                if line.startswith("Verify:"):
                    parts = line.split()
                    maxerr = parts[1].split("=")[1]
                    rmserr = parts[2].split("=")[1]

            results.append((mode, tile, M, maxerr, rmserr))

    # Print Markdown table
    print("\n### Auto-collected Results")
    print("| Mode   | TILE | Matrix Size (M=N=K) | Max Abs Error | RMS Error |")
    print("|--------|------|----------------------|---------------|-----------|")
    for mode, tile, size, maxerr, rmserr in results:
        print(f"| {mode} | {tile} | {size} | {maxerr} | {rmserr} |")
