#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import ctypes
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run a CUDA solution against a problem_* testcase_config.py."
    )
    parser.add_argument("problem", help="Problem directory, for example problem_13")
    parser.add_argument("source", help="CUDA source file that exports extern \"C\" run_kernel")
    parser.add_argument("--case", default="all", help="1-based testcase id, or 'all'")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    parser.add_argument("--build-dir", default=".local_build", help="Directory for compiled shared libraries")
    parser.add_argument("--arch", default="sm_80", help="CUDA arch for nvcc, A800 is sm_80")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed testcase")
    parser.add_argument("--extra-nvcc", action="append", default=[], help="Extra nvcc flag, repeatable")
    parser.add_argument("--skip-compile", action="store_true", help="Reuse the expected .so path")
    parser.add_argument("--time-baseline", action="store_true", help="Also time testcase_config.py baseline()")
    return parser.parse_args()


def load_config(problem_dir: Path) -> Any:
    config_path = problem_dir / "testcase_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing testcase_config.py: {config_path}")
    spec = importlib.util.spec_from_file_location(f"{problem_dir.name}_testcase_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_with_input(func: Any, text: str) -> Any:
    old_input = builtins.input
    builtins.input = lambda prompt="": text
    try:
        return func()
    finally:
        builtins.input = old_input


def split_case_size(case_size: Any) -> tuple[list[tuple[int, ...]], tuple[int, int]]:
    if isinstance(case_size, tuple) and len(case_size) == 2:
        raw_sizes, timing = case_size
        return list(raw_sizes), tuple(timing)
    return list(case_size), (3, 30)


def clone_arg(arg: Any) -> Any:
    if torch.is_tensor(arg):
        return arg.clone()
    return arg


def clone_args(args: list[Any]) -> list[Any]:
    return [clone_arg(arg) for arg in args]


def compile_cuda(source: Path, output: Path, arch: str, extra_nvcc: list[str]) -> None:
    system_cuda_roots = [Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda")]
    full_cuda_root = next(
        (
            root
            for root in system_cuda_roots
            if (root / "bin" / "nvcc").exists()
            and (root / "targets" / "x86_64-linux" / "lib" / "libcudadevrt.a").exists()
        ),
        None,
    )
    nvcc = str(full_cuda_root / "bin" / "nvcc") if full_cuda_root else shutil.which("nvcc")
    if nvcc is None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidate = Path(conda_prefix) / "bin" / "nvcc"
            if candidate.exists():
                nvcc = str(candidate)
    if nvcc is None:
        raise RuntimeError("nvcc not found. Run this script inside the matris311 environment.")

    include_dirs: list[Path] = []
    lib_dirs: list[Path] = []
    if full_cuda_root:
        include_dirs.append(full_cuda_root / "include")
        include_dirs.append(full_cuda_root / "targets" / "x86_64-linux" / "include")
        lib_dirs.append(full_cuda_root / "lib64")
        lib_dirs.append(full_cuda_root / "targets" / "x86_64-linux" / "lib")
    else:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site_packages = Path(conda_prefix) / "lib" / py_ver / "site-packages"
            include_dirs.append(site_packages / "nvidia" / "cuda_runtime" / "include")
            include_dirs.append(site_packages / "nvidia" / "cuda_cccl" / "include")
            lib_dirs.append(Path(conda_prefix) / "lib")
            lib_dirs.append(site_packages / "nvidia" / "cuda_runtime" / "lib")

    include_flags = [f"-I{path}" for path in include_dirs if path.exists()]
    lib_flags = [f"-L{path}" for path in lib_dirs if path.exists()]
    rpath_flags = []
    for path in lib_dirs:
        if path.exists():
            rpath_flags.extend(["-Xlinker", f"-rpath,{path}"])

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-std=c++17",
        "--cudart=shared",
        f"-arch={arch}",
        *include_flags,
        str(source),
        "-o",
        str(output),
        *lib_flags,
        *rpath_flags,
        *extra_nvcc,
    ]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def so_path_for(build_dir: Path, problem_name: str, source: Path, arch: str) -> Path:
    digest = hashlib.sha1((str(source.resolve()) + arch).encode()).hexdigest()[:10]
    return build_dir / problem_name / f"{source.stem}_{digest}.so"


def c_arg(arg: Any) -> Any:
    if torch.is_tensor(arg):
        return ctypes.c_void_p(arg.data_ptr())
    if isinstance(arg, bool):
        return ctypes.c_int64(int(arg))
    if isinstance(arg, int):
        return ctypes.c_int64(arg)
    if isinstance(arg, float):
        return ctypes.c_double(arg)
    raise TypeError(f"Unsupported run_kernel argument type: {type(arg)!r}")


def c_argtype(arg: Any) -> Any:
    if torch.is_tensor(arg):
        return ctypes.c_void_p
    if isinstance(arg, bool):
        return ctypes.c_int64
    if isinstance(arg, int):
        return ctypes.c_int64
    if isinstance(arg, float):
        return ctypes.c_double
    raise TypeError(f"Unsupported run_kernel argument type: {type(arg)!r}")


def run_kernel(func: Any, args: list[Any]) -> None:
    func.argtypes = [c_argtype(arg) for arg in args]
    func.restype = None
    func(*[c_arg(arg) for arg in args])


def format_perf(avg_ms: float, workload: dict[str, Any] | None) -> str:
    parts = [f"{avg_ms:.4f} ms"]
    if workload:
        seconds = avg_ms * 1e-3
        flops = workload.get("flops")
        memory_bytes = workload.get("memory_bytes")
        if flops is not None and seconds > 0:
            parts.append(f"{float(flops) / seconds / 1e12:.3f} TFLOP/s")
        if memory_bytes is not None and seconds > 0:
            parts.append(f"{float(memory_bytes) / seconds / 1e9:.3f} GB/s")
        if workload.get("dtype"):
            parts.append(f"dtype={workload['dtype']}")
    return ", ".join(parts)


def time_baseline(config: Any, original_args: list[Any], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        args = clone_args(original_args)
        config.baseline(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        args = clone_args(original_args)
        config.baseline(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def run_case(config: Any, kernel: Any, case_id: int, device: str, should_time_baseline: bool) -> bool:
    case_size = call_with_input(config.getTestCaseSize, str(case_id))
    raw_sizes, (warmup, iters) = split_case_size(case_size)
    print(f"\n[case {case_id}] warmup={warmup} iters={iters}")
    print(f"[case {case_id}] sizes={raw_sizes}")

    original_args = config.genTestCase(raw_sizes, device=device)
    target_args = clone_args(original_args)
    baseline_args = clone_args(original_args)

    run_kernel(kernel, target_args)
    torch.cuda.synchronize()

    baseline_args = config.baseline(*baseline_args)
    torch.cuda.synchronize()

    ok = config.check(raw_sizes, original_args, target_args, baseline_args)
    print(f"[case {case_id}] correctness={'PASS' if ok else 'FAIL'}")
    if not ok:
        return False

    perf_args = clone_args(original_args)
    for _ in range(warmup):
        run_kernel(kernel, perf_args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_kernel(kernel, perf_args)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / max(iters, 1)
    workload = config.getWorkload(raw_sizes) if hasattr(config, "getWorkload") else None
    print(f"[case {case_id}] runtime={format_perf(avg_ms, workload)}")
    if should_time_baseline:
        baseline_ms = time_baseline(config, original_args, warmup, iters)
        speedup = baseline_ms / avg_ms if avg_ms > 0 else float("inf")
        print(f"[case {case_id}] torch_baseline={baseline_ms:.4f} ms, speedup={speedup:.3f}x")
    return True


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    problem_dir = (repo / args.problem).resolve()
    source = Path(args.source).resolve()
    build_dir = (repo / args.build_dir).resolve()

    if not problem_dir.exists():
        raise FileNotFoundError(f"Problem directory not found: {problem_dir}")
    if not source.exists():
        raise FileNotFoundError(f"CUDA source not found: {source}")

    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"
    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"[env] device={args.gpu} {torch.cuda.get_device_name(args.gpu)}")

    config = load_config(problem_dir)
    out_so = so_path_for(build_dir, problem_dir.name, source, args.arch)
    if not args.skip_compile:
        compile_cuda(source, out_so, args.arch, args.extra_nvcc)

    lib = ctypes.CDLL(str(out_so))
    kernel = lib.run_kernel

    num_cases = int(config.getNumOfTestcases())
    if args.case == "all":
        cases = list(range(1, num_cases + 1))
    else:
        cases = [int(args.case)]

    all_ok = True
    for case_id in cases:
        try:
            ok = run_case(config, kernel, case_id, device, args.time_baseline)
        except Exception as exc:
            ok = False
            print(f"[case {case_id}] ERROR: {exc}", file=sys.stderr)
        all_ok = all_ok and ok
        if not ok and not args.keep_going:
            break

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
