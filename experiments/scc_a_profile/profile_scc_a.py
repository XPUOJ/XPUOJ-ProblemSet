from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
CASE_CONFIG = ROOT / "USTB-SCC-A-fused-swiglu-up-projection" / "testcase_config.py"
SOLUTION = pathlib.Path(__file__).resolve().parent / "solution.py"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cfg = load_module(CASE_CONFIG, "scc_a_testcase_config")
solution = load_module(SOLUTION, "scc_a_solution")


def clone_arg(arg):
    if torch.is_tensor(arg):
        return arg.clone()
    return copy.deepcopy(arg)


def case_shapes(case_id: int):
    M, K, N, warmup, iters = cfg.TESTCASES[case_id - 1]
    shapes = [
        (M, K),
        (K, N),
        (K, N),
        (N,),
        (N,),
        (M, N),
        (),
        (),
        (),
    ]
    return M, K, N, warmup, iters, shapes


def sync():
    torch.cuda.synchronize()


def time_ms(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        sync()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": samples,
    }


def run_case(case_id: int, check_baseline: bool, profile_baseline: bool):
    M, K, N, warmup, iters, shapes = case_shapes(case_id)
    args = cfg.genTestCase(shapes, device="cuda")

    # Compile once.
    solution.run_kernel(*args)
    sync()

    baseline_stats = None
    baseline_args = None
    if check_baseline or profile_baseline:
        baseline_args = [clone_arg(a) for a in args]
        cfg.baseline(*baseline_args)
        sync()

    correct = None
    if check_baseline:
        original_args = [clone_arg(a) for a in args]
        target_args = [clone_arg(a) for a in args]
        reference_args = [clone_arg(a) for a in args]
        cfg.baseline(*reference_args)
        solution.run_kernel(*target_args)
        sync()
        correct = bool(cfg.check(shapes, original_args, target_args, reference_args))

    stats = time_ms(lambda: solution.run_kernel(*args), warmup, iters)

    if profile_baseline:
        assert baseline_args is not None
        # The PyTorch reference is much slower and allocates temporaries; use fewer
        # timing iterations to keep profiling practical.
        baseline_stats = time_ms(lambda: cfg.baseline(*baseline_args), min(warmup, 2), min(iters, 5))

    workload = cfg.getWorkload((shapes, (warmup, iters)))
    flops = int(workload["flops"])
    memory_bytes = int(workload["memory_bytes"])
    result = {
        "case": case_id,
        "M": M,
        "K": K,
        "N": N,
        "warmup": warmup,
        "iters": iters,
        "correct": correct,
        "solution": stats,
        "solution_tflops_mean": flops / (stats["mean_ms"] / 1000.0) / 1e12,
        "solution_tflops_min": flops / (stats["min_ms"] / 1000.0) / 1e12,
        "estimated_io_gb": memory_bytes / 1e9,
        "arithmetic_intensity_flop_per_byte": flops / max(memory_bytes, 1),
    }
    if baseline_stats is not None:
        result["pytorch_baseline"] = baseline_stats
        result["speedup_vs_pytorch_mean"] = baseline_stats["mean_ms"] / stats["mean_ms"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=0, help="1-based case id; 0 means all")
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--profile-baseline", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("highest")
    torch.cuda.empty_cache()

    print("device", torch.cuda.get_device_name())
    print("torch", torch.__version__, "cuda", torch.version.cuda)

    case_ids = range(1, len(cfg.TESTCASES) + 1) if args.case == 0 else [args.case]
    results = []
    for case_id in case_ids:
        t0 = time.time()
        result = run_case(case_id, args.check, args.profile_baseline)
        result["wall_s"] = time.time() - t0
        results.append(result)
        print(
            "CASE {case}: M={M} K={K} N={N} correct={correct} "
            "mean={mean:.4f}ms min={min:.4f}ms tflops_mean={tf:.2f} tflops_min={tfmin:.2f} "
            "AI={ai:.1f}".format(
                case=result["case"],
                M=result["M"],
                K=result["K"],
                N=result["N"],
                correct=result["correct"],
                mean=result["solution"]["mean_ms"],
                min=result["solution"]["min_ms"],
                tf=result["solution_tflops_mean"],
                tfmin=result["solution_tflops_min"],
                ai=result["arithmetic_intensity_flop_per_byte"],
            ),
            flush=True,
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
