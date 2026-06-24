from __future__ import annotations

import argparse
import copy
import importlib.util
import pathlib
import statistics
import sys

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


def clone_arg(arg):
    return arg.clone() if torch.is_tensor(arg) else copy.deepcopy(arg)


def sync():
    torch.cuda.synchronize()


def time_config(fn, warmup: int, iters: int):
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
    return statistics.fmean(samples), min(samples), statistics.pstdev(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=128)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--iters-scale", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_module(CASE_CONFIG, "scc_a_testcase_config_measure")
    solution = load_module(SOLUTION, "scc_a_solution_measure")
    print("device", torch.cuda.get_device_name())
    print(f"config bm={args.bm} bn={args.bn} bk={args.bk} warps={args.warps} stages={args.stages}")

    for case_id, (M, K, N, warmup, iters) in enumerate(cfg.TESTCASES, 1):
        shapes = [(M, K), (K, N), (K, N), (N,), (N,), (M, N), (), (), ()]
        base_args = cfg.genTestCase(shapes, device="cuda")
        ref_args = [clone_arg(a) for a in base_args]
        target_args = [clone_arg(a) for a in base_args]
        cfg.baseline(*ref_args)
        solution.run_kernel_config(
            *target_args,
            args.bm,
            args.bn,
            args.bk,
            args.warps,
            args.stages,
        )
        sync()
        ok = cfg.check(shapes, base_args, target_args, ref_args)
        run_iters = max(5, int(iters * args.iters_scale))
        mean_ms, min_ms, stdev_ms = time_config(
            lambda: solution.run_kernel_config(
                *target_args,
                args.bm,
                args.bn,
                args.bk,
                args.warps,
                args.stages,
            ),
            min(warmup, 3),
            run_iters,
        )
        workload = cfg.getWorkload((shapes, (warmup, iters)))
        tflops = int(workload["flops"]) / (mean_ms / 1000.0) / 1e12
        print(
            f"CASE {case_id}: M={M} K={K} N={N} correct={ok} "
            f"mean={mean_ms:.4f}ms min={min_ms:.4f}ms stdev={stdev_ms:.4f}ms tflops={tflops:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
