from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
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
    return statistics.fmean(samples), min(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--mode", choices=["default", "case4"], default="default")
    args = parser.parse_args()

    cfg = load_module(CASE_CONFIG, "scc_a_testcase_config_sweep")
    solution = load_module(SOLUTION, "scc_a_solution_sweep")
    M, K, N, _, _ = cfg.TESTCASES[args.case - 1]
    shapes = [(M, K), (K, N), (K, N), (N,), (N,), (M, N), (), (), ()]
    base_args = cfg.genTestCase(shapes, device="cuda")
    ref_args = [clone_arg(a) for a in base_args]
    cfg.baseline(*ref_args)

    configs = [
        (16, 16, 32, 4, 3),
        (16, 32, 32, 4, 3),
        (32, 32, 32, 4, 3),
        (32, 64, 32, 4, 3),
        (32, 64, 64, 4, 3),
        (32, 128, 32, 4, 3),
        (32, 128, 64, 4, 3),
        (32, 256, 32, 4, 3),
        (32, 256, 64, 4, 3),
        (64, 64, 32, 4, 3),
        (64, 64, 64, 4, 3),
        (64, 128, 32, 4, 3),
        (64, 128, 64, 4, 3),
        (64, 128, 64, 8, 3),
        (64, 128, 64, 8, 4),
        (64, 256, 32, 4, 3),
        (64, 256, 64, 4, 3),
        (64, 256, 64, 8, 3),
        (128, 64, 32, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 128, 32, 4, 3),
        (128, 128, 64, 4, 3),
        (128, 128, 64, 8, 3),
    ]
    if args.mode == "case4":
        configs = [
            (16, 64, 32, 4, 3),
            (16, 128, 32, 4, 3),
            (16, 256, 32, 4, 3),
            (32, 96, 32, 4, 3),
            (32, 128, 16, 4, 3),
            (32, 128, 32, 4, 3),
            (32, 160, 32, 4, 3),
            (32, 192, 32, 4, 3),
            (64, 96, 32, 4, 3),
            (64, 128, 16, 4, 3),
            (64, 128, 32, 4, 3),
            (64, 160, 32, 4, 3),
            (64, 192, 32, 4, 3),
            (96, 64, 32, 4, 3),
            (96, 96, 32, 4, 3),
            (96, 128, 32, 4, 3),
            (128, 32, 32, 4, 3),
            (128, 48, 32, 4, 3),
            (128, 64, 32, 4, 3),
            (128, 96, 32, 4, 3),
        ]
    workload = cfg.getWorkload((shapes, (0, args.iters)))
    flops = int(workload["flops"])

    rows = []
    for bm, bn, bk, nw, ns in configs:
        kernel_args = [clone_arg(a) for a in base_args]
        try:
            solution.run_kernel_config(*kernel_args, bm, bn, bk, nw, ns)
            sync()
            ok = cfg.check(shapes, base_args, kernel_args, ref_args)
            if not ok:
                print(f"BAD bm={bm} bn={bn} bk={bk} nw={nw} ns={ns}", flush=True)
                continue
            mean_ms, min_ms = time_config(
                lambda: solution.run_kernel_config(*kernel_args, bm, bn, bk, nw, ns),
                3,
                args.iters,
            )
            tflops = flops / (mean_ms / 1000.0) / 1e12
            rows.append((mean_ms, min_ms, tflops, bm, bn, bk, nw, ns))
            print(
                f"CONFIG bm={bm} bn={bn} bk={bk} nw={nw} ns={ns} "
                f"mean={mean_ms:.4f}ms min={min_ms:.4f}ms tflops={tflops:.2f}",
                flush=True,
            )
        except Exception as exc:
            print(f"ERR bm={bm} bn={bn} bk={bk} nw={nw} ns={ns}: {exc!r}", flush=True)
    if rows:
        best = min(rows, key=lambda x: x[0])
        print(
            "BEST mean={:.4f}ms min={:.4f}ms tflops={:.2f} bm={} bn={} bk={} nw={} ns={}".format(
                *best
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
