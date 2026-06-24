from __future__ import annotations

import argparse
import importlib.util
import pathlib
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    cfg = load_module(CASE_CONFIG, "scc_a_testcase_config_ncu")
    solution = load_module(SOLUTION, "scc_a_solution_ncu")
    M, K, N, _, _ = cfg.TESTCASES[args.case - 1]
    shapes = [(M, K), (K, N), (K, N), (N,), (N,), (M, N), (), (), ()]
    kernel_args = cfg.genTestCase(shapes, device="cuda")

    for _ in range(2):
        solution.run_kernel(*kernel_args)
    torch.cuda.synchronize()

    for _ in range(args.repeats):
        solution.run_kernel(*kernel_args)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
