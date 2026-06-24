from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load_config(problem_dir: str):
    path = ROOT / problem_dir / "testcase_config.py"
    spec = importlib.util.spec_from_file_location(f"{problem_dir}_testcase_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clone_arg(arg):
    if torch.is_tensor(arg):
        return arg.clone()
    return arg


def _verify_b(case_id: int) -> bool:
    cfg = _load_config("USTB-SCC-B-online-softmax-gemm")
    B, H, S, D, _, _ = cfg.TESTCASES[case_id - 1]

    args = cfg.genTestCase((B, H, S, D), device="cuda")
    so_args = [_clone_arg(arg) for arg in args]
    torch_args = [_clone_arg(arg) for arg in args]

    cfg.baseline(*so_args)
    torch.cuda.synchronize()
    cfg._torch_reference(*torch_args)
    torch.cuda.synchronize()

    ok = cfg.check((B, H, S, D), args, so_args, torch_args)
    print(f"B case {case_id}: correct={ok}")
    return ok


def _verify_c(case_id: int) -> bool:
    cfg = _load_config("USTB-SCC-C-mixed-precision-grouped-gemm")
    cfg.CURRENT_CASE = cfg.TESTCASES[case_id - 1]
    M_total, K, N, num_groups, *_ = cfg.CURRENT_CASE

    args = cfg.genTestCase(None, device="cuda")
    so_args = [_clone_arg(arg) for arg in args]
    torch_args = [_clone_arg(arg) for arg in args]

    cfg.baseline(*so_args)
    torch.cuda.synchronize()
    cfg._torch_reference(*torch_args)
    torch.cuda.synchronize()

    ok = cfg.check((M_total, K, N, num_groups), args, so_args, torch_args)
    print(f"C case {case_id}: correct={ok}")
    return ok


def main() -> int:
    cases = sys.argv[1:] or ["all"]
    if cases == ["all"]:
        b_cases = range(1, len(_load_config("USTB-SCC-B-online-softmax-gemm").TESTCASES) + 1)
        c_cases = range(1, len(_load_config("USTB-SCC-C-mixed-precision-grouped-gemm").TESTCASES) + 1)
    else:
        parsed = [int(case) for case in cases]
        b_cases = parsed
        c_cases = parsed

    ok = True
    for case_id in b_cases:
        ok = _verify_b(case_id) and ok
    for case_id in c_cases:
        ok = _verify_c(case_id) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
