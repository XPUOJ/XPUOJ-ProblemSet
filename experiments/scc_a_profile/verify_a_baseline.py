from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
CASE_CONFIG = ROOT / "USTB-SCC-A-fused-swiglu-up-projection" / "testcase_config.py"


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


def main():
    cfg = load_module(CASE_CONFIG, "scc_a_testcase_config_verify")
    print("device", torch.cuda.get_device_name())
    ok_all = True
    for case_id, (M, K, N, _, _) in enumerate(cfg.TESTCASES, 1):
        shapes = [(M, K), (K, N), (K, N), (N,), (N,), (M, N), (), (), ()]
        original = cfg.genTestCase(shapes, device="cuda")
        starter_args = [clone_arg(arg) for arg in original]
        torch_args = [clone_arg(arg) for arg in original]
        cfg.baseline(*starter_args)
        cfg._torch_reference(*torch_args)
        torch.cuda.synchronize()
        ok = cfg.check(shapes, original, starter_args, torch_args)
        ok_all = ok_all and ok
        print(f"case {case_id}: correct={ok}", flush=True)
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
