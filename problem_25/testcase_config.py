from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    TESTCASES = [
        (124672, 63232, 4096, 'bfloat16', 10, 292),
        (48128, 12544, 12544, 'float16', 10, 345),
        (87808, 3840, 3840, 'float32', 10, 467),
        (156928, 156928, 2816, 'bfloat16', 10, 215),
        (568576, 2048, 1024, 'float16', 10, 500),
    ]







    def _get_testcase_id() -> int:
        try:
            raw = input().strip()
        except EOFError:
            return 1
        if raw == "":
            return 1
        token = raw.split()[0]
        try:
            testcase_id = int(token)
        except ValueError:
            return 1
        if testcase_id < 1 or testcase_id > len(TESTCASES):
            return 1
        return testcase_id


    def _dtype_from_name(name: str):
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float16":
            return torch.float16
        if name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype name: {name}")


    def _dtype_code_from_name(name: str) -> int:
        if name == "bfloat16":
            return 0
        if name == "float16":
            return 1
        if name == "float32":
            return 2
        raise ValueError(f"Unsupported dtype name: {name}")


    def getTestCaseSize():
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        first_axis_dim, M, H, dtype_name, warmup, iters = TESTCASES[testcase_id - 1]
        M = min(M, first_axis_dim)
        CURRENT_CASE = (first_axis_dim, M, H, dtype_name)
        return [
            (M, H),
            (M,),
            (first_axis_dim, H),
            (),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes, device: str = "cuda"):
        global CURRENT_CASE
        first_axis_dim, M, H, dtype_name = CURRENT_CASE
        assert M <= first_axis_dim, f"M must not exceed first_axis_dim, got M={M}, first_axis_dim={first_axis_dim}"
        dtype = _dtype_from_name(dtype_name)
        dtype_code = _dtype_code_from_name(dtype_name)
        values_shape, indices_shape, output_shape, fad_shape, m_shape, h_shape, dtype_code_shape = testcase_sizes
        assert values_shape == (M, H)
        assert indices_shape == (M,)
        assert output_shape == (first_axis_dim, H)
        values = torch.randn(M, H, dtype=dtype, device=device)
        indices = torch.randperm(first_axis_dim, device=device)[:M].to(torch.int64)
        output = torch.empty(first_axis_dim, H, dtype=dtype, device=device)
        return [values, indices, output, int(first_axis_dim), int(M), int(H), int(dtype_code)]


    def baseline(values, indices, output, first_axis_dim, M, H, dtype_code):
        output.zero_()
        output[indices] = values
        return [values, indices, output, first_axis_dim, M, H, dtype_code]


    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 0.0, atol: float = 0.0):
        output_t = target_kernel_input_tensors[2]
        output_ref = baseline_input_tensors[2]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.equal(output_t, output_ref):
            diff_idx = torch.nonzero(output_t != output_ref, as_tuple=False)
            first = tuple(diff_idx[0].tolist())
            print(f"[FAIL] output mismatch at {first}", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 Index Put First Axis：output[indices] = values

    workload 口径：
      - flops = 0
        理由：这是纯 scatter 写入，没有实质浮点运算。
      - memory_bytes = M * H * dtype_size + M * 8 + first_axis_dim * H * dtype_size
        理由：需要读取 values 和 indices，并写入完整输出张量 output。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    values_shape, indices_shape, output_shape, _, _, _, _ = raw_sizes
    M, H = values_shape
    first_axis_dim, H_out = output_shape
    assert H == H_out and indices_shape == (M,)
    
    # 从CURRENT_CASE获取dtype信息
    global CURRENT_CASE
    if CURRENT_CASE is None:
        raise ValueError("CURRENT_CASE not set")
    _, _, _, dtype_name = CURRENT_CASE
    
    # 根据dtype_name确定dtype_size
    dtype_size_map = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
    dtype_size = dtype_size_map.get(dtype_name, 2)
    
    return {
        "flops": 0,
        "memory_bytes": M * H * dtype_size + M * 8 + first_axis_dim * H * dtype_size,
        "dtype": dtype_name,
    }

DESIGNED_VRAM_SIZE = 48
