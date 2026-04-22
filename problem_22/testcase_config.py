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
        (21504, 21504, 'bfloat16', 3, 32),
        (42752, 11008, 'float16', 3, 31),
        (11008, 42752, 'bfloat16', 3, 31),
        (30720, 15360, 'float16', 3, 31),
        (5632, 85248, 'bfloat16', 3, 31),
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


    def _dtype_from_name(name: str) -> torch.dtype:
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float16":
            return torch.float16
        raise ValueError(f"Unsupported dtype name: {name}")


    def _dtype_code_from_name(name: str) -> int:
        if name == "bfloat16":
            return 0
        if name == "float16":
            return 1
        raise ValueError(f"Unsupported dtype name: {name}")


    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        M, N, dtype_name, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (M, N, dtype_name)
        return [
            (M, N),
            (M, N),
            (M,),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        global CURRENT_CASE
        assert CURRENT_CASE is not None
        M, N, dtype_name = CURRENT_CASE
        dtype = _dtype_from_name(dtype_name)
        dtype_code = _dtype_code_from_name(dtype_name)

        input_shape, output_shape, scale_shape, m_shape, n_shape, dtype_code_shape = testcase_sizes
        assert input_shape == (M, N)
        assert output_shape == (M, N)
        assert scale_shape == (M,)
        assert m_shape == n_shape == dtype_code_shape == ()

        input_tensor = torch.randn(M, N, dtype=dtype, device=device)
        output_q = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty(M, dtype=torch.float32, device=device)
        return [input_tensor, output_q, scale, int(M), int(N), int(dtype_code)]


    def baseline(
        input_tensor: torch.Tensor,
        output_q: torch.Tensor,
        scale: torch.Tensor,
        M: int,
        N: int,
        dtype_code: int,
    ) -> List[KernelArg]:
        amax = input_tensor.abs().float().amax(dim=1)
        scale_ref = torch.clamp(amax / 448.0, min=1e-12)
        output_fp32 = input_tensor.float() * (1.0 / scale_ref).unsqueeze(-1)
        output_clamped = torch.clamp(output_fp32, -448.0, 448.0)
        output_q.copy_(output_clamped.to(torch.float8_e4m3fn))
        scale.copy_(scale_ref)
        return [input_tensor, output_q, scale, M, N, dtype_code]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 32.0,
    ) -> bool:
        _, output_q_t, scale_t, _, _, _ = target_kernel_input_tensors
        _, output_q_ref, scale_ref, _, _, _ = baseline_input_tensors

        if output_q_t.shape != output_q_ref.shape:
            print(f"[FAIL] output_q shape mismatch: target {output_q_t.shape}, ref {output_q_ref.shape}", file=sys.stderr)
            return False
        if output_q_t.dtype != output_q_ref.dtype:
            print(f"[FAIL] output_q dtype mismatch: target {output_q_t.dtype}, ref {output_q_ref.dtype}", file=sys.stderr)
            return False
        if scale_t.shape != scale_ref.shape:
            print(f"[FAIL] scale shape mismatch: target {scale_t.shape}, ref {scale_ref.shape}", file=sys.stderr)
            return False
        if scale_t.dtype != scale_ref.dtype:
            print(f"[FAIL] scale dtype mismatch: target {scale_t.dtype}, ref {scale_ref.dtype}", file=sys.stderr)
            return False

        output_q_t_f = output_q_t.float()
        output_q_ref_f = output_q_ref.float()
        if not torch.allclose(output_q_t_f, output_q_ref_f, rtol=rtol, atol=atol):
            diff = (output_q_t_f - output_q_ref_f).abs()
            max_diff = float(diff.max().item())
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, output_q_t.shape))
            print(f"[FAIL] output_q allclose failed: max_abs_diff={max_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            print(
                f"[FAIL] output_q max diff at position {max_idx_tuple}: target={float(output_q_t_f.flatten()[max_idx].item()):.6f}, ref={float(output_q_ref_f.flatten()[max_idx].item()):.6f}",
                file=sys.stderr,
            )
            return False

        if not torch.allclose(scale_t, scale_ref, rtol=1e-3, atol=1e-6):
            diff = (scale_t - scale_ref).abs()
            max_diff = float(diff.max().item())
            print(f"[FAIL] scale allclose failed: max_abs_diff={max_diff:.6f}", file=sys.stderr)
            return False

        return True
except:
    pass
INPUT_CLASS = ["INPUT", "OUTPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 per-token FP8 quantization

    workload 口径：
      - flops = 3 * M * N
        理由：每一行都要单独求 amax、缩放并量化，因此仍按每个元素常数级近似统计。
      - memory_bytes = M * N * dtype_size + M * N + M * 4
        理由：读取输入张量，写出 fp8 输出 output_q，并为每一行写出 1 个 fp32 scale。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    input_shape, output_shape, scale_shape, _, _, _ = raw_sizes
    M, N = input_shape
    assert output_shape == (M, N)
    
    # 从CURRENT_CASE获取dtype信息
    global CURRENT_CASE
    if CURRENT_CASE is None:
        raise ValueError("CURRENT_CASE not set")
    _, _, dtype_name = CURRENT_CASE
    
    # 根据dtype_name确定dtype_size
    dtype_size_map = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }
    dtype_size = dtype_size_map.get(dtype_name, 2)
    
    return {
        "flops": 3 * M * N,
        "memory_bytes": M * N * dtype_size + M * N + M * 4,
        "dtype": dtype_name,
    }

DESIGNED_VRAM_SIZE = 48
