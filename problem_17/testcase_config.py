from __future__ import annotations

def getNumOfTestcases() -> int:
    return 2

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float, bool]
    CURRENT_CASE = None
    TESTCASES = [
        (21504, 21504, 'bfloat16', 3, 39),
        (21760, 21760, 'bfloat16', 3, 38),
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
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype name: {name}")
        return mapping[name]


    def _dtype_code_from_name(name: str) -> int:
        mapping = {
            "bfloat16": 0,
            "float16": 1,
            "float32": 2,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported dtype name: {name}")
        return mapping[name]


    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """
        参数顺序:
        [input, output_q, scale, is_static, M, N]
        注意：dtype_code 是内部实现细节，不是 kernel 参数
        """
        testcase_id = _get_testcase_id()

        global CURRENT_CASE
        M, N, dtype_name, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (M, N, dtype_name)
        return [
            (M, N),
            (M, N),
            (),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        global CURRENT_CASE
        assert len(testcase_sizes) == 6, "Expect 6 args: input, output_q, scale, is_static, M, N"
        assert CURRENT_CASE is not None, "CURRENT_CASE must be set by getTestCaseSize() before genTestCase()"

        input_shape, output_shape, scale_shape, is_static_shape, m_shape, n_shape = testcase_sizes
        assert scale_shape == () and is_static_shape == () and m_shape == () and n_shape == (), "scalar args must be ()"
        assert input_shape == output_shape, "input and output_q must have same shape"

        M, N, dtype_name = CURRENT_CASE
        dtype = _dtype_from_name(dtype_name)
        dtype_code = _dtype_code_from_name(dtype_name)

        input_tensor = torch.randn(*input_shape, dtype=dtype, device=device)
        output_q = torch.empty(*output_shape, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.zeros((), dtype=torch.float32, device=device)
        is_static = 0
        # dtype_code 是内部实现细节，用于 baseline，但不作为 kernel 参数
        # 将其存储在全局变量中供 baseline 使用
        global _DTYPE_CODE
        _DTYPE_CODE = dtype_code
        return [input_tensor, output_q, scale, is_static, int(M), int(N)]


    def baseline(
        input_tensor: torch.Tensor,
        output_q: torch.Tensor,
        scale: torch.Tensor,
        is_static: bool,
        M: int,
        N: int,
    ) -> List[KernelArg]:
        assert input_tensor.dtype in [torch.bfloat16, torch.float16, torch.float32]
        assert output_q.dtype == torch.float8_e4m3fn
        assert scale.dtype == torch.float32
        assert isinstance(M, int) and isinstance(N, int)
        assert input_tensor.shape == (M, N)

        fp8_max = 448.0
        amax = torch.max(torch.abs(input_tensor)).float()
        scale_ref = amax / fp8_max
        scale_ref = torch.maximum(scale_ref, torch.tensor(1e-12, device=input_tensor.device))

        scale.copy_(scale_ref)
        output_fp32 = input_tensor.float() * (1.0 / scale_ref)
        output_clamped = torch.clamp(output_fp32, -fp8_max, fp8_max)
        output_q.copy_(output_clamped.to(torch.float8_e4m3fn))
        return [input_tensor, output_q, scale, is_static, M, N]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 320.0,
    ) -> bool:
        assert len(testcase_sizes) == 6
        assert len(target_kernel_input_tensors) == 6
        assert len(baseline_input_tensors) == 6

        _, output_q_t, scale_t, _, _, _ = target_kernel_input_tensors
        _, output_q_ref, scale_ref, _, _, _ = baseline_input_tensors

        if not (torch.is_tensor(output_q_t) and torch.is_tensor(output_q_ref)):
            print(f"[FAIL] output_q must be tensor, got {type(output_q_t)} and {type(output_q_ref)}", file=sys.stderr)
            return False
        if not (torch.is_tensor(scale_t) and torch.is_tensor(scale_ref)):
            print(f"[FAIL] scale must be tensor, got {type(scale_t)} and {type(scale_ref)}", file=sys.stderr)
            return False

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
        ok_output = torch.allclose(output_q_t_f, output_q_ref_f, rtol=rtol, atol=atol)
        if not ok_output:
            diff = (output_q_t_f - output_q_ref_f).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] output_q allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, output_q_t.shape))
            print(
                f"[FAIL] output_q max diff at position {max_idx_tuple}: target={float(output_q_t_f.flatten()[max_idx].item()):.6f}, ref={float(output_q_ref_f.flatten()[max_idx].item()):.6f}",
                file=sys.stderr,
            )
            return False

        ok_scale = torch.allclose(scale_t, scale_ref, rtol=rtol, atol=atol)
        if not ok_scale:
            diff = (scale_t - scale_ref).abs()
            max_diff = float(diff.max().item())
            print(f"[FAIL] scale allclose failed: max_abs_diff={max_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False

        return True
except:
    pass
INPUT_CLASS = ["INPUT", "OUTPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 per-tensor FP8 quantization

    workload 口径：
      - flops = 3 * M * N
        理由：需要遍历整张输入张量求 amax、缩放并量化，每个元素按常数级近似统计。
      - memory_bytes = M * N * dtype_size + M * N + 4
        理由：读取原输入，写出 fp8 输出 output_q，并写出 1 个 fp32 scale。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    input_shape, output_shape, scale_shape, _, _, _ = raw_sizes
    M, N = input_shape
    assert output_shape == (M, N)
    assert scale_shape == ()
    dtype_size, dtype_name = 2, "bf16"
    return {
        "flops": 3 * M * N,
        "memory_bytes": M * N * dtype_size + M * N + 4,
        "dtype": dtype_name,
    }

DESIGNED_VRAM_SIZE = 48
