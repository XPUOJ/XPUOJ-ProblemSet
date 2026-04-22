from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float, bool]
    CURRENT_CASE = None
    TESTCASES = [
        (1536, 1536, 1.25, 1, 5, 101),
        (1792, 896, -0.75, 0, 6, 124),
        (2304, 768, 0.5, 1, 4, 92),
        (1280, 2304, -1.5, 0, 5, 97),
        (2048, 512, 1.0, 1, 8, 161),
    ]



    def _get_testcase_id() -> int:
        try:
            raw = input().strip()
        except EOFError:
            return 1
        if raw == "":
            return 1
        try:
            testcase_id = int(raw.split()[0])
        except ValueError:
            return 1
        return testcase_id if 1 <= testcase_id <= len(TESTCASES) else 1

    def getTestCaseSize():
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        M, N, alpha, upper, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (M, N, float(alpha), int(upper))
        return [
            (M, M),
            (M, N),
            (M, N),
            (),
            (),
            (),
            (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        M, N, alpha, upper = CURRENT_CASE
        a = torch.randn(M, M, dtype=torch.float64, device=device)
        b = torch.randn(M, N, dtype=torch.float64, device=device)
        output = torch.empty(M, N, dtype=torch.float64, device=device)
        alpha_tensor = torch.tensor(alpha, dtype=torch.float64, device=device)
        upper_tensor = torch.tensor(upper, dtype=torch.int64, device=device)
        return [a, b, output, alpha_tensor, upper_tensor, int(M), int(N)]

    def baseline(a, b, output, alpha, upper, M, N):
        alpha_val = float(alpha.item()) if torch.is_tensor(alpha) else float(alpha)
        upper_val = bool(upper.item()) if torch.is_tensor(upper) else bool(upper)
        idx = torch.arange(M, device=a.device)
        if upper_val:
            mask = idx[:, None] <= idx[None, :]
        else:
            mask = idx[:, None] >= idx[None, :]
        a_tri = a.double() * mask
        output.copy_(alpha_val * torch.matmul(a_tri, b.double()))
        return [a, b, output, alpha, upper, M, N]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 1e-8, atol: float = 1e-8):
        output_t = target_kernel_input_tensors[2]
        output_ref = baseline_input_tensors[2]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(output_t, output_ref, rtol=rtol, atol=atol):
            diff = (output_t - output_ref).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6e} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 TRMM：先根据 upper/lower 取 a 的三角部分，再计算 output = alpha * a_tri @ b

    workload 口径：
      - flops = M * (M + 1) * N
        理由：只有三角部分参与乘法，等价于每列只使用 M*(M+1)/2 个有效元素。
      - memory_bytes = (M * M + M * N + M * N) * 8
        理由：需要读取 fp64 的 a、b，并写出 fp64 的 output。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, output_shape, _, _, _, _ = raw_sizes
    M, M_b = a_shape
    M_b2, N = b_shape
    assert M == M_b == M_b2 and output_shape == (M, N)
    return {
        "flops": M * (M + 1) * N,
        "memory_bytes": (M * M + M * N + M * N) * 8,
        "dtype": "fp64",
    }

DESIGNED_VRAM_SIZE = 48
