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
        (4, 1024, 1536, 2048, 5, 114),
        (6, 1536, 1024, 2048, 3, 76),
        (8, 512, 1024, 4096, 4, 84),
        (4, 2048, 2048, 1024, 4, 85),
        (2, 2304, 2304, 1536, 4, 89),
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
        G, M, N, K, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (G, M, N, K)
        return [
            (G, M, K),
            (G, K, N),
            (G, M, N),
            (),
            (),
            (),
            (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        G, M, N, K = CURRENT_CASE
        a = torch.randint(-8, 8, (G, M, K), dtype=torch.int8, device=device)
        b_fp16 = torch.randn(G, K, N, device=device).to(torch.float16)
        b = b_fp16.to(torch.float8_e4m3fn)
        output = torch.empty(G, M, N, dtype=torch.float32, device=device)
        return [a, b, output, int(G), int(M), int(N), int(K)]

    def baseline(a, b, output, G, M, N, K):
        output.copy_(torch.matmul(a.float(), b.float()))
        return [a, b, output, G, M, N, K]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 1e-2, atol: float = 1.0):
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
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 grouped GEMM：output = matmul(a.float(), b.float())

    workload 口径：
      - flops = 2 * G * M * N * K
        理由：每个 batch/group 的每个输出元素都是长度 K 的点积。
      - memory_bytes = G * M * K + G * K * N + G * M * N * 4
        理由：需要读取 int8 的 a、fp8 的 b，并写出 fp32 的 output。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, output_shape, _, _, _, _ = raw_sizes
    G, M, K = a_shape
    G_b, K_b, N = b_shape
    assert G == G_b and K == K_b and output_shape == (G, M, N)
    return {
        "flops": 2 * G * M * N * K,
        "memory_bytes": G * M * K + G * K * N + G * M * N * 4,
    }

DESIGNED_VRAM_SIZE = 48
