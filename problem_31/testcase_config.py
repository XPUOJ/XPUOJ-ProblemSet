from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    BLOCK_SIZE = 128
    TESTCASES = [
        (1024, 1024, 1024, 9, 174),
        (2048, 1024, 1024, 4, 87),
        (1024, 1280, 1280, 5, 97),
        (768, 1536, 1536, 4, 90),
        (2304, 768, 768, 6, 129),
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
        M, N, K, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (M, N, K)
        K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        return [
            (M, K),
            (M, K_blocks),
            (N, K),
            (N_blocks, K_blocks),
            (M, N),
            (), (), (), (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        M, N, K = CURRENT_CASE
        K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        A = torch.randn(M, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
        B = torch.randn(N, K, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
        A_scale = torch.rand(M, K_blocks, dtype=torch.float32, device=device) + 0.5
        B_scale = torch.rand(N_blocks, K_blocks, dtype=torch.float32, device=device) + 0.5
        output = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        return [A, A_scale, B, B_scale, output, int(M), int(N), int(K), int(BLOCK_SIZE)]

    def baseline(A, A_scale, B, B_scale, output, M, N, K, block_size):
        A_scaled = A.float() * A_scale.repeat_interleave(block_size, dim=1)[:, :K]
        B_scaled = B.float() * B_scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:N, :K]
        output.copy_(torch.matmul(A_scaled, B_scaled.t()).to(output.dtype))
        return [A, A_scale, B, B_scale, output, M, N, K, block_size]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 1e-1, atol: float = 1e-1):
        output_t = target_kernel_input_tensors[4]
        output_ref = baseline_input_tensors[4]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(output_t.float(), output_ref.float(), rtol=rtol, atol=atol):
            diff = (output_t.float() - output_ref.float()).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 FP8 groupwise GEMM：先按 A_scale / B_scale 做缩放，再执行 GEMM

    workload 口径：
      - flops = 2 * M * N * K
        理由：主计算仍是标准 GEMM，scale 只影响访存，不改变乘加数量。
      - memory_bytes = M * K + M * K_blocks * 4 + N * K + N_blocks * K_blocks * 4 + M * N * 2
        理由：需要读取 fp8 的 A/B、fp32 的 scale，并写出 bf16 输出。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, a_scale_shape, b_shape, b_scale_shape, output_shape, _, _, _, _ = raw_sizes
    M, K = a_shape
    N, K_b = b_shape
    assert K == K_b and output_shape == (M, N)
    K_blocks = a_scale_shape[1]
    N_blocks = b_scale_shape[0]
    return {
        "flops": 2 * M * N * K,
        "memory_bytes": M * K + M * K_blocks * 4 + N * K + N_blocks * K_blocks * 4 + M * N * 2,
        "dtype": "fp8",
    }

DESIGNED_VRAM_SIZE = 48
