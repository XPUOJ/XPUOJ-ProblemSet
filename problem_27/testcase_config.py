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
        (3072, 3072, 1536, 5, 111),
        (3072, 2304, 1536, 7, 148),
        (3072, 4096, 1536, 4, 83),
        (6144, 2048, 1024, 6, 124),
        (1536, 4352, 1536, 8, 158),
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
        return [
            (M, K),
            (K, N),
            (M, N),
            (),
            (),
            (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        M, N, K = CURRENT_CASE
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
        a4 = a.view(M, K // 4, 4).float()
        _, idx = torch.topk(a4.abs(), k=2, dim=-1)
        mask = torch.zeros_like(a4, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        a = (a4 * mask).view(M, K).to(torch.int8)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device)
        output = torch.empty(M, N, dtype=torch.int32, device=device)
        return [a, b, output, int(M), int(N), int(K)]

    def baseline(a, b, output, M, N, K):
        output.copy_(torch.matmul(a.float(), b.float()).to(output.dtype))
        return [a, b, output, M, N, K]

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
            diff = (output_t - output_ref).abs()
            print(f"[FAIL] output mismatch: max_abs_diff={int(diff.max().item())}", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 sparse int8 GEMM 的参考计算：output = a.float() @ b.float()

    workload 口径：
      - flops = 2 * M * N * K
        理由：虽然输入 a 是 2:4 稀疏构造，但 baseline 仍按标准 GEMM 参考实现统计。
      - memory_bytes = (M * K + K * N) + M * N * 4
        理由：需要读取两个 int8 输入矩阵，并写出 int32 输出。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, output_shape, _, _, _ = raw_sizes
    M, K = a_shape
    K_b, N = b_shape
    assert K == K_b and output_shape == (M, N)
    return {
        "flops": 2 * M * N * K,
        "memory_bytes": (M * K + K * N) + M * N * 4,
    }

DESIGNED_VRAM_SIZE = 48
