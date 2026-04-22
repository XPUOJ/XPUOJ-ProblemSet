# torchgen.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Any, Union
import torch

KernelArg = Union[torch.Tensor, int, float]


def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 5


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    """
    返回每个参数的"尺寸描述"：
      - tensor 参数：返回 shape tuple
      - scalar 参数：返回 ()
    这里我们定义参数为 [A, B, C, M, N, K]
    
    返回格式：(shapes, (warmup, iters))
    """
    testcase_id = int(input())
    
    # 定义多个大规模测试点，每个有不同的shape
    # 矩阵乘法复杂度O(M*N*K)，需要平衡规模
    testcases = [
        (6400, 6400, 1792, 8, 155),
        (20480, 2048, 3072, 4, 88),
        (1792, 18176, 2816, 5, 106),
        (9248, 6176, 1248, 8, 158),
        (113408, 128, 5120, 6, 119),
    ]











    
    if testcase_id < 1 or testcase_id > len(testcases):
        raise ValueError(f"Invalid testcase_id: {testcase_id}")
    
    M, N, K, warmup, iters = testcases[testcase_id - 1]
    return [
        (M, K),  # A: (M, K)
        (N, K),  # B: (N, K)
        (M, N),  # C: (M, N)
        (),      # M scalar
        (),      # N scalar
        (),      # K scalar
    ], (warmup, iters)


def genTestCase(testcase_sizes: List[Tuple[int, ...]]) -> List[KernelArg]:
    """
    生成 testcase： [A, B, C, M, N, K]
    A/B/C 为 bfloat16 CUDA tensor，M/N/K 为 python int
    """
    assert len(testcase_sizes) == 6, "Expect 6 args: A, B, C, M, N, K"
    a_shape, b_shape, c_shape, m_shape, n_shape, k_shape = testcase_sizes
    assert m_shape == () and n_shape == () and k_shape == (), "M, N, K must be scalars ()"

    # 验证形状一致性
    M, K = a_shape
    N, K_B = b_shape
    M_C, N_C = c_shape
    assert K == K_B, f"A's K {K} must equal B's K {K_B}"
    assert M == M_C, f"A's M {M} must equal C's M {M_C}"
    assert N == N_C, f"B's N {N} must equal C's N {N_C}"

    device = torch.device("cuda")
    A = torch.randn(*a_shape, device=device, dtype=torch.bfloat16)
    B = torch.randn(*b_shape, device=device, dtype=torch.bfloat16)
    C = torch.zeros(*c_shape, device=device, dtype=torch.bfloat16)
    M_val = int(M)
    N_val = int(N)
    K_val = int(K)
    return [A, B, C, M_val, N_val, K_val]


def baseline(*inputs: List[KernelArg]):
    """
    baseline 用来算正确结果。
    计算 C = A @ B.T
    """
    A, B, C, M, N, K = inputs
    assert torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(C)
    assert isinstance(M, int) and isinstance(N, int) and isinstance(K, int)

    # Reference implementation
    C_ref = torch.matmul(A, B.T)
    C.copy_(C_ref)
    return inputs


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> bool:
    """
    你的 run_cuda_kernel.py 会把三份输入都传进来：
      - original_input_tensors：原始输入（通常不用于比较，但保留以便你做更复杂逻辑）
      - target_kernel_input_tensors：跑过 target_kernel 后的输入（C 被改）
      - baseline_input_tensors：跑过 baseline 后的输入（C 被改为正确值）
    这里比较 C 是否一致。
    """
    assert len(testcase_sizes) == 6
    assert len(target_kernel_input_tensors) == 6
    assert len(baseline_input_tensors) == 6

    A_t, B_t, C_t, M_t, N_t, K_t = target_kernel_input_tensors
    A_ref, B_ref, C_ref, M_ref, N_ref, K_ref = baseline_input_tensors

    if not (torch.is_tensor(C_t) and torch.is_tensor(C_ref)):
        raise TypeError("C must be tensor")
    if C_t.shape != C_ref.shape:
        print(f"[FAIL] shape mismatch: target {C_t.shape}, ref {C_ref.shape}")
        return False
    if C_t.dtype != C_ref.dtype:
        print(f"[FAIL] dtype mismatch: target {C_t.dtype}, ref {C_ref.dtype}")
        return False

    # 数值比较
    ok = torch.allclose(C_t, C_ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (C_t - C_ref).abs()
        max_diff = float(diff.max().item())
        print(f"[FAIL] allclose failed: max_abs_diff={max_diff} (rtol={rtol}, atol={atol})")
        return False

    return True
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 计算 Dense GEMM：C = A @ B.T

    workload 口径：
      - flops = 2 * M * N * K
        理由：每个输出元素是长度为 K 的点积，按 1 次乘法 + 1 次加法记作 2K FLOPs。
      - memory_bytes = (M * K + N * K + M * N) * 2
        理由：需要读取 A 和 B，并写出 C，三者都按 bf16 的 2 Bytes/elem 统计。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, c_shape, _, _, _ = raw_sizes
    M, K = a_shape
    N, K_b = b_shape
    M_c, N_c = c_shape
    assert K == K_b and M == M_c and N == N_c
    return {
        "flops": 2 * M * N * K,
        "memory_bytes": (M * K + N * K + M * N) * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
