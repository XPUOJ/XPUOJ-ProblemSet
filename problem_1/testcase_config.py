# torchgen.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Any, Union
import torch

KernelArg = Union[torch.Tensor, int, float]

# (shape, warmup, iters)
# Note: problem_1 uses an in-place INOUT tensor; overly large shapes + large warmup/iters
# will explode memory due to per-iteration cloning in the runner. Keep shapes moderate.
TESTCASES = [
    ((8192, 8192), 3, 30),
    ((12288, 8192), 3, 30),
    ((16384, 8192), 3, 30),
    ((8192, 12288), 3, 30),
    ((12288, 12288), 3, 30),
]


def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 5


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    """
    返回每个参数的"尺寸描述"：
      - tensor 参数：返回 shape tuple
      - scalar 参数：返回 ()
    这里我们定义参数为 [A, B, numel]
    
    返回格式：(shapes, (warmup, iters))
    """
    testcase_id = int(input())

    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase_id: {testcase_id}")
    
    shape, warmup, iters = TESTCASES[testcase_id - 1]
    return [
        shape,  # A
        shape,  # B
        (),     # numel scalar
    ], (warmup, iters)


def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
    """
    生成 testcase： [A, B, numel]
    A/B 为 float32 CUDA tensor，numel 为 python int
    """
    assert len(testcase_sizes) == 3, "Expect 3 args: A, B, numel"
    a_shape, b_shape, n_shape = testcase_sizes
    assert n_shape == (), "Third arg must be scalar ()"

    # 这里要求 A/B shape 一致（a+b）
    assert a_shape == b_shape, f"A shape {a_shape} must equal B shape {b_shape}"

    A = torch.randn(*a_shape, device=device, dtype=torch.float16)
    B = torch.randn(*b_shape, device=device, dtype=torch.float16)
    numel = int(A.numel())
    return [A, B, numel]


# Reduce cloning pressure in cuda_runner.py:
# - A is INOUT (must be cloned)
# - B is read-only INPUT (can be shared)
# - numel is scalar INPUT
INPUT_CLASS = ["INOUT", "INPUT", "INPUT"]


def baseline(*inputs: List[KernelArg]):
    """
    baseline 用来算正确结果。
    由于目标 kernel 是 inplace：A += B
    我们也在 baseline 上做同样的 inplace，并返回修改后的 inputs。
    """
    A, B, numel = inputs
    assert torch.is_tensor(A) and torch.is_tensor(B)
    assert isinstance(numel, int)

    # inplace reference
    A.add_(B)
    return inputs


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """
    你的 run_cuda_kernel.py 会把三份输入都传进来：
      - original_input_tensors：原始输入（通常不用于比较，但保留以便你做更复杂逻辑）
      - target_kernel_input_tensors：跑过 target_kernel 后的输入（A 被改）
      - baseline_input_tensors：跑过 baseline 后的输入（A 被改为正确值）
    这里比较 A 是否一致。
    """
    assert len(testcase_sizes) == 3
    assert len(target_kernel_input_tensors) == 3
    assert len(baseline_input_tensors) == 3

    A_t, B_t, n_t = target_kernel_input_tensors
    A_ref, B_ref, n_ref = baseline_input_tensors

    if not (torch.is_tensor(A_t) and torch.is_tensor(A_ref)):
        raise TypeError("A must be tensor")
    if A_t.shape != A_ref.shape:
        print(f"[FAIL] shape mismatch: target {A_t.shape}, ref {A_ref.shape}")
        return False
    if A_t.dtype != A_ref.dtype:
        print(f"[FAIL] dtype mismatch: target {A_t.dtype}, ref {A_ref.dtype}")
        return False

    # 数值比较
    ok = torch.allclose(A_t, A_ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (A_t - A_ref).abs()
        max_diff = float(diff.max().item())
        print(f"[FAIL] allclose failed: max_abs_diff={max_diff} (rtol={rtol}, atol={atol})")
        return False

    return True
INPUT_CLASS = ["INOUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:

    """
    计算本题 workload。
    本题计算逻辑：
      - kernel 执行 fp16 原地逐元素加法：A += B

    workload 口径：
      - flops = numel
        理由：每个元素只做 1 次加法。
      - memory_bytes = numel * 2 * 3
        理由：每个 fp16 元素 2 Bytes，访存包含读 A、读 B、写回 A。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    assert len(raw_sizes) == 3, "Expect 3 args: A, B, numel"
    a_shape, b_shape, n_shape = raw_sizes
    assert a_shape == b_shape
    assert n_shape == ()
    numel = 1
    for dim in a_shape:
        numel *= dim
    return {
        "flops": numel,
        "memory_bytes": numel * 2 * 3,
        "dtype": "fp16",
    }

DESIGNED_VRAM_SIZE = 48
