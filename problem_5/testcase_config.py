# torchgen.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Any, Union
import torch
import torch.nn.functional as F

KernelArg = Union[torch.Tensor, int, float]


def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 5


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    """
    返回每个参数的"尺寸描述"：
      - tensor 参数：返回 shape tuple
      - scalar 参数：返回 ()
    这里我们定义参数为 [x, y, T, H]
    
    返回格式：(shapes, (warmup, iters))
    """
    testcase_id = int(input())
    
    # 定义多个大规模测试点，每个有不同的shape
    testcases = [
        (23552, 11776, 16, 323),
        (93184, 3072, 16, 312),
        (3072, 93184, 16, 313),
        (29568, 9920, 15, 301),
        (230400, 1280, 16, 304),
    ]





    
    if testcase_id < 1 or testcase_id > len(testcases):
        raise ValueError(f"Invalid testcase_id: {testcase_id}")
    
    T, H, warmup, iters = testcases[testcase_id - 1]
    return [
        (T, 2*H),  # x: (T, 2*H)
        (T, H),    # y: (T, H)
        (),        # T scalar
        (),        # H scalar
    ], (warmup, iters)


def genTestCase(testcase_sizes: List[Tuple[int, ...]]) -> List[KernelArg]:
    """
    生成 testcase： [x, y, T, H]
    x/y 为 bfloat16 CUDA tensor，T/H 为 python int
    """
    assert len(testcase_sizes) == 4, "Expect 4 args: x, y, T, H"
    x_shape, y_shape, t_shape, h_shape = testcase_sizes
    assert t_shape == () and h_shape == (), "T, H must be scalars ()"

    # 验证形状一致性
    T, H2 = x_shape
    T_y, H = y_shape
    assert T == T_y, f"T must match: x has {T}, y has {T_y}"
    assert H2 == 2 * H, f"x's last dim {H2} must equal 2*H={2*H}"

    device = torch.device("cuda")
    x = torch.randn(x_shape, device=device, dtype=torch.bfloat16)
    y = torch.zeros(y_shape, device=device, dtype=torch.bfloat16)
    T_val = int(T)
    H_val = int(H)
    return [x, y, T_val, H_val]


def baseline(*inputs: List[KernelArg]):
    """
    baseline 用来算正确结果。
    实现 SwiGLU: SiLU(x_1) * x_2
    """
    x, y, T, H = inputs
    assert torch.is_tensor(x) and torch.is_tensor(y)
    assert isinstance(T, int) and isinstance(H, int)

    # Reference implementation
    x_1 = x[:, :H]
    x_2 = x[:, H:]
    silu_x1 = F.silu(x_1)
    y_ref = silu_x1 * x_2
    y.copy_(y_ref)
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
      - target_kernel_input_tensors：跑过 target_kernel 后的输入（y 被改）
      - baseline_input_tensors：跑过 baseline 后的输入（y 被改为正确值）
    这里比较 y 是否一致。
    """
    assert len(testcase_sizes) == 4
    assert len(target_kernel_input_tensors) == 4
    assert len(baseline_input_tensors) == 4

    x_t, y_t, T_t, H_t = target_kernel_input_tensors
    x_ref, y_ref, T_ref, H_ref = baseline_input_tensors

    if not (torch.is_tensor(y_t) and torch.is_tensor(y_ref)):
        raise TypeError("y must be tensor")
    if y_t.shape != y_ref.shape:
        print(f"[FAIL] shape mismatch: target {y_t.shape}, ref {y_ref.shape}")
        return False
    if y_t.dtype != y_ref.dtype:
        print(f"[FAIL] dtype mismatch: target {y_t.dtype}, ref {y_ref.dtype}")
        return False

    # 数值比较
    ok = torch.allclose(y_t, y_ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (y_t - y_ref).abs()
        max_diff = float(diff.max().item())
        print(f"[FAIL] allclose failed: max_abs_diff={max_diff} (rtol={rtol}, atol={atol})")
        return False

    return True
INPUT_CLASS = ["INPUT", "OUTPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 SwiGLU：y = SiLU(x_1) * x_2

    workload 口径：
      - flops = 4 * T * H
        理由：每个输出元素包含 1 次 SiLU 和 1 次乘法，这里按 SiLU 3 FLOPs、乘法 1 FLOP 统计。
      - memory_bytes = T * (2 * H) * 2 + T * H * 2
        理由：需要读取完整输入 x（宽度 2H）并写出输出 y（宽度 H），均按 bf16 统计。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, y_shape, _, _ = raw_sizes
    T, H2 = x_shape
    T_y, H = y_shape
    assert T == T_y and H2 == 2 * H
    return {
        "flops": 4 * T * H,
        "memory_bytes": T * H2 * 2 + T * H * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
