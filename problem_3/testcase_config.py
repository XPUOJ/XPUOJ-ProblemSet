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
    这里我们定义参数为 [probs, top_p, renorm_probs, batch_size, num_classes]
    
    返回格式：(shapes, (warmup, iters))
    """
    testcase_id = int(input())
    
    # 定义多个大规模测试点，每个有不同的shape
    testcases = [
        (1536, 11776, 7, 146),
        (7936, 4096, 16, 316),
        (256, 19712, 16, 305),
        (2316, 7712, 12, 242),
        (33024, 1280, 11, 222),
    ]









    
    if testcase_id < 1 or testcase_id > len(testcases):
        raise ValueError(f"Invalid testcase_id: {testcase_id}")
    
    batch_size, num_classes, warmup, iters = testcases[testcase_id - 1]
    return [
        (batch_size, num_classes),  # probs: (batch_size, num_classes)
        (batch_size,),               # top_p: (batch_size,)
        (batch_size, num_classes),  # renorm_probs: (batch_size, num_classes)
        (),                          # batch_size scalar
        (),                          # num_classes scalar
    ], (warmup, iters)


def genTestCase(testcase_sizes: List[Tuple[int, ...]]) -> List[KernelArg]:
    """
    生成 testcase： [probs, top_p, renorm_probs, batch_size, num_classes]
    probs/top_p/renorm_probs 为 float32 CUDA tensor，其余为 python int
    """
    assert len(testcase_sizes) == 5, "Expect 5 args: probs, top_p, renorm_probs, batch_size, num_classes"
    probs_shape, top_p_shape, renorm_shape, bs_shape, nc_shape = testcase_sizes
    assert bs_shape == () and nc_shape == (), "batch_size, num_classes must be scalars ()"

    # 验证形状一致性
    batch_size, num_classes = probs_shape
    batch_size_tp, = top_p_shape
    batch_size_r, num_classes_r = renorm_shape
    assert batch_size == batch_size_tp == batch_size_r, "batch_size must match"
    assert num_classes == num_classes_r, "num_classes must match"

    device = torch.device("cuda")
    # 生成有效的概率分布（每行和为1）
    probs = torch.rand(probs_shape, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=1, keepdim=True)
    # top_p 在 (0, 1) 范围内
    top_p = torch.rand((batch_size,), device=device, dtype=torch.float32) * 0.5 + 0.3
    renorm_probs = torch.zeros(renorm_shape, device=device, dtype=torch.float32)
    batch_size_val = int(batch_size)
    num_classes_val = int(num_classes)
    return [probs, top_p, renorm_probs, batch_size_val, num_classes_val]


def baseline(probs, top_p, renorm_probs, batch_size, num_classes):
    """
    向量化 Top-p 重归一化（in-place 写入 renorm_probs）
    probs:        [B, C]
    top_p:        [B] 或 [B, 1]
    renorm_probs: [B, C]  (输出)
    """
    assert torch.is_tensor(probs) and torch.is_tensor(top_p) and torch.is_tensor(renorm_probs)
    assert probs.dim() == 2, f"probs must be [B, C], got {probs.shape}"
    B, C = probs.shape
    assert isinstance(batch_size, int) and isinstance(num_classes, int)
    assert batch_size == B and num_classes == C, f"batch_size/num_classes mismatch: ({batch_size},{num_classes}) vs ({B},{C})"
    assert top_p.numel() == B, f"top_p must have B elements, got {top_p.shape}"
    assert renorm_probs.shape == probs.shape

    # 统一 top_p 形状为 [B, 1] 以便广播
    p = top_p.reshape(B, 1).to(dtype=probs.dtype, device=probs.device)

    # 每行降序排序
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)  # [B, C]

    # 累积和
    cumsum = torch.cumsum(sorted_probs, dim=-1)  # [B, C]

    # nucleus mask: cumulative <= top_p
    mask = cumsum <= p  # [B, C] bool

    # 保证每行至少保留 1 个（即使 top_p 很小）
    # 等价于 baseline 里：if not mask.any(): mask[0]=True
    mask[:, 0] = True

    # 过滤：不在 top-p 里的置零
    filtered_sorted = sorted_probs * mask.to(sorted_probs.dtype)  # [B, C]

    # 计算每行归一化系数（过滤后总和）
    denom = filtered_sorted.sum(dim=-1, keepdim=True)  # [B, 1]

    # 避免除 0：如果 denom==0（极端情况 probs 全 0），保持全 0
    renorm_sorted = torch.where(
        denom > 0,
        filtered_sorted / denom,
        torch.zeros_like(filtered_sorted)
    )  # [B, C]

    # scatter 回原 vocab 顺序
    renorm_probs.zero_()
    renorm_probs.scatter_(dim=-1, index=sorted_idx, src=renorm_sorted)


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """
    你的 run_cuda_kernel.py 会把三份输入都传进来：
      - original_input_tensors：原始输入（通常不用于比较，但保留以便你做更复杂逻辑）
      - target_kernel_input_tensors：跑过 target_kernel 后的输入（renorm_probs 被改）
      - baseline_input_tensors：跑过 baseline 后的输入（renorm_probs 被改为正确值）
    这里比较 renorm_probs 是否一致。
    """
    assert len(testcase_sizes) == 5
    assert len(target_kernel_input_tensors) == 5
    assert len(baseline_input_tensors) == 5

    probs_t, top_p_t, renorm_probs_t, bs_t, nc_t = target_kernel_input_tensors
    probs_ref, top_p_ref, renorm_probs_ref, bs_ref, nc_ref = baseline_input_tensors

    if not (torch.is_tensor(renorm_probs_t) and torch.is_tensor(renorm_probs_ref)):
        raise TypeError("renorm_probs must be tensor")
    if renorm_probs_t.shape != renorm_probs_ref.shape:
        print(f"[FAIL] shape mismatch: target {renorm_probs_t.shape}, ref {renorm_probs_ref.shape}")
        return False
    if renorm_probs_t.dtype != renorm_probs_ref.dtype:
        print(f"[FAIL] dtype mismatch: target {renorm_probs_t.dtype}, ref {renorm_probs_ref.dtype}")
        return False

    # 数值比较
    ok = torch.allclose(renorm_probs_t, renorm_probs_ref, rtol=rtol, atol=atol)
    if not ok:
        diff = (renorm_probs_t - renorm_probs_ref).abs()
        max_diff = float(diff.max().item())
        print(f"[FAIL] allclose failed: max_abs_diff={max_diff} (rtol={rtol}, atol={atol})")
        return False

    return True
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 对每行概率做 Top-p 过滤并重归一化

    workload 口径：
      - flops = 20 * batch_size * num_classes
        理由：主要开销来自排序、前缀和、阈值判断和归一化，按每个概率元素常数级近似统计。
      - memory_bytes = batch_size * num_classes * 4 * 2 + batch_size * 4
        理由：主要读取 probs 并写回 renorm_probs，另读取每行一个 top_p，均按 fp32 统计。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    probs_shape, top_p_shape, renorm_shape, _, _ = raw_sizes
    batch_size, num_classes = probs_shape
    assert top_p_shape == (batch_size,)
    assert renorm_shape == (batch_size, num_classes)
    numel = batch_size * num_classes
    return {
        "flops": 20 * numel,
        "memory_bytes": numel * 4 * 2 + batch_size * 4,
        "dtype": "fp32",
    }

DESIGNED_VRAM_SIZE = 48
