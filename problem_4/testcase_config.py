# torchgen.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Tuple, Any, Union
import torch

KernelArg = Union[torch.Tensor, int, float]

# (batch_size, num_classes, warmup, iters)
TESTCASES = [
    (5120, 1024, 3, 21),
    (17152, 2048, 3, 24),
    (32768, 4096, 3, 39),
    (7144, 4906, 3, 20),
    (70400, 1024, 3, 51),
]






def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 5


def getTestCaseSize() -> Tuple[List[Tuple[int, ...]], Tuple[int, int]]:
    """
    返回每个参数的"尺寸描述"：
      - tensor 参数：返回 shape tuple
      - scalar 参数：返回 ()
    这里我们定义参数为 [probs, top_k, renorm_probs, batch_size, num_classes]
    
    返回格式：(shapes, (warmup, iters))
    """
    testcase_id = int(input())

    if testcase_id < 1 or testcase_id > len(TESTCASES):
        raise ValueError(f"Invalid testcase_id: {testcase_id}")
    
    batch_size, num_classes, warmup, iters = TESTCASES[testcase_id - 1]
    return [
        (batch_size, num_classes),  # probs: (batch_size, num_classes)
        (batch_size,),               # top_k: (batch_size,)
        (batch_size, num_classes),  # renorm_probs: (batch_size, num_classes)
        (),                          # batch_size scalar
        (),                          # num_classes scalar
    ], (warmup, iters)


def genTestCase(testcase_sizes: List[Tuple[int, ...]]) -> List[KernelArg]:
    """
    生成 testcase： [probs, top_k, renorm_probs, batch_size, num_classes]
    probs/renorm_probs 为 float32 CUDA tensor，top_k 为 int32 CUDA tensor，其余为 python int
    """
    assert len(testcase_sizes) == 5, "Expect 5 args: probs, top_k, renorm_probs, batch_size, num_classes"
    probs_shape, top_k_shape, renorm_shape, bs_shape, nc_shape = testcase_sizes
    assert bs_shape == () and nc_shape == (), "batch_size, num_classes must be scalars ()"

    # 验证形状一致性
    batch_size, num_classes = probs_shape
    batch_size_tk, = top_k_shape
    batch_size_r, num_classes_r = renorm_shape
    assert batch_size == batch_size_tk == batch_size_r, "batch_size must match"
    assert num_classes == num_classes_r, "num_classes must match"

    device = torch.device("cuda")
    seed = int((batch_size * 1_000_003 + num_classes) & 0x7FFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 生成有效的概率分布（每行和为1）
    probs = torch.rand(probs_shape, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=1, keepdim=True)
    # top_k 在 [1, num_classes) 范围内
    top_k = torch.randint(1, num_classes, (batch_size,), device=device, dtype=torch.int32)
    renorm_probs = torch.zeros(renorm_shape, device=device, dtype=torch.float32)
    batch_size_val = int(batch_size)
    num_classes_val = int(num_classes)
    return [probs, top_k, renorm_probs, batch_size_val, num_classes_val]


def baseline(probs, top_k, renorm_probs, batch_size, num_classes):
    """
    baseline 用来算正确结果。
    实现 Top-k 重归一化
    """
    assert torch.is_tensor(probs) and torch.is_tensor(top_k) and torch.is_tensor(renorm_probs)
    assert probs.dim() == 2, f"probs must be [B, C], got {probs.shape}"
    B, C = probs.shape
    assert isinstance(batch_size, int) and isinstance(num_classes, int)
    assert batch_size == B and num_classes == C, f"batch_size/num_classes mismatch: ({batch_size},{num_classes}) vs ({B},{C})"
    assert top_k.numel() == B, f"top_k must have B elements, got {top_k.shape}"
    assert renorm_probs.shape == probs.shape

    # k 处理：转为 [B,1]，并 clamp 到 [0, C]
    k = top_k.reshape(B, 1).to(device=probs.device)
    k = k.clamp(min=0, max=C).to(dtype=torch.long)

    # 为了向量化：统一取每行最大 Kmax 的 topk（Kmax = batch 内最大 k）
    Kmax = int(k.max().item())
    renorm_probs.zero_()
    if Kmax == 0:
        return

    topv, topi = torch.topk(probs, k=Kmax, dim=-1, largest=True, sorted=True)  # [B, Kmax]

    # 每行只保留前 k[i] 个：构造 mask（[B,Kmax]）
    # mask[b, j] = (j < k[b])
    arange = torch.arange(Kmax, device=probs.device).view(1, Kmax)
    mask = arange < k  # bool, [B, Kmax]

    # 过滤并计算归一化分母
    topv_masked = topv * mask.to(topv.dtype)                 # [B, Kmax]
    denom = topv_masked.sum(dim=-1, keepdim=True)            # [B, 1]

    # 重归一化：denom==0 的行保持全 0（等价 baseline）
    topv_renorm = torch.where(
        denom > 0,
        topv_masked / denom,
        torch.zeros_like(topv_masked)
    )  # [B, Kmax]

    # scatter 回原 vocab 位置
    renorm_probs.scatter_(dim=-1, index=topi, src=topv_renorm)


def check(
    testcase_sizes: List[Tuple[int, ...]],
    original_input_tensors: List[KernelArg],
    target_kernel_input_tensors: List[KernelArg],
    baseline_input_tensors: List[KernelArg],
    rtol: float = 2e-2,
    atol: float = 1.5e-1,
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

    probs_t, top_k_t, renorm_probs_t, bs_t, nc_t = target_kernel_input_tensors
    probs_ref, top_k_ref, renorm_probs_ref, bs_ref, nc_ref = baseline_input_tensors

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
      - 对每行概率做 Top-k 过滤并重归一化

    workload 口径：
      - flops = 20 * batch_size * num_classes
        理由：主要开销来自 topk 选择、归一化和 scatter 回写，按每个概率元素常数级近似统计。
      - memory_bytes = batch_size * num_classes * 4 * 2 + batch_size * 4
        理由：主要读取 probs 并写回 renorm_probs，另读取每行一个 int32 的 top_k。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    probs_shape, top_k_shape, renorm_shape, _, _ = raw_sizes
    batch_size, num_classes = probs_shape
    assert top_k_shape == (batch_size,)
    assert renorm_shape == (batch_size, num_classes)
    numel = batch_size * num_classes
    return {
        "flops": 20 * numel,
        "memory_bytes": numel * 4 * 2 + batch_size * 4,
        "dtype": "fp32",
    }

DESIGNED_VRAM_SIZE = 48
