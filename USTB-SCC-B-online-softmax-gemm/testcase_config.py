from __future__ import annotations

from typing import List, Tuple, Union
import torch

KernelArg = Union[torch.Tensor, int, float]

# (B, H, S, D, warmup, iters)
# 所有测试点均在单张 A800 VRAM 限制内
TESTCASES = [
    (1, 16,  512,   64,  5, 100),   # 小尺寸正确性基准
    (1, 16,  1024,  64,  5, 100),   # 中等序列长度
    (1, 16,  2048,  64,  5,  50),   # 典型推理长度
    (1, 32,  4096,  128, 5,  30),   # 大规模，多 head
    (2, 16,  2048,  64,  5,  30),   # batch > 1
    (1, 16,  1987,  64,  5, 100),   # 非整除 S，mask = nullptr
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize():
    """从标准输入读取测试点编号（1-based），返回参数 shape 列表和 (warmup, iters)。"""
    try:
        s = input().strip()
        if s:
            testcase_id = int(s)
        else:
            testcase_id = 1
    except (EOFError, ValueError):
        testcase_id = 1

    if testcase_id < 1 or testcase_id > len(TESTCASES):
        testcase_id = 1

    B, H, S, D, warmup, iters = TESTCASES[testcase_id - 1]

    shapes = [
        (B, H, S, D),          # Q
        (B, H, S, D),          # K
        (B, H, S, D),          # V
        (B, 1, S, S),          # mask
        (),                     # B  (scalar)
        (),                     # H  (scalar)
        (),                     # S  (scalar)
        (),                     # D  (scalar)
        (),                     # alpha (scalar)
        (B, H, S, D),          # O
    ]
    return shapes, (warmup, iters)


def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
    B, H, S, D = testcase_sizes[:4]

    # 输入值控制在 [-1, 1] 范围，避免 softmax 溢出
    Q = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device).uniform_(-1.0, 1.0)
    K = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device).uniform_(-1.0, 1.0)
    V = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device).uniform_(-1.0, 1.0)

    # mask: 测试点 6 使用 nullptr/None，其余测试点使用 causal mask
    if testcase_sizes[0] == 1 and testcase_sizes[2] == 1987:
        # 测试点 6：无 mask
        mask = None
    else:
        # Causal mask: 上三角 -inf，下三角 + 对角线 0
        mask = torch.triu(
            torch.full((S, S), float('-inf'), dtype=torch.bfloat16, device=device),
            diagonal=1
        )
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).contiguous()

    alpha = 1.0 / (D ** 0.5)
    O = torch.empty(B, H, S, D, dtype=torch.bfloat16, device=device)

    return [Q, K, V, mask, B, H, S, D, alpha, O]


def baseline(*args):
    """PyTorch 参考实现：O = softmax(α · Q @ K^T + mask) @ V"""
    Q, K, V, mask, B, H, S, D, alpha, O = args

    Q_t = Q.view(B, H, S, D).float()
    K_t = K.view(B, H, S, D).float()
    V_t = V.view(B, H, S, D).float()

    # Score = α × Q @ K^T
    Score = alpha * torch.matmul(Q_t, K_t.transpose(-2, -1))

    # 加 mask
    if mask is not None:
        mask_t = mask.view(B, 1, S, S).float()
        Score = Score + mask_t

    # P = softmax(Score)
    P = torch.softmax(Score, dim=-1)

    # O = P @ V
    O_t = torch.matmul(P, V_t)
    O.copy_(O_t.to(torch.bfloat16))

    return [Q, K, V, mask, B, H, S, D, alpha, O]


def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors,
          baseline_input_tensors, rtol=1e-2, atol=1e-2) -> bool:
    """比较用户 kernel 输出 (target) 与 baseline 输出"""
    # 输出 O 是最后一个参数
    baseline_O = baseline_input_tensors[-1]
    target_O = target_kernel_input_tensors[-1]

    # 检查 shape
    if baseline_O.shape != target_O.shape:
        print(f"Shape mismatch: baseline {list(baseline_O.shape)} vs "
              f"target {list(target_O.shape)}")
        return False

    # 检查 dtype
    if baseline_O.dtype != target_O.dtype:
        print(f"Dtype mismatch: baseline {baseline_O.dtype} vs "
              f"target {target_O.dtype}")
        return False

    # 数值比较（转为 float32 后比较）
    baseline_f32 = baseline_O.float()
    target_f32 = target_O.float()

    if not torch.allclose(baseline_f32, target_f32, rtol=rtol, atol=atol):
        diff = (baseline_f32 - target_f32).abs()
        print(f"max_abs_diff: {diff.max().item():.6e}")
        print(f"mean_abs_diff: {diff.mean().item():.6e}")
        # 打印差异最大的 3 个位置
        flat_idx = diff.flatten().argsort(descending=True)[:3]
        for idx in flat_idx:
            pos = torch.unravel_index(idx, diff.shape)
            print(f"  pos={tuple(p.item() for p in pos)}: "
                  f"baseline={baseline_f32[pos].item():.6f} "
                  f"target={target_f32[pos].item():.6f} "
                  f"diff={diff[pos].item():.6e}")
        return False

    return True


INPUT_CLASS = [
    "INPUT",    # Q
    "INPUT",    # K
    "INPUT",    # V
    "INPUT",    # mask (可为 None)
    "INPUT",    # B
    "INPUT",    # H
    "INPUT",    # S
    "INPUT",    # D
    "INPUT",    # alpha
    "OUTPUT",   # O
]


def getWorkload(testcase_sizes) -> dict:
    """
    计算 FLOPs 和访存量口径。

    FLOPs:
      - S = α · Q @ K^T: [B*H, S, D] × [B*H, D, S] → 2·B·H·S·S·D
      - O = softmax(S) @ V: [B*H, S, S] × [B*H, S, D] → 2·B·H·S·S·D
      - 融合算子 FLOPs = 4·B·H·S·S·D（softmax 的 exp/sum 为 SFU，不在此口径）

    Memory (baseline / 分立 kernel 口径):
      - 读 Q, K, V + 写/读 S, P + 写 O
      - = (3 + 1) × B·H·S·D × 2 + 4 × B·H·S·S × 2 bytes (bf16)

    Memory (融合后 / 参赛者目标口径):
      - 读 Q, K, V, mask + 写 O
      - = (3 × B·H·S·D + B·S·S) × 2 + B·H·S·D × 2 bytes (bf16)
      - 此处按 baseline 口径统计以便公平比较
    """
    B, H, S, D = testcase_sizes[:4]

    flops = 4 * B * H * S * S * D

    # 按 baseline 口径：包含 S 和 P 的 HBM 往返
    memory_bytes = (3 * B * H * S * D + 4 * B * H * S * S + B * H * S * D) * 2

    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "bf16",
    }


DESIGNED_VRAM_SIZE = 80
