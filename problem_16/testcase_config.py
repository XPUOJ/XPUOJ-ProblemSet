from __future__ import annotations

def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 4

try:
    from typing import List, Tuple, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]

    # Standard TESTCASES so auto_tune_testcases.py can edit sizes.
    # (total_tokens, in_features, out_features, num_groups, warmup, iters)
    # NOTE: original values could OOM in baseline due to w_expanded materialization; start smaller and let tuner scale.
    TESTCASES = [
        (19712, 1280, 128, 4, 3, 10),
        (211456, 128, 128, 8, 3, 10),
        (6144, 768, 768, 16, 3, 10),
        (20480, 128, 1280, 32, 3, 10),
    ]



    def getTestCaseSize():
        """返回每个参数的尺寸描述及 (预热轮数, 测试轮数)。"""
        testcase_id = int(input())
        if testcase_id < 1 or testcase_id > len(TESTCASES):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(TESTCASES)}]")
        total_tokens, in_features, out_features, num_groups, warmup, iters = TESTCASES[testcase_id - 1]
        sizes = [
            (total_tokens, in_features),
            (num_groups, out_features, in_features),
            (num_groups, out_features, in_features),
            (num_groups,),
            (total_tokens, out_features * 2),
            (), (), (), (),
        ]
        return sizes, (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        """根据尺寸生成测试数据，使用 tensor 操作。"""
        raw = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw) == 9, f"Expect 9 shape entries, got {len(raw)}"
        x_shape, w1_shape, w3_shape, gs_shape, out_shape = raw[:5]
        total_tokens, in_features = x_shape
        num_groups, out_features, _ = w1_shape
        assert w3_shape == w1_shape and gs_shape == (num_groups,) and out_shape == (total_tokens, out_features * 2)
        
        x = torch.randn(*x_shape, dtype=torch.bfloat16, device=device) * 0.5
        w1 = torch.randn(*w1_shape, dtype=torch.bfloat16, device=device) * 0.5
        w3 = torch.randn(*w3_shape, dtype=torch.bfloat16, device=device) * 0.5
        
        base = total_tokens // num_groups
        rem = total_tokens - base * num_groups
        # Tensor operation for gs
        gs = torch.full((num_groups,), base, dtype=torch.int32, device=device)
        gs[:rem] += 1
        group_sizes = gs
        
        out = torch.zeros(*out_shape, dtype=torch.bfloat16, device=device)
        return [x, w1, w3, group_sizes, out, total_tokens, in_features, out_features, num_groups]

    def baseline(
        x: torch.Tensor,
        w1: torch.Tensor,
        w3: torch.Tensor,
        group_sizes: torch.Tensor,
        out: torch.Tensor,
        total_tokens: int,
        in_features: int,
        out_features: int,
        num_groups: int,
    ) -> List[KernelArg]:
        """baseline：按组计算 out[token_start:token_end] = x[token_start:token_end] @ concat(w1[g],w3[g]).T"""
        assert torch.is_tensor(x) and torch.is_tensor(w1) and torch.is_tensor(w3)
        assert torch.is_tensor(group_sizes) and torch.is_tensor(out)
        assert x.shape == (total_tokens, in_features) and out.shape == (total_tokens, out_features * 2)
        
        # Vectorized implementation
        # 1. Create group indices for each token
        # group_sizes: (G,) -> repeat each group index size times
        group_indices = torch.repeat_interleave(
            torch.arange(num_groups, device=x.device),
            group_sizes.long()
        ) # (M_total,)
        
        # 2. Gather weights
        # w1: (G, Out, In), w3: (G, Out, In)
        # w_concat: (G, 2*Out, In)
        w_concat = torch.cat([w1, w3], dim=1)
        
        # w_expanded: (M_total, 2*Out, In)
        w_expanded = w_concat[group_indices]
        
        # 3. Compute out = x @ w.T
        # x: (M, In) -> (M, 1, In)
        # w_expanded: (M, 2*Out, In) -> transpose -> (M, In, 2*Out)
        # matmul -> (M, 1, 2*Out) -> squeeze -> (M, 2*Out)
        
        res = torch.matmul(x.unsqueeze(1).float(), w_expanded.transpose(1, 2).float()).squeeze(1)
        out.copy_(res.to(torch.bfloat16))
        
        return [x, w1, w3, group_sizes, out, total_tokens, in_features, out_features, num_groups]

    def check(
        testcase_sizes,
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        """比较用户 kernel 输出 out 与 baseline 输出。"""
        raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(target_kernel_input_tensors) >= 5 and len(baseline_input_tensors) >= 5
        out_t = target_kernel_input_tensors[4]
        out_ref = baseline_input_tensors[4]
        if not (torch.is_tensor(out_t) and torch.is_tensor(out_ref)):
            print(f"[FAIL] out must be tensor, got types: {type(out_t)}, {type(out_ref)}", file=sys.stderr)
            return False
        if out_t.shape != out_ref.shape:
            print(f"[FAIL] shape mismatch: target {out_t.shape}, ref {out_ref.shape}", file=sys.stderr)
            return False
        if out_t.dtype != out_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {out_t.dtype}, ref {out_ref.dtype}", file=sys.stderr)
            return False
        ok = torch.allclose(out_t.float(), out_ref.float(), rtol=rtol, atol=atol)
        if not ok:
            diff = (out_t.float() - out_ref.float()).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except Exception:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 按 group_sizes 为每个 token 选择所属 group 的权重，再计算输出

    workload 口径：
      - flops = 4 * total_tokens * in_features * out_features
        理由：每个 token 最终会产生 2 * out_features 个输出，因此按一次 (in_features -> 2*out_features) 的 matmul 统计。
      - memory_bytes = (total_tokens * in_features + 2 * num_groups * out_features * in_features + total_tokens * out_features * 2) * 2 + num_groups * 4
        理由：需要读取输入、两组权重 w1/w3、group_sizes，并写出拼接后的输出 out。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, w1_shape, w3_shape, group_sizes_shape, out_shape, _, _, _, _ = raw_sizes
    total_tokens, in_features = x_shape
    num_groups, out_features, in_features_w = w1_shape
    assert w3_shape == w1_shape and in_features == in_features_w
    assert group_sizes_shape == (num_groups,) and out_shape == (total_tokens, out_features * 2)
    return {
        "flops": 4 * total_tokens * in_features * out_features,
        "memory_bytes": (total_tokens * in_features + 2 * num_groups * out_features * in_features + total_tokens * out_features * 2) * 2 + num_groups * 4,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
