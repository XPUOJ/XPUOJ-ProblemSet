from __future__ import annotations


def getNumOfTestcases() -> int:
    return 4


try:
    from typing import List, Tuple, Union
    import math
    import sys
    import torch
    import torch.nn.functional as F

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    BLOCK_TOKEN = 128

    # (d_hidden, d_expert, n_routed_experts, group_sum, seed, warmup, iters)
    # group_sum corresponds to batch_size * seq_len * n_experts_per_token after routing.
    TESTCASES = [
        (3584, 1024, 4, 8192, 81394, 2, 8),
        (3584, 1024, 4, 16384, 81395, 1, 5),
        (7168, 2048, 8, 8192, 81396, 1, 3),
        (7168, 2048, 8, 32768, 81397, 0, 1),
    ]

    def _get_testcase_index() -> int:
        try:
            raw = input().strip()
        except EOFError:
            return 0
        if raw == "":
            return 0
        try:
            testcase_id = int(raw.split()[0])
        except ValueError:
            return 0
        if 0 <= testcase_id < len(TESTCASES):
            return testcase_id
        return 0

    def _group_metadata(group_sum: int, n_experts: int, seed: int, device: str):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        if group_sum < n_experts:
            raise ValueError("group_sum must be at least n_experts")
        base = torch.arange(n_experts, device=device, dtype=torch.int64)
        rest = torch.randint(0, n_experts, (group_sum - n_experts,), device=device, generator=gen)
        assignments = torch.cat([base, rest], dim=0)
        perm = torch.randperm(group_sum, device=device, generator=gen)
        assignments = assignments[perm]
        group_sizes = torch.bincount(assignments, minlength=n_experts).to(torch.int32)
        group_offsets = torch.cumsum(group_sizes, dim=0) - group_sizes

        counts = group_sizes.cpu().tolist()
        padded_offsets = [0 for _ in range(n_experts)]
        for i in range(1, n_experts):
            padded_offsets[i] = padded_offsets[i - 1] + math.ceil((counts[i - 1] + 1) / BLOCK_TOKEN) * BLOCK_TOKEN
        group_padded_offsets = torch.tensor(padded_offsets, dtype=torch.int32, device=device)

        num_blocks = math.ceil(group_sum / BLOCK_TOKEN) + n_experts
        group_idx_for_bx = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        for bx in range(num_blocks):
            m_start_padded = bx * BLOCK_TOKEN
            cur = 0
            for i, off in enumerate(padded_offsets):
                if m_start_padded >= off:
                    cur = i
            group_idx_for_bx[bx] = cur
        return group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx

    def getTestCaseSize():
        testcase_id = _get_testcase_index()
        global CURRENT_CASE
        d_hidden, d_expert, n_experts, group_sum, seed, warmup, iters = TESTCASES[testcase_id]
        CURRENT_CASE = (d_hidden, d_expert, n_experts, group_sum, seed)
        num_blocks = math.ceil(group_sum / BLOCK_TOKEN) + n_experts
        return [
            (group_sum, d_hidden),
            (n_experts, d_expert, d_hidden),
            (n_experts, d_expert, d_hidden),
            (n_experts, d_hidden, d_expert),
            (group_sum,),
            (n_experts,),
            (n_experts,),
            (n_experts,),
            (num_blocks,),
            (group_sum, d_expert),
            (group_sum, d_hidden),
            (), (), (), (), (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
        d_hidden, d_expert, n_experts, group_sum, seed = CURRENT_CASE
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        expert_tokens = (
            torch.randn(group_sum, d_hidden, dtype=torch.float16, device=device, generator=gen) * 0.02
        ).contiguous()
        routed_expert_gate = (
            torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16, device=device, generator=gen)
            / math.sqrt(d_expert)
        ).contiguous()
        routed_expert_up = (
            torch.randn(n_experts, d_expert, d_hidden, dtype=torch.float16, device=device, generator=gen)
            / math.sqrt(d_expert)
        ).contiguous()
        routed_expert_down = (
            torch.randn(n_experts, d_hidden, d_expert, dtype=torch.float16, device=device, generator=gen)
            / math.sqrt(d_hidden)
        ).contiguous()
        routed_expert_weights = torch.rand(group_sum, dtype=torch.float16, device=device, generator=gen).contiguous()
        group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx = _group_metadata(
            group_sum, n_experts, seed + 17, device
        )
        up_logits = torch.empty(group_sum, d_expert, dtype=torch.float16, device=device)
        output = torch.empty(group_sum, d_hidden, dtype=torch.float16, device=device)

        return [
            expert_tokens,
            routed_expert_gate,
            routed_expert_up,
            routed_expert_down,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            group_idx_for_bx,
            up_logits,
            output,
            int(group_sum),
            int(d_hidden),
            int(d_expert),
            int(n_experts),
            int(BLOCK_TOKEN),
        ]

    def baseline(
        expert_tokens,
        routed_expert_gate,
        routed_expert_up,
        routed_expert_down,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
        output,
        group_sum,
        d_hidden,
        d_expert,
        n_experts,
        block_token,
    ):
        output.zero_()
        for expert_id in range(int(n_experts)):
            count = int(group_sizes[expert_id].item())
            if count == 0:
                continue
            start = int(group_offsets[expert_id].item())
            end = start + count
            x = expert_tokens[start:end].float()
            gate = F.silu(x @ routed_expert_gate[expert_id].float().t())
            up = x @ routed_expert_up[expert_id].float().t()
            hidden = (gate * up).to(torch.float16)
            up_logits[start:end].copy_(hidden)
            out = hidden.float() @ routed_expert_down[expert_id].float().t()
            out = out * routed_expert_weights[start:end].float().unsqueeze(1)
            output[start:end].copy_(out.to(output.dtype))
        return [
            expert_tokens,
            routed_expert_gate,
            routed_expert_up,
            routed_expert_down,
            routed_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            group_idx_for_bx,
            up_logits,
            output,
            group_sum,
            d_hidden,
            d_expert,
            n_experts,
            block_token,
        ]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol=3e-2, atol=3e-2):
        del testcase_sizes, original_input_tensors
        output_t = target_kernel_input_tensors[10]
        output_ref = baseline_input_tensors[10]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(output_t.float(), output_ref.float(), rtol=rtol, atol=atol):
            diff = (output_t.float() - output_ref.float()).abs()
            print(
                f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f}, "
                f"mean_abs_diff={float(diff.mean().item()):.6f} (rtol={rtol}, atol={atol})",
                file=sys.stderr,
            )
            return False
        return True
except:
    pass


INPUT_CLASS = [
    "INPUT", "INPUT", "INPUT", "INPUT", "INPUT",
    "INPUT", "INPUT", "INPUT", "INPUT",
    "OUTPUT", "OUTPUT",
    "INPUT", "INPUT", "INPUT", "INPUT", "INPUT",
]


def getWorkload(testcase_sizes) -> dict:
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    token_shape, gate_shape, _, down_shape, weight_shape = raw_sizes[:5]
    group_sum, d_hidden = token_shape
    n_experts, d_expert, d_hidden_w = gate_shape
    assert d_hidden == d_hidden_w
    assert down_shape == (n_experts, d_hidden, d_expert)
    assert weight_shape == (group_sum,)
    flops = 6 * group_sum * d_hidden * d_expert
    memory_bytes = (
        group_sum * d_hidden * 2
        + n_experts * d_expert * d_hidden * 2 * 2
        + n_experts * d_hidden * d_expert * 2
        + group_sum * 2
        + group_sum * d_expert * 2
        + group_sum * d_hidden * 2
    )
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "fp16",
    }


DESIGNED_VRAM_SIZE = 48
