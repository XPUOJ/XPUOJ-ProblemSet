from __future__ import annotations


TESTCASES = [
    (1, 64, 1, 16, 32, 1, 16, 1, 2, 10),
    (1, 128, 1, 16, 64, 1, 16, 1, 2, 10),
    (1, 256, 1, 16, 128, 1, 16, 1, 1, 6),
    (2, 512, 1, 16, 64, 1, 16, 1, 1, 5),
    (4, 1024, 1, 16, 64, 1, 16, 1, 0, 2),
    (8, 1024, 1, 16, 128, 1, 32, 1, 0, 1),
    (1, 4096, 1, 16, 64, 1, 16, 1, 0, 1),
    (2, 4096, 1, 16, 64, 1, 16, 1, 0, 1),
    (1, 8192, 1, 16, 64, 1, 16, 1, 0, 1),
    (1, 256, 1, 16, 64, 2, 16, 1, 1, 5),
    (2, 512, 1, 16, 64, 4, 16, 1, 0, 2),
    (4, 1024, 1, 16, 64, 8, 16, 1, 0, 1),
    (1, 256, 2, 32, 64, 1, 16, 1, 1, 5),
    (2, 512, 2, 32, 64, 1, 16, 1, 0, 2),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


try:
    from typing import List, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None

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

    def getTestCaseSize():
        testcase_id = _get_testcase_index()
        global CURRENT_CASE
        B, seq_len, H, HQ, D, S, block_size, is_causal, warmup, iters = TESTCASES[testcase_id]
        CURRENT_CASE = (B, seq_len, H, HQ, D, S, block_size, is_causal, 20250301 + testcase_id)
        return [
            (B, seq_len, HQ, D),
            (B, seq_len, H, D),
            (B, seq_len, H, D),
            (B, seq_len, H, S),
            (B, seq_len, HQ, D),
            (), (), (), (), (), (), (), (),
        ], (warmup, iters)

    def _make_block_indices(B: int, seq_len: int, H: int, S: int, block_size: int, device: str):
        data = torch.full((B, seq_len, H, S), seq_len, dtype=torch.int32)
        for t in range(seq_len):
            cur_block = t // block_size
            if cur_block == 0:
                vals = [0]
            else:
                start = max(0, cur_block - S)
                vals = list(range(start, cur_block))
            vals = vals[-S:]
            for b in range(B):
                for h in range(H):
                    data[b, t, h, : len(vals)] = torch.tensor(vals, dtype=torch.int32)
        return data.to(device=device)

    def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
        B, seq_len, H, HQ, D, S, block_size, is_causal, seed = CURRENT_CASE
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        q = (torch.randn(B, seq_len, HQ, D, dtype=torch.float16, device=device, generator=gen) * 0.125).contiguous()
        k = (torch.randn(B, seq_len, H, D, dtype=torch.float16, device=device, generator=gen) * 0.125).contiguous()
        v = (torch.randn(B, seq_len, H, D, dtype=torch.float16, device=device, generator=gen) * 0.125).contiguous()
        block_indices = _make_block_indices(B, seq_len, H, S, block_size, device).contiguous()
        output = torch.empty(B, seq_len, HQ, D, dtype=torch.float16, device=device)
        return [q, k, v, block_indices, output, B, seq_len, H, HQ, D, S, block_size, int(is_causal)]

    def baseline(q, k, v, block_indices, output, B, seq_len, H, HQ, D, S, block_size, is_causal):
        G = HQ // H
        scale = float(D ** -0.5)
        out = torch.empty_like(output)
        offsets = torch.arange(block_size, device=q.device, dtype=torch.long)
        query_pos = torch.arange(seq_len, device=q.device, dtype=torch.long)[:, None]
        for b in range(B):
            for h in range(H):
                idx = block_indices[b, :, h, :].long().unsqueeze(-1) * block_size + offsets
                idx = idx.reshape(seq_len, S * block_size)
                valid = (idx >= 0) & (idx < seq_len)
                if int(is_causal):
                    valid = valid & (idx <= query_pos)
                idx_clamped = idx.clamp(0, seq_len - 1)
                k_sel = k[b, :, h, :].float()[idx_clamped]
                v_sel = v[b, :, h, :].float()[idx_clamped]
                q_group = q[b, :, h * G : (h + 1) * G, :].float()
                scores = torch.einsum("tgd,tld->tgl", q_group, k_sel) * scale
                scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
                attn = torch.softmax(scores, dim=-1)
                out_group = torch.einsum("tgl,tld->tgd", attn, v_sel)
                out[b, :, h * G : (h + 1) * G, :].copy_(out_group.to(out.dtype))
        output.copy_(out)
        return [q, k, v, block_indices, output, B, seq_len, H, HQ, D, S, block_size, is_causal]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol=1e-2, atol=1e-2):
        del testcase_sizes, original_input_tensors
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
            print(
                f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f}, "
                f"mean_abs_diff={float(diff.mean().item()):.6f} (rtol={rtol}, atol={atol})",
                file=sys.stderr,
            )
            return False
        return True
except:
    pass


INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    q_shape, k_shape, v_shape, block_shape, output_shape = raw_sizes[:5]
    B, seq_len, HQ, D = q_shape
    _, _, H, D_k = k_shape
    _, _, H_v, D_v = v_shape
    _, _, _, S = block_shape
    assert D == D_k == D_v and H == H_v
    assert output_shape == q_shape
    try:
        block_size = CURRENT_CASE[6]
    except Exception:
        block_size = 16
    selected = S * block_size
    flops = 4 * B * seq_len * HQ * selected * D
    memory_bytes = (
        B * seq_len * HQ * D * 2
        + B * seq_len * H * D * 2 * 2
        + B * seq_len * H * S * 4
        + B * seq_len * HQ * D * 2
    )
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "fp16",
    }


DESIGNED_VRAM_SIZE = 48
