from __future__ import annotations


def _build_cases():
    cases = []
    for batch, ctxs in [
        (1, [2048, 4096, 8192, 16384, 32768, 65536]),
        (2, [2048, 8192, 16384, 32768, 65536]),
        (4, [2048, 8192, 16384, 32768, 65536]),
        (8, [2048, 8192, 16384, 32768, 65536]),
        (16, [2048, 8192, 16384, 32768, 65536]),
        (32, [2048, 8192, 16384, 32768, 65536]),
    ]:
        for kv_ctx in ctxs:
            cases.append((batch, 16, 1, kv_ctx, 512, 64))
    return cases


TESTCASES = _build_cases()


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

    def _timing(batch: int, kv_ctx: int, dim: int, pe_dim: int):
        bytes_per_case = batch * kv_ctx * (dim + pe_dim) * 2
        if bytes_per_case > 1536 * 1024 * 1024:
            return 0, 1
        if bytes_per_case > 512 * 1024 * 1024:
            return 0, 2
        if bytes_per_case > 128 * 1024 * 1024:
            return 1, 4
        return 2, 10

    def getTestCaseSize():
        testcase_id = _get_testcase_index()
        global CURRENT_CASE
        batch, heads, kv_heads, kv_ctx, dim, pe_dim = TESTCASES[testcase_id]
        CURRENT_CASE = (batch, heads, kv_heads, kv_ctx, dim, pe_dim, 20250217 + testcase_id)
        warmup, iters = _timing(batch, kv_ctx, dim, pe_dim)
        return [
            (batch, heads, dim),
            (batch, heads, pe_dim),
            (batch, kv_ctx, kv_heads, dim),
            (batch, kv_ctx, kv_heads, pe_dim),
            (batch, heads, dim),
            (), (), (), (), (), (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
        batch, heads, kv_heads, kv_ctx, dim, pe_dim, seed = CURRENT_CASE
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        q = (torch.randn(batch, heads, dim, dtype=torch.float16, device=device, generator=gen) * 0.125).contiguous()
        q_pe = (torch.randn(batch, heads, pe_dim, dtype=torch.float16, device=device, generator=gen) * 0.125).contiguous()
        kv = (
            torch.randn(batch, kv_ctx, kv_heads, dim, dtype=torch.float16, device=device, generator=gen) * 0.125
        ).contiguous()
        k_pe = (
            torch.randn(batch, kv_ctx, kv_heads, pe_dim, dtype=torch.float16, device=device, generator=gen) * 0.125
        ).contiguous()
        output = torch.empty(batch, heads, dim, dtype=torch.float16, device=device)
        return [q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim]

    def baseline(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
        group_num = heads // kv_heads
        q_main = q.reshape(batch, kv_heads, group_num, dim).permute(0, 2, 1, 3).float()
        q_pos = q_pe.reshape(batch, kv_heads, group_num, pe_dim).permute(0, 2, 1, 3).float()
        kv_main = kv.permute(0, 2, 1, 3).float()
        k_pos = k_pe.permute(0, 2, 1, 3).float()
        query = torch.cat([q_main, q_pos], dim=-1)
        key = torch.cat([kv_main, k_pos], dim=-1)
        scale = float((dim + pe_dim) ** -0.5)
        scores = torch.einsum("bghd,bhsd->bghs", query, key)
        attention = torch.softmax(scores * scale, dim=-1)
        out = torch.einsum("bghs,bhsd->bghd", attention, kv_main)
        out = out.permute(0, 2, 1, 3).reshape(batch, heads, dim)
        output.copy_(out.to(output.dtype))
        return [q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol=2e-3, atol=2e-3):
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


INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    q_shape, q_pe_shape, kv_shape, k_pe_shape, output_shape = raw_sizes[:5]
    batch, heads, dim = q_shape
    _, _, pe_dim = q_pe_shape
    _, kv_ctx, kv_heads, dim_kv = kv_shape
    assert dim == dim_kv
    assert k_pe_shape == (batch, kv_ctx, kv_heads, pe_dim)
    assert output_shape == (batch, heads, dim)
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    return {
        "flops": qk_flops + pv_flops,
        "memory_bytes": (
            batch * heads * (dim + pe_dim) * 2
            + batch * kv_ctx * kv_heads * (dim + pe_dim) * 2
            + batch * heads * dim * 2
        ),
        "dtype": "fp16",
    }


DESIGNED_VRAM_SIZE = 48
