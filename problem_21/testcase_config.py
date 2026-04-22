from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    TESTCASES = [
        (4096, 2048, 16, 64, 4, 91),
        (2048, 4096, 8, 128, 4, 91),
        (8192, 1024, 16, 64, 4, 91),
        (3072, 1536, 32, 64, 4, 78),
        (1536, 4096, 8, 128, 6, 121),
    ]






    def _get_testcase_id() -> int:
        try:
            raw = input().strip()
        except EOFError:
            return 1
        if raw == "":
            return 1
        token = raw.split()[0]
        try:
            testcase_id = int(token)
        except ValueError:
            return 1
        if testcase_id < 1 or testcase_id > len(TESTCASES):
            return 1
        return testcase_id


    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        M, seq_len_kv, num_heads, head_dim, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (M, seq_len_kv, num_heads, head_dim)
        return [
            (M, num_heads, head_dim),
            (seq_len_kv, head_dim),
            (seq_len_kv,),
            (M, num_heads),
            (M,),
            (M,),
            (M, seq_len_kv),
            (),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        global CURRENT_CASE
        assert CURRENT_CASE is not None
        M, seq_len_kv, num_heads, head_dim = CURRENT_CASE
        (
            q_shape,
            kv_shape,
            kv_sf_shape,
            weights_shape,
            ks_shape,
            ke_shape,
            logits_shape,
            m_shape,
            seq_shape,
            heads_shape,
            dim_shape,
        ) = testcase_sizes

        assert q_shape == (M, num_heads, head_dim)
        assert kv_shape == (seq_len_kv, head_dim)
        assert kv_sf_shape == (seq_len_kv,)
        assert weights_shape == (M, num_heads)
        assert ks_shape == (M,)
        assert ke_shape == (M,)
        assert logits_shape == (M, seq_len_kv)
        assert m_shape == seq_shape == heads_shape == dim_shape == ()

        q_bf16 = torch.empty(M, num_heads, head_dim, dtype=torch.bfloat16, device=device).uniform_(-0.1, 0.1)
        q = q_bf16.to(torch.float8_e4m3fn)
        kv_bf16 = torch.empty(seq_len_kv, head_dim, dtype=torch.bfloat16, device=device).uniform_(-0.1, 0.1)
        kv_amax = kv_bf16.abs().float().amax(dim=1, keepdim=True).clamp(min=1e-4)
        kv_sf = (kv_amax / 448.0).squeeze(-1)
        kv = (kv_bf16 * (1.0 / kv_sf.unsqueeze(-1))).to(torch.float8_e4m3fn)
        weights = torch.empty(M, num_heads, dtype=torch.float32, device=device).uniform_(-0.1, 0.1)
        cu_seqlen_ks = torch.randint(0, max(1, seq_len_kv // 2), (M,), dtype=torch.int32, device=device)
        window_size = torch.randint(max(1, seq_len_kv // 4), max(2, seq_len_kv // 2 + 1), (M,), dtype=torch.int32, device=device)
        cu_seqlen_ke = torch.minimum(cu_seqlen_ks + window_size, torch.tensor(seq_len_kv, dtype=torch.int32, device=device))
        cu_seqlen_ke = torch.maximum(cu_seqlen_ke, cu_seqlen_ks + 1)
        logits = torch.empty(M, seq_len_kv, dtype=torch.float32, device=device)

        return [q, kv, kv_sf, weights, cu_seqlen_ks, cu_seqlen_ke, logits, M, seq_len_kv, num_heads, head_dim]


    def baseline(
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_sf: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        logits: torch.Tensor,
        M: int,
        seq_len_kv: int,
        num_heads: int,
        head_dim: int,
    ) -> List[KernelArg]:
        q_fp32 = q.float()
        kv_fp32 = kv.float() * kv_sf.unsqueeze(-1)
        out = torch.zeros(M, seq_len_kv, dtype=torch.float32, device=q.device)
        for h in range(num_heads):
            score_h = torch.matmul(q_fp32[:, h, :], kv_fp32.t())
            score_h = score_h.relu()
            out = out + score_h * weights[:, h:h + 1]
        ar = torch.arange(0, seq_len_kv, device=q.device, dtype=torch.int32)[None, :]
        mask = (ar >= cu_seqlen_ks[:, None]) & (ar < cu_seqlen_ke[:, None])
        out = out.masked_fill(~mask, float("-inf"))
        logits.copy_(out)
        return [q, kv, kv_sf, weights, cu_seqlen_ks, cu_seqlen_ke, logits, M, seq_len_kv, num_heads, head_dim]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        logits_t = target_kernel_input_tensors[6]
        logits_ref = baseline_input_tensors[6]

        if not (torch.is_tensor(logits_t) and torch.is_tensor(logits_ref)):
            print(f"[FAIL] logits must be tensor, got {type(logits_t)} and {type(logits_ref)}", file=sys.stderr)
            return False
        if logits_t.shape != logits_ref.shape:
            print(f"[FAIL] shape mismatch: target {logits_t.shape}, ref {logits_ref.shape}", file=sys.stderr)
            return False
        if logits_t.dtype != logits_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {logits_t.dtype}, ref {logits_ref.dtype}", file=sys.stderr)
            return False

        finite_mask = torch.isfinite(logits_ref)
        inf_mask = ~finite_mask
        if inf_mask.any():
            same_inf = torch.equal(torch.isneginf(logits_t[inf_mask]), torch.isneginf(logits_ref[inf_mask]))
            if not same_inf:
                print("[FAIL] masked -inf positions mismatch", file=sys.stderr)
                return False
        if finite_mask.any():
            ok = torch.allclose(logits_t[finite_mask], logits_ref[finite_mask], rtol=rtol, atol=atol)
            if not ok:
                diff = (logits_t[finite_mask] - logits_ref[finite_mask]).abs()
                max_diff = float(diff.max().item())
                print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
                return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 按 head 计算 q @ kv.T，做 ReLU，再按 weights 加权求和，最后按区间 mask

    workload 口径：
      - flops = num_heads * (2 * M * seq_len_kv * head_dim + 3 * M * seq_len_kv) + M * seq_len_kv
        理由：每个 head 的主开销是矩阵乘，再加上逐元素 ReLU、加权累加和最终 mask 处理。
      - memory_bytes = q_bytes + kv_bytes + kv_sf_bytes + weights_bytes + ks_ke_bytes + logits_bytes
        理由：需要读取 q、量化 kv 与 kv_sf、weights、区间边界 ks/ke，并写出 logits。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    q_shape, kv_shape, kv_sf_shape, weights_shape, ks_shape, ke_shape, logits_shape, _, _, _, _ = raw_sizes
    M, num_heads, head_dim = q_shape
    seq_len_kv, head_dim_kv = kv_shape
    assert head_dim == head_dim_kv
    assert kv_sf_shape == (seq_len_kv,)
    assert weights_shape == (M, num_heads)
    assert ks_shape == (M,) and ke_shape == (M,)
    assert logits_shape == (M, seq_len_kv)
    score_elems = M * seq_len_kv
    flops = num_heads * (2 * score_elems * head_dim + 3 * score_elems) + score_elems
    memory_bytes = M * num_heads * head_dim + seq_len_kv * head_dim + seq_len_kv * 4 + M * num_heads * 4 + M * 4 + M * 4 + M * seq_len_kv * 4
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "fp8",
    }

DESIGNED_VRAM_SIZE = 48
