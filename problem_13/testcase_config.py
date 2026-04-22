from __future__ import annotations

def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 4

try:
    from typing import List, Tuple, Any, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]

    BLOCK_SIZE = 128

    # Standard TESTCASES so auto_tune_testcases.py can edit sizes.
    # (M_total, K, N, num_groups, warmup, iters)
    # NOTE: original values were large enough to OOM during genTestCase; start smaller and let the tuner scale.
    TESTCASES = [
        (3072, 768, 768, 16, 3, 37),
        (3072, 256, 1024, 32, 5, 100),
        (3072, 1024, 256, 32, 5, 96),
        (5376, 512, 1024, 32, 3, 27),
    ]




    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """返回每个参数的尺寸描述及可选的 (预热轮数, 测试轮数)。"""
        testcase_id = int(input())
        if testcase_id < 1 or testcase_id > len(TESTCASES):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(TESTCASES)}]")
        M_total, K, N, num_groups, warmup, iters = TESTCASES[testcase_id - 1]
        K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        sizes = [
            (M_total, K),                           # a_data
            (M_total, K_blocks),                    # a_sf
            (num_groups, N, K),                     # b_data
            (num_groups, N_blocks, K_blocks),       # b_sf
            (M_total,),                             # m_indices
            (M_total, N),                           # D
            (),                                     # M_total (scalar)
            (),                                     # K (scalar)
            (),                                     # N (scalar)
            (),                                     # num_groups (scalar)
        ]
        return sizes, (warmup, iters)

    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """根据尺寸生成测试数据，使用 tensor 操作。"""
        raw = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw) == 10, f"Expect 10 shape entries, got {len(raw)}"
        a_shape, a_sf_shape, b_shape, b_sf_shape, m_indices_shape, d_shape = raw[:6]
        m_total, k = a_shape
        num_groups, n, _ = b_shape
        assert m_indices_shape == (m_total,) and d_shape == (m_total, n)
        
        # a_data: FP8
        a_fp32 = torch.randn(*a_shape, device=device, dtype=torch.float32) * 0.5
        a_fp32 = torch.clamp(a_fp32, -448.0, 448.0)
        a_data = a_fp32.to(torch.float8_e4m3fn)
        a_sf = torch.rand(*a_sf_shape, device=device, dtype=torch.float32) * 9.9 + 0.1

        # b_data, b_sf
        b_fp32 = torch.randn(*b_shape, device=device, dtype=torch.float32) * 0.5
        b_fp32 = torch.clamp(b_fp32, -448.0, 448.0)
        b_data = b_fp32.to(torch.float8_e4m3fn)
        b_sf = torch.rand(*b_sf_shape, device=device, dtype=torch.float32) * 9.9 + 0.1

        # m_indices: 
        m_indices = torch.randint(0, num_groups, (m_total,), device=device, dtype=torch.int32)
        mask = torch.rand(m_total, device=device) < 0.1
        m_indices[mask] = -1

        d = torch.zeros(*d_shape, device=device, dtype=torch.bfloat16)
        return [a_data, a_sf, b_data, b_sf, m_indices, d, m_total, k, n, num_groups]

    def _expand_a_sf(a_sf: torch.Tensor, k: int) -> torch.Tensor:
        """将 a_sf [..., k_blocks] 沿最后一维扩展到 k。"""
        # Batched expansion
        ex = a_sf.repeat_interleave(BLOCK_SIZE, dim=-1)
        if ex.shape[-1] >= k:
            return ex[..., :k].contiguous()
        return torch.cat([ex, ex[..., -1:].expand(*ex.shape[:-1], k - ex.shape[-1])], dim=-1)

    def baseline(
        a_data: torch.Tensor,
        a_sf: torch.Tensor,
        b_data: torch.Tensor,
        b_sf: torch.Tensor,
        m_indices: torch.Tensor,
        d: torch.Tensor,
        m_total: int,
        k: int,
        n: int,
        num_groups: int,
    ) -> List[KernelArg]:
        """baseline：按 M-grouped FP8 GEMM NT contiguous 规则计算 d，原地修改 d。"""
        assert torch.is_tensor(a_data) and torch.is_tensor(a_sf) and torch.is_tensor(b_data)
        assert torch.is_tensor(b_sf) and torch.is_tensor(m_indices) and torch.is_tensor(d)
        assert a_data.shape == (m_total, k) and d.shape == (m_total, n)
        assert b_data.shape == (num_groups, n, k)

        # a 反量化
        a_sf_exp = _expand_a_sf(a_sf, k)  # (m_total, k)
        a_fp32 = a_data.float() * a_sf_exp
        
        # b 反量化
        # b_sf: (num_groups, n_blocks, k_blocks)
        b_sf_exp = b_sf.repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)
        b_sf_exp = b_sf_exp[:, :n, :k]
        b_fp32 = b_data.float() * b_sf_exp # (G, N, K)

        d.zero_()
        
        # Gather B matrices
        valid_mask = (m_indices >= 0) & (m_indices < num_groups)
        safe_indices = m_indices.clone()
        safe_indices[~valid_mask] = 0
        
        B_gathered = b_fp32[safe_indices.long()] # (M, N, K)
        
        # Compute D = A @ B.T
        # A: (M, K) -> (M, 1, K)
        # B_gathered: (M, N, K) -> (M, K, N) (transpose)
        res = torch.matmul(a_fp32.unsqueeze(1), B_gathered.transpose(1, 2)).squeeze(1)
        
        # Apply mask
        res = res * valid_mask.unsqueeze(1).float()
        
        d.copy_(res.to(torch.bfloat16))
        
        return [a_data, a_sf, b_data, b_sf, m_indices, d, m_total, k, n, num_groups]

    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        """比较用户 kernel 输出 D 与 baseline 输出。"""
        raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw_sizes) >= 6
        assert len(target_kernel_input_tensors) >= 6 and len(baseline_input_tensors) >= 6
        D_t = target_kernel_input_tensors[5]
        D_ref = baseline_input_tensors[5]
        if not (torch.is_tensor(D_t) and torch.is_tensor(D_ref)):
            print(f"[FAIL] D must be tensor, got D_t type: {type(D_t)}, D_ref type: {type(D_ref)}", file=sys.stderr)
            return False
        if D_t.shape != D_ref.shape:
            print(f"[FAIL] shape mismatch: target {D_t.shape}, ref {D_ref.shape}", file=sys.stderr)
            return False
        if D_t.dtype != D_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {D_t.dtype}, ref {D_ref.dtype}", file=sys.stderr)
            return False
        D_t_f = D_t.float()
        D_ref_f = D_ref.float()
        ok = torch.allclose(D_t_f, D_ref_f, rtol=rtol, atol=atol)
        if not ok:
            diff = (D_t_f - D_ref_f).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            max_idx = diff.argmax()
            max_pos = tuple(torch.unravel_index(max_idx, D_t.shape))
            print(f"[FAIL] max diff at {max_pos}: target={float(D_t_f.flatten()[max_idx].item()):.6f}, ref={float(D_ref_f.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        return True
except Exception:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 M-grouped FP8 GEMM NT contiguous，先按 scale 反量化，再做 grouped GEMM

    workload 口径：
      - flops = 2 * expected_valid_m * N * K
        理由：主体仍是按有效行执行的 GEMM；无效行比例沿用 testcase_config 的约 10% 估计。
      - memory_bytes = M_total * K + M_total * K_blocks * 4 + num_groups * N * K + num_groups * N_blocks * K_blocks * 4 + M_total * 4 + M_total * N * 2
        理由：需要读取 fp8 数据、对应 scale、m_indices，并写出 bf16 输出 D。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, a_sf_shape, b_shape, b_sf_shape, m_indices_shape, d_shape, _, _, _, _ = raw_sizes
    M_total, K = a_shape
    num_groups, N, K_b = b_shape
    assert K == K_b and m_indices_shape == (M_total,) and d_shape == (M_total, N)
    K_blocks = a_sf_shape[1]
    N_blocks = b_sf_shape[1]
    expected_valid_m = (9 * M_total + 5) // 10
    return {
        "flops": 2 * expected_valid_m * N * K,
        "memory_bytes": M_total * K + M_total * K_blocks * 4 + num_groups * N * K + num_groups * N_blocks * K_blocks * 4 + M_total * 4 + M_total * N * 2,
        "dtype": "fp8",
    }

DESIGNED_VRAM_SIZE = 48
