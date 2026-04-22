from __future__ import annotations

def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 4

try:
    from typing import List, Tuple, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]

    BLOCK_SIZE = 128

    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """返回每个参数的尺寸描述及 (预热轮数, 测试轮数)。"""
        testcase_id = int(input())
        testcases = [
            (4, 1536, 1536, 1536, 11, 209),
            (8, 512, 2048, 1024, 18, 356),
            (16, 256, 4096, 512, 19, 364),
            (32, 128, 8192, 256, 19, 367),
        ]



        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        num_groups, max_m, n, k, warmup, iters = testcases[testcase_id - 1]
        k_blocks = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
        sizes = [
            (num_groups, max_m, k),                     # a_data
            (num_groups, max_m, k_blocks),             # a_sf
            (num_groups, n, k),                        # b_data
            (num_groups, n_blocks, k_blocks),         # b_sf
            (num_groups,),                             # masked_m
            (num_groups, max_m, n),                    # d
            (), (), (), (),                            # scalars
        ]
        return sizes, (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
        """根据尺寸生成测试数据，使用 tensor 操作。"""
        raw = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw) == 10, f"Expect 10 shape entries, got {len(raw)}"
        a_shape, a_sf_shape, b_shape, b_sf_shape, masked_m_shape, d_shape = raw[:6]
        num_groups, max_m, k = a_shape
        _, n, _ = b_shape
        assert masked_m_shape == (num_groups,) and d_shape == (num_groups, max_m, n)
        
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

        # masked_m: 每组的实际行数 [1, max_m]
        masked_m = torch.randint(1, max_m + 1, (num_groups,), device=device, dtype=torch.int32)

        d = torch.zeros(*d_shape, device=device, dtype=torch.bfloat16)
        return [a_data, a_sf, b_data, b_sf, masked_m, d, num_groups, max_m, n, k]

    def _expand_a_sf(a_sf: torch.Tensor, k: int) -> torch.Tensor:
        """将 a_sf [..., k_blocks] 沿最后一维扩展到 k。"""
        # Batched expansion
        # a_sf: (..., kb)
        # repeat interleave 128 on last dim
        ex = a_sf.repeat_interleave(BLOCK_SIZE, dim=-1)
        if ex.shape[-1] >= k:
            return ex[..., :k].contiguous()
        # Fallback if padding needed (should not happen with standard blocking)
        return torch.cat([ex, ex[..., -1:].expand(*ex.shape[:-1], k - ex.shape[-1])], dim=-1)

    def baseline(
        a_data: torch.Tensor,
        a_sf: torch.Tensor,
        b_data: torch.Tensor,
        b_sf: torch.Tensor,
        masked_m: torch.Tensor,
        d: torch.Tensor,
        num_groups: int,
        max_m: int,
        n: int,
        k: int,
    ) -> List[KernelArg]:
        """baseline：按 M-grouped FP8 GEMM NT masked 规则计算 d，原地修改 d。"""
        assert torch.is_tensor(a_data) and torch.is_tensor(a_sf) and torch.is_tensor(b_data)
        assert torch.is_tensor(b_sf) and torch.is_tensor(masked_m) and torch.is_tensor(d)
        assert a_data.shape == (num_groups, max_m, k) and d.shape == (num_groups, max_m, n)
        assert b_data.shape == (num_groups, n, k)

        # a 反量化：逐组逐行与 a_sf 扩展后相乘
        a_sf_exp = _expand_a_sf(a_sf, k)  # (num_groups, max_m, k)
        a_fp32 = a_data.float() * a_sf_exp
        
        # b 反量化
        # b_sf: (num_groups, n_blocks, k_blocks)
        # expand to (num_groups, n, k)
        b_sf_exp = b_sf.repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)
        b_sf_exp = b_sf_exp[:, :n, :k]
        b_fp32 = b_data.float() * b_sf_exp

        d.zero_()
        
        # res = a @ b.T
        res = torch.matmul(a_fp32, b_fp32.transpose(1, 2))
        
        # Masking
        m_indices = torch.arange(max_m, device=a_data.device).unsqueeze(0) # (1, M)
        mask = m_indices < masked_m.unsqueeze(1) # (G, M)
        
        res = res * mask.unsqueeze(2).float()
        d.copy_(res.to(torch.bfloat16))
        
        return [a_data, a_sf, b_data, b_sf, masked_m, d, num_groups, max_m, n, k]

    def check(
        testcase_sizes,
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        """比较用户 kernel 输出 d 与 baseline 输出。"""
        raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw_sizes) >= 6 and len(target_kernel_input_tensors) >= 6 and len(baseline_input_tensors) >= 6
        d_t = target_kernel_input_tensors[5]
        d_ref = baseline_input_tensors[5]
        if not (torch.is_tensor(d_t) and torch.is_tensor(d_ref)):
            print(f"[FAIL] d must be tensor, got D_t type: {type(d_t)}, D_ref type: {type(d_ref)}", file=sys.stderr)
            return False
        if d_t.shape != d_ref.shape:
            print(f"[FAIL] shape mismatch: target {d_t.shape}, ref {d_ref.shape}", file=sys.stderr)
            return False
        if d_t.dtype != d_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {d_t.dtype}, ref {d_ref.dtype}", file=sys.stderr)
            return False
        d_t_f = d_t.float()
        d_ref_f = d_ref.float()
        ok = torch.allclose(d_t_f, d_ref_f, rtol=rtol, atol=atol)
        if not ok:
            diff = (d_t_f - d_ref_f).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            max_idx = diff.argmax()
            max_pos = tuple(torch.unravel_index(max_idx, d_t.shape))
            print(f"[FAIL] max diff at {max_pos}: target={float(d_t_f.flatten()[max_idx].item()):.6f}, ref={float(d_ref_f.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        return True
except Exception:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 M-grouped FP8 GEMM NT masked，先按 scale 反量化，再按 masked_m 做分组 GEMM

    workload 口径：
      - flops = num_groups * (max_m + 1) * n * k
        理由：实际有效行数由 masked_m 决定，这里按随机 masked_m 的平均有效行数统计。
      - memory_bytes = num_groups * max_m * k + num_groups * max_m * k_blocks * 4 + num_groups * n * k + num_groups * n_blocks * k_blocks * 4 + num_groups * 4 + num_groups * max_m * n * 2
        理由：需要读取 fp8 数据、对应 scale、masked_m，并写出 bf16 输出 d。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, a_sf_shape, b_shape, b_sf_shape, masked_m_shape, d_shape, _, _, _, _ = raw_sizes
    num_groups, max_m, k = a_shape
    _, n, k_b = b_shape
    assert k == k_b and masked_m_shape == (num_groups,) and d_shape == (num_groups, max_m, n)
    k_blocks = a_sf_shape[2]
    n_blocks = b_sf_shape[1]
    return {
        "flops": num_groups * (max_m + 1) * n * k,
        "memory_bytes": num_groups * max_m * k + num_groups * max_m * k_blocks * 4 + num_groups * n * k + num_groups * n_blocks * k_blocks * 4 + num_groups * 4 + num_groups * max_m * n * 2,
        "dtype": "fp8",
    }

DESIGNED_VRAM_SIZE = 48
