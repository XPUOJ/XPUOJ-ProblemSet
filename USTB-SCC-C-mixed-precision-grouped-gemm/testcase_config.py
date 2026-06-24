from __future__ import annotations


def getNumOfTestcases() -> int:
    return len(TESTCASES)


try:
    import ctypes
    from pathlib import Path
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    SCALE_BLOCK_A = 128
    GROUP_K = 64
    _BASELINE_SO_PATH = Path(__file__).resolve().parent / "baseline_lib" / "libscc_c_baseline.so"
    _BASELINE_LIB = None

    # Contest-style ranking cases.
    # (
    #   M_total, K, N, num_groups, distribution,
    #   padding_ratio_permille, A_zero_scale_permille, B_zero_scale_permille,
    #   seed, warmup, iters
    # )
    TESTCASES = [
        # Main path: K is a multiple of 128 and a hot expert dominates.
        (8192, 1024, 1024, 32, "long_tail", 100, 60, 60, 20260201, 3, 16),

        # Deep K + skinny N: unpack/dequant cost is high and many groups are small.
        (16384, 2048, 512, 64, "many_small", 250, 50, 50, 20260202, 2, 10),

        # Wide N: stresses output tiling, B bandwidth, and sparse group skipping.
        (8192, 768, 2048, 48, "sparse_groups", 150, 30, 40, 20260203, 2, 10),

        # Counter case: large balanced GEMM, low padding, no zero scales.
        (12288, 1536, 1536, 96, "balanced", 20, 0, 0, 20260205, 2, 4),

        # Counter case: K is not a multiple of 128, so no-tail-only kernels must fall back.
        (14336, 960, 1408, 96, "sparse_groups", 220, 20, 20, 20260301, 2, 8),

        # Many rows + many groups + high padding: rewards padding skip and launch balance.
        (24576, 640, 768, 128, "long_tail", 350, 40, 40, 20260302, 2, 8),

        # Very deep K: emphasizes scale reuse, INT4 unpack fusion, and K-loop efficiency.
        (6144, 3072, 640, 64, "many_small", 120, 30, 30, 20260303, 2, 6),

        # Counter case: high arithmetic density with almost no padding and no zero scales.
        (10240, 1152, 1792, 64, "balanced", 0, 0, 0, 20260304, 2, 6),
    ]

    CURRENT_CASE = None

    def _get_testcase_id() -> int:
        try:
            raw = input().strip()
        except EOFError:
            return 1
        if raw == "":
            return 1
        try:
            testcase_id = int(raw.split()[0])
        except ValueError:
            return 1
        return testcase_id if 1 <= testcase_id <= len(TESTCASES) else 1

    def getTestCaseSize():
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        M_total, K, N, num_groups, dist, padding_pm, A_zero_pm, B_zero_pm, seed, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = TESTCASES[testcase_id - 1]
        A_k_blocks = (K + SCALE_BLOCK_A - 1) // SCALE_BLOCK_A
        B_k_blocks = (K + GROUP_K - 1) // GROUP_K
        K_packed = (K + 1) // 2
        return [
            (M_total, K),                    # A_fp8
            (M_total, A_k_blocks),           # A_scale
            (num_groups, N, K_packed),       # B_int4_packed, two signed int4 values per byte
            (num_groups, N, B_k_blocks),     # B_scale
            (M_total,),                      # m_indices
            (M_total, N),                    # D
            (), (), (), (), (),              # M_total, K, N, num_groups, group_k
        ], (warmup, iters)

    def _pack_int4(q: torch.Tensor) -> torch.Tensor:
        """Pack signed int4 values in [-8, 7], low nibble first."""
        assert q.dtype == torch.int8
        if q.shape[-1] % 2 != 0:
            pad = torch.zeros(*q.shape[:-1], 1, device=q.device, dtype=torch.int8)
            q = torch.cat([q, pad], dim=-1)
        q_u = q.to(torch.int16) & 0xF
        low = q_u[..., 0::2]
        high = q_u[..., 1::2]
        return (low | (high << 4)).to(torch.uint8).contiguous()

    def _unpack_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
        bytes_i16 = packed.to(torch.int16)
        low = bytes_i16 & 0xF
        high = (bytes_i16 >> 4) & 0xF
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, device=packed.device, dtype=torch.int8)
        out[..., 0::2] = low.to(torch.int8)
        out[..., 1::2] = high.to(torch.int8)
        return out[..., :K].contiguous()

    def _make_group_indices(
        M_total: int,
        num_groups: int,
        distribution: str,
        padding_pm: int,
        seed: int,
        device: str,
    ):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 17)

        if distribution == "balanced":
            base = torch.arange(M_total, device=device, dtype=torch.int64) % num_groups
            perm_rows = torch.randperm(M_total, device=device, generator=gen)
            m_indices = base[perm_rows].to(torch.int32)
        elif distribution == "long_tail":
            ranks = torch.arange(1, num_groups + 1, device=device, dtype=torch.float32)
            probs = 1.0 / torch.pow(ranks, 1.45)
            probs = probs / probs.sum()
            m_indices = torch.multinomial(probs, M_total, replacement=True, generator=gen).to(torch.int32)
        elif distribution == "many_small":
            hot_groups = max(1, num_groups // 8)
            probs = torch.full((num_groups,), 0.0, device=device, dtype=torch.float32)
            probs[:hot_groups] = 0.88 / hot_groups
            probs[hot_groups:] = 0.12 / max(1, num_groups - hot_groups)
            m_indices = torch.multinomial(probs, M_total, replacement=True, generator=gen).to(torch.int32)
        elif distribution == "sparse_groups":
            active_groups = max(1, (num_groups * 45) // 100)
            active = torch.randperm(num_groups, device=device, generator=gen)[:active_groups]
            ranks = torch.arange(1, active_groups + 1, device=device, dtype=torch.float32)
            probs = 1.0 / torch.pow(ranks, 1.25)
            probs = probs / probs.sum()
            sampled = torch.multinomial(probs, M_total, replacement=True, generator=gen)
            m_indices = active[sampled].to(torch.int32)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Shuffle group labels so the hot group is not always group 0.
        if distribution != "sparse_groups":
            perm = torch.randperm(num_groups, device=device, generator=gen).to(torch.int32)
            m_indices = perm[m_indices.long()]

        padding_mask = torch.rand(M_total, device=device, generator=gen) < (padding_pm / 1000.0)
        m_indices[padding_mask] = -1
        return m_indices.contiguous()

    def _apply_zero_scale(scale: torch.Tensor, zero_pm: int, gen: torch.Generator) -> torch.Tensor:
        if zero_pm <= 0:
            return scale
        zero_mask = torch.rand(scale.shape, device=scale.device, generator=gen) < (zero_pm / 1000.0)
        scale[zero_mask] = 0.0
        return scale

    def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
        del testcase_sizes
        M_total, K, N, num_groups, distribution, padding_pm, A_zero_pm, B_zero_pm, seed, _, _ = CURRENT_CASE
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        A_fp32 = torch.randn(M_total, K, device=device, dtype=torch.float32, generator=gen) * 0.5
        A_fp32 = torch.clamp(A_fp32, -448.0, 448.0)
        A_fp8 = A_fp32.to(torch.float8_e4m3fn).contiguous()

        A_k_blocks = (K + SCALE_BLOCK_A - 1) // SCALE_BLOCK_A
        A_scale = (torch.rand(M_total, A_k_blocks, device=device, dtype=torch.float32, generator=gen) * 1.8 + 0.1).contiguous()
        A_scale = _apply_zero_scale(A_scale, A_zero_pm, gen)

        B_q = torch.randint(-8, 8, (num_groups, N, K), device=device, dtype=torch.int8, generator=gen).contiguous()
        B_packed = _pack_int4(B_q)

        B_k_blocks = (K + GROUP_K - 1) // GROUP_K
        B_scale = (torch.rand(num_groups, N, B_k_blocks, device=device, dtype=torch.float32, generator=gen) * 0.9 + 0.05).contiguous()
        B_scale = _apply_zero_scale(B_scale, B_zero_pm, gen)

        m_indices = _make_group_indices(M_total, num_groups, distribution, padding_pm, seed, device)
        D = torch.empty(M_total, N, device=device, dtype=torch.bfloat16)

        return [A_fp8, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, GROUP_K]

    def _expand_scale(scale: torch.Tensor, block: int, K: int) -> torch.Tensor:
        return scale.repeat_interleave(block, dim=-1)[..., :K].contiguous()

    def _torch_reference(
        A_fp8,
        A_scale,
        B_packed,
        B_scale,
        m_indices,
        D,
        M_total,
        K,
        N,
        num_groups,
        group_k,
    ) -> List[KernelArg]:
        assert group_k == GROUP_K
        assert A_fp8.shape == (M_total, K)
        assert B_packed.shape == (num_groups, N, (K + 1) // 2)
        assert D.shape == (M_total, N)

        A_real = A_fp8.float() * _expand_scale(A_scale, SCALE_BLOCK_A, K)
        B_q = _unpack_int4(B_packed, K).float()
        B_real = B_q * _expand_scale(B_scale, group_k, K)

        D.zero_()
        for g in range(num_groups):
            rows = torch.nonzero(m_indices == g, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            out = A_real.index_select(0, rows) @ B_real[g].transpose(0, 1)
            D.index_copy_(0, rows, out.to(torch.bfloat16))
        return [A_fp8, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k]

    def _load_cuda_baseline():
        global _BASELINE_LIB
        if _BASELINE_LIB is not None:
            return _BASELINE_LIB

        if not _BASELINE_SO_PATH.exists():
            raise RuntimeError(
                "SCC C starter CUDA baseline shared library is missing. "
                f"Build it first with: bash {_BASELINE_SO_PATH.parent / 'build.sh'}"
            )

        lib = ctypes.CDLL(str(_BASELINE_SO_PATH))
        lib.run_kernel.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
        ]
        lib.run_kernel.restype = None
        _BASELINE_LIB = lib
        return lib

    def baseline(
        A_fp8,
        A_scale,
        B_packed,
        B_scale,
        m_indices,
        D,
        M_total,
        K,
        N,
        num_groups,
        group_k,
    ) -> List[KernelArg]:
        assert group_k == GROUP_K
        assert A_fp8.shape == (M_total, K)
        assert B_packed.shape == (num_groups, N, (K + 1) // 2)
        assert D.shape == (M_total, N)

        lib = _load_cuda_baseline()
        lib.run_kernel(
            ctypes.c_void_p(A_fp8.data_ptr()),
            ctypes.c_void_p(A_scale.data_ptr()),
            ctypes.c_void_p(B_packed.data_ptr()),
            ctypes.c_void_p(B_scale.data_ptr()),
            ctypes.c_void_p(m_indices.data_ptr()),
            ctypes.c_void_p(D.data_ptr()),
            ctypes.c_int64(M_total),
            ctypes.c_int64(K),
            ctypes.c_int64(N),
            ctypes.c_int64(num_groups),
            ctypes.c_int64(group_k),
        )
        return [A_fp8, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k]

    def check(
        testcase_sizes,
        original_input_tensors,
        target_kernel_input_tensors,
        baseline_input_tensors,
        rtol: float = 2e-2,
        atol: float = 2e-2,
    ) -> bool:
        del testcase_sizes, original_input_tensors
        D_t = target_kernel_input_tensors[5]
        D_ref = baseline_input_tensors[5]
        if D_t.shape != D_ref.shape:
            print(f"[FAIL] shape mismatch: target {D_t.shape}, ref {D_ref.shape}", file=sys.stderr)
            return False
        if D_t.dtype != D_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {D_t.dtype}, ref {D_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(D_t.float(), D_ref.float(), rtol=rtol, atol=atol):
            diff = (D_t.float() - D_ref.float()).abs()
            print(
                f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f}, "
                f"mean_abs_diff={float(diff.mean().item()):.6f} (rtol={rtol}, atol={atol})",
                file=sys.stderr,
            )
            return False
        return True
except Exception:
    pass


INPUT_CLASS = [
    "INPUT", "INPUT", "INPUT", "INPUT", "INPUT",
    "OUTPUT",
    "INPUT", "INPUT", "INPUT", "INPUT", "INPUT",
]


def getWorkload(testcase_sizes) -> dict:
    raw = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    A_shape, A_scale_shape, B_shape, B_scale_shape, m_shape, D_shape = raw[:6]
    M_total, K = A_shape
    num_groups, N, K_packed = B_shape
    assert K_packed == (K + 1) // 2
    assert D_shape == (M_total, N) and m_shape == (M_total,)
    expected_valid_m = (85 * M_total) // 100
    flops = 2 * expected_valid_m * N * K
    memory_bytes = (
        M_total * K
        + M_total * A_scale_shape[1] * 4
        + num_groups * N * K_packed
        + num_groups * N * B_scale_shape[2] * 4
        + M_total * 4
        + M_total * N * 2
    )
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "fp8_int4",
    }


DESIGNED_VRAM_SIZE = 48
