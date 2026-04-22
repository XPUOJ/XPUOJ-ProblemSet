from __future__ import annotations

def getNumOfTestcases() -> int:
    """
    返回测试点数量
    """
    return 4

try:
    from typing import List, Tuple, Any, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]


    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """
        返回每个参数的"尺寸描述"：
        - tensor 参数：返回 shape tuple
        - scalar 参数：返回 ()
        这里我们定义参数为 [A, A_scale, B, B_scale, D, M, K, N]
        """
        testcase_id = int(input())
        
        # 定义测试点配置: (M, K, N, warmup, iters)
        # 所有测试点都是大规模（> 4M elements）
        testcases = [
            (4096, 4096, 4096, 5, 114),
            (8192, 2048, 4096, 6, 114),
            (2048, 4096, 8192, 6, 116),
            (5376, 2816, 5376, 5, 102),
        ]



        
        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        
        M, K, N, warmup, iters = testcases[testcase_id - 1]
        
        # 计算 scale 的维度
        K_blocks = (K + 127) // 128  # ceil(K/128)
        N_blocks = (N + 127) // 128  # ceil(N/128)
        
        return [
            (M, K),              # A
            (M, K_blocks),       # A_scale
            (N, K),              # B
            (N_blocks, K_blocks), # B_scale
            (M, N),              # D
            (),                  # M (scalar)
            (),                  # K (scalar)
            (),                  # N (scalar)
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [A, A_scale, B, B_scale, D, M, K, N]
        A: FP8E4M3 CUDA tensor, shape (M, K)
        A_scale: FP32 CUDA tensor, shape (M, ceil(K/128))
        B: FP8E4M3 CUDA tensor, shape (N, K)
        B_scale: FP32 CUDA tensor, shape (ceil(N/128), ceil(K/128))
        D: BF16 CUDA tensor, shape (M, N) - 输出，初始值会被覆盖
        M, K, N: python int
        """
        assert len(testcase_sizes) == 8, "Expect 8 args: A, A_scale, B, B_scale, D, M, K, N"
        a_shape, a_scale_shape, b_shape, b_scale_shape, d_shape, m_shape, k_shape, n_shape = testcase_sizes
        assert m_shape == () and k_shape == () and n_shape == (), "M, K, N must be scalars"
        
        M = a_shape[0]
        K = a_shape[1]
        N = b_shape[0]
        assert b_shape[1] == K, f"B shape {b_shape} must have K={K} as second dimension"
        assert d_shape == (M, N), f"D shape {d_shape} must be ({M}, {N})"
        
        # 生成随机 FP32 数据，然后转换为 FP8
        A_fp32 = torch.randn(*a_shape, dtype=torch.float32, device=device) * 0.5
        A_fp32 = torch.clamp(A_fp32, -448.0, 448.0)  # FP8 E4M3 范围
        A = A_fp32.to(torch.float8_e4m3fn)
        
        # 生成 A_scale
        K_blocks = a_scale_shape[1]
        A_scale = torch.rand(*a_scale_shape, dtype=torch.float32, device=device)
        A_scale.view(-1)[::2] = 0
        
        # 生成随机 FP32 数据，然后转换为 FP8
        B_fp32 = torch.randn(*b_shape, dtype=torch.float32, device=device) * 0.5
        B_fp32 = torch.clamp(B_fp32, -448.0, 448.0)  # FP8 E4M3 范围
        B = B_fp32.to(torch.float8_e4m3fn)
        
        # 生成 B_scale
        B_scale = torch.rand(*b_scale_shape, dtype=torch.float32, device=device)
        B_scale.view(-1)[::2] = 0
        
        # D 作为输出，初始化为零（实际会被 kernel 覆盖）
        D = torch.zeros(*d_shape, dtype=torch.bfloat16, device=device)
        
        return [A, A_scale, B, B_scale, D, M, K, N]


    def baseline(A: torch.Tensor, A_scale: torch.Tensor, B: torch.Tensor, B_scale: torch.Tensor, 
                 D: torch.Tensor, M: int, K: int, N: int) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 FP8 GEMM with 1D-2D scaling: D = (A * A_scale) @ (B * B_scale).T
        
        Args:
            A: FP8E4M3 tensor of shape (M, K)
            A_scale: FP32 tensor of shape (M, ceil(K/128)) - per-token scale factors
            B: FP8E4M3 tensor of shape (N, K)
            B_scale: FP32 tensor of shape (ceil(N/128), ceil(K/128)) - per-block scale factors
            D: BF16 tensor of shape (M, N) - output (will be modified inplace)
            M: int - first dimension of A
            K: int - second dimension of A and B
            N: int - first dimension of B
        
        Returns:
            List containing modified D tensor
        """
        assert torch.is_tensor(A) and torch.is_tensor(A_scale) and torch.is_tensor(B) and torch.is_tensor(B_scale) and torch.is_tensor(D)
        assert isinstance(M, int) and isinstance(K, int) and isinstance(N, int)
        assert A.shape == (M, K)
        assert B.shape == (N, K)
        assert D.shape == (M, N)
        
        BLOCK_SIZE = 128
        K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Convert FP8 to FP32
        A_fp32 = A.to(torch.float32)
        B_fp32 = B.to(torch.float32)
        
        # Compute block sizes for K dimension
        k_block_sizes = [BLOCK_SIZE] * (K_blocks - 1) + [(K % BLOCK_SIZE) or BLOCK_SIZE]
        k_repeats = torch.tensor(k_block_sizes, device=A.device)
        
        # Expand A_scale to (M, K)
        expanded_A_scale = torch.repeat_interleave(A_scale, k_repeats, dim=1)
        A_scaled = A_fp32 * expanded_A_scale
        
        # Compute block sizes for N dimension
        n_block_sizes = [BLOCK_SIZE] * (N_blocks - 1) + [(N % BLOCK_SIZE) or BLOCK_SIZE]
        n_repeats = torch.tensor(n_block_sizes, device=A.device)
        
        # Expand B_scale: first along K (dim=1), then along N (dim=0)
        expanded_B_scale_k = torch.repeat_interleave(B_scale, k_repeats, dim=1)
        expanded_B_scale = torch.repeat_interleave(expanded_B_scale_k, n_repeats, dim=0)
        B_scaled = B_fp32 * expanded_B_scale
        
        # Compute GEMM: D = A_scaled @ B_scaled.T
        D_result = torch.matmul(A_scaled, B_scaled.T)
        D.copy_(D_result.to(torch.bfloat16))
        
        return [A, A_scale, B, B_scale, D, M, K, N]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        """
        对用户 kernel 输出与标准答案进行比较。
        
        Args:
            testcase_sizes: getTestCaseSize() 的返回值
            original_input_tensors: 原始输入（通常不用于比较）
            target_kernel_input_tensors: 跑过 target_kernel 后的输入（D 被修改）
            baseline_input_tensors: 跑过 baseline 后的输入（D 被改为正确值）
        """
        assert len(testcase_sizes) == 8
        assert len(target_kernel_input_tensors) == 8
        assert len(baseline_input_tensors) == 8
        
        _, _, _, _, D_t, _, _, _ = target_kernel_input_tensors
        _, _, _, _, D_ref, _, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(D_t) and torch.is_tensor(D_ref)):
            print(f"[FAIL] D must be tensor, got D_t type: {type(D_t)}, D_ref type: {type(D_ref)}", file=sys.stderr)
            return False
        
        if D_t.shape != D_ref.shape:
            print(f"[FAIL] shape mismatch: target {D_t.shape}, ref {D_ref.shape}", file=sys.stderr)
            return False
        
        if D_t.dtype != D_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {D_t.dtype}, ref {D_ref.dtype}", file=sys.stderr)
            return False
        D_t = D_t.to(torch.float32)
        D_ref = D_ref.to(torch.float32)
        # 数值比较
        ok = torch.allclose(D_t, D_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (D_t - D_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, D_t.shape))
            print(f"[FAIL] max diff at position {max_idx_tuple}: target={float(D_t.flatten()[max_idx].item()):.6f}, ref={float(D_ref.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 FP8 GEMM NT：D = (A * A_scale) @ (B * B_scale).T

    workload 口径：
      - flops = 2 * M * N * K
        理由：主体仍是 GEMM，每个输出元素对应长度 K 的点积。
      - memory_bytes = M * K + M * K_blocks * 4 + N * K + N_blocks * K_blocks * 4 + M * N * 2
        理由：需要读取 A/B 的 fp8 数据和对应 scale，并写出 bf16 输出 D。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, a_scale_shape, b_shape, b_scale_shape, d_shape, _, _, _ = raw_sizes
    M, K = a_shape
    N, K_b = b_shape
    assert K == K_b and d_shape == (M, N)
    K_blocks = a_scale_shape[1]
    N_blocks = b_scale_shape[0]
    return {
        "flops": 2 * M * N * K,
        "memory_bytes": M * K + M * K_blocks * 4 + N * K + N_blocks * K_blocks * 4 + M * N * 2,
        "dtype": "fp8",
    }

DESIGNED_VRAM_SIZE = 48
