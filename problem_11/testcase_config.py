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
        这里我们定义参数为 [A, B, m_indices, D, M_total, K, N, num_groups]
        """
        testcase_id = int(input())
        
        # 定义测试点配置: (M_total, K, N, num_groups, warmup, iters)
        # 所有测试点都是大规模（> 4M elements）
        testcases = [
            (1024, 1024, 1024, 2, 5, 99),
            (2048, 512, 1024, 4, 5, 98),
            (768, 768, 2816, 2, 3, 61),
            (1536, 768, 1536, 2, 3, 54),
        ]



        
        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        
        M_total, K, N, num_groups, warmup, iters = testcases[testcase_id - 1]
        
        return [
            (M_total, K),        # A
            (num_groups, N, K),  # B
            (M_total,),          # m_indices
            (M_total, N),        # D
            (),                  # M_total (scalar)
            (),                  # K (scalar)
            (),                  # N (scalar)
            (),                  # num_groups (scalar)
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [A, B, m_indices, D, M_total, K, N, num_groups]
        A: BF16 CUDA tensor, shape (M_total, K)
        B: BF16 CUDA tensor, shape (num_groups, N, K)
        m_indices: INT32 CUDA tensor, shape (M_total,) - 每个行属于哪个 group
        D: BF16 CUDA tensor, shape (M_total, N) - 输出，初始值会被覆盖
        M_total, K, N, num_groups: python int
        """
        assert len(testcase_sizes) == 8, "Expect 8 args"
        a_shape, b_shape, m_indices_shape, d_shape, m_total_shape, k_shape, n_shape, num_groups_shape = testcase_sizes
        assert m_total_shape == () and k_shape == () and n_shape == () and num_groups_shape == (), "M_total, K, N, num_groups must be scalars"
        
        M_total = a_shape[0]
        K = a_shape[1]
        num_groups = b_shape[0]
        N = b_shape[1]
        
        assert a_shape == (M_total, K), f"A shape {a_shape} must be ({M_total}, {K})"
        assert b_shape == (num_groups, N, K), f"B shape {b_shape} must be ({num_groups}, {N}, {K})"
        assert m_indices_shape == (M_total,), f"m_indices shape {m_indices_shape} must be ({M_total},)"
        assert d_shape == (M_total, N), f"D shape {d_shape} must be ({M_total}, {N})"
        
        
        # 生成随机输入 A
        A = torch.randn(*a_shape, dtype=torch.bfloat16, device=device) * 0.5
        
        # 生成随机权重 B
        B = torch.randn(*b_shape, dtype=torch.bfloat16, device=device) * 0.5
        
        # 生成 m_indices：每个行属于哪个 group，范围 [0, num_groups - 1]
        # 为了模拟 padding，可以设置一些无效的 group_id (例如 -1)
        m_indices = torch.randint(0, num_groups, (M_total,), dtype=torch.int32, device=device)
        # 随机设置 10% 的行无效
        mask = torch.rand(M_total, device=device) < 0.1
        m_indices[mask] = -1
        
        # D 作为输出，初始化为零（实际会被 kernel 覆盖）
        D = torch.zeros(*d_shape, dtype=torch.bfloat16, device=device)
        
        return [A, B, m_indices, D, M_total, K, N, num_groups]


    def baseline(A: torch.Tensor, B: torch.Tensor, m_indices: torch.Tensor, D: torch.Tensor,
                 M_total: int, K: int, N: int, num_groups: int) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 M-grouped BF16 GEMM NT contiguous: D[i] = A[i] @ B[m_indices[i]].T
        如果 m_indices[i] 无效 (<0 or >=num_groups)，则 D[i] 不计算（保持原值或置零，这里置零）
        
        Args:
            A: BF16 tensor of shape (M_total, K)
            B: BF16 tensor of shape (num_groups, N, K)
            m_indices: INT32 tensor of shape (M_total,)
            D: BF16 tensor of shape (M_total, N) - output (will be modified inplace)
            M_total, K, N, num_groups: int
        
        Returns:
            List containing modified D tensor
        """
        assert torch.is_tensor(A) and torch.is_tensor(B) and torch.is_tensor(m_indices) and torch.is_tensor(D)
        assert isinstance(M_total, int) and isinstance(K, int) and isinstance(N, int) and isinstance(num_groups, int)
        assert A.shape == (M_total, K)
        assert B.shape == (num_groups, N, K)
        assert m_indices.shape == (M_total,)
        assert D.shape == (M_total, N)
        
        # 初始化输出为零
        D.zero_()
        
        # D[i] = A[i] @ B[m_indices[i]].T
        # A: (M, K)
        # B: (G, N, K)
        # m_indices: (M,)
        
        # 1. Gather B matrices
        valid_mask = (m_indices >= 0) & (m_indices < num_groups)
        safe_indices = m_indices.clone()
        safe_indices[~valid_mask] = 0
        
        B_gathered = B[safe_indices.long()] # (M, N, K)
        
        # 2. Compute D = A @ B.T
        # A.unsqueeze(1): (M, 1, K)
        # B_gathered.transpose(1, 2): (M, K, N)
        # Matmul -> (M, 1, N) -> squeeze -> (M, N)
        res = torch.matmul(A.unsqueeze(1).float(), B_gathered.transpose(1, 2).float()).squeeze(1)
        
        # 3. Apply mask
        res = res * valid_mask.unsqueeze(1).float()
        
        D.copy_(res.to(torch.bfloat16))
        
        return [A, B, m_indices, D, M_total, K, N, num_groups]


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
        
        _, _, _, D_t, _, _, _, _ = target_kernel_input_tensors
        _, _, _, D_ref, _, _, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(D_t) and torch.is_tensor(D_ref)):
            print(f"[FAIL] D must be tensor, got D_t type: {type(D_t)}, D_ref type: {type(D_ref)}", file=sys.stderr)
            return False
        
        if D_t.shape != D_ref.shape:
            print(f"[FAIL] shape mismatch: target {D_t.shape}, ref {D_ref.shape}", file=sys.stderr)
            return False
        
        if D_t.dtype != D_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {D_t.dtype}, ref {D_ref.dtype}", file=sys.stderr)
            return False
        
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
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 M-grouped BF16 GEMM NT contiguous：D[i] = A[i] @ B[m_indices[i]].T

    workload 口径：
      - flops = 2 * expected_valid_m * N * K
        理由：只有 m_indices 有效的行参与 GEMM；当前 testcase_config 大约 10% 行无效，因此按 90% 有效行估算。
      - memory_bytes = M_total * K * 2 + num_groups * N * K * 2 + M_total * 4 + M_total * N * 2
        理由：需要读取 A、所有组权重 B、每行的 m_indices，并写出输出 D。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, m_indices_shape, d_shape, _, _, _, _ = raw_sizes
    M_total, K = a_shape
    num_groups, N, K_b = b_shape
    assert K == K_b and m_indices_shape == (M_total,) and d_shape == (M_total, N)
    expected_valid_m = (9 * M_total + 5) // 10
    return {
        "flops": 2 * expected_valid_m * N * K,
        "memory_bytes": M_total * K * 2 + num_groups * N * K * 2 + M_total * 4 + M_total * N * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
