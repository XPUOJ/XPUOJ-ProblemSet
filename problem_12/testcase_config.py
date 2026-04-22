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
        这里我们定义参数为 [a, b, masked_m, d, num_groups, max_m, n, k]
        """
        testcase_id = int(input())
        
        # 定义测试点配置: (num_groups, max_m, n, k, warmup, iters)
        # 所有测试点都是大规模（> 4M elements）
        testcases = [
            (2, 4352, 4352, 4352, 4, 80),
            (4, 2048, 8192, 2048, 4, 87),
            (4, 1536, 10496, 2816, 3, 60),
            (2, 4096, 7680, 4096, 3, 47),
        ]



        
        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        
        num_groups, max_m, n, k, warmup, iters = testcases[testcase_id - 1]
        
        return [
            (num_groups, max_m, k),  # a
            (num_groups, n, k),      # b
            (num_groups,),           # masked_m
            (num_groups, max_m, n),  # d
            (),                      # num_groups (scalar)
            (),                      # max_m (scalar)
            (),                      # n (scalar)
            (),                      # k (scalar)
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [a, b, masked_m, d, num_groups, max_m, n, k]
        a: BF16 CUDA tensor, shape (num_groups, max_m, k)
        b: BF16 CUDA tensor, shape (num_groups, n, k)
        masked_m: INT32 CUDA tensor, shape (num_groups,)
        d: BF16 CUDA tensor, shape (num_groups, max_m, n) - 输出，初始值会被覆盖
        num_groups, max_m, n, k: python int
        """
        assert len(testcase_sizes) == 8, "Expect 8 args"
        a_shape, b_shape, masked_m_shape, d_shape, num_groups_shape, max_m_shape, n_shape, k_shape = testcase_sizes
        assert num_groups_shape == () and max_m_shape == () and n_shape == () and k_shape == (), "num_groups, max_m, n, k must be scalars"
        
        num_groups = a_shape[0]
        max_m = a_shape[1]
        k = a_shape[2]
        n = d_shape[2]
        
        assert a_shape == (num_groups, max_m, k), f"a shape {a_shape} must be ({num_groups}, {max_m}, {k})"
        assert b_shape == (num_groups, n, k), f"b shape {b_shape} must be ({num_groups}, {n}, {k})"
        assert masked_m_shape == (num_groups,), f"masked_m shape {masked_m_shape} must be ({num_groups},)"
        assert d_shape == (num_groups, max_m, n), f"d shape {d_shape} must be ({num_groups}, {max_m}, {n})"
        
        
        # 生成随机输入 a
        a = torch.randn(*a_shape, dtype=torch.bfloat16, device=device) * 0.5
        
        # 生成随机权重 b
        b = torch.randn(*b_shape, dtype=torch.bfloat16, device=device) * 0.5
        
        # 生成 masked_m：每个组的实际处理行数，范围 [1, max_m]
        masked_m = torch.randint(1, max_m + 1, (num_groups,), dtype=torch.int32, device=device)
        
        # d 作为输出，初始化为零（实际会被 kernel 覆盖）
        d = torch.zeros(*d_shape, dtype=torch.bfloat16, device=device)
        
        return [a, b, masked_m, d, num_groups, max_m, n, k]


    def baseline(a: torch.Tensor, b: torch.Tensor, masked_m: torch.Tensor, d: torch.Tensor,
                 num_groups: int, max_m: int, n: int, k: int) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 M-grouped BF16 GEMM NT with masked M: d[g, :masked_m[g], :] = a[g, :masked_m[g], :] @ b[g].T
        
        Args:
            a: BF16 tensor of shape (num_groups, max_m, k)
            b: BF16 tensor of shape (num_groups, n, k)
            masked_m: INT32 tensor of shape (num_groups,)
            d: BF16 tensor of shape (num_groups, max_m, n) - output (will be modified inplace)
            num_groups, max_m, n, k: int
        
        Returns:
            List containing modified d tensor
        """
        assert torch.is_tensor(a) and torch.is_tensor(b) and torch.is_tensor(masked_m) and torch.is_tensor(d)
        assert isinstance(num_groups, int) and isinstance(max_m, int) and isinstance(n, int) and isinstance(k, int)
        assert a.shape == (num_groups, max_m, k)
        assert b.shape == (num_groups, n, k)
        assert masked_m.shape == (num_groups,)
        assert d.shape == (num_groups, max_m, n)
        
        # 初始化输出为零
        d.zero_()
        
        # d = (a @ b.T) * mask
        # a: (G, M, K), b: (G, N, K) -> b.T: (G, K, N)
        # a @ b.T -> (G, M, N)
        res = torch.matmul(a.float(), b.transpose(1, 2).float())
        
        # Create mask (G, M)
        # masked_m: (G,)
        # Need mask[g, m] = 1 if m < masked_m[g] else 0
        m_indices = torch.arange(max_m, device=a.device).unsqueeze(0) # (1, M)
        mask = m_indices < masked_m.unsqueeze(1) # (G, M)
        
        # Apply mask (expand to N)
        res = res * mask.unsqueeze(2).float()
        d.copy_(res.to(torch.bfloat16))
        
        return [a, b, masked_m, d, num_groups, max_m, n, k]


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
            target_kernel_input_tensors: 跑过 target_kernel 后的输入（d 被修改）
            baseline_input_tensors: 跑过 baseline 后的输入（d 被改为正确值）
        """
        assert len(testcase_sizes) == 8
        assert len(target_kernel_input_tensors) == 8
        assert len(baseline_input_tensors) == 8
        
        _, _, _, d_t, _, _, _, _ = target_kernel_input_tensors
        _, _, _, d_ref, _, _, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(d_t) and torch.is_tensor(d_ref)):
            print(f"[FAIL] d must be tensor, got d_t type: {type(d_t)}, d_ref type: {type(d_ref)}", file=sys.stderr)
            return False
        
        if d_t.shape != d_ref.shape:
            print(f"[FAIL] shape mismatch: target {d_t.shape}, ref {d_ref.shape}", file=sys.stderr)
            return False
        
        if d_t.dtype != d_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {d_t.dtype}, ref {d_ref.dtype}", file=sys.stderr)
            return False
        
        # 数值比较
        ok = torch.allclose(d_t, d_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (d_t - d_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, d_t.shape))
            print(f"[FAIL] max diff at position {max_idx_tuple}: target={float(d_t.flatten()[max_idx].item()):.6f}, ref={float(d_ref.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 M-grouped BF16 GEMM NT masked：仅 d[g, :masked_m[g], :] 参与有效计算

    workload 口径：
      - flops = num_groups * (max_m + 1) * n * k
        理由：实际有效行数由随机 masked_m 决定，这里按区间 [1, max_m] 的平均有效行数统计。
      - memory_bytes = num_groups * max_m * k * 2 + num_groups * n * k * 2 + num_groups * 4 + num_groups * max_m * n * 2
        理由：需要读取 a、b、每组的 masked_m，并写出完整输出 d。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    a_shape, b_shape, masked_m_shape, d_shape, _, _, _, _ = raw_sizes
    num_groups, max_m, k = a_shape
    _, n, k_b = b_shape
    assert k == k_b and masked_m_shape == (num_groups,) and d_shape == (num_groups, max_m, n)
    return {
        "flops": num_groups * (max_m + 1) * n * k,
        "memory_bytes": num_groups * max_m * k * 2 + num_groups * n * k * 2 + num_groups * 4 + num_groups * max_m * n * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
