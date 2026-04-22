from __future__ import annotations

def getNumOfTestcases() -> int:
    """
    返回测试点数量
    """
    return 8

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
        这里我们定义参数为 [x, y, T, H]
        x: (T, H)
        y: (T, H)
        T: scalar
        H: scalar
        """
        testcase_id = int(input())
        
        # 定义测试点配置: (T, H, warmup, iters)
        # 所有测试点都是大规模（> 4M elements）
        testcases = [
            (27904, 14080, 20, 402),
            (103680, 3584, 20, 427),
            (3584, 103680, 20, 428),
            (34144, 11392, 20, 416),
            (283136, 1280, 20, 438),
            (27648, 14336, 20, 399),
            (28160, 14080, 20, 399),
            (1280, 283136, 20, 438),
        ]




        
        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        
        T, H, warmup, iters = testcases[testcase_id - 1]
        
        return [
            (T, H),  # x
            (T, H),  # y
            (),      # T (scalar)
            (),      # H (scalar)
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [x, y, T, H]
        x: BF16 CUDA tensor, shape (T, H)
        y: BF16 CUDA tensor, shape (T, H) - 输出，初始值会被覆盖
        T: python int
        H: python int
        """
        assert len(testcase_sizes) == 4, "Expect 4 args: x, y, T, H"
        x_shape, y_shape, t_shape, h_shape = testcase_sizes
        assert t_shape == () and h_shape == (), "T and H must be scalars"
        
        T = x_shape[0]
        H = x_shape[1]
        assert y_shape == (T, H), f"y shape {y_shape} must be ({T}, {H})"
        
        # 生成随机输入数据，范围在 [-5, 5] 之间，避免极端值导致数值问题
        x = torch.randn(*x_shape, dtype=torch.bfloat16, device=device) * 2.0 - 1.0
        x = x * 5.0  # 缩放到 [-5, 5]
        
        # y 作为输出，初始化为零（实际会被 kernel 覆盖）
        y = torch.zeros(*y_shape, dtype=torch.bfloat16, device=device)
        
        return [x, y, T, H]


    def baseline(x: torch.Tensor, y: torch.Tensor, T: int, H: int) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 SiLU: y = x * sigmoid(x)
        
        Args:
            x: BF16 tensor of shape (T, H) - input
            y: BF16 tensor of shape (T, H) - output (will be modified inplace)
            T: int - first dimension size
            H: int - second dimension size
        
        Returns:
            List containing modified y tensor
        """
        assert torch.is_tensor(x) and torch.is_tensor(y)
        assert isinstance(T, int) and isinstance(H, int)
        assert x.shape == (T, H)
        assert y.shape == (T, H)
        
        # SiLU: y = x * sigmoid(x)
        y.copy_(torch.nn.functional.silu(x))
        
        return [x, y, T, H]


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
            target_kernel_input_tensors: 跑过 target_kernel 后的输入（y 被修改）
            baseline_input_tensors: 跑过 baseline 后的输入（y 被改为正确值）
        """
        assert len(testcase_sizes) == 4
        assert len(target_kernel_input_tensors) == 4
        assert len(baseline_input_tensors) == 4
        
        _, y_t, _, _ = target_kernel_input_tensors
        _, y_ref, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(y_t) and torch.is_tensor(y_ref)):
            print(f"[FAIL] y must be tensor, got y_t type: {type(y_t)}, y_ref type: {type(y_ref)}", file=sys.stderr)
            return False
        
        if y_t.shape != y_ref.shape:
            print(f"[FAIL] shape mismatch: target {y_t.shape}, ref {y_ref.shape}", file=sys.stderr)
            return False
        
        if y_t.dtype != y_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {y_t.dtype}, ref {y_ref.dtype}", file=sys.stderr)
            return False
        
        # 数值比较
        ok = torch.allclose(y_t, y_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (y_t - y_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, y_t.shape))
            print(f"[FAIL] max diff at position {max_idx_tuple}: target={float(y_t.flatten()[max_idx].item()):.6f}, ref={float(y_ref.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "OUTPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现逐元素 SiLU：y = x * sigmoid(x)

    workload 口径：
      - flops = 3 * numel
        理由：按当前统一口径，SiLU 记 3 FLOPs/elem。
      - memory_bytes = numel * 2 * 2
        理由：读取输入 x，写出输出 y，按 bf16 的输入读 + 输出写统计。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, y_shape, _, _ = raw_sizes
    assert x_shape == y_shape
    numel = 1
    for dim in x_shape:
        numel *= dim
    return {
        "flops": 3 * numel,
        "memory_bytes": numel * 2 * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
