from __future__ import annotations


def getNumOfTestcases() -> int:
    """
    返回测试点数量
    """
    return 3


try:
    from typing import List, Tuple, Any, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]


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
        if testcase_id < 1 or testcase_id > getNumOfTestcases():
            return 1
        return testcase_id


    # Standard TESTCASES format so auto_tune_testcases.py can edit it.
    # (batch_size, hidden_size, warmup, iters)
    TESTCASES = [
        (17152, 17152, 8, 152),
        (68608, 4352, 7, 151),
        (34304, 8704, 7, 150),
    ]




    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """
        返回每个参数的"尺寸描述"：
        - tensor 参数：返回 shape tuple
        - scalar 参数：返回 ()
        这里我们定义参数为 [input, out, batch_size, hidden_size]
        input: (batch_size, 2*hidden_size)
        out: (batch_size, hidden_size)
        batch_size: scalar
        hidden_size: scalar
        """
        testcase_id = _get_testcase_id()
        batch_size, hidden_size, warmup, iters = TESTCASES[testcase_id - 1]
        return [
            (batch_size, 2 * hidden_size),
            (batch_size, hidden_size),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [input, out, batch_size, hidden_size]
        input: BF16 CUDA tensor, shape (batch_size, 2*hidden_size)
        out: BF16 CUDA tensor, shape (batch_size, hidden_size) - 输出，初始值会被覆盖
        batch_size: python int
        hidden_size: python int
        """
        assert len(testcase_sizes) == 4, "Expect 4 args: input, out, batch_size, hidden_size"
        input_shape, output_shape, batch_shape, hidden_shape = testcase_sizes
        assert batch_shape == () and hidden_shape == (), "batch_size and hidden_size must be scalars"
        
        batch_size = input_shape[0]
        hidden_size = input_shape[1] // 2
        assert output_shape == (batch_size, hidden_size), f"output shape {output_shape} must be ({batch_size}, {hidden_size})"
        
        # 生成随机输入数据，范围在 [-1, 1] 之间
        input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16, device=device)
        # 缩放到 [-1, 1]
        input_tensor = input_tensor.clamp(-1, 1)
        
        # out 作为输出，初始化为零（实际会被 kernel 覆盖）
        output_tensor = torch.zeros(*output_shape, dtype=torch.bfloat16, device=device)
        
        return [input_tensor, output_tensor, batch_size, hidden_size]


    def baseline(*inputs: List[KernelArg]) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 fused GeLU Tanh and Mul: out[..., h] = gelu_tanh(input[..., h]) * input[..., h+hidden_size]
        
        Args:
            input: BF16 tensor of shape (batch_size, 2*hidden_size) - input
            out: BF16 tensor of shape (batch_size, hidden_size) - output (will be modified inplace)
            batch_size: int - batch size
            hidden_size: int - hidden size
        
        Returns:
            List containing modified out tensor
        """
        input_tensor, output_tensor, batch_size, hidden_size = inputs
        
        assert torch.is_tensor(input_tensor) and torch.is_tensor(output_tensor)
        assert isinstance(batch_size, int) and isinstance(hidden_size, int)
        assert input_tensor.shape == (batch_size, 2 * hidden_size)
        assert output_tensor.shape == (batch_size, hidden_size)
        
        first_half = input_tensor[..., :hidden_size]
        second_half = input_tensor[..., hidden_size:]
        
        gelu_result = torch.nn.functional.gelu(first_half)
        result = gelu_result * second_half
        result = result.to(input_tensor.dtype)
        
        output_tensor.copy_(result)
        
        return [input_tensor, output_tensor, batch_size, hidden_size]


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
            target_kernel_input_tensors: 跑过 target_kernel 后的输入（out 被修改）
            baseline_input_tensors: 跑过 baseline 后的输入（out 被改为正确值）
        """
        assert len(testcase_sizes) == 4
        assert len(target_kernel_input_tensors) == 4
        assert len(baseline_input_tensors) == 4
        
        _, out_t, _, _ = target_kernel_input_tensors
        _, out_ref, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(out_t) and torch.is_tensor(out_ref)):
            print(f"[FAIL] out must be tensor, got out_t type: {type(out_t)}, out_ref type: {type(out_ref)}", file=sys.stderr)
            return False
        
        if out_t.shape != out_ref.shape:
            print(f"[FAIL] shape mismatch: target {out_t.shape}, ref {out_ref.shape}", file=sys.stderr)
            return False
        
        if out_t.dtype != out_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {out_t.dtype}, ref {out_ref.dtype}", file=sys.stderr)
            return False
        
        # 数值比较
        ok = torch.allclose(out_t, out_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (out_t - out_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, out_t.shape))
            print(f"[FAIL] max diff at position {max_idx_tuple}: target={float(out_t.flatten()[max_idx].item()):.6f}, ref={float(out_ref.flatten()[max_idx].item()):.6f}", file=sys.stderr)
            return False
        
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "OUTPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 fused GeLU Tanh and Mul：out[..., h] = gelu_tanh(input[..., h]) * input[..., h + hidden_size]

    workload 口径：
      - flops = 8 * batch_size * hidden_size
        理由：每个输出元素包含 1 次 GELU tanh 近似和 1 次乘法，这里按 7 + 1 FLOPs 统计。
      - memory_bytes = batch_size * (2 * hidden_size) * 2 + batch_size * hidden_size * 2
        理由：需要读取完整输入张量 input，并写出输出 out，均按 bf16 统计。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    input_shape, output_shape, _, _ = raw_sizes
    batch_size, hidden_twice = input_shape
    hidden_size = hidden_twice // 2
    assert output_shape == (batch_size, hidden_size)
    return {
        "flops": 8 * batch_size * hidden_size,
        "memory_bytes": batch_size * hidden_twice * 2 + batch_size * hidden_size * 2,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
