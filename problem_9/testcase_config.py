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
        这里我们定义参数为 [logits, labels, loss, M, N]
        logits: (M, N)
        labels: (M,)
        loss: (M,)
        M: scalar
        N: scalar
        """
        testcase_id = int(input())
        
        # 定义测试点配置: (M, N, warmup, iters)
        # 所有测试点都是大规模（> 4M elements）
        testcases = [
            (32000, 16128, 16, 304),
            (120576, 4096, 16, 316),
            (4352, 120832, 12, 236),
            (38672, 12896, 16, 314),
            (262656, 1280, 11, 217),
            (32000, 16128, 16, 304),
            (31744, 15872, 16, 311),
            (1536, 340224, 8, 165),
        ]




        
        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        
        M, N, warmup, iters = testcases[testcase_id - 1]
        
        return [
            (M, N),  # logits
            (M,),    # labels
            (M,),    # loss
            (),      # M (scalar)
            (),      # N (scalar)
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        """
        生成 testcase: [logits, labels, loss, M, N]
        logits: FP32 CUDA tensor, shape (M, N)
        labels: INT64 CUDA tensor, shape (M,)
        loss: FP32 CUDA tensor, shape (M,) - 输出，初始值会被覆盖
        M: python int
        N: python int
        """
        assert len(testcase_sizes) == 5, "Expect 5 args: logits, labels, loss, M, N"
        logits_shape, labels_shape, loss_shape, m_shape, n_shape = testcase_sizes
        assert m_shape == () and n_shape == (), "M and N must be scalars"
        
        M = logits_shape[0]
        N = logits_shape[1]
        assert labels_shape == (M,), f"labels shape {labels_shape} must be ({M},)"
        assert loss_shape == (M,), f"loss shape {loss_shape} must be ({M},)"
        
        # 生成随机 logits，范围在 [-5, 5] 之间
        logits = torch.randn(*logits_shape, dtype=torch.float32, device=device) * 2.0 - 1.0
        logits = logits * 5.0  # 缩放到 [-5, 5]
        
        # 生成随机标签，范围在 [0, N)
        labels = torch.randint(0, N, (M,), dtype=torch.int64, device=device)
        
        # loss 作为输出，初始化为零（实际会被 kernel 覆盖）
        loss = torch.zeros(*loss_shape, dtype=torch.float32, device=device)
        
        return [logits, labels, loss, M, N]


    def baseline(logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor, M: int, N: int) -> List[KernelArg]:
        """
        baseline 用来算正确结果。
        实现 Cross Entropy: loss[i] = -log(softmax(logits[i])[labels[i]])
        
        Args:
            logits: FP32 tensor of shape (M, N) - input logits
            labels: INT64 tensor of shape (M,) - target labels
            loss: FP32 tensor of shape (M,) - output (will be modified inplace)
            M: int - number of samples
            N: int - number of classes
        
        Returns:
            List containing modified loss tensor
        """
        assert torch.is_tensor(logits) and torch.is_tensor(labels) and torch.is_tensor(loss)
        assert isinstance(M, int) and isinstance(N, int)
        assert logits.shape == (M, N)
        assert labels.shape == (M,)
        assert loss.shape == (M,)
        
        # Cross Entropy: loss = -log(softmax(logits)[labels])
        loss.copy_(torch.nn.functional.cross_entropy(logits, labels, reduction='none'))
        
        return [logits, labels, loss, M, N]


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
            target_kernel_input_tensors: 跑过 target_kernel 后的输入（loss 被修改）
            baseline_input_tensors: 跑过 baseline 后的输入（loss 被改为正确值）
        """
        assert len(testcase_sizes) == 5
        assert len(target_kernel_input_tensors) == 5
        assert len(baseline_input_tensors) == 5
        
        _, _, loss_t, _, _ = target_kernel_input_tensors
        _, _, loss_ref, _, _ = baseline_input_tensors
        
        if not (torch.is_tensor(loss_t) and torch.is_tensor(loss_ref)):
            print(f"[FAIL] loss must be tensor, got loss_t type: {type(loss_t)}, loss_ref type: {type(loss_ref)}", file=sys.stderr)
            return False
        
        if loss_t.shape != loss_ref.shape:
            print(f"[FAIL] shape mismatch: target {loss_t.shape}, ref {loss_ref.shape}", file=sys.stderr)
            return False
        
        if loss_t.dtype != loss_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {loss_t.dtype}, ref {loss_ref.dtype}", file=sys.stderr)
            return False
        
        # 数值比较
        ok = torch.allclose(loss_t, loss_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (loss_t - loss_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            
            # 找出差异最大的位置
            max_idx = diff.argmax()
            print(f"[FAIL] max diff at position {max_idx.item()}: target={float(loss_t[max_idx].item()):.6f}, ref={float(loss_ref[max_idx].item()):.6f}", file=sys.stderr)
            return False
        
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现逐样本 CrossEntropy：loss[i] = -log(softmax(logits[i])[labels[i]])

    workload 口径：
      - flops = 10 * M * N
        理由：主开销来自每行 logits 的 softmax/log 运算和按标签取值，按每个 logits 元素常数级近似统计。
      - memory_bytes = M * N * 4 + M * 8 + M * 4
        理由：读取 fp32 的 logits、int64 的 labels，并写出 fp32 的 loss。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    logits_shape, labels_shape, loss_shape, _, _ = raw_sizes
    M, N = logits_shape
    assert labels_shape == (M,)
    assert loss_shape == (M,)
    return {
        "flops": 10 * M * N,
        "memory_bytes": M * N * 4 + M * 8 + M * 4,
        "dtype": "fp32",
    }

DESIGNED_VRAM_SIZE = 48
