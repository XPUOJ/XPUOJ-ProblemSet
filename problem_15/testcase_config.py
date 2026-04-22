from __future__ import annotations

def getNumOfTestcases() -> int:
    """返回测试点数量"""
    return 4

try:
    from typing import List, Tuple, Union
    import torch
    import sys
    KernelArg = Union[torch.Tensor, int, float]

    def getTestCaseSize():
        """返回每个参数的尺寸描述及 (预热轮数, 测试轮数)。"""
        testcase_id = int(input())
        testcases = [
            (6144, 1536, 2, [3072, 3072], 11, 209),
            (23296, 512, 2, [1536, 1536], 17, 338),
            (2816, 1024, 4, [5376, 5376, 5376, 5376], 13, 261),
            (9216, 768, 4, [2304, 2304, 2304, 2304], 11, 209),
        ]



        if testcase_id < 1 or testcase_id > len(testcases):
            raise ValueError(f"Invalid testcase_id: {testcase_id}, must be in [1, {len(testcases)}]")
        batch_size, in_features, K, out_dims_list, warmup, iters = testcases[testcase_id - 1]
        total_out = sum(out_dims_list)
        sizes = [
            (batch_size, in_features),
            (total_out, in_features),
            (batch_size, total_out),
            (K,),
            (), (), (),
        ]
        return sizes, (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        """根据尺寸生成测试数据，使用 tensor 操作。"""
        raw = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(raw) == 7, f"Expect 7 shape entries, got {len(raw)}"
        x_shape, w_shape, out_shape, out_dims_shape = raw[:4]
        batch_size, in_features = x_shape
        total_out, _ = w_shape
        K = out_dims_shape[0]
        assert out_shape == (batch_size, total_out)
        x = torch.randn(*x_shape, dtype=torch.bfloat16, device=device) * 0.5
        w_all = torch.randn(*w_shape, dtype=torch.bfloat16, device=device) * 0.5
        out = torch.zeros(*out_shape, dtype=torch.bfloat16, device=device)
        base = total_out // K
        rem = total_out - base * K
        out_dims = torch.tensor([base] * (K - rem) + [base + 1] * rem, dtype=torch.int64, device=device)
        return [x, w_all, out, out_dims, batch_size, in_features, K]

    def baseline(
        x: torch.Tensor,
        w_all: torch.Tensor,
        out: torch.Tensor,
        out_dims: torch.Tensor,
        batch_size: int,
        in_features: int,
        K: int,
    ) -> List[KernelArg]:
        """baseline：按路分段计算 out = x @ w_all.T"""
        assert torch.is_tensor(x) and torch.is_tensor(w_all) and torch.is_tensor(out) and torch.is_tensor(out_dims)
        assert x.shape == (batch_size, in_features) and out.shape == (batch_size, out_dims.sum().item())
        
        # Replace loop with single matmul
        out.copy_(torch.matmul(x, w_all.t()).to(torch.bfloat16))
        
        return [x, w_all, out, out_dims, batch_size, in_features, K]

    def check(
        testcase_sizes,
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> bool:
        """比较用户 kernel 输出 out 与 baseline 输出。"""
        raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
        assert len(target_kernel_input_tensors) >= 3 and len(baseline_input_tensors) >= 3
        out_t = target_kernel_input_tensors[2]
        out_ref = baseline_input_tensors[2]
        if not (torch.is_tensor(out_t) and torch.is_tensor(out_ref)):
            print(f"[FAIL] out must be tensor, got types: {type(out_t)}, {type(out_ref)}", file=sys.stderr)
            return False
        if out_t.shape != out_ref.shape:
            print(f"[FAIL] shape mismatch: target {out_t.shape}, ref {out_ref.shape}", file=sys.stderr)
            return False
        if out_t.dtype != out_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {out_t.dtype}, ref {out_ref.dtype}", file=sys.stderr)
            return False
        ok = torch.allclose(out_t.float(), out_ref.float(), rtol=rtol, atol=atol)
        if not ok:
            diff = (out_t.float() - out_ref.float()).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except Exception:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现线性变换：out = x @ w_all.T

    workload 口径：
      - flops = 2 * batch_size * in_features * total_out
        理由：当前 baseline 已把 variadic weights 向量化为一次标准 matmul。
      - memory_bytes = (batch_size * in_features + total_out * in_features + batch_size * total_out) * 2 + K * 8
        理由：需要读取输入 x、权重 w_all、分段信息 out_dims，并写出输出 out。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, w_shape, out_shape, out_dims_shape, _, _, _ = raw_sizes
    batch_size, in_features = x_shape
    total_out, in_features_w = w_shape
    assert in_features == in_features_w and out_shape == (batch_size, total_out)
    K = out_dims_shape[0]
    return {
        "flops": 2 * batch_size * in_features * total_out,
        "memory_bytes": (batch_size * in_features + total_out * in_features + batch_size * total_out) * 2 + K * 8,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
