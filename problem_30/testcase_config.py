from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch
    import torch.nn.functional as F

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    TESTCASES = [
        (1, 48, 80, 32, 32, 32, 3, 3, 3, 1, 1, 8, 153),
        (2, 48, 80, 32, 32, 32, 3, 3, 3, 1, 1, 4, 77),
        (1, 64, 96, 24, 24, 24, 3, 3, 3, 1, 1, 10, 220),
        (2, 32, 48, 48, 32, 32, 3, 3, 3, 1, 1, 6, 127),
        (1, 48, 80, 32, 48, 48, 3, 3, 3, 1, 1, 3, 68),
    ]







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
        case = TESTCASES[testcase_id - 1]
        N, C_in, C_out, D, H, W, T, R, S, stride, padding, warmup, iters = case
        CURRENT_CASE = case[:-2]
        D_out = (D + 2 * padding - T) // stride + 1
        H_out = (H + 2 * padding - R) // stride + 1
        W_out = (W + 2 * padding - S) // stride + 1
        return [
            (N, C_in, D, H, W),
            (C_out, C_in, T, R, S),
            (C_out,),
            (N, C_out, D_out, H_out, W_out),
            (), (), (), (), (), (), (), (), (), (), (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        N, C_in, C_out, D, H, W, T, R, S, stride, padding = CURRENT_CASE
        x = torch.randn(N, C_in, D, H, W, dtype=torch.float32, device=device)
        weight = torch.randn(C_out, C_in, T, R, S, dtype=torch.float32, device=device)
        bias = torch.randn(C_out, dtype=torch.float32, device=device)
        D_out = (D + 2 * padding - T) // stride + 1
        H_out = (H + 2 * padding - R) // stride + 1
        W_out = (W + 2 * padding - S) // stride + 1
        output = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.float32, device=device)
        return [x, weight, bias, output, int(N), int(C_in), int(C_out), int(D), int(H), int(W), int(T), int(R), int(S), int(stride), int(padding)]

    def baseline(x, weight, bias, output, N, C_in, C_out, D, H, W, T, R, S, stride, padding):
        output.copy_(F.conv3d(x.float(), weight.float(), bias.float(), stride, padding))
        return [x, weight, bias, output, N, C_in, C_out, D, H, W, T, R, S, stride, padding]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 1e-2, atol: float = 1e-2):
        output_t = target_kernel_input_tensors[3]
        output_ref = baseline_input_tensors[3]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(output_t, output_ref, rtol=rtol, atol=atol):
            diff = (output_t - output_ref).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 Conv3d forward：output = conv3d(x, weight, bias, stride, padding)

    workload 口径：
      - flops = 2 * out_elems * C_in * T * R * S + out_elems
        理由：每个输出元素都要遍历一个大小为 C_in*T*R*S 的卷积窗口做乘加，再加 1 次 bias。
      - memory_bytes = N * C_in * D * H * W * 4 + C_out * C_in * T * R * S * 4 + C_out * 4 + out_elems * 4
        理由：需要读取输入 x、卷积核 weight、bias，并写出 fp32 输出 output。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, weight_shape, bias_shape, output_shape = raw_sizes[:4]
    N, C_in, D, H, W = x_shape
    C_out, C_in_w, T, R, S = weight_shape
    N_out, C_out_out, D_out, H_out, W_out = output_shape
    assert C_in == C_in_w and bias_shape == (C_out,)
    assert (N_out, C_out_out) == (N, C_out)
    out_elems = N * C_out * D_out * H_out * W_out
    return {
        "flops": 2 * out_elems * C_in * T * R * S + out_elems,
        "memory_bytes": N * C_in * D * H * W * 4 + C_out * C_in * T * R * S * 4 + C_out * 4 + out_elems * 4,
        "dtype": "fp32",
    }

DESIGNED_VRAM_SIZE = 48
