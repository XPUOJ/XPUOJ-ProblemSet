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
        (1, 80, 160, 160, 160, 3, 3, 1, 1, 10, 263),
        (2, 64, 128, 160, 160, 3, 3, 1, 1, 10, 202),
        (1, 128, 192, 112, 112, 3, 3, 1, 1, 10, 278),
        (2, 8, 128, 640, 512, 3, 3, 1, 1, 6, 116),
        (1, 64, 96, 224, 224, 3, 3, 1, 1, 10, 277),
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
        N, C_in, C_out, H, W, R, S, stride, padding, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (N, C_in, C_out, H, W, R, S, stride, padding)
        H_out = (H + 2 * padding - R) // stride + 1
        W_out = (W + 2 * padding - S) // stride + 1
        return [
            (N, C_in, H, W),
            (C_out, C_in, R, S),
            (C_out,),
            (N, C_out, H_out, W_out),
            (), (), (), (), (), (), (), (), (),
        ], (warmup, iters)

    def genTestCase(testcase_sizes, device: str = "cuda"):
        N, C_in, C_out, H, W, R, S, stride, padding = CURRENT_CASE
        x = torch.randint(-8, 8, (N, C_in, H, W), dtype=torch.int32, device=device)
        weight = torch.randint(-8, 8, (C_out, C_in, R, S), dtype=torch.int32, device=device)
        bias = torch.randint(-8, 8, (C_out,), dtype=torch.int32, device=device)
        H_out = (H + 2 * padding - R) // stride + 1
        W_out = (W + 2 * padding - S) // stride + 1
        output = torch.empty(N, C_out, H_out, W_out, dtype=torch.int32, device=device)
        return [x, weight, bias, output, int(N), int(C_in), int(C_out), int(H), int(W), int(R), int(S), int(stride), int(padding)]

    def baseline(x, weight, bias, output, N, C_in, C_out, H, W, R, S, stride, padding):
        output.copy_(F.conv2d(x.float(), weight.float(), bias.float(), stride, padding).to(output.dtype))
        return [x, weight, bias, output, N, C_in, C_out, H, W, R, S, stride, padding]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 0.0, atol: float = 0.0):
        output_t = target_kernel_input_tensors[3]
        output_ref = baseline_input_tensors[3]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.equal(output_t, output_ref):
            diff = (output_t - output_ref).abs()
            print(f"[FAIL] output mismatch: max_abs_diff={int(diff.max().item())}", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 Conv2d forward：output = conv2d(x, weight, bias, stride, padding)

    workload 口径：
      - flops = 2 * out_elems * C_in * R * S + out_elems
        理由：每个输出元素都要遍历一个大小为 C_in*R*S 的卷积窗口做乘加，再加 1 次 bias。
      - memory_bytes = N * C_in * H * W * 4 + C_out * C_in * R * S * 4 + C_out * 4 + out_elems * 4
        理由：需要读取 int32 的输入、权重、bias，并写出 int32 输出。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, weight_shape, bias_shape, output_shape = raw_sizes[:4]
    N, C_in, H, W = x_shape
    C_out, C_in_w, R, S = weight_shape
    N_out, C_out_out, H_out, W_out = output_shape
    assert C_in == C_in_w and bias_shape == (C_out,)
    assert (N_out, C_out_out) == (N, C_out)
    out_elems = N * C_out * H_out * W_out
    return {
        "flops": 2 * out_elems * C_in * R * S + out_elems,
        "memory_bytes": N * C_in * H * W * 4 + C_out * C_in * R * S * 4 + C_out * 4 + out_elems * 4,
    }

DESIGNED_VRAM_SIZE = 48
