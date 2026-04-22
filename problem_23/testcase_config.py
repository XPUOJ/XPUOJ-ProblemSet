from __future__ import annotations

def getNumOfTestcases() -> int:
    return 5

try:
    from typing import List, Tuple, Union
    import sys
    import torch

    KernelArg = Union[torch.Tensor, int, float]
    CURRENT_CASE = None
    TESTCASES = [
        (48896, 24832, 12800, 10, 253),
        (54528, 14080, 14080, 10, 404),
        (41472, 41472, 9472, 10, 199),
        (55040, 55040, 7680, 9, 189),
        (22016, 22016, 19456, 9, 186),
    ]






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
        if testcase_id < 1 or testcase_id > len(TESTCASES):
            return 1
        return testcase_id


    def getTestCaseSize():
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        N, M, H, warmup, iters = TESTCASES[testcase_id - 1]
        M = min(M, N)
        CURRENT_CASE = (N, M, H)
        return [
            (N, H),
            (M,),
            (M, H),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes, device: str = "cuda"):
        global CURRENT_CASE
        N, M, H = CURRENT_CASE
        assert M <= N, f"indices length M must not exceed first axis N, got M={M}, N={N}"
        input_shape, indices_shape, output_shape, n_shape, m_shape, h_shape = testcase_sizes
        assert input_shape == (N, H)
        assert indices_shape == (M,)
        assert output_shape == (M, H)
        input_tensor = torch.randn(N, H, dtype=torch.bfloat16, device=device)
        indices = torch.randperm(N, device=device)[:M].to(torch.int64)
        output = torch.empty(M, H, dtype=torch.bfloat16, device=device)
        return [input_tensor, indices, output, int(N), int(M), int(H)]


    def baseline(input_tensor, indices, output, N, M, H):
        output.copy_(input_tensor[indices])
        return [input_tensor, indices, output, N, M, H]


    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 0.0, atol: float = 0.0):
        output_t = target_kernel_input_tensors[2]
        output_ref = baseline_input_tensors[2]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.equal(output_t, output_ref):
            diff_idx = torch.nonzero(output_t != output_ref, as_tuple=False)
            first = tuple(diff_idx[0].tolist())
            print(f"[FAIL] output mismatch at {first}", file=sys.stderr)
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 Index First Axis：output = input_tensor[indices]

    workload 口径：
      - flops = 0
        理由：这是纯 gather，没有实质浮点运算。
      - memory_bytes = M * H * 2 * 2 + M * 8
        理由：需要读取被索引到的输入行和 int64 的 indices，并写出输出 output。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    input_shape, indices_shape, output_shape, _, _, _ = raw_sizes
    N, H = input_shape
    M = indices_shape[0]
    assert output_shape == (M, H)
    return {
        "flops": 0,
        "memory_bytes": M * H * 2 * 2 + M * 8,
        "dtype": "bf16",
    }

DESIGNED_VRAM_SIZE = 48
