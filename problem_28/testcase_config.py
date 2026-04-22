from __future__ import annotations

designed_vram_size = 256
DESIGNED_VRAM_SIZE = designed_vram_size


def getNumOfTestcases() -> int:
    return 6


try:
    import sys
    import torch

    CURRENT_CASE = None
    TESTCASES = [
        (1, 4, 256, 256, 384, 10, 100),
        (1, 6, 384, 384, 256, 8, 80),
        (2, 8, 192, 256, 320, 8, 80),
        (2, 4, 512, 192, 256, 6, 60),
        (3, 5, 256, 320, 384, 6, 60),
        (3, 3, 640, 256, 128, 4, 40),
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

    def _get_shapes(pattern_id: int, B: int, M: int, N: int, K: int):
        if pattern_id == 1:
            return (B, M, K), (B, K, N), (B, M, N)
        if pattern_id == 2:
            return (B, K, M), (B, K, N), (B, M, N)
        return (B, M, K), (B, N, K), (B, M, N)

    def getTestCaseSize():
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        case = TESTCASES[testcase_id - 1]
        pattern_id, B, M, N, K, warmup, iters = case
        CURRENT_CASE = case
        a_shape, b_shape, out_shape = _get_shapes(pattern_id, B, M, N, K)
        return [
            a_shape,
            b_shape,
            out_shape,
            (), (), (), (), ()
        ], (warmup, iters)

    def getDesignedVramSize():
        return designed_vram_size

    def get_designed_vram_size():
        return designed_vram_size

    def genTestCase(testcase_sizes, device: str = "cuda"):
        pattern_id, B, M, N, K, warmup, iters = CURRENT_CASE
        a_shape, b_shape, out_shape = _get_shapes(pattern_id, B, M, N, K)
        a = torch.randn(*a_shape, dtype=torch.bfloat16, device=device)
        b = torch.randn(*b_shape, dtype=torch.bfloat16, device=device)
        output = torch.empty(*out_shape, dtype=torch.bfloat16, device=device)
        return [a, b, output, int(pattern_id), int(B), int(M), int(N), int(K)]

    def baseline(a, b, output, pattern_id, B, M, N, K):
        if pattern_id == 1:
            output.copy_(torch.einsum("bmk,bkn->bmn", a, b))
        elif pattern_id == 2:
            output.copy_(torch.einsum("bkm,bkn->bmn", a, b))
        elif pattern_id == 3:
            output.copy_(torch.einsum("bmk,bnk->bmn", a, b))
        else:
            raise ValueError(f"Unsupported pattern_id: {pattern_id}")
        return [a, b, output, pattern_id, B, M, N, K]

    def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol: float = 1e-2, atol: float = 1e-2):
        output_t = target_kernel_input_tensors[2]
        output_ref = baseline_input_tensors[2]
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False
        if not torch.allclose(output_t, output_ref, rtol=rtol, atol=atol):
            diff = (output_t.float() - output_ref.float()).abs()
            print(f"[FAIL] allclose failed: max_abs_diff={float(diff.max().item()):.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            return False
        return True
except:
    pass
