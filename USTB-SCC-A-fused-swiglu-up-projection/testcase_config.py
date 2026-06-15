from __future__ import annotations

from typing import List, Union

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

KernelArg = Union["torch.Tensor", int, float]

# Each testcase is (M, K, N, warmup, iters).
TESTCASES = [
    (1024, 1024, 2048, 5, 50),
    (2048, 4096, 3072, 5, 30),
    (4096, 4096, 4096, 5, 20),
    (4097, 3073, 2305, 5, 20),
    (8192, 2048, 4096, 5, 15),
    (1536, 6144, 6144, 5, 15),
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def _read_testcase_id() -> int:
    try:
        raw = input().strip()
    except EOFError:
        return 1

    try:
        testcase_id = int(raw.split()[0]) if raw else 1
    except ValueError:
        return 1

    if testcase_id < 1 or testcase_id > len(TESTCASES):
        return 1
    return testcase_id


def getTestCaseSize():
    testcase_id = _read_testcase_id()
    M, K, N, warmup, iters = TESTCASES[testcase_id - 1]
    return [
        (M, K),
        (K, N),
        (K, N),
        (N,),
        (N,),
        (M, N),
        (),
        (),
        (),
    ], (warmup, iters)


def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
    if torch is None:
        raise RuntimeError("torch is required to generate test cases")

    M, K = testcase_sizes[0]
    N = testcase_sizes[1][1]

    generator = torch.Generator(device=device)
    generator.manual_seed(20240605 + M * 17 + K * 31 + N * 47)

    x = torch.randn((M, K), device=device, dtype=torch.float32, generator=generator).to(torch.bfloat16)

    weight_scale = K ** -0.5
    w_gate = (torch.randn((K, N), device=device, dtype=torch.float32, generator=generator) * weight_scale).to(torch.bfloat16)
    w_up = (torch.randn((K, N), device=device, dtype=torch.float32, generator=generator) * weight_scale).to(torch.bfloat16)

    b_gate = torch.empty((N,), device=device, dtype=torch.float32)
    b_gate.uniform_(-0.5, 0.5, generator=generator)
    b_up = torch.empty((N,), device=device, dtype=torch.float32)
    b_up.uniform_(-0.5, 0.5, generator=generator)

    y = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    return [x, w_gate, w_up, b_gate, b_up, y, int(M), int(K), int(N)]


def _set_ieee_matmul_precision(enabled: bool):
    if torch is None or not hasattr(torch.backends, "cuda"):
        return None

    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is None:
        return None

    if hasattr(matmul_backend, "fp32_precision"):
        old_value = matmul_backend.fp32_precision
        if enabled:
            matmul_backend.fp32_precision = "ieee"
        return ("fp32_precision", old_value)

    if hasattr(matmul_backend, "allow_tf32"):
        old_value = matmul_backend.allow_tf32
        if enabled:
            matmul_backend.allow_tf32 = False
        return ("allow_tf32", old_value)

    return None


def _restore_matmul_precision(state) -> None:
    if state is None or torch is None or not hasattr(torch.backends, "cuda"):
        return

    name, old_value = state
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None:
        setattr(matmul_backend, name, old_value)


def baseline(x, w_gate, w_up, b_gate, b_up, y, M, K, N):
    if torch is None or F is None:
        raise RuntimeError("torch is required to run baseline")

    precision_state = _set_ieee_matmul_precision(x.is_cuda)
    try:
        x_fp32 = x.to(torch.float32)
        gate = x_fp32 @ w_gate.to(torch.float32)
        gate = gate + b_gate
        up = x_fp32 @ w_up.to(torch.float32)
        up = up + b_up
        y.copy_((F.silu(gate) * up).to(torch.bfloat16))
    finally:
        _restore_matmul_precision(precision_state)

    return [x, w_gate, w_up, b_gate, b_up, y, M, K, N]


def _check_same_tensor(name: str, actual: "torch.Tensor", expected: "torch.Tensor") -> bool:
    if actual.shape != expected.shape:
        print(f"[FAIL] {name} shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}")
        return False
    if actual.dtype != expected.dtype:
        print(f"[FAIL] {name} dtype mismatch: got {actual.dtype}, expected {expected.dtype}")
        return False
    if not torch.equal(actual, expected):
        diff = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
        print(
            f"[FAIL] {name} was modified: max_abs_diff={diff.max().item():.6g}, "
            f"mean_abs_diff={diff.mean().item():.6g}"
        )
        return False
    return True


def _check_close(name: str, actual: "torch.Tensor", expected: "torch.Tensor", rtol: float, atol: float) -> bool:
    if actual.shape != expected.shape:
        print(f"[FAIL] {name} shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}")
        return False
    if actual.dtype != expected.dtype:
        print(f"[FAIL] {name} dtype mismatch: got {actual.dtype}, expected {expected.dtype}")
        return False

    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    if not torch.allclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol):
        diff = (actual_fp32 - expected_fp32).abs()
        max_idx = int(diff.argmax().item())
        print(
            f"[FAIL] {name} mismatch: max_abs_diff={diff.max().item():.6g}, "
            f"mean_abs_diff={diff.mean().item():.6g}, rtol={rtol}, atol={atol}"
        )
        print(
            f"[FAIL] max diff flat index {max_idx}: "
            f"target={actual_fp32.flatten()[max_idx].item():.6g}, "
            f"ref={expected_fp32.flatten()[max_idx].item():.6g}"
        )
        return False
    return True


def check(
    testcase_sizes,
    original_input_tensors,
    target_kernel_input_tensors,
    baseline_input_tensors,
    rtol=5e-2,
    atol=5e-2,
) -> bool:
    if torch is None:
        raise RuntimeError("torch is required to check results")

    del testcase_sizes
    input_names = ["x", "w_gate", "w_up", "b_gate", "b_up"]
    for idx, name in enumerate(input_names):
        if not _check_same_tensor(name, target_kernel_input_tensors[idx], original_input_tensors[idx]):
            return False

    return _check_close("y", target_kernel_input_tensors[5], baseline_input_tensors[5], rtol, atol)


INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    M, K = raw_sizes[0]
    N = raw_sizes[1][1]

    # Two MxK by KxN GEMMs plus a small fused bias/SwiGLU epilogue.
    flops = 4 * M * K * N + 8 * M * N
    memory_bytes = 2 * (M * K + 2 * K * N + M * N) + 4 * (2 * N)
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "bf16",
    }


DESIGNED_VRAM_SIZE = 48
