from __future__ import annotations

from typing import List, Union

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

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


def _torch_reference(x, w_gate, w_up, b_gate, b_up, y, M, K, N):
    if torch is None or F is None:
        raise RuntimeError("torch is required to run the PyTorch reference")

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


if tl is not None:
    @triton.jit
    def _swiglu_kernel(
        x,
        w_gate,
        w_up,
        b_gate,
        b_up,
        y,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x_tile = tl.load(
                x + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            wg_tile = tl.load(
                w_gate + k[:, None] * N + offs_n[None, :],
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            wu_tile = tl.load(
                w_up + k[:, None] * N + offs_n[None, :],
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc_gate += tl.dot(x_tile, wg_tile, out_dtype=tl.float32)
            acc_up += tl.dot(x_tile, wu_tile, out_dtype=tl.float32)

        acc_gate += tl.load(b_gate + offs_n, mask=offs_n < N, other=0.0)[None, :]
        acc_up += tl.load(b_up + offs_n, mask=offs_n < N, other=0.0)[None, :]

        out = (acc_gate / (1.0 + tl.exp(-acc_gate))) * acc_up
        tl.store(
            y + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )
else:
    _swiglu_kernel = None


def _starter_triton_baseline(x, w_gate, w_up, b_gate, b_up, y, M: int, K: int, N: int):
    if triton is None or tl is None or _swiglu_kernel is None:
        raise RuntimeError("triton is required to run the starter-code baseline")

    block_m = 16
    block_n = 16
    block_k = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _swiglu_kernel[grid](
        x,
        w_gate,
        w_up,
        b_gate,
        b_up,
        y,
        M,
        K,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return [x, w_gate, w_up, b_gate, b_up, y, M, K, N]


def baseline(x, w_gate, w_up, b_gate, b_up, y, M, K, N):
    return _starter_triton_baseline(x, w_gate, w_up, b_gate, b_up, y, M, K, N)


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
