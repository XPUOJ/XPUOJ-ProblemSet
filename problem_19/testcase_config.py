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
        (128, 3072, 512, 4, 1, 1, 4, 84),
        (256, 3072, 256, 4, 0, 1, 5, 110),
        (64, 6144, 512, 4, 1, 1, 4, 84),
        (192, 2048, 512, 4, 0, 1, 5, 108),
        (96, 4096, 512, 4, 1, 1, 4, 84),
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


    def getTestCaseSize() -> Union[Tuple[List[Tuple[int, ...]], Tuple[int, int]], List[Tuple[int, ...]]]:
        """
        参数顺序:
        [x, weight, bias, conv_states, output, batch_size, dim, seq_len, kernel_size, has_initial_state, silu_activation]
        """
        testcase_id = _get_testcase_id()

        global CURRENT_CASE
        batch_size, seq_len, dim, kernel_size, has_initial_state, silu_activation, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = (batch_size, seq_len, dim, kernel_size, has_initial_state, silu_activation)
        return [
            (batch_size, dim, seq_len),
            (dim, kernel_size),
            (dim,),
            (batch_size, dim, kernel_size - 1),
            (batch_size, dim, seq_len),
            (),
            (),
            (),
            (),
            (),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        global CURRENT_CASE
        assert len(testcase_sizes) == 11, "Expect 11 args"
        assert CURRENT_CASE is not None, "CURRENT_CASE must be set by getTestCaseSize() before genTestCase()"
        (
            x_shape,
            weight_shape,
            bias_shape,
            conv_states_shape,
            output_shape,
            batch_size_shape,
            dim_shape,
            seq_len_shape,
            kernel_size_shape,
            has_initial_state_shape,
            silu_activation_shape,
        ) = testcase_sizes

        assert batch_size_shape == ()
        assert dim_shape == ()
        assert seq_len_shape == ()
        assert kernel_size_shape == ()
        assert has_initial_state_shape == ()
        assert silu_activation_shape == ()

        batch_size, dim, seq_len = x_shape
        weight_dim, kernel_size = weight_shape
        assert weight_dim == dim
        assert bias_shape == (dim,)
        assert conv_states_shape == (batch_size, dim, kernel_size - 1)
        assert output_shape == (batch_size, dim, seq_len)
        _, _, _, _, has_initial_state, silu_activation = CURRENT_CASE

        x = torch.randn(*x_shape, dtype=torch.float32, device=device)
        weight = torch.randn(*weight_shape, dtype=torch.float32, device=device)
        bias = torch.randn(*bias_shape, dtype=torch.float32, device=device)
        conv_states = torch.randn(*conv_states_shape, dtype=torch.float32, device=device)
        output = torch.zeros(*output_shape, dtype=torch.float32, device=device)

        return [
            x,
            weight,
            bias,
            conv_states,
            output,
            int(batch_size),
            int(dim),
            int(seq_len),
            int(kernel_size),
            int(has_initial_state),
            int(silu_activation),
        ]


    def baseline(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        conv_states: torch.Tensor,
        output: torch.Tensor,
        batch_size: int,
        dim: int,
        seq_len: int,
        kernel_size: int,
        has_initial_state: int,
        silu_activation: int,
    ) -> List[KernelArg]:
        assert x.shape == (batch_size, dim, seq_len)
        assert weight.shape == (dim, kernel_size)
        assert bias.shape == (dim,)
        assert conv_states.shape == (batch_size, dim, kernel_size - 1)
        assert output.shape == (batch_size, dim, seq_len)

        weight_reshaped = weight.unsqueeze(1)
        if has_initial_state:
            x_with_states = torch.cat([conv_states, x], dim=-1)
            ref = F.conv1d(x_with_states, weight_reshaped, bias, groups=dim, padding=0)
            ref = ref[..., :seq_len]
        else:
            ref = F.conv1d(x, weight_reshaped, bias, groups=dim, padding=kernel_size - 1)
            ref = ref[..., :seq_len]

        if silu_activation:
            ref = F.silu(ref)

        output.copy_(ref)
        return [x, weight, bias, conv_states, output, batch_size, dim, seq_len, kernel_size, has_initial_state, silu_activation]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> bool:
        assert len(testcase_sizes) == 11
        assert len(target_kernel_input_tensors) == 11
        assert len(baseline_input_tensors) == 11

        output_t = target_kernel_input_tensors[4]
        output_ref = baseline_input_tensors[4]

        if not (torch.is_tensor(output_t) and torch.is_tensor(output_ref)):
            print(f"[FAIL] output must be tensor, got {type(output_t)} and {type(output_ref)}", file=sys.stderr)
            return False
        if output_t.shape != output_ref.shape:
            print(f"[FAIL] shape mismatch: target {output_t.shape}, ref {output_ref.shape}", file=sys.stderr)
            return False
        if output_t.dtype != output_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {output_t.dtype}, ref {output_ref.dtype}", file=sys.stderr)
            return False

        ok = torch.allclose(output_t, output_ref, rtol=rtol, atol=atol)
        if not ok:
            diff = (output_t - output_ref).abs()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            print(f"[FAIL] allclose failed: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f} (rtol={rtol}, atol={atol})", file=sys.stderr)
            max_idx = diff.argmax()
            max_idx_tuple = tuple(torch.unravel_index(max_idx, output_t.shape))
            print(
                f"[FAIL] max diff at position {max_idx_tuple}: target={float(output_t.flatten()[max_idx].item()):.6f}, ref={float(output_ref.flatten()[max_idx].item()):.6f}",
                file=sys.stderr,
            )
            return False

        return True
except:
    pass
INPUT_CLASS = ["INPUT", "INPUT", "INPUT", "INPUT", "OUTPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 causal depthwise Conv1d forward，可选拼接 conv_states，可选再做 SiLU

    workload 口径：
      - flops = 2 * output_elems * kernel_size + output_elems + optional_silu_flops
        理由：每个输出元素要对 kernel_size 个位置做乘加，再加一次 bias；若开启 SiLU，还要再加激活开销。
      - memory_bytes = x_bytes + weight_bytes + bias_bytes + output_bytes + optional_conv_states_bytes
        理由：需要读取输入 x、卷积核、bias，并写出 output；若有初始状态，还要额外读取 conv_states。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    x_shape, weight_shape, bias_shape, conv_states_shape, output_shape, _, _, _, _, _, _ = raw_sizes
    batch_size, dim, seq_len = x_shape
    weight_dim, kernel_size = weight_shape
    assert weight_dim == dim and bias_shape == (dim,) and output_shape == (batch_size, dim, seq_len)
    
    # 从CURRENT_CASE获取has_initial_state和silu_activation
    global CURRENT_CASE
    if CURRENT_CASE is None:
        raise ValueError("CURRENT_CASE not set")
    _, _, _, _, has_initial_state, silu_activation = CURRENT_CASE
    
    output_elems = batch_size * dim * seq_len
    flops = 2 * output_elems * kernel_size + output_elems
    if silu_activation:
        flops += 3 * output_elems
    memory_bytes = batch_size * dim * seq_len * 4 + dim * kernel_size * 4 + dim * 4 + output_elems * 4
    if has_initial_state:
        memory_bytes += batch_size * dim * (kernel_size - 1) * 4
    return {
        "flops": flops,
        "memory_bytes": memory_bytes,
        "dtype": "fp32",
    }

DESIGNED_VRAM_SIZE = 48
