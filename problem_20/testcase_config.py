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
        (1767116544, 3, 25),
        (1764532992, 3, 25),
        (1767116800, 3, 25),
        (1759871744, 3, 25),
        (1771405056, 3, 25),
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
        [bits, packed, num_bits]
        """
        testcase_id = _get_testcase_id()
        global CURRENT_CASE
        num_bits, warmup, iters = TESTCASES[testcase_id - 1]
        CURRENT_CASE = num_bits
        num_packed = (num_bits + 7) // 8
        return [
            (num_bits,),
            (num_packed,),
            (),
        ], (warmup, iters)


    def genTestCase(testcase_sizes: List[Tuple[int, ...]], device: str = "cuda") -> List[KernelArg]:
        global CURRENT_CASE
        assert len(testcase_sizes) == 3, "Expect 3 args: bits, packed, num_bits"
        assert CURRENT_CASE is not None, "CURRENT_CASE must be set by getTestCaseSize() before genTestCase()"

        bits_shape, packed_shape, num_bits_shape = testcase_sizes
        assert num_bits_shape == (), "num_bits must be scalar"

        num_bits = CURRENT_CASE
        num_packed = (num_bits + 7) // 8
        assert bits_shape == (num_bits,)
        assert packed_shape == (num_packed,)

        bits = torch.randint(0, 2, bits_shape, dtype=torch.bool, device=device)
        packed = torch.zeros(packed_shape, dtype=torch.uint8, device=device)
        return [bits, packed, int(num_bits)]


    def baseline(
        bits: torch.Tensor,
        packed: torch.Tensor,
        num_bits: int,
    ) -> List[KernelArg]:
        assert bits.dtype == torch.bool
        assert packed.dtype == torch.uint8
        assert bits.shape == (num_bits,)

        num_packed = (num_bits + 7) // 8
        bits_uint8 = bits.to(torch.uint8)
        if num_bits % 8 != 0:
            pad_size = 8 - (num_bits % 8)
            bits_uint8 = torch.cat([bits_uint8, torch.zeros(pad_size, dtype=torch.uint8, device=bits.device)])
        bits_reshaped = bits_uint8.view(num_packed, 8)

        packed_ref = torch.zeros(num_packed, dtype=torch.uint8, device=bits.device)
        for i in range(8):
            packed_ref = packed_ref | (bits_reshaped[:, i] << (7 - i))

        packed.copy_(packed_ref)
        return [bits, packed, num_bits]


    def check(
        testcase_sizes: List[Tuple[int, ...]],
        original_input_tensors: List[KernelArg],
        target_kernel_input_tensors: List[KernelArg],
        baseline_input_tensors: List[KernelArg],
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> bool:
        assert len(testcase_sizes) == 3
        assert len(target_kernel_input_tensors) == 3
        assert len(baseline_input_tensors) == 3

        _, packed_t, _ = target_kernel_input_tensors
        _, packed_ref, _ = baseline_input_tensors

        if not (torch.is_tensor(packed_t) and torch.is_tensor(packed_ref)):
            print(f"[FAIL] packed must be tensor, got {type(packed_t)} and {type(packed_ref)}", file=sys.stderr)
            return False
        if packed_t.shape != packed_ref.shape:
            print(f"[FAIL] shape mismatch: target {packed_t.shape}, ref {packed_ref.shape}", file=sys.stderr)
            return False
        if packed_t.dtype != packed_ref.dtype:
            print(f"[FAIL] dtype mismatch: target {packed_t.dtype}, ref {packed_ref.dtype}", file=sys.stderr)
            return False

        ok = torch.equal(packed_t, packed_ref)
        if not ok:
            diff_idx = torch.nonzero(packed_t != packed_ref, as_tuple=False)
            first = diff_idx[0].item()
            print(
                f"[FAIL] packed mismatch at index {first}: target={int(packed_t[first].item())}, ref={int(packed_ref[first].item())}",
                file=sys.stderr,
            )
            return False
        return True
except:
    pass
INPUT_CLASS = ["INPUT", "OUTPUT", "INPUT"]


def getWorkload(testcase_sizes) -> dict:
    """
    计算本题 workload。

    本题计算逻辑：
      - 实现 packbits：把 bool bit 序列按 8 个一组打包成 uint8

    workload 口径：
      - flops = num_bits
        理由：每个输入 bit 只参与一次打包写入。
      - memory_bytes = num_bits + num_packed
        理由：需要读取全部 bits，并写出打包后的 uint8 结果。
    """
    raw_sizes = testcase_sizes[0] if isinstance(testcase_sizes, tuple) and len(testcase_sizes) == 2 else testcase_sizes
    bits_shape, packed_shape, _ = raw_sizes
    num_bits = bits_shape[0]
    num_packed = packed_shape[0]
    assert num_packed == (num_bits + 7) // 8
    return {
        "flops": num_bits,
        "memory_bytes": num_bits + num_packed,
    }

DESIGNED_VRAM_SIZE = 48
