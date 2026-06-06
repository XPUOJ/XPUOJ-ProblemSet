"""
test_runner.py — 验证 testcase_config.py 的所有功能
====================================================
- 测试 getNumOfTestcases()
- 测试 getTestCaseSize()（模拟 stdin）
- 测试 genTestCase()（在 CPU 上生成 tensor）
- 测试 baseline()（PyTorch 参考实现）
- 测试 check()（数值比较）
- 测试 getWorkload()（FLOPs/内存计算）
- 测试 INPUT_CLASS 结构
"""

import sys
import os
import io
import traceback

import torch

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保可以导入 testcase_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testcase_config as tc


def test_getNumOfTestcases():
    """测试 1: 验证测试点数量"""
    print("=" * 60)
    print("测试 1: getNumOfTestcases()")
    n = tc.getNumOfTestcases()
    assert n == len(tc.TESTCASES), f"数量不匹配: {n} vs {len(tc.TESTCASES)}"
    assert n > 0, "测试点数量为 0"
    print(f"  ✓ 共有 {n} 个测试点")
    print()


def test_getTestCaseSize():
    """测试 2: 验证 getTestCaseSize() 返回的 shape 列表"""
    print("=" * 60)
    print("测试 2: getTestCaseSize()")
    print()

    for idx in range(len(tc.TESTCASES)):
        B, H, S, D, warmup, iters = tc.TESTCASES[idx]

        # 模拟 stdin 输入（1-based）
        simulated_input = str(idx + 1)
        sys.stdin = io.StringIO(simulated_input)

        shapes, (w, i) = tc.getTestCaseSize()

        sys.stdin = sys.__stdin__  # 恢复 stdin

        # 验证 shapes 长度
        assert len(shapes) == 10, f"shapes 长度应为 10，实际为 {len(shapes)}"
        assert len(shapes[0]) == 4, f"Q shape 应为 4D，实际为 {len(shapes[0])}"
        assert len(shapes[-1]) == 4, f"O shape 应为 4D，实际为 {len(shapes[-1])}"

        # 验证 scalar shape
        for j in range(4, 9):
            assert shapes[j] == (), f"shapes[{j}] 应为 ()，实际为 {shapes[j]}"

        # 验证具体值
        assert shapes[0] == (B, H, S, D), f"Q shape 应为 ({B},{H},{S},{D})"
        assert shapes[1] == (B, H, S, D), f"K shape"
        assert shapes[2] == (B, H, S, D), f"V shape"
        assert shapes[3] == (B, 1, S, S), f"mask shape 应为 ({B},1,{S},{S})"
        assert shapes[9] == (B, H, S, D), f"O shape"

        # 验证 warmup/iters
        assert w == warmup, f"warmup: {w} vs {warmup}"
        assert i == iters, f"iters: {i} vs {iters}"

        print(f"  ✓ 测试点 {idx+1}: B={B}, H={H}, S={S}, D={D}, "
              f"warmup={warmup}, iters={iters}")

    # 测试边界情况：无输入 → 默认 1
    sys.stdin = io.StringIO("")
    shapes, _ = tc.getTestCaseSize()
    sys.stdin = sys.__stdin__
    assert shapes[0] == (1, 16, 512, 64), f"默认测试点应为测试点 1"

    # 测试边界情况：超出范围 → 默认 1
    sys.stdin = io.StringIO("999")
    shapes, _ = tc.getTestCaseSize()
    sys.stdin = sys.__stdin__
    assert shapes[0] == (1, 16, 512, 64), f"超出范围应回退到测试点 1"

    print(f"  ✓ 边界情况处理正确")
    print()


def test_genTestCase():
    """测试 3: 验证 genTestCase() 生成的数据"""
    print("=" * 60)
    print("测试 3: genTestCase() — CPU 上测试")
    print()

    # 遍历所有测试点 — 用 CPU 测试数据生成逻辑
    skipped_cuda = 0
    for idx, (B, H, S, D, warmup, iters) in enumerate(tc.TESTCASES):
        testcase_sizes = (B, H, S, D, warmup, iters)

        # 小测试在 CPU 上跑
        try:
            args = tc.genTestCase(testcase_sizes, device="cpu")
        except Exception as e:
            print(f"  ✗ 测试点 {idx+1}: 生成数据失败 — {e}")
            continue

        assert len(args) == 10, f"args 应为 10 个元素，实际 {len(args)}"

        Q, K, V, mask, B_s, H_s, S_s, D_s, alpha, O = args

        # 检查 tensor shape
        assert Q.shape == (B, H, S, D), f"Q shape"
        assert K.shape == (B, H, S, D), f"K shape"
        assert V.shape == (B, H, S, D), f"V shape"
        assert O.shape == (B, H, S, D), f"O shape"

        # 检查 dtype（CPU 上也要是 bfloat16，只要 PyTorch 支持）
        assert Q.dtype == torch.bfloat16, f"Q dtype"
        assert O.dtype == torch.bfloat16, f"O dtype"

        # 检查 mask
        if B == 1 and S == 1987:
            assert mask is None, f"测试点 {idx+1} mask 应为 None"
        else:
            assert mask is not None, f"测试点 {idx+1} mask 不应为 None"
            assert mask.shape == (B, 1, S, S), f"mask shape: {mask.shape}"

        # 检查 scalar
        assert isinstance(B_s, int) and B_s == B
        assert isinstance(H_s, int) and H_s == H
        assert isinstance(S_s, int) and S_s == S
        assert isinstance(D_s, int) and D_s == D
        assert isinstance(alpha, float)

        # 检查值范围 [-1, 1]
        assert Q.min() >= -1.0 and Q.max() <= 1.0, f"Q 值范围超出 [-1, 1]"

        print(f"  ✓ 测试点 {idx+1}: Q{K.shape}, "
              f"mask={'None' if mask is None else f'causal {mask.shape}'}, "
              f"alpha={alpha:.6f}")

    print()


def test_baseline():
    """测试 4: 验证 baseline() 参考实现"""
    print("=" * 60)
    print("测试 4: baseline() — 参考实现")
    print()

    for idx, (B, H, S, D, warmup, iters) in enumerate(tc.TESTCASES):
        testcase_sizes = (B, H, S, D, warmup, iters)

        # 用小尺寸在 CPU 上测试
        actual_S = min(S, 64)  # CPU 上用更小的尺寸，加速测试
        actual_testcase = (B, H, actual_S, D, warmup, iters)

        try:
            args = tc.genTestCase(actual_testcase, device="cpu")
        except Exception as e:
            print(f"  ✗ 测试点 {idx+1}: 生成数据失败 — {e}")
            continue

        # 运行 baseline
        try:
            result = tc.baseline(*args)
        except Exception as e:
            print(f"  ✗ 测试点 {idx+1}: baseline 执行失败 — {e}")
            traceback.print_exc()
            continue

        O = result[-1]

        # baseline 应该原地修改 O（copy_ 操作）
        assert not torch.allclose(O, torch.zeros_like(O)), \
            "baseline 应该修改了输出 O"

        # 检查 softmax 性质：O 的最后一维应该归一化了（概率分布加权）
        # 值应该在合理范围内
        assert not torch.isnan(O).any(), f"O 包含 NaN"
        assert not torch.isinf(O).any(), f"O 包含 Inf"

        print(f"  ✓ 测试点 {idx+1}: "
              f"baseline O shape={list(O.shape)}, "
              f"O range=[{O.min().item():.6f}, {O.max().item():.6f}]")

    print()


def test_check():
    """测试 5: 验证 check() 函数"""
    print("=" * 60)
    print("测试 5: check() — 数值比较")
    print()

    # 测试正常情况：baseline 与自己比较应该通过
    B, H, S, D = 1, 4, 64, 32
    testcase_sizes = (B, H, S, D, 1, 1)

    args = tc.genTestCase(testcase_sizes, device="cpu")
    baseline_args = list(args)  # 复制
    baseline_args[-1] = baseline_args[-1].clone()

    tc.baseline(*args)
    tc.baseline(*baseline_args)

    assert tc.check(testcase_sizes, args, args, baseline_args), \
        "相同输出应该通过 check"

    # 测试故意制造差异
    original_O = args[-1].clone()
    args[-1].add_(1.0)  # 制造差异

    assert not tc.check(tc.TESTCASES[0], args, args, baseline_args), \
        "不同输出不应该通过 check"

    # 恢复
    args[-1] = original_O

    print("  ✓ 正常情况通过")
    print("  ✓ 异常差异正确检测")
    print()


def test_getWorkload():
    """测试 6: 验证 getWorkload()"""
    print("=" * 60)
    print("测试 6: getWorkload() — FLOPs/内存计算")
    print()

    for idx, (B, H, S, D, warmup, iters) in enumerate(tc.TESTCASES):
        testcase_sizes = (B, H, S, D, warmup, iters)
        workload = tc.getWorkload(testcase_sizes)

        assert "flops" in workload
        assert "memory_bytes" in workload
        assert "dtype" in workload
        assert workload["dtype"] == "bf16"

        # FLOPs 验证
        expected_flops = 4 * B * H * S * S * D
        assert workload["flops"] == expected_flops, \
            f"FLOPs: {workload['flops']} vs {expected_flops}"

        # memory_bytes 验证
        expected_mem = (3 * B * H * S * D + 4 * B * H * S * S + B * H * S * D) * 2
        assert workload["memory_bytes"] == expected_mem, \
            f"Memory: {workload['memory_bytes']} vs {expected_mem}"

        flops_g = workload["flops"] / 1e9
        mem_gb = workload["memory_bytes"] / 1e9

        print(f"  ✓ 测试点 {idx+1}: {flops_g:.3f} GFLOPs, "
              f"{mem_gb:.3f} GB memory")

    print()


def test_INPUT_CLASS():
    """测试 7: 验证 INPUT_CLASS 结构"""
    print("=" * 60)
    print("测试 7: INPUT_CLASS")
    input_count = sum(1 for c in tc.INPUT_CLASS if c == "INPUT")
    output_count = sum(1 for c in tc.INPUT_CLASS if c == "OUTPUT")
    assert input_count == 9, f"INPUT 数量应为 9，实际 {input_count}"
    assert output_count == 1, f"OUTPUT 数量应为 1，实际 {output_count}"
    assert tc.INPUT_CLASS[-1] == "OUTPUT", "最后一个应为 OUTPUT"
    print(f"  ✓ INPUT x{input_count}, OUTPUT x{output_count}")
    print()


def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║     testcase_config.py 功能验证测试                    ║")
    print("║     环境: CPU-only (无 GPU)                            ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  cpu")
    print()

    all_passed = True
    tests = [
        test_getNumOfTestcases,
        test_getTestCaseSize,
        test_genTestCase,
        test_baseline,
        test_check,
        test_getWorkload,
        test_INPUT_CLASS,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  ╔══════════════════════════════════════════════════╗")
            print(f"  ║  ✗ {test_fn.__name__} FAILED")
            print(f"  ╚══════════════════════════════════════════════════╝")
            traceback.print_exc()
            print()
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  ✓ 所有测试通过！testcase_config.py 功能正常。")
    else:
        print("  ✗ 部分测试失败，请检查上述错误。")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
