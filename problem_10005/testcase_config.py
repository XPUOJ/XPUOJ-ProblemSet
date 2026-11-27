"""S5000 correctness and scale-up cases; published timing is 20 warmups / 50 repeats."""
import torch

DESIGNED_VRAM_SIZE = 80
DESIGNED_TESTDATA_GROUPS = 2
TIMING_WARMUP = 20
TIMING_REPEATS = 50
INPUT_CLASS = ["INPUT"] * 5 + ["OUTPUT"] + ["INPUT"] * 4
# Cases 1-6 check correctness; cases 7-12 carry performance weights.
# score_weight is metadata for OJ aggregation, not consumed by the generic Runner.
CASES = [
    {'name': 'ds-v4-fp16',
     'M': 1,
     'E': 32,
     'K': 16,
     'N': 8,
     'topk': 6,
     'dtype': 'float16',
     'family': 'v4',
     'score_weight': 0.0},
    {'name': 'ds-v4-bf16',
     'M': 3,
     'E': 32,
     'K': 17,
     'N': 13,
     'topk': 6,
     'dtype': 'bfloat16',
     'family': 'v4',
     'score_weight': 0.0},
    {'name': 'ds-r1-fp16',
     'M': 3,
     'E': 32,
     'K': 32,
     'N': 16,
     'topk': 8,
     'dtype': 'float16',
     'family': 'r1',
     'score_weight': 0.0},
    {'name': 'ds-r1-bf16',
     'M': 9,
     'E': 64,
     'K': 32,
     'N': 16,
     'topk': 8,
     'dtype': 'bfloat16',
     'family': 'r1',
     'score_weight': 0.0},
    {'name': 'qwen-fp16',
     'M': 9,
     'E': 64,
     'K': 33,
     'N': 17,
     'topk': 10,
     'dtype': 'float16',
     'family': 'qwen',
     'score_weight': 0.0},
    {'name': 'qwen-bf16',
     'M': 17,
     'E': 64,
     'K': 64,
     'N': 32,
     'topk': 10,
     'dtype': 'bfloat16',
     'family': 'qwen',
     'score_weight': 0.0},
    {'name': 'r1-fp16-decode-4k',
     'M': 1,
     'E': 64,
     'K': 4096,
     'N': 1024,
     'topk': 8,
     'dtype': 'float16',
     'family': 'r1',
     'score_weight': 0.2},
    {'name': 'r1-bf16-m16-4k',
     'M': 16,
     'E': 128,
     'K': 4096,
     'N': 1024,
     'topk': 8,
     'dtype': 'bfloat16',
     'family': 'r1',
     'score_weight': 0.2},
    {'name': 'v4-fp16-m8-e256',
     'M': 8,
     'E': 256,
     'K': 2048,
     'N': 512,
     'topk': 6,
     'dtype': 'float16',
     'family': 'v4',
     'score_weight': 0.175},
    {'name': 'v4-bf16-m64-e256',
     'M': 64,
     'E': 256,
     'K': 2048,
     'N': 1024,
     'topk': 6,
     'dtype': 'bfloat16',
     'family': 'v4',
     'score_weight': 0.175},
    {'name': 'qwen-fp16-m16-e256-4k',
     'M': 16,
     'E': 256,
     'K': 4096,
     'N': 1024,
     'topk': 10,
     'dtype': 'float16',
     'family': 'qwen',
     'score_weight': 0.125},
    {'name': 'qwen-bf16-m256-e256',
     'M': 256,
     'E': 256,
     'K': 2048,
     'N': 512,
     'topk': 10,
     'dtype': 'bfloat16',
     'family': 'qwen',
     'score_weight': 0.125},
]


def getNumOfTestcases():
    return len(CASES)


def getTestCaseSize():
    case_id = int(input())
    if not 1 <= case_id <= len(CASES):
        raise ValueError("invalid case id")
    return [case_id], (TIMING_WARMUP, TIMING_REPEATS)


def genTestCase(testcase_sizes, device="musa"):
    c = CASES[testcase_sizes[0] - 1]
    M, E, K, N = (c[k] for k in ["M", "E", "K", "N"])
    dtype = getattr(torch, c["dtype"])
    x = torch.randn((M, K), device=device, dtype=dtype)
    w1 = torch.randn((E, 2 * N, K), device=device, dtype=dtype) / K ** 0.5
    w2 = torch.randn((E, K, N), device=device, dtype=dtype) / N ** 0.5
    logits = torch.randn((M, E), device=device, dtype=torch.float32)
    bias = torch.empty(E, device=device, dtype=torch.float32).uniform_(-0.4, 0.4)
    out = torch.full((M, K), float("nan"), device=device, dtype=dtype)
    return [x, w1, w2, logits, bias, out, c["topk"], 8, 4, 1.0]


def routing(router_logits, routing_bias, topk, n_groups, topk_groups, scale):
    M, E = router_logits.shape
    scores = torch.sigmoid(router_logits)
    choice_scores = scores + routing_bias.view(1, E)
    grouped = choice_scores.view(M, n_groups, E // n_groups)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(topk_groups, dim=-1).indices
    mask = torch.zeros((M, n_groups), device=scores.device, dtype=torch.bool)
    mask.scatter_(1, selected_groups, True)
    expert_mask = mask[:, :, None].expand(M, n_groups, E // n_groups).reshape(M, E)
    ids = choice_scores.masked_fill(~expert_mask, float("-inf")).topk(topk, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * scale
    return ids, weights


def baseline(x, w1, w2, router_logits, routing_bias, out, topk, n_groups, topk_groups, scale):
    ids, weights = routing(router_logits, routing_bias, topk, n_groups, topk_groups, scale)
    fp32 = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for m in range(x.shape[0]):
        for t in range(topk):
            e = int(ids[m, t])
            hidden = x[m].float() @ w1[e].float().T
            gate, up = hidden.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_out = activated @ w2[e].float().T
            fp32[m] += weights[m, t].float() * expert_out
    out.copy_(fp32.to(x.dtype))


def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors,
          baseline_input_tensors, rtol=None, atol=None):
    a, t, ref = original_input_tensors, target_kernel_input_tensors, baseline_input_tensors
    for i in range(len(a)):
        if i == 5:
            continue
        if torch.is_tensor(a[i]):
            if not torch.equal(a[i], t[i]):
                return False
        elif a[i] != t[i]:
            return False
    out = t[5]
    if out.shape != a[0].shape or out.dtype != a[0].dtype or out.device != a[0].device:
        return False
    if not bool(torch.isfinite(out).all()):
        return False
    tol = (2e-2, 3e-2) if out.dtype == torch.bfloat16 else (1e-2, 2e-2)
    return bool(torch.allclose(out.float(), ref[5].float(), rtol=tol[0], atol=tol[1]))


def getWorkload(testcase_sizes):
    c = CASES[testcase_sizes[0] - 1]
    return {"flops": int(6 * c["M"] * c["topk"] * c["K"] * c["N"]),
            "memory_bytes": int(2 * c["E"] * 3 * c["K"] * c["N"]), "dtype": c["dtype"]}
