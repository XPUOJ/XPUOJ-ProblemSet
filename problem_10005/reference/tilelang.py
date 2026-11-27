"""Simple FP32-accumulating TileLang baseline; intended for small validation cases."""
import torch
import tilelang
import tilelang.language as T

_KERNELS = {}


def _build(M, E, K, N, TOPK, GROUPS, TOP_GROUPS, SCALE, DTYPE):
    GROUP_SIZE = E // GROUPS

    @T.prim_func
    def route(
        Logits: T.Tensor((M, E), "float32"),
        Bias: T.Tensor((E,), "float32"),
        Ids: T.Tensor((M, TOPK), "int32"),
        Weights: T.Tensor((M, TOPK), "float32"),
    ):
        with T.Kernel(M, threads=128) as row:
            # One GPU thread per token: deliberately simple selection baseline.
            tx = T.get_thread_binding()
            raw = T.alloc_local((E,), "float32")
            choice = T.alloc_local((E,), "float32")
            group_scores = T.alloc_local((GROUPS,), "float32")
            selected = T.alloc_local((GROUPS,), "int32")
            used = T.alloc_local((E,), "int32")
            first = T.alloc_local((1,), "float32")
            second = T.alloc_local((1,), "float32")
            best = T.alloc_local((1,), "float32")
            best_id = T.alloc_local((1,), "int32")
            denominator = T.alloc_local((1,), "float32")
            if tx == 0:
                for e in T.serial(E):
                    raw[e] = 1.0 / (1.0 + T.exp(-Logits[row, e]))
                    choice[e] = raw[e] + Bias[e]
                    used[e] = 0
                for g in T.serial(GROUPS):
                    first[0] = -T.infinity("float32")
                    second[0] = -T.infinity("float32")
                    selected[g] = 0
                    for j in T.serial(GROUP_SIZE):
                        if choice[g * GROUP_SIZE + j] > first[0]:
                            second[0] = first[0]
                            first[0] = choice[g * GROUP_SIZE + j]
                        else:
                            second[0] = T.max(second[0], choice[g * GROUP_SIZE + j])
                    group_scores[g] = first[0] + second[0]
                for t in T.serial(TOP_GROUPS):
                    best[0] = -T.infinity("float32")
                    best_id[0] = 0
                    for g in T.serial(GROUPS):
                        if (selected[g] == 0) and (group_scores[g] > best[0]):
                            best[0] = group_scores[g]
                            best_id[0] = g
                    selected[best_id[0]] = 1
                denominator[0] = 0.0
                for t in T.serial(TOPK):
                    best[0] = -T.infinity("float32")
                    best_id[0] = 0
                    for e in T.serial(E):
                        if (selected[e // GROUP_SIZE] != 0) and (used[e] == 0) and (choice[e] > best[0]):
                            best[0] = choice[e]
                            best_id[0] = e
                    Ids[row, t] = best_id[0]
                    Weights[row, t] = raw[best_id[0]]
                    denominator[0] += raw[best_id[0]]
                    used[best_id[0]] = 1
                for t in T.serial(TOPK):
                    Weights[row, t] = Weights[row, t] / denominator[0] * SCALE

    @T.prim_func
    def activate(
        X: T.Tensor((M, K), DTYPE),
        W1: T.Tensor((E, 2 * N, K), DTYPE),
        Ids: T.Tensor((M, TOPK), "int32"),
        Hidden: T.Tensor((M, TOPK, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, 128), M, TOPK, threads=128) as (block, row, t):
            gate = T.alloc_local((1,), "float32")
            up = T.alloc_local((1,), "float32")
            for lane in T.Parallel(128):
                j = block * 128 + lane
                if j < N:
                    gate[0] = 0.0
                    up[0] = 0.0
                    for k in T.serial(K):
                        gate[0] += T.cast(X[row, k], "float32") * T.cast(W1[Ids[row, t], j, k], "float32")
                        up[0] += T.cast(X[row, k], "float32") * T.cast(W1[Ids[row, t], N + j, k], "float32")
                    Hidden[row, t, j] = gate[0] / (1.0 + T.exp(-gate[0])) * up[0]

    @T.prim_func
    def combine(
        Hidden: T.Tensor((M, TOPK, N), "float32"),
        W2: T.Tensor((E, K, N), DTYPE),
        Ids: T.Tensor((M, TOPK), "int32"),
        Weights: T.Tensor((M, TOPK), "float32"),
        Out: T.Tensor((M, K), DTYPE),
    ):
        with T.Kernel(T.ceildiv(K, 128), M, threads=128) as (block, row):
            total = T.alloc_local((1,), "float32")
            expert = T.alloc_local((1,), "float32")
            for lane in T.Parallel(128):
                k = block * 128 + lane
                if k < K:
                    total[0] = 0.0
                    for t in T.serial(TOPK):
                        expert[0] = 0.0
                        for j in T.serial(N):
                            expert[0] += Hidden[row, t, j] * T.cast(W2[Ids[row, t], k, j], "float32")
                        total[0] += Weights[row, t] * expert[0]
                    Out[row, k] = T.cast(total[0], DTYPE)

    options = {"target": {"kind": "musa", "arch": "mp_31"}, "execution_backend": "cython"}
    return tuple(tilelang.compile(f, **options) for f in (route, activate, combine))


def run_kernel(x, w1, w2, router_logits, routing_bias, out, topk, n_groups, topk_groups, routed_scaling_factor):
    key = (x.shape[0], w1.shape[0], x.shape[1], w2.shape[2], topk, n_groups, topk_groups,
           routed_scaling_factor, "bfloat16" if x.dtype == torch.bfloat16 else "float16")
    kernels = _KERNELS.get(key)
    if kernels is None:
        kernels = _build(*key)
        _KERNELS[key] = kernels
    ids = torch.empty((x.shape[0], topk), device=x.device, dtype=torch.int32)
    weights = torch.empty((x.shape[0], topk), device=x.device, dtype=torch.float32)
    hidden = torch.empty((x.shape[0], topk, w2.shape[2]), device=x.device, dtype=torch.float32)
    kernels[0](router_logits, routing_bias, ids, weights)
    kernels[1](x, w1, ids, hidden)
    kernels[2](hidden, w2, ids, weights, out)
    return out
