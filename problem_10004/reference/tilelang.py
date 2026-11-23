"""TileLang correctness baseline for the draft interface; not performance tuned."""
import tilelang
import tilelang.language as T

_KERNELS = {}


def _build(B, S, STRIDE, K, PAGES, PAGE_SIZE, MODE):
    @T.prim_func
    def initialize(Out: T.Tensor((B, K), "int32")):
        with T.Kernel(T.ceildiv(B * K, 128), threads=128) as block:
            for i in T.Parallel(128):
                idx = block * 128 + i
                if idx < B * K:
                    Out[idx // K, idx % K] = -1

    @T.prim_func
    def select(
        Scores: T.StridedTensor((B, S), (STRIDE, 1), "float32"),
        Lengths: T.Tensor((B,), "int32"),
        Pages: T.Tensor((B, PAGES), "int32"),
        Offsets: T.Tensor((B,), "int32"),
        Starts: T.Tensor((B,), "int32"),
        Out: T.Tensor((B, K), "int32"),
    ):
        with T.Kernel(T.ceildiv(S, 128), B, threads=128) as (block, row):
            rank = T.alloc_local((1,), "int32")
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < Lengths[row]:
                    rank[0] = 0
                    if Lengths[row] > K:
                        for j in T.serial(Lengths[row]):
                            if (Scores[row, Starts[row] + j] > Scores[row, Starts[row] + i]) or (
                                (Scores[row, Starts[row] + j] == Scores[row, Starts[row] + i]) and (j < i)
                            ):
                                rank[0] += 1
                    else:
                        rank[0] = i
                    if rank[0] < K:
                        if MODE == 0:
                            Out[row, rank[0]] = Pages[row, i // PAGE_SIZE] * PAGE_SIZE + i % PAGE_SIZE
                        else:
                            Out[row, rank[0]] = i + Offsets[row]

    options = {"target": {"kind": "musa", "arch": "mp_31"}, "execution_backend": "cython"}
    return tilelang.compile(initialize, **options), tilelang.compile(select, **options)


def run_kernel(scores, lengths, page_table, offsets, row_starts, out, topk, page_size, mode):
    key = (scores.shape[0], scores.shape[1], scores.stride(0), topk, page_table.shape[1], page_size, mode)
    kernels = _KERNELS.get(key)
    if kernels is None:
        kernels = _build(*key)
        _KERNELS[key] = kernels
    kernels[0](out)
    kernels[1](scores, lengths, page_table, offsets, row_starts, out)
    return out
