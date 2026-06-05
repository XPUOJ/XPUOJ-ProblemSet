# Iterations

## Baseline

- Source: copied `optimized_warp_row.cu` into `solution/kernel.cu`.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 30.0115, case2 49.8523, case3 43.1511, case4 146.8872, case5 46.1524, case6 26.7491, case7 44.1982.
- Notes: one warp computes one row and 64 output columns; A and A_scale are broadcast within the warp; each packed INT4 byte is consumed for two K values.

## Iteration 1

- Change: tried a warp-dot mapping where one warp computes one `(row, col)` output and lanes split the K dimension with a warp reduction.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 42.8951, case2 68.1484, case3 60.2290, case4 143.0584, case5 76.7946, case6 44.6753, case7 55.9514.
- Result: mostly slower than baseline; only case4 improved slightly. The higher warp count and reduction overhead outweighed the more contiguous K access.

## Iteration 2

- Change: attempted a shared-A row kernel where one block handles one row and 256 output columns, caching each 64-wide K chunk of A in shared memory for reuse across 4 warps.
- Correctness: not reached.
- Runtime ms: not available.
- Result: compile failed because the CUDA device code used `min<int64_t>(...)` in a form nvcc rejected. Need to fix the chunk-size expression before evaluating the idea.

## Iteration 3

- Change: fixed the shared-A row kernel compile issue and corrected per-warp group broadcast.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 33.7872, case2 59.9280, case3 49.6861, case4 210.3798, case5 64.9992, case6 32.6787, case7 49.4587.
- Result: slower than baseline. The shared-memory staging and `__syncthreads()` overhead outweighed reduced A loads.

## Iteration 4

- Change: restored the warp-row baseline and added a `group_k=64, K%64==0` fast path that removes tail checks and uses shift-based block indexing.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 31.8344, case2 54.4660, case3 44.9280, case4 212.5309, case5 53.6488, case6 29.4772, case7 47.9669.
- Result: slower than baseline. The specialized path likely increased unrolling/register pressure enough to lose more than it saved.

## Iteration 5

- Change: restored the best warp-row implementation and changed `WARPS_PER_BLOCK` from 4 to 8.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 30.9379, case2 49.9755, case3 42.4748, case4 123.1606, case5 41.0047, case6 24.7258, case7 43.7798.
- Result: new best overall. Larger blocks improved throughput on the large/wide cases while keeping small regressions on case1/case2 modest.

## Iteration 6

- Change: increased `WARPS_PER_BLOCK` from 8 to 16.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 29.6545, case2 49.0086, case3 41.9314, case4 113.4264, case5 38.9944, case6 22.8970, case7 42.5578.
- Result: new best overall. A larger block continues to improve occupancy/scheduling efficiency for this warp-row mapping without hurting the smaller cases.

## Iteration 7

- Change: increased `WARPS_PER_BLOCK` from 16 to 32, using the maximum 1024 threads per block.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 29.0188, case2 48.5079, case3 41.3999, case4 112.9782, case5 37.9597, case6 22.7165, case7 42.8612.
- Result: new best overall by geometric mean. Cases 1-6 improved, while the very deep-K case7 regressed slightly.

## Iteration 8

- Change: tested the middle point `WARPS_PER_BLOCK=24`.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 30.0973, case2 50.0225, case3 41.5496, case4 117.2122, case5 41.0749, case6 25.3628, case7 43.6994.
- Result: worse than iter 7. The 32-warp maximum block remains the best tested configuration.

## Final

- Selected: iteration 7.
- Reason: best geometric mean among tested variants. Iteration 8 was worse, so `solution/kernel.cu` was restored from the iter-7 commit and verified again.
- Final verification: PASS on all 7 cases.
- Final runtime ms: case1 29.5521, case2 48.5707, case3 41.4841, case4 112.9508, case5 37.9255, case6 22.6913, case7 42.9745.

## Iteration 9

- Change: converted most kernel-internal indexing and loops from `int64_t` to 32-bit `int`, relying on the problem bounds.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 28.8208, case2 49.0392, case3 42.2279, case4 253.8696, case5 36.7183, case6 21.8295, case7 42.9360.
- Result: rejected. Some cases improved, but case4 regressed badly, so the mixed/address calculation change is not robust as-is.

## Iteration 10

- Change: restored the iter-7 best version and added `__ldg` read-only load hints for A, scales, B, and m_indices.
- Correctness: not reached.
- Runtime ms: not available.
- Result: compile failed because CUDA does not provide an `__ldg` overload for `const __nv_fp8_e4m3*`. Need to keep A as a normal FP8 load or load through a supported byte type.

## Iteration 11

- Change: restored the iter-7 best version and added `__ldg` only for supported read-only types: `m_indices`, `A_scale`, `B_scale`, and `B_packed`; FP8 `A` remains a normal load.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 30.1310, case2 48.5445, case3 41.4943, case4 112.9503, case5 38.0017, case6 22.7096, case7 42.8853.
- Result: very close to iter 7 but not better overall. Read-only cache hints do not provide a reliable win on this kernel.

## Iteration 12

- Change: tested `COLS_PER_WARP=32`, with each thread computing only one output column instead of two.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 33.0216, case2 56.9321, case3 46.8157, case4 122.2676, case5 48.1036, case6 30.4022, case7 46.0662.
- Result: slower than iter 7. Reusing each A value across 64 columns is more valuable than reducing per-thread output/register work.

## Iteration 13

- Change: restored the iter-7 best version, added `cudaMemsetAsync` to zero the whole output, and changed padding rows to return without per-tile zero stores.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 28.7618, case2 48.5171, case3 41.6291, case4 112.6595, case5 37.8021, case6 22.6257, case7 42.9486.
- Result: slight new best candidate by geometric mean. The memset cost is offset by removing scattered padding zero stores, especially on cases with nontrivial padding.

## Iteration 14

- Change: kept the iter-13 memset/padding strategy, but changed the K loop to load `A_scale` once per 128-wide A scale block and process the two 64-wide B scale blocks inside it.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 29.0660, case2 48.3745, case3 41.4074, case4 112.4470, case5 37.4414, case6 22.3569, case7 42.7513.
- Result: new best overall. Reusing `A_scale` across two `group_k=64` chunks improves most cases with only a small case1 regression.

## Iteration 15

- Change: added zero-scale skipping. If an `A_scale` block is zero, the whole 128-wide A block is skipped. If a `B_scale` block is zero for an output column, that column's INT4 unpack and FMA work for the 64-wide B block is skipped.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 21.5874, case2 35.8673, case3 29.8730, case4 84.2227, case5 28.7433, case6 18.5261, case7 31.2960.
- Result: new best by a large margin. Skipping mathematically zero contribution blocks removes substantial unpack and multiply-add work while preserving exact semantics.
