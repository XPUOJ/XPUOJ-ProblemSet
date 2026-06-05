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
