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
