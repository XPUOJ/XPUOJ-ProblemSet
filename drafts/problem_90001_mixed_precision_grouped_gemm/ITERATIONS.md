# Iterations

## Baseline

- Source: copied `optimized_warp_row.cu` into `solution/kernel.cu`.
- Correctness: PASS on all 7 cases.
- Runtime ms: case1 30.0115, case2 49.8523, case3 43.1511, case4 146.8872, case5 46.1524, case6 26.7491, case7 44.1982.
- Notes: one warp computes one row and 64 output columns; A and A_scale are broadcast within the warp; each packed INT4 byte is consumed for two K values.
