#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int A_SCALE_BLOCK = 128;
constexpr int COLS_PER_WARP = 64;
constexpr int WARPS_PER_BLOCK = 8;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ int signed_low4(uint8_t packed) {
    int v = int(packed & 0x0F);
    return v >= 8 ? v - 16 : v;
}

__device__ __forceinline__ int signed_high4(uint8_t packed) {
    int v = int((packed >> 4) & 0x0F);
    return v >= 8 ? v - 16 : v;
}

__global__ void warp_row_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const float* __restrict__ A_scale,
    const uint8_t* __restrict__ B_packed,
    const float* __restrict__ B_scale,
    const int32_t* __restrict__ m_indices,
    __nv_bfloat16* __restrict__ D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups,
    int64_t group_k
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t row = int64_t(blockIdx.y) * WARPS_PER_BLOCK + warp_in_block;
    const int64_t col_base = int64_t(blockIdx.x) * COLS_PER_WARP;
    const int64_t col0 = col_base + lane;
    const int64_t col1 = col0 + 32;

    if (row >= M_total) {
        return;
    }

    int group = 0;
    if (lane == 0) {
        group = m_indices[row];
    }
    group = __shfl_sync(FULL_MASK, group, 0);

    if (group < 0 || group >= num_groups) {
        if (col0 < N) {
            D[row * N + col0] = __float2bfloat16(0.0f);
        }
        if (col1 < N) {
            D[row * N + col1] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int64_t a_k_blocks = (K + A_SCALE_BLOCK - 1) / A_SCALE_BLOCK;
    const int64_t b_k_blocks = (K + group_k - 1) / group_k;
    const int64_t k_packed = (K + 1) / 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const bool valid0 = col0 < N;
    const bool valid1 = col1 < N;

    for (int64_t kb = 0; kb < K; kb += group_k) {
        const int64_t kend = min(kb + group_k, K);
        float b_scale0 = 0.0f;
        float b_scale1 = 0.0f;
        if (valid0) {
            b_scale0 = B_scale[(int64_t(group) * N + col0) * b_k_blocks + kb / group_k];
        }
        if (valid1) {
            b_scale1 = B_scale[(int64_t(group) * N + col1) * b_k_blocks + kb / group_k];
        }

        float a_scale_v = 1.0f;
        if (lane == 0) {
            a_scale_v = A_scale[row * a_k_blocks + kb / A_SCALE_BLOCK];
        }
        a_scale_v = __shfl_sync(FULL_MASK, a_scale_v, 0);

        #pragma unroll 4
        for (int64_t k = kb; k < kend; k += 2) {
            float a0 = 0.0f;
            float a1 = 0.0f;
            if (lane == 0) {
                a0 = static_cast<float>(A[row * K + k]);
                if (k + 1 < kend) {
                    a1 = static_cast<float>(A[row * K + k + 1]);
                }
            }
            a0 = __shfl_sync(FULL_MASK, a0, 0) * a_scale_v;
            a1 = __shfl_sync(FULL_MASK, a1, 0) * a_scale_v;

            const int64_t kp = k >> 1;

            if (valid0) {
                const uint8_t packed0 = B_packed[(int64_t(group) * N + col0) * k_packed + kp];
                const int bq00 = signed_low4(packed0);
                const int bq01 = signed_high4(packed0);
                acc0 += a0 * (float(bq00) * b_scale0);
                if (k + 1 < kend) {
                    acc0 += a1 * (float(bq01) * b_scale0);
                }
            }
            if (valid1) {
                const uint8_t packed1 = B_packed[(int64_t(group) * N + col1) * k_packed + kp];
                const int bq10 = signed_low4(packed1);
                const int bq11 = signed_high4(packed1);
                acc1 += a0 * (float(bq10) * b_scale1);
                if (k + 1 < kend) {
                    acc1 += a1 * (float(bq11) * b_scale1);
                }
            }
        }
    }

    if (valid0) {
        D[row * N + col0] = __float2bfloat16(acc0);
    }
    if (valid1) {
        D[row * N + col1] = __float2bfloat16(acc1);
    }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_fp8_e4m3* A,
    const float* A_scale,
    const uint8_t* B_packed,
    const float* B_scale,
    const int32_t* m_indices,
    __nv_bfloat16* D,
    int64_t M_total,
    int64_t K,
    int64_t N,
    int64_t num_groups,
    int64_t group_k
) {
    dim3 block(32 * WARPS_PER_BLOCK);
    dim3 grid(
        static_cast<unsigned int>((N + COLS_PER_WARP - 1) / COLS_PER_WARP),
        static_cast<unsigned int>((M_total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)
    );
    warp_row_kernel<<<grid, block>>>(
        A, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k
    );
}
