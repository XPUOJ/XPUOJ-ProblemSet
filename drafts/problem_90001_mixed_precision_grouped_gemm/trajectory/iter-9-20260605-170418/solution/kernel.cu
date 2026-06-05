#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int A_SCALE_BLOCK = 128;
constexpr int COLS_PER_WARP = 64;
constexpr int WARPS_PER_BLOCK = 32;
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
    const int M = int(M_total);
    const int K_i = int(K);
    const int N_i = int(N);
    const int num_groups_i = int(num_groups);
    const int group_k_i = int(group_k);
    const int row = int(blockIdx.y) * WARPS_PER_BLOCK + warp_in_block;
    const int col_base = int(blockIdx.x) * COLS_PER_WARP;
    const int col0 = col_base + lane;
    const int col1 = col0 + 32;

    if (row >= M) {
        return;
    }

    int group = 0;
    if (lane == 0) {
        group = m_indices[row];
    }
    group = __shfl_sync(FULL_MASK, group, 0);

    if (group < 0 || group >= num_groups_i) {
        if (col0 < N_i) {
            D[row * N_i + col0] = __float2bfloat16(0.0f);
        }
        if (col1 < N_i) {
            D[row * N_i + col1] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int a_k_blocks = (K_i + A_SCALE_BLOCK - 1) / A_SCALE_BLOCK;
    const int b_k_blocks = (K_i + group_k_i - 1) / group_k_i;
    const int k_packed = (K_i + 1) >> 1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const bool valid0 = col0 < N_i;
    const bool valid1 = col1 < N_i;

    for (int kb = 0; kb < K_i; kb += group_k_i) {
        const int kend = min(kb + group_k_i, K_i);
        float b_scale0 = 0.0f;
        float b_scale1 = 0.0f;
        if (valid0) {
            b_scale0 = B_scale[(group * N_i + col0) * b_k_blocks + kb / group_k_i];
        }
        if (valid1) {
            b_scale1 = B_scale[(group * N_i + col1) * b_k_blocks + kb / group_k_i];
        }

        float a_scale_v = 1.0f;
        if (lane == 0) {
            a_scale_v = A_scale[row * a_k_blocks + kb / A_SCALE_BLOCK];
        }
        a_scale_v = __shfl_sync(FULL_MASK, a_scale_v, 0);

        #pragma unroll 4
        for (int k = kb; k < kend; k += 2) {
            float a0 = 0.0f;
            float a1 = 0.0f;
            if (lane == 0) {
                a0 = static_cast<float>(A[row * K_i + k]);
                if (k + 1 < kend) {
                    a1 = static_cast<float>(A[row * K_i + k + 1]);
                }
            }
            a0 = __shfl_sync(FULL_MASK, a0, 0) * a_scale_v;
            a1 = __shfl_sync(FULL_MASK, a1, 0) * a_scale_v;

            const int kp = k >> 1;

            if (valid0) {
                const uint8_t packed0 = B_packed[(group * N_i + col0) * k_packed + kp];
                const int bq00 = signed_low4(packed0);
                const int bq01 = signed_high4(packed0);
                acc0 += a0 * (float(bq00) * b_scale0);
                if (k + 1 < kend) {
                    acc0 += a1 * (float(bq01) * b_scale0);
                }
            }
            if (valid1) {
                const uint8_t packed1 = B_packed[(group * N_i + col1) * k_packed + kp];
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
        D[row * N_i + col0] = __float2bfloat16(acc0);
    }
    if (valid1) {
        D[row * N_i + col1] = __float2bfloat16(acc1);
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
