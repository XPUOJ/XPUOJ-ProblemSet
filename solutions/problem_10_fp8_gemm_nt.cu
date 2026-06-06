#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int BLOCK_K = 16;
constexpr int SCALE_BLOCK = 128;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int WARPS_M = BLOCK_M / WMMA_M;
constexpr int WARP_TILE_N = 32;
constexpr int WARPS_N = BLOCK_N / WARP_TILE_N;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 x) {
    return static_cast<float>(x);
}

template <bool CHECK_BOUNDS>
__global__ void fp8_gemm_nt_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const float* __restrict__ A_scale,
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ B_scale,
    __nv_bfloat16* __restrict__ D,
    int64_t M,
    int64_t K,
    int64_t N
) {
    const int tile_m = blockIdx.y * BLOCK_M;
    const int tile_n = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id - warp_m * WARPS_N;
    const int warp_n0 = warp_n * WARP_TILE_N;
    const int K_blocks = static_cast<int>((K + SCALE_BLOCK - 1) / SCALE_BLOCK);

    __shared__ half As[BLOCK_M * BLOCK_K];
    __shared__ half Bs[BLOCK_K * BLOCK_N];
    __shared__ float Cs[BLOCK_M * BLOCK_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag0;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag0;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag1;

    wmma::fill_fragment(acc_frag0, 0.0f);
    wmma::fill_fragment(acc_frag1, 0.0f);

    for (int64_t kk = 0; kk < K; kk += BLOCK_K) {
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += blockDim.x) {
            const int mi = idx / BLOCK_K;
            const int ki = idx % BLOCK_K;
            const int64_t row = tile_m + mi;
            const int64_t k = kk + ki;
            if constexpr (CHECK_BOUNDS) {
                float value = 0.0f;
                if (row < M && k < K) {
                    const int64_t k_block = k / SCALE_BLOCK;
                    value = fp8_to_float(A[row * K + k]) * A_scale[row * K_blocks + k_block];
                }
                As[idx] = __float2half_rn(value);
            } else {
                const int64_t k_block = k / SCALE_BLOCK;
                const float value = fp8_to_float(A[row * K + k]) * A_scale[row * K_blocks + k_block];
                As[idx] = __float2half_rn(value);
            }
        }

        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += blockDim.x) {
            const int ki = idx / BLOCK_N;
            const int nj = idx % BLOCK_N;
            const int64_t k = kk + ki;
            const int64_t col = tile_n + nj;
            if constexpr (CHECK_BOUNDS) {
                float value = 0.0f;
                if (col < N && k < K) {
                    const int64_t n_block = col / SCALE_BLOCK;
                    const int64_t k_block = k / SCALE_BLOCK;
                    value = fp8_to_float(B[col * K + k]) * B_scale[n_block * K_blocks + k_block];
                }
                Bs[idx] = __float2half_rn(value);
            } else {
                const int64_t n_block = col / SCALE_BLOCK;
                const int64_t k_block = k / SCALE_BLOCK;
                const float value = fp8_to_float(B[col * K + k]) * B_scale[n_block * K_blocks + k_block];
                Bs[idx] = __float2half_rn(value);
            }
        }

        __syncthreads();
        if (warp_id < WARPS_PER_BLOCK) {
            #pragma unroll
            for (int ko = 0; ko < BLOCK_K; ko += WMMA_K) {
                const half* a_ptr = As + (warp_m * WMMA_M) * BLOCK_K + ko;
                const half* b_ptr0 = Bs + ko * BLOCK_N + warp_n0;
                const half* b_ptr1 = b_ptr0 + WMMA_N;
                wmma::load_matrix_sync(a_frag, a_ptr, BLOCK_K);
                wmma::load_matrix_sync(b_frag0, b_ptr0, BLOCK_N);
                wmma::load_matrix_sync(b_frag1, b_ptr1, BLOCK_N);
                wmma::mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
                wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
            }
        }
        __syncthreads();
    }

    if (warp_id < WARPS_PER_BLOCK) {
        float* c_ptr0 = Cs + (warp_m * WMMA_M) * BLOCK_N + warp_n0;
        float* c_ptr1 = c_ptr0 + WMMA_N;
        wmma::store_matrix_sync(c_ptr0, acc_frag0, BLOCK_N, wmma::mem_row_major);
        wmma::store_matrix_sync(c_ptr1, acc_frag1, BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
        const int mi = idx / BLOCK_N;
        const int nj = idx % BLOCK_N;
        const int64_t row = tile_m + mi;
        const int64_t col = tile_n + nj;
        if constexpr (CHECK_BOUNDS) {
            if (row < M && col < N) {
                D[row * N + col] = __float2bfloat16(Cs[idx]);
            }
        } else {
            D[row * N + col] = __float2bfloat16(Cs[idx]);
        }
    }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_fp8_e4m3* A,
    const float* A_scale,
    const __nv_fp8_e4m3* B,
    const float* B_scale,
    __nv_bfloat16* D,
    int64_t M,
    int64_t K,
    int64_t N
) {
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid(
        static_cast<unsigned int>((N + BLOCK_N - 1) / BLOCK_N),
        static_cast<unsigned int>((M + BLOCK_M - 1) / BLOCK_M)
    );
    if ((M % BLOCK_M == 0) && (N % BLOCK_N == 0) && (K % BLOCK_K == 0)) {
        fp8_gemm_nt_kernel<false><<<grid, block>>>(A, A_scale, B, B_scale, D, M, K, N);
    } else {
        fp8_gemm_nt_kernel<true><<<grid, block>>>(A, A_scale, B, B_scale, D, M, K, N);
    }
}
