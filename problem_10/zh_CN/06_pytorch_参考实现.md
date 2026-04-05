---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A, A_scale, B, B_scale, D, M, K, N):
    """
    PyTorch 参考实现：FP8 GEMM with 1D-2D scaling
    
    参数说明：
    * A: 输入张量（FP8E4M3），shape (M, K)，行优先连续存储，只读
    * A_scale: 输入张量（float32），shape (M, ceil(K/128))，A 矩阵的 1D 缩放因子
    * B: 输入张量（FP8E4M3），shape (N, K)，行优先连续存储，只读
    * B_scale: 输入张量（float32），shape (ceil(N/128), ceil(K/128))，B 矩阵的 2D 缩放因子
    * D: 输出张量（bfloat16），shape (M, N)，行优先连续存储，需写入结果
    * M: A 的第一维大小（int64）
    * K: A 的第二维和 B 的第二维大小（int64）
    * N: B 的第一维大小（int64）
    """
    BLOCK_SIZE = 128
    K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Convert FP8 to FP32
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    
    # Compute block sizes for K dimension
    k_block_sizes = [BLOCK_SIZE] * (K_blocks - 1) + [(K % BLOCK_SIZE) or BLOCK_SIZE]
    k_repeats = torch.tensor(k_block_sizes, device=A.device)
    
    # Expand A_scale to (M, K)
    expanded_A_scale = torch.repeat_interleave(A_scale, k_repeats, dim=1)
    A_scaled = A_fp32 * expanded_A_scale
    
    # Compute block sizes for N dimension
    n_block_sizes = [BLOCK_SIZE] * (N_blocks - 1) + [(N % BLOCK_SIZE) or BLOCK_SIZE]
    n_repeats = torch.tensor(n_block_sizes, device=A.device)
    
    # Expand B_scale: first along K (dim=1), then along N (dim=0)
    expanded_B_scale_k = torch.repeat_interleave(B_scale, k_repeats, dim=1)
    expanded_B_scale = torch.repeat_interleave(expanded_B_scale_k, n_repeats, dim=0)
    B_scaled = B_fp32 * expanded_B_scale
    
    # Compute GEMM: D = A_scaled @ B_scaled.T
    D_result = torch.matmul(A_scaled, B_scaled.T)
    D.copy_(D_result.to(torch.bfloat16))
```
