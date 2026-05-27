---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    group_num = heads // kv_heads
    q_main = q.reshape(batch, kv_heads, group_num, dim).permute(0, 2, 1, 3).float()
    q_pos = q_pe.reshape(batch, kv_heads, group_num, pe_dim).permute(0, 2, 1, 3).float()
    kv_main = kv.permute(0, 2, 1, 3).float()
    k_pos = k_pe.permute(0, 2, 1, 3).float()
    query = torch.cat([q_main, q_pos], dim=-1)
    key = torch.cat([kv_main, k_pos], dim=-1)
    scores = torch.einsum("bghd,bhsd->bghs", query, key)
    attention = torch.softmax(scores * ((dim + pe_dim) ** -0.5), dim=-1)
    out = torch.einsum("bghs,bhsd->bghd", attention, kv_main)
    output.copy_(out.permute(0, 2, 1, 3).reshape(batch, heads, dim).to(output.dtype))
```
