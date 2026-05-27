---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(q, k, v, block_indices, output, B, seq_len, H, HQ, D, S, block_size, is_causal):
    G = HQ // H
    scale = D ** -0.5
    offsets = torch.arange(block_size, device=q.device, dtype=torch.long)
    query_pos = torch.arange(seq_len, device=q.device, dtype=torch.long)[:, None]
    out = torch.empty_like(output)
    for b in range(B):
        for h in range(H):
            idx = block_indices[b, :, h, :].long().unsqueeze(-1) * block_size + offsets
            idx = idx.reshape(seq_len, S * block_size)
            valid = (idx >= 0) & (idx < seq_len)
            if int(is_causal):
                valid = valid & (idx <= query_pos)
            idx_clamped = idx.clamp(0, seq_len - 1)
            k_sel = k[b, :, h, :].float()[idx_clamped]
            v_sel = v[b, :, h, :].float()[idx_clamped]
            q_group = q[b, :, h * G : (h + 1) * G, :].float()
            scores = torch.einsum("tgd,tld->tgl", q_group, k_sel) * scale
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out_group = torch.einsum("tgl,tld->tgd", attn, v_sel)
            out[b, :, h * G : (h + 1) * G, :].copy_(out_group.to(out.dtype))
    output.copy_(out)
```
