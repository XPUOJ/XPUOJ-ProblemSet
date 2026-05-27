---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(expert_tokens, routed_expert_gate, routed_expert_up, routed_expert_down,
             routed_expert_weights, group_sizes, group_offsets, group_padded_offsets,
             group_idx_for_bx, up_logits, output, group_sum, d_hidden, d_expert,
             n_routed_experts, block_token):
    output.zero_()
    for expert_id in range(int(n_routed_experts)):
        count = int(group_sizes[expert_id].item())
        if count == 0:
            continue
        start = int(group_offsets[expert_id].item())
        end = start + count
        x = expert_tokens[start:end].float()
        gate = torch.nn.functional.silu(x @ routed_expert_gate[expert_id].float().t())
        up = x @ routed_expert_up[expert_id].float().t()
        hidden = (gate * up).to(torch.float16)
        up_logits[start:end].copy_(hidden)
        out = hidden.float() @ routed_expert_down[expert_id].float().t()
        out = out * routed_expert_weights[start:end].float().unsqueeze(1)
        output[start:end].copy_(out.to(output.dtype))
```
