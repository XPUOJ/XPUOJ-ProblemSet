---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
import torch

def routing(router_logits, routing_bias, topk, n_groups, topk_groups, scale):
    M, E = router_logits.shape
    scores = torch.sigmoid(router_logits)
    choice_scores = scores + routing_bias.view(1, E)
    grouped = choice_scores.view(M, n_groups, E // n_groups)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(topk_groups, dim=-1).indices
    mask = torch.zeros((M, n_groups), device=scores.device, dtype=torch.bool)
    mask.scatter_(1, selected_groups, True)
    expert_mask = mask[:, :, None].expand(M, n_groups, E // n_groups).reshape(M, E)
    ids = choice_scores.masked_fill(~expert_mask, float("-inf")).topk(topk, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True) * scale
    return ids, weights

def baseline(x, w1, w2, router_logits, routing_bias, out, topk, n_groups, topk_groups, scale):
    ids, weights = routing(router_logits, routing_bias, topk, n_groups, topk_groups, scale)
    fp32 = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for m in range(x.shape[0]):
        for t in range(topk):
            e = int(ids[m, t])
            hidden = x[m].float() @ w1[e].float().T
            gate, up = hidden.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_out = activated @ w2[e].float().T
            fp32[m] += weights[m, t].float() * expert_out
    out.copy_(fp32.to(x.dtype))
```

以上代码用于说明计算语义。参赛实现的核心计算须使用 TileLang-MUSA；性能成绩按“评分规则”计算。
