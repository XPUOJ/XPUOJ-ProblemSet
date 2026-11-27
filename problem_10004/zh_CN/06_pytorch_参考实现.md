---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
import torch

def baseline(scores, lengths, page_table, offsets, row_starts, out, topk, page_size, mode):
    out.fill_(-1)
    for b in range(scores.shape[0]):
        L, start = int(lengths[b]), int(row_starts[b])
        k = min(topk, L)
        if k == 0:
            continue
        raw = (torch.arange(L, device=scores.device, dtype=torch.int32) if L <= topk else
               torch.topk(scores[b, start:start + L], topk, sorted=False).indices.to(torch.int32))
        transformed = (page_table[b, raw // page_size] * page_size + raw % page_size if mode == 0
                       else raw + offsets[b])
        out[b, :k] = transformed[:k]
```

以上代码用于说明计算语义。参赛实现的核心计算须使用 TileLang-MUSA；性能成绩按“评分规则”计算。
