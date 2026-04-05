---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(bits, packed, num_bits):
    """
    PyTorch 参考实现：Packbits
    
    参数说明：
    * bits: 输入张量（bool），shape (num_bits,)，底层按 1 字节布尔值存储，只读
    * packed: 输出张量（uint8），shape (ceil(num_bits / 8),)，需写入结果
    * num_bits: 输入 bit 数量（int64）
    """
    num_packed = (num_bits + 7) // 8
    bits_uint8 = bits.to(torch.uint8)
    if num_bits % 8 != 0:
        pad_size = 8 - (num_bits % 8)
        bits_uint8 = torch.cat([bits_uint8, torch.zeros(pad_size, dtype=torch.uint8, device=bits.device)])
    bits_reshaped = bits_uint8.view(num_packed, 8)
    
    packed_ref = torch.zeros(num_packed, dtype=torch.uint8, device=bits.device)
    for i in range(8):
        packed_ref = packed_ref | (bits_reshaped[:, i] << (7 - i))
    
    packed.copy_(packed_ref)
```
