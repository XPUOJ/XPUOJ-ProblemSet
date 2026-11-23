"""S5000 correctness and scale-up cases; published timing is 20 warmups / 50 repeats."""
import torch

DESIGNED_VRAM_SIZE = 80
DESIGNED_TESTDATA_GROUPS = 2
TIMING_WARMUP = 20
TIMING_REPEATS = 50
INPUT_CLASS = ["INPUT"] * 5 + ["OUTPUT"] + ["INPUT"] * 3
# Cases 1-6 check correctness; cases 7-12 carry performance weights.
# score_weight is metadata for OJ aggregation, not consumed by the generic Runner.
CASES = [
    {'name': 'dsv4-paged-boundaries',
     'B': 6,
     'S': 537,
     'topk': 512,
     'page_size': 64,
     'mode': 0,
     'lengths': [0, 1, 511, 512, 513, 529],
     'score_weight': 0.0},
    {'name': 'dsv4-paged-selection',
     'B': 3,
     'S': 1039,
     'topk': 512,
     'page_size': 64,
     'mode': 0,
     'lengths': [1031, 777, 513],
     'score_weight': 0.0},
    {'name': 'nsa-paged',
     'B': 4,
     'S': 2089,
     'topk': 2048,
     'page_size': 1,
     'mode': 0,
     'lengths': [0, 2048, 2049, 2081],
     'score_weight': 0.0},
    {'name': 'nsa-ragged',
     'B': 4,
     'S': 2089,
     'topk': 2048,
     'page_size': 1,
     'mode': 1,
     'lengths': [0, 1, 2048, 2081],
     'score_weight': 0.0},
    {'name': 'tie-membership',
     'B': 2,
     'S': 29,
     'topk': 4,
     'page_size': 4,
     'mode': 0,
     'lengths': [17, 21],
     'ties': True,
     'score_weight': 0.0},
    {'name': 'row-stride-and-offset',
     'B': 3,
     'S': 47,
     'topk': 8,
     'page_size': 1,
     'mode': 1,
     'lengths': [0, 17, 41],
     'row_padding': 11,
     'score_weight': 0.0},
    {'name': 'paged512-16k',
     'B': 1,
     'S': 16387,
     'topk': 512,
     'page_size': 64,
     'mode': 0,
     'lengths': [16384],
     'score_weight': 0.225},
    {'name': 'paged512-b8-32k',
     'B': 8,
     'S': 32779,
     'topk': 512,
     'page_size': 64,
     'mode': 0,
     'lengths': [32768, 32761, 30000, 28672, 24576, 16384, 8192, 512],
     'score_weight': 0.225},
    {'name': 'paged2048-32k',
     'B': 1,
     'S': 32771,
     'topk': 2048,
     'page_size': 1,
     'mode': 0,
     'lengths': [32768],
     'score_weight': 0.175},
    {'name': 'paged2048-b4-64k',
     'B': 4,
     'S': 65543,
     'topk': 2048,
     'page_size': 1,
     'mode': 0,
     'lengths': [65536, 60001, 49152, 32768],
     'score_weight': 0.175},
    {'name': 'ragged2048-64k',
     'B': 1,
     'S': 65539,
     'topk': 2048,
     'page_size': 1,
     'mode': 1,
     'lengths': [65536],
     'score_weight': 0.1},
    {'name': 'ragged2048-b8-32k-strided',
     'B': 8,
     'S': 32779,
     'topk': 2048,
     'page_size': 1,
     'mode': 1,
     'row_padding': 17,
     'lengths': [32768, 32763, 30000, 28672, 24576, 16384, 2048, 0],
     'score_weight': 0.1},
]


def getNumOfTestcases():
    return len(CASES)


def getTestCaseSize():
    case_id = int(input())
    if not 1 <= case_id <= len(CASES):
        raise ValueError("invalid case id")
    return [case_id], (TIMING_WARMUP, TIMING_REPEATS)


def genTestCase(testcase_sizes, device="musa"):
    c = CASES[testcase_sizes[0] - 1]
    B, S, K = c["B"], c["S"], c["topk"]
    # Do not reseed here: the Runner must generate independent input groups.
    scores = torch.randn((B, S + c.get("row_padding", 0)), device=device, dtype=torch.float32)[:, :S]
    if c.get("ties"):
        scores.copy_((torch.arange(S, device=device) % 3).float().expand(B, S))
    lengths = torch.tensor(c["lengths"], dtype=torch.int32, device=device)
    starts = torch.tensor([min(3 + b, S - L) for b, L in enumerate(c["lengths"])],
                          dtype=torch.int32, device=device)
    pages = (max(c["lengths"]) + c["page_size"] - 1) // c["page_size"]
    pages = max(1, pages)
    page_table = torch.stack([torch.randperm(pages, device=device).to(torch.int32) + 17 + b * pages
                              for b in range(B)])
    offsets = torch.tensor([71 + 503 * b for b in range(B)], dtype=torch.int32, device=device)
    out = torch.full((B, K), -123456, dtype=torch.int32, device=device)
    return [scores, lengths, page_table, offsets, starts, out, K, c["page_size"], c["mode"]]


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


def _same_inputs(original, target):
    return all(torch.equal(a, b) if torch.is_tensor(a) else a == b
               for i, (a, b) in enumerate(zip(original, target)) if i != 5)


def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors,
          baseline_input_tensors, rtol=None, atol=None):
    a, t = original_input_tensors, target_kernel_input_tensors
    if not _same_inputs(a, t):
        return False
    scores, lengths, pages, offsets, starts, _, topk, page_size, mode = a
    out = t[5]
    if out.shape != (scores.shape[0], topk) or out.dtype != torch.int32 or out.device != scores.device:
        return False
    if not out.is_contiguous():
        return False
    for b in range(scores.shape[0]):
        L, start = int(lengths[b]), int(starts[b])
        k = min(topk, L)
        # Padding must occupy the suffix, and selected members must be unique.
        if not bool(torch.all(out[b, k:] == -1)):
            return False
        picked = out[b, :k]
        if torch.unique(picked).numel() != k:
            return False
        raw = torch.arange(L, device=scores.device, dtype=torch.int32)
        valid = (pages[b, raw // page_size] * page_size + raw % page_size if mode == 0
                 else raw + offsets[b])
        if k == 0:
            continue
        # Invert the transform instead of requiring torch.topk's arbitrary tie selection.
        ordered, permutation = torch.sort(valid)
        pos = torch.searchsorted(ordered, picked)
        if bool(torch.any(pos >= L)) or not torch.equal(ordered[pos.clamp(max=L - 1)], picked):
            return False
        selected_raw = permutation[pos]
        if L > topk:
            values = scores[b, start:start + L]
            threshold = torch.topk(values, topk, sorted=True).values[-1]
            if bool(torch.any(values[selected_raw] < threshold)):
                return False
            # Include every strictly-better member; threshold ties may vary freely.
            selected = torch.zeros(L, dtype=torch.bool, device=scores.device)
            selected[selected_raw] = True
            if bool(torch.any((values > threshold) & ~selected)):
                return False
    return True


def getWorkload(testcase_sizes):
    c = CASES[testcase_sizes[0] - 1]
    return {"flops": int(sum(c["lengths"])), "memory_bytes": int(c["B"] * c["S"] * 4),
            "dtype": "float32"}
