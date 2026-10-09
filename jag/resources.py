"""Bound an evaluation process's CUDA allocator before model loading."""

import math


def configure_cuda_memory(device, memory_gib=None, memory_fraction=None):
    if memory_gib is not None and memory_fraction is not None:
        raise ValueError("Specify only one CUDA memory limit")
    if memory_gib is None and memory_fraction is None:
        return {"allocator_limit_applied": False}
    value = memory_gib if memory_gib is not None else memory_fraction
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("CUDA memory limit must be a finite positive number")
    if memory_fraction is not None and memory_fraction > 1:
        raise ValueError("CUDA memory fraction must not exceed 1")
    import torch

    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("CUDA memory limits require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested evaluation GPU is unavailable")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = memory_fraction if memory_fraction is not None else memory_gib * 1024**3 / total
    if fraction > 1:
        raise ValueError("Requested allocator limit exceeds device memory")
    free, _ = torch.cuda.mem_get_info(index)
    budget = int(total * fraction)
    # Leave space outside our allocator for CUDA context/library allocations.
    if free < budget + 2 * 1024**3:
        raise RuntimeError("Insufficient free GPU memory for capped evaluation plus 2 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return {"allocator_limit_applied": True, "device": str(selected),
            "allocator_limit_bytes": budget, "allocator_limit_fraction": fraction,
            "free_device_bytes_before_load": free,
            "note": "PyTorch allocator cap; CUDA context/library allocations are additional"}
