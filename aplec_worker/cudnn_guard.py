"""The cuDNN settings every short-clip model in the worker runs under.

Every clip has its own length, and what cuDNN does with a new length depends on
two process-wide switches other libraries set. wespeaker's load turns benchmark
mode on: a search per length (10 clips took 103 s against 3 s, one run peaked at
58 GB). pyannote turns TF32 off, and without the search the default algorithm
for some lengths then wants over 100 GB at once: a single 1.8 s clip exhausted
the host's memory, and the kernel killed the worker with it. With TF32 on and no
search: 1.2 GB, 0.1 s, the same embeddings.

The language-ID model ran outside this guard and spiked the same way: on a
56 min meeting the stage peaked at 22.8 GB, in 1-2 s bursts while ECAPA scored
spans, against 8.3 GB with it, the same language decisions and the same
transcript size.
"""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def cudnn_for_clips():
    """Run the block with cuDNN's benchmark off and TF32 on, then put back
    whatever the process had, also after an exception."""
    import torch

    cudnn = torch.backends.cudnn
    saved = cudnn.benchmark, cudnn.allow_tf32
    cudnn.benchmark, cudnn.allow_tf32 = False, True
    try:
        yield
    finally:
        cudnn.benchmark, cudnn.allow_tf32 = saved
