"""Pick the best available accelerator so the same code runs on the Mac and the desktop."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    # Explicit choice in the config wins ("cuda", "mps" or "cpu").
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():          # NVIDIA GPU (desktop)
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU (MacBook)
        return torch.device("mps")
    return torch.device("cpu")
