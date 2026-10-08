"""Dataset + DataLoader over the tokenized .bin files.

Same idea as study/datasetv1.py (GPTDatasetV1): slide a window of `max_length`
tokens over the corpus with a given `stride`; the target is the input shifted by
one token. The difference: GPTDatasetV1 tokenizes everything and stores every
window as a tensor up front, which doesn't scale past a few MB of text. Here
windows are cut on demand from a memory-mapped file, so memory use stays flat
no matter how big the corpus gets.

Quick check (prints shapes and decodes one sample):
    python -m llmbr.data.loader
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, bin_path: str, max_length: int, stride: int):
        self.bin_path = bin_path
        self.max_length = max_length
        self.stride = stride
        n_tokens = len(np.memmap(bin_path, dtype=np.uint16, mode="r"))
        # Each window needs max_length + 1 tokens (input plus the one-step-ahead target).
        self.n_windows = max(0, (n_tokens - max_length - 1) // stride + 1)
        # Opened lazily in each worker: pickling a memmap to send it to DataLoader
        # workers would copy the whole array.
        self._data = None

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        start = idx * self.stride
        # uint16 -> int64: embedding layers and cross_entropy want long indices.
        chunk = torch.from_numpy(
            self._data[start:start + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def create_dataloader(bin_path: str, batch_size: int = 32, max_length: int = 256,
                      stride: int | None = None, shuffle: bool = True,
                      drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """`stride` defaults to max_length: windows touch but don't overlap, so each
    token is seen once per epoch. A smaller stride gives more (overlapping) windows."""
    dataset = GPTDataset(bin_path, max_length, stride or max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def main() -> None:
    from llmbr.config import load_config
    from llmbr.tokenizer import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.toml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = cfg["model"]["context_length"]
    B = cfg["train"]["batch_size"]
    tok = load_tokenizer(cfg)

    for split in ("train", "val"):
        path = f"{cfg['paths']['processed_dir']}/{split}.bin"
        dl = create_dataloader(path, batch_size=B, max_length=T,
                               shuffle=(split == "train"), drop_last=(split == "train"))
        print(f"{split}: {len(dl.dataset):,} windows of {T} tokens -> {len(dl):,} batches of {B}")

    x, y = next(iter(dl))
    print(f"\nbatch shapes: x={tuple(x.shape)} y={tuple(y.shape)} dtype={x.dtype}")
    assert torch.equal(x[0, 1:], y[0, :-1]), "target must be input shifted by one"
    print("x[0] decoded:\n ", tok.decode(x[0, :60].tolist()), "...")
    print("y[0] decoded (shifted by one token):\n ", tok.decode(y[0, :60].tolist()), "...")


if __name__ == "__main__":
    main()
