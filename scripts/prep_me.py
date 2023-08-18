import json
import sys
from pathlib import Path

import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from lit_gpt.tokenizer import Tokenizer

DATA_DIR = "data"
DATA_FILE_NAME = "obsidianconcat.txt"
CHECKPOINT_DIR = "checkpoints/tiiuae/falcon-7b"
SEED = 42

def prepare(
    data_dir: Path = Path(DATA_DIR),
    checkpoint_dir: Path = Path(CHECKPOINT_DIR),
    test_split_size: int = 2000,
    max_length: int = -1,
    seed: int = SEED,
    mask_inputs: bool = False, # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME,
) -> None:
    """Prepare the Connor dataset for instruction tuning.
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / data_file_name
    tokenizer = Tokenizer(checkpoint_dir)
    with open(file_path, "r") as file:
        data = file.readlines()
    
    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, lengths=(train_split_size, test_split_size), generator=torch.Generator().manual_seed(seed)
        )
    train_set, test_set = list(train_set), list(test_set)
    
    # Now save motherfucker
    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")
    print("Processing train split ...")
    train_set = [tokenizer.encode(sample, max_length=max_length) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")
    print("Processing test split ...")
    test_set = [tokenizer.encode(sample, max_length=max_length) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
