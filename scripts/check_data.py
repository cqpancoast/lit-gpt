from pathlib import Path
import torch
from random import choice
from lit_gpt.tokenizer import Tokenizer

# loads data
file = "data/test.pt"
data = torch.load(file)
num_samples = 10

# randomly select samples
idxs = [choice(range(len(data))) for _ in range(num_samples)]

# decode and print
checkpoint_dir = Path("checkpoints/tiiuae/falcon-7b")
tokenizer = Tokenizer(checkpoint_dir)
for idx in idxs:
    text = tokenizer.decode(data[idx])
    print(text)
