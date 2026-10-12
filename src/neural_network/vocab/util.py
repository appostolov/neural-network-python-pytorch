import json
import torch
from torch import Tensor
from pathlib import Path

def text_to_tensor(text: str, max: int)->Tensor:
    text = text[:max]
    dir = Path(__file__).parent
    path = dir / "map.json"
    with open(path, "r", encoding="utf-8") as file:
        map = json.load(file)
        char_to_idx = {char: idx for idx, char in enumerate(map)}
        unk = char_to_idx.get('<UNK>', 0)
        indexes = [char_to_idx.get(char, unk) for char in text]
        if len(indexes) < max:
            pad = char_to_idx.get('<PAD>', 1)
            indexes += [pad] * (max - len(indexes))
        return torch.tensor(indexes, dtype=torch.long)