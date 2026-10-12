import json
import torch
from torch import Tensor

def text_to_tensor(text: str, max: int)->Tensor:
    text = text[:max]
    with open("map.json", "r", encoding="utf-8") as file:
        map = json.load(file)
        char_to_idx = {char: idx for idx, char in enumerate(map)}
        unk = char_to_idx.get('<UNK>', 0)
        indexes = [char_to_idx.get(char, unk) for char in text]
        if len(indexes) < max:
            pad = char_to_idx.get('<PAD>', 1)
            indexes += [pad] * (max - len(indexes))
        return torch.tensor(indexes, dtype=torch.long)