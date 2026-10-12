import json
import torch
from torch import Tensor

def text_to_tensor(text: str)->Tensor:
    with open("map.json", "r", encoding="utf-8") as file:
        map = json.load(file)
        char_to_idx = {char: idx for idx, char in enumerate(map)}
        unk = char_to_idx.get('<UNK>', 0)
        indexes = [char_to_idx.get(char, unk) for char in text]
        return torch.tensor(indexes, dtype=torch.long)