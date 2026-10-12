import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from .vocab.util import text_to_tensor

class EmailDataset(Dataset):
    def __init__(self, csv: str, vocab: list[str], max: int):

        df = pd.read_csv(csv)

        vocab_dict = {char: idx for idx, char in enumerate(vocab)}

        self.features = []
        for email in df["email"]:
            tensor = text_to_tensor(email, vocab_dict, 254)
            self.features.append(tensor)

        self.features = torch.stack(self.features)

        self.labels = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        
        return self.features[idx], self.labels[idx]
