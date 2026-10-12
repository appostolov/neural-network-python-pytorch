import torch
import torch.nn as nn
from torch import Tensor

class EmailValidator(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int = 64):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=1
        )

        self.conv_block = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim, 
                out_channels=num_filters, 
                kernel_size=5, 
                padding=2
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.classifier = nn.Linear(in_features=num_filters, out_features=1)

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.embedding(x)
        
        x = x.transpose(1, 2)
        
        x = self.conv_block(x)

        x = self.pool(x).squeeze(-1)
        
        return self.classifier(x)

