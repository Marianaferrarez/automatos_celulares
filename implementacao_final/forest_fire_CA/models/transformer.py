import torch
import torch.nn as nn

class FireTransformer(nn.Module):
    def __init__(
        self,
        input_dim=8,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_classes=3,
        seq_len=9,
    ):

        super().__init__()

        self.seq_len = seq_len

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding

        x = self.transformer(x)

        center_token = x[:, -1, :]

        out = self.classifier(center_token)

        return out