import torch
import torch.nn as nn


class FireTransformer(nn.Module):

    def __init__(
        self,
        input_dim=8,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_classes=3
    ):

        super().__init__()

        self.embedding = nn.Linear(
            input_dim,
            d_model
        )

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

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        x = self.classifier(x)

        return x