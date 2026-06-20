import torch
import torch.nn as nn


class FireTransformer(nn.Module):
    """
    Transformer que recebe a vizinhança de Moore completa (8 vizinhos +
    célula central = 9 "tokens") e usa self-attention para relacionar
    a célula central com seu entorno espacial.

    Entrada esperada: (batch, seq_len=9, input_dim=8)
      posições 0-7 = vizinhos na ordem fixa (N, NE, E, SE, S, SW, W, NW)
      posição 8    = célula central

    Isso corrige a limitação da versão anterior, em que a sequência
    tinha comprimento 1 (só a célula central) e o TransformerEncoder
    não tinha nada de fato para relacionar via atenção — funcionando,
    na prática, como uma MLP.
    """

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

        # Embedding posicional aprendido: diferencia "vizinho norte" de
        # "vizinho sul" etc., já que a ordem da sequência é fixa e
        # carrega significado espacial (não é uma sequência temporal
        # arbitrária).
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
        """
        x: (batch, seq_len, input_dim)
        """

        x = self.embedding(x)            # (batch, seq_len, d_model)
        x = x + self.pos_embedding        # injeta posição espacial

        x = self.transformer(x)           # self-attention entre os 9 tokens

        # A predição é sobre a célula central — pega só o token
        # correspondente a ela (última posição, índice seq_len-1)
        # após o self-attention já ter misturado informação dos
        # vizinhos nela.
        center_token = x[:, -1, :]        # (batch, d_model)

        out = self.classifier(center_token)

        return out