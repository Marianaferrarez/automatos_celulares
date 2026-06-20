import numpy as np

from data.neighborhood import extract_neighborhood_sequence

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

EMPTY = 0          # usado como estado NONFLAMMABLE (Regra 2 do artigo)
TREE = 1
BURNING = 2
BURNED = 3
NONFLAMMABLE = EMPTY


class ForestCA:

    def __init__(
        self,
        environment,
        rows=100,
        cols=100,
        model=None,
        scaler=None,
        alpha=0.3,
        threshold=0.20,   # calibrado para Pt = RA*Ft*Wt ~ 0.29 médio
    ):
        self.rows = rows
        self.cols = cols
        self.environment = environment
        self.model = model
        self.scaler = scaler
        self.alpha = alpha
        self.threshold = threshold

        self.grid = np.full((rows, cols), TREE, dtype=int)

        # Regra 2 do artigo: células sem material combustível (água,
        # rocha, clareiras) começam e permanecem NONFLAMMABLE.
        flammable_mask = self.environment.get("flammable_mask")
        if flammable_mask is not None:
            self.grid[~flammable_mask] = NONFLAMMABLE

        center_i = rows // 2
        center_j = cols // 2

        # Garante que o foco de ignição e sua vizinhança imediata
        # sejam inflamáveis, mesmo que a máscara (por azar do sorteio)
        # tenha isolado essa região. Sem isso, o incêndio poderia
        # nascer cercado de NONFLAMMABLE e apagar no primeiro passo.
        for ni, nj in self.neighbors(center_i, center_j):
            if self.grid[ni, nj] == NONFLAMMABLE:
                self.grid[ni, nj] = TREE

        self.grid[center_i, center_j] = BURNING

    # ------------------------------------------------------------------
    # Vizinhança Moore
    # ------------------------------------------------------------------

    def neighbors(self, i, j):
        neigh = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    neigh.append((ni, nj))
        return neigh

    # ------------------------------------------------------------------
    # Fator de aleatoriedade RA  (eq. 6 — White & Engelen, 1994)
    # RA = 1 + (−ln γ)^α,   γ ∈ U(0, 1)
    # ------------------------------------------------------------------

    def _random_factor(self):
        gamma = np.random.uniform(1e-4, 1.0)
        return 1.0 + (-np.log(gamma)) ** self.alpha

    # ------------------------------------------------------------------
    # Efeito direcional do vento Wt  (eq. 5 do artigo)
    # ------------------------------------------------------------------

    def _wind_factor(self, i, j, burning_i, burning_j):
        wind_speed     = self.environment["wind_speed"][i, j]
        wind_direction = self.environment["wind_direction"][i, j]

        spread_di = i - burning_i
        spread_dj = j - burning_j

        angle = (np.degrees(np.arctan2(spread_di, spread_dj)) + 360) % 360
        diff  = min(abs(angle - wind_direction), 360 - abs(angle - wind_direction))

        # cos ∈ [-1, 1]: vento favorável → positivo; contrário → negativo
        wind_alignment = np.cos(np.radians(diff))

        # normaliza pela velocidade máxima; usa 0.5 como neutro para
        # não zerar Wt quando o vento sopra perpendicular
        wind_factor = 0.5 + 0.5 * (wind_speed / 25.0) * wind_alignment

        return float(np.clip(wind_factor, 0.1, 1.0))

    # ------------------------------------------------------------------
    # Probabilidade local Ft(i,j) — Transformer ou fórmula analítica
    # ------------------------------------------------------------------

    def _local_prob_analytical(self, i, j):
        """Fórmula analítica de fallback."""
        temperature = self.environment["temperature"][i, j]
        humidity    = self.environment["humidity"][i, j]
        vegetation  = self.environment["vegetation"][i, j]
        elevation   = self.environment["elevation"][i, j]
        slope       = self.environment["slope"][i, j]

        temp_factor      = (temperature - 15.0) / 30.0
        humidity_factor  = 1.0 - (humidity - 10.0) / 90.0
        vegetation_factor = float(vegetation)
        slope_factor      = slope / 45.0
        elevation_factor  = elevation / 2000.0

        score = (
            0.25 * temp_factor
            + 0.25 * humidity_factor
            + 0.30 * vegetation_factor
            + 0.15 * slope_factor
            + 0.05 * elevation_factor
        )
        return float(np.clip(score, 0.0, 1.0))

    def _local_prob_transformer(self, i, j):
        """
        Usa o Transformer treinado para obter Ft(i,j) = P(próx. estado = BURNING).

        A entrada é a vizinhança de Moore completa da célula (i,j) —
        8 vizinhos + célula central, montada exatamente na mesma ordem
        usada no treino (ver data/neighborhood.py). Isso é o que torna
        o self-attention do Transformer útil de fato: ele relaciona a
        célula central com seu entorno espacial, em vez de receber só
        um vetor de 8 atributos isolados.

        O scaler deve ser o objeto salvo em saved_models/scaler.pkl,
        ajustado sobre as mesmas 8 features (agregando as 9 posições).
        """
        if not _TORCH_AVAILABLE:
            return self._local_prob_analytical(i, j)

        sequence = extract_neighborhood_sequence(
            self.environment, self.grid, i, j
        )   # (seq_len=9, n_features=8)

        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)

            # Proteção defensiva: se alguma feature normalizada sair
            # muito fora da faixa que o scaler "viu" no treino (ex:
            # uma variável era praticamente constante no dataset de
            # treino e o ambiente de inferência trouxe um valor bem
            # diferente), o z-score pode saturar a rede e produzir
            # previsões inúteis. Aqui limitamos a ±5 desvios-padrão,
            # o que ainda cobre outliers genuínos sem deixar o valor
            # explodir.
            sequence = np.clip(sequence, -5.0, 5.0)

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        # x: (1, seq_len, n_features) — batch de 1 amostra

        with torch.no_grad():
            logits = self.model(x)             # (1, 3)
            probs  = torch.softmax(logits, dim=1)

        # índice 1 = BURNING  (mapeamento: next_state - 1)
        return float(probs[0, 1].item())

    def _local_prob(self, i, j):
        if self.model is not None:
            return self._local_prob_transformer(i, j)
        return self._local_prob_analytical(i, j)

    # ------------------------------------------------------------------
    # Passo do CA  (eqs. 1, 6, 7 do artigo)
    # ------------------------------------------------------------------

    def step(self):
        new_grid = self.grid.copy()

        for i in range(self.rows):
            for j in range(self.cols):

                state = self.grid[i, j]

                # Regra 2: célula sem material combustível permanece
                # NONFLAMMABLE para sempre (não participa da dinâmica).
                if state == NONFLAMMABLE:
                    continue

                # Regra 1: queimando → queimado
                elif state == BURNING:
                    new_grid[i, j] = BURNED

                # Regra 4: árvore com vizinho queimando → calcula prob
                elif state == TREE:

                    burning_neighbors = [
                        (ni, nj)
                        for ni, nj in self.neighbors(i, j)
                        if self.grid[ni, nj] == BURNING
                    ]

                    if not burning_neighbors:
                        continue

                    # Ft(i,j) — probabilidade local via Transformer (ou analítica)
                    ft = self._local_prob(i, j)

                    # Wt(i,j) — maior influência de vento entre vizinhos queimando
                    wt = max(
                        self._wind_factor(i, j, ni, nj)
                        for ni, nj in burning_neighbors
                    )

                    # RA — fator de aleatoriedade (eq. 6)
                    ra = self._random_factor()

                    # Pt(i,j) = RA × Ft × Wt  (eq. 7)
                    pt = float(np.clip(ra * ft * wt, 0.0, 1.0))

                    # Eq. 1: ignição se Pt >= threshold
                    if pt >= self.threshold:
                        new_grid[i, j] = BURNING

        self.grid = new_grid
