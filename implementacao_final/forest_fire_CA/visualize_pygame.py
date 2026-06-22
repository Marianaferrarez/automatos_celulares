import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from data.dataset_generator import DatasetGenerator
from models.transformer import FireTransformer
from simulation.automata import ForestCA, NONFLAMMABLE, TREE, BURNING, BURNED

ROWS        = 80
COLS        = 80
THRESHOLD   = 0.20
ALPHA       = 0.3
SAVED_DIR   = ROOT_DIR / "saved_models"

CELL_SIZE   = 8
PANEL_H     = 80
WIN_W       = COLS * CELL_SIZE
WIN_H       = ROWS * CELL_SIZE + PANEL_H

FPS_OPTIONS = [2, 4, 8, 16, 30]
FPS_DEFAULT = 2

COLOR = {
    NONFLAMMABLE: (120, 120, 120),   # cinza
    TREE:         ( 34, 139,  34),   # verde
    BURNING:      (255, 102,   0),   # laranja
    BURNED:       ( 59,  42,  26),   # marrom escuro
    "bg":         ( 15,  15,  15),   # fundo do painel
    "text":       (220, 220, 220),   # texto principal
    "dim":        (140, 140, 140),   # texto secundário
    "accent":     (255, 180,  50),   # destaque (pausa, valores)
}

def load_model():
    model_path  = SAVED_DIR / "fire_transformer.pth"
    scaler_path = SAVED_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print(
            "ERRO: modelo treinado não encontrado em saved_models/.\n"
            "Execute main.py pelo menos uma vez antes deste script."
        )
        sys.exit(1)

    model = FireTransformer(num_classes=3)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

def new_simulation(model, scaler):
    generator   = DatasetGenerator(ROWS, COLS)
    environment = generator.generate_all()
    ca          = ForestCA(
        environment=environment,
        rows=ROWS, cols=COLS,
        model=model, scaler=scaler,
        alpha=ALPHA, threshold=THRESHOLD,
    )
    return ca

def draw_grid(surface, grid, pygame):
    for i in range(ROWS):
        for j in range(COLS):
            color = COLOR[grid[i, j]]
            rect  = (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1)
            pygame.draw.rect(surface, color, rect)

def draw_panel(surface, pygame, font_big, font_sm, step, ca, paused, fps_idx):
    import pygame as pg

    panel_y = ROWS * CELL_SIZE
    pg.draw.rect(surface, COLOR["bg"], (0, panel_y, WIN_W, PANEL_H))

    grid = ca.grid
    n_tree    = int((grid == TREE).sum())
    n_burning = int((grid == BURNING).sum())
    n_burned  = int((grid == BURNED).sum())
    n_nf      = int((grid == NONFLAMMABLE).sum())
    total_flammable = ROWS * COLS - n_nf
    pct_burned = 100 * n_burned / total_flammable if total_flammable > 0 else 0

    status = "  ⏸ PAUSADO" if paused else "  ▶ RODANDO"
    color_status = COLOR["accent"] if paused else COLOR["text"]

    txt = font_big.render(f"Passo {step:3d}{status}", True, color_status)
    surface.blit(txt, (10, panel_y + 8))

    fps_txt = font_sm.render(
        f"Velocidade: {FPS_OPTIONS[fps_idx]} pps   [↑↓ ajustar]",
        True, COLOR["dim"]
    )
    surface.blit(fps_txt, (WIN_W - 220, panel_y + 10))

    items = [
        (COLOR[TREE],         f"Árvore {n_tree}"),
        (COLOR[BURNING],      f"Queimando {n_burning}"),
        (COLOR[BURNED],       f"Queimado {n_burned}"),
        (COLOR[NONFLAMMABLE], f"Inerte {n_nf}"),
    ]

    x = 10
    for color, label in items:
        pg.draw.rect(surface, color, (x, panel_y + 38, 12, 12))
        lbl = font_sm.render(label, True, COLOR["text"])
        surface.blit(lbl, (x + 16, panel_y + 37))
        x += lbl.get_width() + 36

    pct_txt = font_sm.render(
        f"Área queimada: {pct_burned:.1f}%   "
        f"[ESPAÇO pausar]  [R reiniciar]  [Q sair]",
        True, COLOR["dim"]
    )
    surface.blit(pct_txt, (10, panel_y + 58))

def main():
    try:
        import pygame
    except ImportError:
        print("pygame não está instalado. Execute:  pip install pygame")
        sys.exit(1)

    print("Carregando Transformer treinado...")
    model, scaler = load_model()

    pygame.init()
    surface = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Transformer-CA — Simulação de Incêndio Florestal")

    font_big = pygame.font.SysFont("monospace", 15, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 12)

    clock   = pygame.time.Clock()
    fps_idx = FPS_DEFAULT

    ca      = new_simulation(model, scaler)
    step    = 0
    paused  = False
    running = True

    print("Janela aberta. Controles: ESPAÇO pausar | ↑↓ velocidade | R reiniciar | Q sair")

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_UP:
                    fps_idx = min(fps_idx + 1, len(FPS_OPTIONS) - 1)

                elif event.key == pygame.K_DOWN:
                    fps_idx = max(fps_idx - 1, 0)

                elif event.key == pygame.K_r:
                    ca    = new_simulation(model, scaler)
                    step  = 0
                    paused = False

        fire_active = int((ca.grid == BURNING).sum()) > 0

        if not paused and fire_active:
            ca.step()
            step += 1
        elif not paused and not fire_active and step > 0:
            paused = True

        surface.fill(COLOR["bg"])
        draw_grid(surface, ca.grid, pygame)
        draw_panel(surface, pygame, font_big, font_sm, step, ca, paused, fps_idx)
        pygame.display.flip()

        clock.tick(FPS_OPTIONS[fps_idx])

    pygame.quit()
    print("Simulação encerrada.")

if __name__ == "__main__":
    main()
