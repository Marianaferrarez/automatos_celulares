from pathlib import Path

import torch

from models.transformer import FireTransformer


def main():
    model = FireTransformer(num_classes=3)

    output_dir = Path(__file__).resolve().parent / "saved_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        output_dir / "fire_transformer.pth"
    )

    print(f"Modelo salvo em {output_dir / 'fire_transformer.pth'}")


if __name__ == "__main__":
    main()