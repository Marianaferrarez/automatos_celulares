import matplotlib.pyplot as plt


class MapVisualizer:

    @staticmethod
    def show_map(matrix, title="Mapa", cmap="viridis"):

        plt.figure(figsize=(8, 6))

        plt.imshow(matrix, cmap=cmap)

        plt.colorbar()

        plt.title(title)

        plt.tight_layout()

        plt.show()

    @staticmethod
    def show_multiple_maps(maps):

        total = len(maps)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        axes = axes.flatten()

        for ax, (name, matrix) in zip(axes, maps.items()):

            img = ax.imshow(matrix)

            ax.set_title(name)

            fig.colorbar(img, ax=ax)

        for i in range(len(maps), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        plt.show()
        
    @staticmethod
    def show_vegetation(matrix):

        plt.figure(figsize=(6,6))

        plt.imshow(matrix, cmap="Greens")

        plt.title("Vegetação")

        plt.colorbar()

        plt.show()
        
    @staticmethod
    def show_forest(grid):

        plt.figure(figsize=(8,8))

        plt.imshow(grid, cmap="hot")

        plt.title("Estado da Floresta")

        plt.colorbar()

        plt.show()