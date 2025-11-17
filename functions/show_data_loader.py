from matplotlib import pyplot as plt

def show_data_loader(data_loader) -> None:
    batch = next(iter(data_loader))
    plt.figure(figsize=(10, 4))
    for i in range(2):
        plt.subplot(2, 4, i+1)
        plt.imshow(((batch["sketch"][i].permute(1, 2, 0) * 0.5) + 0.5))
        plt.title("Sketch")
        plt.axis("off")
        plt.subplot(2, 4, i+5)
        plt.imshow(((batch["anime"][i].permute(1, 2, 0) * 0.5) + 0.5))
        plt.title("Anime")
        plt.axis("off")
    plt.show()