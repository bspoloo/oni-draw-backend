

import torch
from classes.sketch_2_anime_dataset import SketchToAnimeSDDataset
from app.config import Config
from torch.utils.data import DataLoader

def get_data_loader(number_of_images=None):
    dataset = SketchToAnimeSDDataset(
        sketch_dir=Config.SKETCH_DIR,
        anime_dir=Config.ANIME_DIR,
        image_size=Config.IMAGE_SIZE,
        tokenizer=Config.TOKENIZER
    )
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    if number_of_images is not None:
        # Usar solo un número específico de imágenes
        train_dataset = torch.utils.data.Subset(dataset, range(min(number_of_images, len(dataset))))
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    return train_loader