import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from transformers import AutoTokenizer

class SketchToAnimeSDDataset(Dataset):
    def __init__(self, sketch_dir, anime_dir, image_size=512, tokenizer=None):
        self.sketch_dir = sketch_dir
        self.anime_dir = anime_dir
        
        # Asegurarse de que los archivos estén alineados
        self.sketches = sorted([f for f in os.listdir(sketch_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.animes = sorted([f for f in os.listdir(anime_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Verificar que tengamos el mismo número de archivos
        assert len(self.sketches) == len(self.animes), "Número diferente de sketches y animes"
        
        self.image_size = image_size
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        
        # Transformaciones para las imágenes
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.prompt = "anime style, high quality, detailed"

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketches[idx])
        anime_path = os.path.join(self.anime_dir, self.animes[idx])

        # Cargar imágenes
        sketch = Image.open(sketch_path).convert("RGB")
        anime = Image.open(anime_path).convert("RGB")
        
        # Aplicar transformaciones
        sketch_tensor = self.transform(sketch)
        anime_tensor = self.transform(anime)
        
        # Tokenizar el prompt
        text_inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "sketch": sketch_tensor,
            "anime": anime_tensor,
            "input_ids": text_inputs.input_ids[0],
            "prompt": self.prompt
        }