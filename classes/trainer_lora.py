import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import os
from tqdm import tqdm
from app.config import Config
from torch.utils.data import DataLoader
from torch import device

class TrainerLora:
    def __init__(self):
        self.device : device = Config.DEVICE
        self.unet : UNet2DConditionModel = Config.UNET
        self.text_encoder : CLIPTextModel = Config.TEXT_ENCODER
        self.vae : AutoencoderKL = Config.VAE
        self.scheduler : DDPMScheduler = Config.SCHEDULER

        self.unet, self.text_encoder = self.setup_lora(self.unet, self.text_encoder)

    def setup_lora(self, unet, text_encoder):
        # Configuración LoRA para UNet
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
            lora_dropout=0.1,
        )
        
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        
        return unet, text_encoder

    def train_sketch_to_anime(self, train_loader : DataLoader):
        # Mover a GPU
        self.vae.to(self.device)
        self.unet.to(self.device)
        self.text_encoder.to(self.device)

        # Congelar VAE y text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Optimizador solo para parámetros entrenables
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=Config.LEARNING_RATE)

        self.unet.train()

        for epoch in range(Config.NUM_EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")

            for batch in progress_bar:
                # Mover datos a GPU
                sketches = batch["sketch"].to(self.device)
                animes = batch["anime"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                
                # Codificar imágenes con VAE (usamos los animes como target)
                with torch.no_grad():
                    # Codificar imágenes anime a latents
                    anime_latents = self.vae.encode(animes).latent_dist.sample()
                    anime_latents = anime_latents * self.vae.config.scaling_factor

                    # Codificar sketches para condición
                    sketch_latents = self.vae.encode(sketches).latent_dist.sample()
                    sketch_latents = sketch_latents * self.vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(anime_latents)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (anime_latents.shape[0],), device=self.device)
                
                # Add noise to latents
                noisy_latents = self.scheduler.add_noise(anime_latents, noise, timesteps)
                
                # Codificar texto
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(input_ids)[0]
                
                # Predicción del noise - CORREGIDO
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs={"image_embeds": sketch_latents}
                ).sample
                
                # Loss - comparar con el noise original
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            # Guardar checkpoint cada 1 épocas
            if (epoch + 1) % 1 == 0:
                save_path = f"../ai_models/lora_checkpoint_epoch_{epoch+1}"
                os.makedirs(save_path, exist_ok=True)
                self.unet.save_pretrained(save_path)
                print(f"Checkpoint guardado en {save_path}")
        
        # Guardar modelo final
        self.unet.save_pretrained("../ai_models/sketch_to_anime_lora_final")
        print("Entrenamiento completado!")
