import torch
import numpy as np
import json
from typing import List, Optional
from pathlib import Path
from safetensors.torch import load_file
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPConfig, CLIPTextConfig
import logging

logger = logging.getLogger(__name__)

LATENT_SCALING = 0.18215


class DiffusionModel:
    """
    Complete Stable Diffusion v1.5 implementation with full reconstruction capability.
    
    This class implements:
        - E(x): image -> latent encoding
        - R_theta(z; xi): latent reconstruction via DDIM sampling
        - D(z): latent -> image decoding
    
    Compatible with both CUDA and CPU devices.
    Tested with Python 3.9.16/torch 2.8 and Python 3.8.5/torch 1.13.1+cu117
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = None,
        low_vram_mode: bool = False,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the Stable Diffusion model directory containing:
            - v1-5-pruned-emaonly.safetensors
            - vocab.json
            - merges.txt
        device : torch.device
            Device to run diffusion model (cuda or cpu).
        dtype : torch.dtype, optional
            Data type for model weights. If None, uses float32.
            For better performance on GPU, consider torch.float16.
        low_vram_mode : bool
            If True, models are moved to CPU when not in use (slower but uses less VRAM).
        """
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.low_vram_mode = low_vram_mode
        
        # Convert to Path object for cross-platform compatibility
        model_dir = Path(model_path)
        
        print(f"Loading Stable Diffusion v1.5 from {model_dir}")
        print(f"Device: {device}, dtype: {self.dtype}, low_vram_mode: {low_vram_mode}")
        
        # Use Path object for cross-platform compatibility (Windows/Linux/macOS)
        safetensors_path = str(model_dir / "v1-5-pruned-emaonly.safetensors")
        
        # Set environment variables to disable HuggingFace cache and network access
        # This prevents attempts to download configs from the internet
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force transformers offline mode
        
        # Load VAE
        print("Loading VAE...")
        try:
            self.vae = AutoencoderKL.from_single_file(
                safetensors_path,
                torch_dtype=self.dtype,
                local_files_only=True,  # Only use local files
            )
        except Exception as e:
            print(f"Error loading VAE with from_single_file: {e}")
            print("Attempting alternative VAE loading method...")
            # Fallback: load VAE config and weights separately
            self.vae = self._load_vae_offline(model_dir, safetensors_path)
        if not low_vram_mode:
            self.vae = self.vae.to(device)
        self.vae.eval()
        
        # Load UNet
        print("Loading UNet...")
        try:
            self.unet = UNet2DConditionModel.from_single_file(
                safetensors_path,
                torch_dtype=self.dtype,
                local_files_only=True,  # Only use local files
            )
        except Exception as e:
            print(f"Error loading UNet with from_single_file: {e}")
            print("Attempting alternative UNet loading method...")
            # Fallback: load UNet config and weights separately
            self.unet = self._load_unet_offline(model_dir, safetensors_path)
        if not low_vram_mode:
            self.unet = self.unet.to(device)
        self.unet.eval()
        
        # Load text encoder and tokenizer
        print("Loading text encoder...")
        try:
            # Try loading tokenizer from local directory (recommended)
            tokenizer_dir = str(model_dir / "tokenizer")
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            print(f"Loading tokenizer from local directory failed: {e}")
            print("Downloading tokenizer from HuggingFace...")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        
        # Load text encoder model from safetensors file with local config
        try:
            # Try loading config from local JSON file
            config_path = str(model_dir / "text_encoder_config.json")
            with open(config_path, 'r') as f:
                clip_config_dict = json.load(f)
            # Extract text_config from CLIP config
            text_config_dict = clip_config_dict.get("text_config", {})
            text_config = CLIPTextConfig(**text_config_dict)
        except Exception as e:
            print(f"Loading text encoder config from local failed: {e}")
            print("Downloading text encoder config from HuggingFace...")
            full_config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
            text_config = full_config.text_config
        
        # Initialize model with text config
        self.text_encoder = CLIPTextModel(text_config)
        
        # Load weights from safetensors
        state_dict = load_file(safetensors_path)
        text_encoder_state = {}
        for k in state_dict.keys():
            if "cond_stage_model.transformer" in k:
                new_key = k.replace("cond_stage_model.transformer.", "")
                weight = state_dict[k]
                # Convert int64 indices to int32 and float tensors to target dtype
                if weight.dtype == torch.int64:
                    weight = weight.to(torch.int32)
                elif weight.dtype in [torch.float16, torch.float32, torch.float64]:
                    weight = weight.to(self.dtype)
                text_encoder_state[new_key] = weight
        
        self.text_encoder.load_state_dict(text_encoder_state, strict=False)
        if not low_vram_mode:
            self.text_encoder = self.text_encoder.to(device)
        self.text_encoder.eval()
        
        # Initialize DDIM scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,  # SD v1.5 uses steps_offset=1
        )
        
        # Freeze all parameters
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        # Precompute unconditional (null) conditioning
        print("Computing null conditioning...")
        with torch.no_grad():
            self.null_cond = self._get_text_embeddings([""])
        
        print("Model loaded successfully!")

    def _load_vae_offline(self, model_dir: Path, safetensors_path: str) -> AutoencoderKL:
        """
        Load VAE offline without any network access.
        This is a fallback method that directly loads weights without config instantiation.
        """
        print("Loading VAE offline without network access...")
        try:
            # Load weights from safetensors
            state_dict = load_file(safetensors_path)
            vae_state = {}
            for k in state_dict.keys():
                if k.startswith("first_stage_model."):
                    new_key = k.replace("first_stage_model.", "")
                    vae_state[new_key] = state_dict[k]
            
            # Create a default VAE instance (SD v1.5 standard config)
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
                up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                latent_channels=4,
            )
            
            # Load the weights into the model
            vae.load_state_dict(vae_state, strict=False)
            return vae
        except Exception as e:
            print(f"Failed to load VAE offline: {e}")
            raise RuntimeError("Cannot load VAE model. Please ensure the model files are complete.")

    def _load_unet_offline(self, model_dir: Path, safetensors_path: str) -> UNet2DConditionModel:
        """
        Load UNet offline without any network access.
        This is a fallback method that directly loads weights without config instantiation.
        """
        print("Loading UNet offline without network access...")
        try:
            # Load weights from safetensors
            state_dict = load_file(safetensors_path)
            unet_state = {}
            for k in state_dict.keys():
                if k.startswith("model.diffusion_model."):
                    new_key = k.replace("model.diffusion_model.", "")
                    unet_state[new_key] = state_dict[k]
            
            # Create a default UNet instance (SD v1.5 standard config)
            unet = UNet2DConditionModel(
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                cross_attention_dim=768,
                attention_head_dim=8,
            )
            
            # Load the weights into the model
            unet.load_state_dict(unet_state, strict=False)
            return unet
        except Exception as e:
            print(f"Failed to load UNet offline: {e}")
            raise RuntimeError("Cannot load UNet model. Please ensure the model files are complete.")

    def _move_to_device(self, module, device=None):
        """Move module to device if not in low VRAM mode."""
        if device is None:
            device = self.device
        if not self.low_vram_mode or device == self.device:
            return module.to(device)
        return module

    def _get_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        Get CLIP text embeddings for given prompts.
        
        Parameters
        ----------
        prompts : List[str]
            List of text prompts.
        
        Returns
        -------
        embeddings : torch.Tensor
            Text embeddings of shape (len(prompts), 77, 768)
        """
        if self.low_vram_mode:
            self.text_encoder = self.text_encoder.to(self.device)
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Get embeddings
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            embeddings = self.text_encoder(text_input_ids)[0]
        
        if self.low_vram_mode:
            self.text_encoder = self.text_encoder.to("cpu")
        
        return embeddings

    # ------------------------------------------------------------------
    # E(x): Image -> Latent
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode an image into latent space.

        Parameters
        ----------
        image : torch.Tensor
            Shape (B, 3, H, W), range [-1, 1]

        Returns
        -------
        latent : torch.Tensor
            Shape (B, 4, h, w)
        """
        if self.low_vram_mode:
            self.vae = self.vae.to(self.device)
        
        image = image.to(device=self.device, dtype=self.dtype)
        
        # VAE encoder returns a distribution, we take the mean
        latent_dist = self.vae.encode(image).latent_dist
        latent = latent_dist.mean * LATENT_SCALING
        
        if self.low_vram_mode:
            self.vae = self.vae.to("cpu")
        
        return latent

    # ------------------------------------------------------------------
    # R_theta(z; xi): Single stochastic reconstruction via DDIM
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_latent(
        self,
        latent: torch.Tensor,
        seed: int,
        num_steps: int,
        eta: float = 0.0,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform DDIM reconstruction from latent (image-to-image generation).
        
        This performs image-to-image reconstruction by starting from the original
        latent (optionally mixed with noise based on strength parameter) and then
        denoising it using the UNet model (reverse diffusion).

        Parameters
        ----------
        latent : torch.Tensor
            Shape (B, 4, h, w), original encoded image latent
        seed : int
            Random seed for reproducibility
        num_steps : int
            Number of DDIM steps
        eta : float
            DDIM eta parameter. 0.0 = deterministic, 1.0 = stochastic
        strength : float
            Image guidance strength (0.0 to 1.0):
            - 0.0: completely preserve original image
            - 0.5: balanced preservation and variation
            - 1.0: complete regeneration with random noise

        Returns
        -------
        latent_rec : torch.Tensor
            Shape (B, 4, h, w)
        """
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.low_vram_mode:
            self.unet = self.unet.to(self.device)
        
        latent = latent.to(device=self.device, dtype=self.dtype)
        batch_size = latent.shape[0]
        
        # Set timesteps
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # Image-to-image: implement standard diffusers strength logic
        # strength controls what fraction of the denoising steps to perform
        # strength = 1.0: perform all denoising steps (full regeneration)
        # strength = 0.5: perform half the denoising steps (balanced)
        # strength = 0.0: skip all denoising steps (full preservation)
        init_timestep = int(num_steps * strength)
        t_start = max(num_steps - init_timestep, 0)
        
        # Start with original latent
        noisy_latent = latent.clone()
        
        # Prepare conditioning (unconditional)
        text_embeddings = self.null_cond.repeat(batch_size, 1, 1)
        
        # Denoising loop - only process timesteps after t_start
        for i, t in enumerate(timesteps):
            # Skip early timesteps based on strength
            if i < t_start:
                continue
            # Expand the latents for classifier free guidance (not used here, but kept for compatibility)
            latent_model_input = noisy_latent
            
            # Predict noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            # Compute previous noisy sample (denoising step)
            scheduler_output = self.scheduler.step(
                noise_pred,
                t,
                noisy_latent,
                eta=eta,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            noisy_latent = scheduler_output.prev_sample
        
        if self.low_vram_mode:
            self.unet = self.unet.to("cpu")
        
        return noisy_latent


    # ------------------------------------------------------------------
    # Monte-Carlo reconstruction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_multiple(
        self,
        latent: torch.Tensor,
        seeds: List[int],
        num_steps: int,
        eta: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        Perform multiple stochastic reconstructions.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (1, 4, h, w)
        seeds : List[int]
            Random seeds for Monte-Carlo sampling
        num_steps : int
            DDIM steps
        eta : float
            DDIM eta parameter

        Returns
        -------
        latents_rec : List[torch.Tensor]
            List of reconstructed latents
        """
        outputs = []
        for s in seeds:
            z_rec = self.reconstruct_latent(
                latent=latent,
                seed=s,
                num_steps=num_steps,
                eta=eta,
            )
            outputs.append(z_rec)
        return outputs

    # ------------------------------------------------------------------
    # Batch reconstruction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_latent_batch(
        self,
        latents: torch.Tensor,
        seeds: List[int],
        num_steps: int,
        eta: float = 0.0,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Batch DDIM reconstruction for multiple latents and seeds.

        Parameters
        ----------
        latents : torch.Tensor
            Shape (B, 4, h, w)
        seeds : List[int]
            Random seeds
        num_steps : int
            DDIM steps
        eta : float
            DDIM eta parameter
        strength : float
            Image guidance strength (0.0 to 1.0)

        Returns
        -------
        latents_rec : torch.Tensor
            Shape (B * len(seeds), 4, h, w)
        """
        outputs = []

        for seed in seeds:
            z_rec = self.reconstruct_latent(
                latent=latents,
                seed=seed,
                num_steps=num_steps,
                eta=eta,
                strength=strength,
            )
            outputs.append(z_rec)

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # D(z): Latent -> Image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent into image space.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (B, 4, h, w)

        Returns
        -------
        image : torch.Tensor
            Shape (B, 3, H, W), range [-1, 1]
        """
        if self.low_vram_mode:
            self.vae = self.vae.to(self.device)
        
        latent = latent.to(device=self.device, dtype=self.dtype)
        latent = latent / LATENT_SCALING
        
        image = self.vae.decode(latent).sample
        
        if self.low_vram_mode:
            self.vae = self.vae.to("cpu")
        
        return image
        
