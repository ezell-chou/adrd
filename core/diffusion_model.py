import torch
import numpy as np
from typing import List
from ldm.models.diffusion.ddim import DDIMSampler

LATENT_SCALING = 0.18215


class DiffusionModel:
    """
    Encapsulates diffusion encoding and stochastic reconstruction
    in latent space.

    This class implements the operators:
        - E(x): image -> latent
        - R_theta(z; xi): latent reconstruction via diffusion dynamics
    """

    def __init__(
        self,
        model,
        device: torch.device,
    ):
        """
        Parameters
        ----------
        model :
            LatentDiffusion model.
        device : torch.device
            Device to run diffusion model.
        """
        self.model = model.to(device).eval()
        self.device = device

        # Ensure all submodules are on the same device
        # This is critical for models with multiple submodules (VAE, UNet, text encoder)
        for module in self.model.modules():
            module.to(device)
        
        # Explicitly move problematic submodules
        if hasattr(self.model, 'first_stage_model'):
            self.model.first_stage_model.to(device)
        if hasattr(self.model, 'cond_stage_model') and self.model.cond_stage_model is not None:
            self.model.cond_stage_model.to(device)

        # VAE (first-stage)
        self.vae = model.first_stage_model

        # DDIM sampler (LDM-native)
        self.sampler = DDIMSampler(self.model)

        for p in self.model.parameters():
            p.requires_grad = False

        # Precompute unconditional (null) conditioning
        with torch.no_grad():
            self.null_cond = self.model.get_learned_conditioning([""])


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
        image = image.to(self.device)
        latent_dist = self.vae.encode(image)
        latent = latent_dist.mean * LATENT_SCALING
        return latent

    # ------------------------------------------------------------------
    # R_theta(z; xi): Single stochastic reconstruction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_latent(
        self,
        latent: torch.Tensor,
        seed: int,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Perform one stochastic DDIM reconstruction.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (1, 4, h, w)
        seed : int
            Random seed (xi)
        num_steps : int
            Number of DDIM steps

        Returns
        -------
        latent_rec : torch.Tensor
            Shape (1, 4, h, w)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        latent = latent.to(self.device)

        # DDIM forward + reverse (eta=0 → deterministic DDIM)
        z_rec, _ = self.sampler.sample(
            S=num_steps,
            batch_size=1,
            shape=latent.shape[1:],
            conditioning=self.null_cond,
            x_T=latent,
            verbose=False,
            eta=0.0,
        )

        return z_rec

    # ------------------------------------------------------------------
    # Monte-Carlo reconstruction (Expectation over xi)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_multiple(
        self,
        latent: torch.Tensor,
        seeds: List[int],
        num_steps: int,
    ) -> List[torch.Tensor]:
        """
        Perform multiple stochastic reconstructions.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (1, 4, h, w)
        seeds : List[int]
            Random seeds for Monte-Carlo sampling.
        num_steps : int
            DDIM steps.

        Returns
        -------
        latents_rec : List[torch.Tensor]
            List of reconstructed latents.
        """
        outputs = []
        for s in seeds:
            z_rec = self.reconstruct_latent(
                latent=latent,
                seed=s,
                num_steps=num_steps,
            )
            outputs.append(z_rec)
        return outputs

    # ------------------------------------------------------------------
    # Batch reconstruction (optimized for GPU)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_latent_batch(
        self,
        latents: torch.Tensor,
        seeds: List[int],
        num_steps: int,
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

        Returns
        -------
        latents_rec : torch.Tensor
            Shape (B * len(seeds), 4, h, w)
        """
        B = latents.shape[0]
        latents = latents.to(self.device)

        outputs = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            z_rec, _ = self.sampler.sample(
                S=num_steps,
                batch_size=B,
                shape=latents.shape[1:],
                conditioning=self.null_cond.repeat(B, 1, 1),
                x_T=latents,
                verbose=False,
                eta=0.0,
            )
            outputs.append(z_rec)

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # (Optional) Latent -> Image (for visualization only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent into image space.
        NOT used for quantitative evaluation.

        Parameters
        ----------
        latent : torch.Tensor
            Shape (B, C, h, w)

        Returns
        -------
        image : torch.Tensor
            Shape (B, 3, H, W), range [-1, 1]
        """
        latent = latent / LATENT_SCALING
        image = self.vae.decode(latent)
        return image
