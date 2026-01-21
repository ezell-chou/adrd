import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import json

from core.utilities import (
    load_image_list,
    load_image,
)
from core.diffusion_model_v3 import DiffusionModel
from configs.config import OptConfig


# ============================================================================
# Training Function
# ============================================================================

def train_universal_probe(
    real_images: List[torch.Tensor],
    fake_images: List[torch.Tensor],
    model: DiffusionModel,
    config: OptConfig,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Train universal adversarial perturbation using parameterized optimization.
    
    Parameters
    ----------
    real_images : List[torch.Tensor]
        List of real images
    fake_images : List[torch.Tensor]
        List of fake images
    model : DiffusionModel
        Diffusion model
    config : Config
        Training configuration
    
    Returns
    -------
    direction : torch.Tensor
        Optimized probing direction, shape (1, C, h, w)
    history : List[dict]
        Training history
    """
    device = model.device
    
    # Initialize unnormalized direction parameter
    # Get latent shape by encoding a sample image
    sample_latent = model.encode_image(real_images[0])  # (1, C, h, w)
    
    direction_unnormalized = torch.randn_like(sample_latent)
    direction_unnormalized = direction_unnormalized.to(device)
    direction_unnormalized.requires_grad = True
    
    optimizer = torch.optim.Adam(
        [direction_unnormalized], 
        lr=config.LEARNING_RATE,
        betas=(0.95, 0.999),
        eps=1e-8
    )
    
    # Training loop
    history = []
    
    print("\n" + "="*70)
    print("Training Universal Adversarial Probe")
    print("="*70)
    print(f"Real images:     {len(real_images)}")
    print(f"Fake images:     {len(fake_images)}")
    print(f"Iterations:      {config.N_ITERATIONS}")
    print(f"Epsilon:         {config.EPSILON}")
    print(f"Strength:         {config.STRENGTH}")
    print(f"Learning rate:   {config.LEARNING_RATE}")
    print(f"MC samples:      {config.MC_SAMPLES}")
    print("="*70)

    print("\n📊 Optimization Strategy:")
    print("   Objective: max(R_real - R_fake)")
    print("   Expectation: Real images have higher reconstruction error")
    print("   Goal: Make gap more positive (amplify the difference)\n")
    
    pbar = tqdm(range(config.N_ITERATIONS), desc="Training")
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        # ✅ SCHEME C: Parameterized optimization
        # Step 1: Parameterize unnormalized direction to normalized direction
        norm = direction_unnormalized.norm()
        direction_normalized = direction_unnormalized / (norm + 1e-8)
        direction = config.EPSILON * direction_normalized
        
        # Step 2: Compute reconstruction responses (gradient flows through parameterization)
        response_fake = compute_response_with_grad(
            images=fake_images,
            direction=direction,
            epsilon=config.EPSILON,
            model=model,
            num_steps=config.NUM_STEPS,
            mc_samples=config.MC_SAMPLES,
            seed_base=config.SEED_BASE,
            strength=config.STRENGTH,
            batch_size=config.BATCH_SIZE,
        )
        
        response_real = compute_response_with_grad(
            images=real_images,
            direction=direction,
            epsilon=config.EPSILON,
            model=model,
            num_steps=config.NUM_STEPS,
            mc_samples=config.MC_SAMPLES,
            seed_base=config.SEED_BASE + 1000,
            strength=config.STRENGTH,
            batch_size=config.BATCH_SIZE,
        )
        
        # Step 3: Gradient ascent (maximize gap)
        gap = response_real - response_fake
        loss = -gap
        
        # Step 4: Backward pass (gradient flows back to direction_unnormalized)
        loss.backward()
        
        # Debug: log gradient norm
        grad_norm = direction_unnormalized.grad.norm().item() if direction_unnormalized.grad is not None else 0.0
        
        if grad_norm < 1e-8:
            tqdm.write(f"  ⚠️ WARNING: Gradient is zero or too small! (norm={grad_norm:.6e})")
        else:
            tqdm.write(f"  ✓ Gradient norm: {grad_norm:.6e}")
        
        # Step 5: Optimizer step (no manual projection needed, constraint satisfied by parameterization)
        optimizer.step()
        
        # Record history
        history.append({
            "iteration": iteration,
            "gap": gap.item(),
            "response_fake": response_fake.item(),
            "response_real": response_real.item(),
            "direction_norm": direction.view(1, -1).norm(p=2).item(),
            "grad_norm": grad_norm,
        })
        
        # Update progress bar
        pbar.set_postfix({
            "gap": f"{gap.item():.4f}",
            "R_fake": f"{response_fake.item():.4f}",
            "R_real": f"{response_real.item():.4f}",
            "grad_norm": f"{grad_norm:.2e}",
        })
        
        # Log every iteration
        tqdm.write(f"Iter {iteration+1:3d}/{config.N_ITERATIONS}: gap={gap.item():.4f}, R_fake={response_fake.item():.4f}, R_real={response_real.item():.4f}, grad_norm={grad_norm:.2e}")
        
        # Save checkpoint
        if (iteration + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(direction, history, config.OUTPUT_DIR, iteration + 1)
    
    print("\n✓ Training completed\n")
    
    return direction.detach(), history



def compute_response_with_grad(
    images: List[torch.Tensor],
    direction: torch.Tensor,
    epsilon: float,
    model: DiffusionModel,
    num_steps: int,
    mc_samples: int,
    seed_base: int = 42,
    strength: float = 1.0,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Compute reconstruction response with gradient tracking.
    
    Key insight: Gradients must flow through the PROBED latents, not the reconstruction.
    We use the reconstruction to compute a target, then compute loss against probed latents.
    
    Parameters
    ----------
    images : List[torch.Tensor]
        List of input images in tensor format
    direction : torch.Tensor
        Probing direction
    epsilon : float
        Perturbation strength
    model : DiffusionModel
        Diffusion model for encoding/decoding
    num_steps : int
        Number of DDIM steps
    mc_samples : int
        Number of MC samples
    seed_base : int
        Base seed for random number generation
    strength : float
        Reconstruction strength parameter
    batch_size : int
        Batch size for processing
    """
    device = model.device
    responses = []
    
    n_images = len(images)
    n_batches = (n_images + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_images)
        batch_images = images[start_idx:end_idx]
        
        # Load and encode batch
        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        
        with torch.no_grad():
            latents_orig = model.encode_image(batch_tensor)
        
        # Expand direction
        B = latents_orig.shape[0]
        direction_batch = direction.expand(B, -1, -1, -1)
        
        latents_orig_grad = latents_orig.detach().requires_grad_(False)
        
        # Apply probing (KEEP GRADIENT through direction)
        # This is where the gradient path starts
        latents_probed = latents_orig.detach() + direction_batch

        # Reconstruction (no_grad - diffusion is not differentiable)
        mc_seeds = [seed_base + i for i in range(mc_samples)]
        with torch.no_grad():
            # Detach probed latents for reconstruction
            latents_recon = model.reconstruct_latent_batch(
                latents=latents_probed.detach(),
                seeds=mc_seeds,
                num_steps=num_steps,
                strength=strength,
            )   
        
        # Reshape reconstructed latents
        latents_recon = latents_recon.view(B, mc_samples, *latents_recon.shape[1:])
        
        # Compute distances between probed and reconstructed latents
        # This is the KEY: we compute distance from PROBED (has gradient) to recon (no gradient)
        # The loss will be: distance = ||latents_recon - latents_probed||
        # Since latents_probed depends on direction, gradients flow back through it
        
        distances = []
        for i in range(mc_samples):
            recon_latent = latents_recon[:, i]  # constant

            # Key identity:
            # || recon - z || = || recon - (z+εδ) + εδ ||
            diff = (latents_probed - recon_latent)

            dist = torch.norm(diff.reshape(B, -1), p=2, dim=1)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=1)
        avg_dist_per_image = distances.mean(dim=1)
        
        responses.append(avg_dist_per_image)
    
    all_responses = torch.cat(responses, dim=0)
    avg_response = all_responses.mean()
    
    return avg_response


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
    direction: torch.Tensor,
    history: List[dict],
    output_dir: str,
    iteration: int,
) -> None:
    """Save training checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save direction
    torch.save(
        direction.cpu(),
        output_dir / f"direction_iter{iteration}.pt"
    )
    
    # Save history
    with open(output_dir / f"history_iter{iteration}.json", "w") as f:
        json.dump(history, f, indent=2)
    
    tqdm.write(f"  ✓ Checkpoint saved: iter {iteration}")



# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(
    real_csv: str,
    fake_csv: str,
    image_root: str,
    n_samples: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load training images.
    
    Returns
    -------
    real_images, fake_images : Tuple
        Lists of image tensors
    """
    print("Loading training data...")
    
    # Load image lists
    real_paths, _ = load_image_list(Path(real_csv), Path(image_root))
    fake_paths, _ = load_image_list(Path(fake_csv), Path(image_root))
    
    # Sample subset
    if len(real_paths) > n_samples:
        indices = np.random.choice(len(real_paths), n_samples, replace=False)
        real_paths = [real_paths[i] for i in indices]
    
    if len(fake_paths) > n_samples:
        indices = np.random.choice(len(fake_paths), n_samples, replace=False)
        fake_paths = [fake_paths[i] for i in indices]
    
    # Load images
    real_images = []
    fake_images = []
    
    print(f"  Loading {len(real_paths)} real images...")
    for path in tqdm(real_paths, desc="  Real", leave=False):
        try:
            img = load_image(Path(path), device)
            real_images.append(img)
        except Exception as e:
            print(f"  ✗ Failed to load {path}: {e}")
    
    print(f"  Loading {len(fake_paths)} fake images...")
    for path in tqdm(fake_paths, desc="  Fake", leave=False):
        try:
            img = load_image(Path(path), device)
            fake_images.append(img)
        except Exception as e:
            print(f"  ✗ Failed to load {path}: {e}")
    
    print(f"✓ Loaded {len(real_images)} real, {len(fake_images)} fake images\n")
    
    return real_images, fake_images


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main training pipeline."""
    config = OptConfig()
    
    # Setup device
    if config.DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.DEVICE)
    
    print(f"\nDevice: {device}\n")
    
    # Load model
    print("Loading diffusion model...")
    model = DiffusionModel(str(config.CHECKPOINT), device)
    print("✓ Model loaded\n")
    
    # Load training data
    real_images, fake_images = load_training_data(
        real_csv=config.REAL_CSV,
        fake_csv=config.FAKE_CSV,
        image_root=config.IMAGE_ROOT,
        n_samples=config.N_SAMPLES_REAL,
        device=device,
    )
    
    # Train
    direction, history = train_universal_probe(
        real_images=real_images,
        fake_images=fake_images,
        model=model,
        config=config,
    )
    
    # Save final results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(direction.cpu(), output_dir / "direction_final.pt")
    
    with open(output_dir / "history_final.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Final results saved to {output_dir}")
    print(f"  - direction_final.pt")
    print(f"  - history_final.json\n")


if __name__ == "__main__":
    main()
