import argparse
import csv
import torch
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from core.utilities import (
    get_list_files,
    load_image_list,
    load_image
)
from core.diffusion_model_v3 import DiffusionModel
from core.latent_probe import load_probe_direction, apply_latent_probe
from configs.config import ReconConfig

# ============================================================================
# Command-line argument parser
# ============================================================================

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch Latent Probing Pipeline - Reconstruct images with latent probing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python recon.py
  
  # Specify custom paths
  python recon.py --recon_target path/to/images --probe_path path/to/probe.pt --save_img_tensor path/to/output
        """
    )
    
    parser.add_argument('--recon_target', type=str, default=None,
                       help='Path to CSV root directory containing image lists')
    parser.add_argument('--probe_path', type=str, default=None,
                       help='Path to probe file (.pt)')
    parser.add_argument('--save_img_tensor', type=str, default=None,
                       help='Path to output directory for saving image tensors')
    
    return parser.parse_args()


# ============================================================================
# Tensor I/O utilities
# ============================================================================

def save_tensor_tuple(
    tensor: torch.Tensor,
    is_fake: int,
    save_path: Path,
) -> None:
    """
    Save a tuple (tensor, isFake) to disk using torch.save.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to save (will be moved to CPU before saving).
    is_fake : int
        Label (0=real, 1=fake).
    save_path : Path
        Output file path (including .pt extension).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU and detach
    tensor_to_save = tensor.detach().cpu()
    torch.save((tensor_to_save, int(is_fake)), save_path)


# ------------------------------------------------------------
# Batch processing: encode -> apply probe -> reconstruct (latent) -> save diff
# ------------------------------------------------------------

@torch.no_grad()
def reconstruct_from_csv(
    image_list: Path,
    model: DiffusionModel,
    output_root: Path,
    image_root: Optional[Path] = None,
    batch_size: int = 16,
    num_steps: int = 50,
    seed: int = 42,
    epsilon: float = 0.0,
    strength: float = 1.0,
    probe_direction: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, any]:
    """
    Batch process images: encode -> apply probe -> reconstruct latents -> compute diff -> save (diff, isFake)

    Notes:
    - We keep the encode step, but we DO NOT decode latents to images.
    - The saved tensor is: latents_recon_probed - latents_probed

    Returns a stats dict.
    """
    if device is None:
        device = model.device

    # Load image list and arch
    try:
        images, arch = load_image_list(image_list, image_root)
    except Exception as e:
        print(f"✗ Failed to load {image_list.name}: {e}")
        return {"image_list": str(image_list), "saved": 0, "failed": 0}

    # Load IsFake labels from CSV in same order
    labels: List[int] = []
    try:
        with image_list.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("IsFake", "0")
                try:
                    labels.append(int(val))
                except Exception:
                    labels.append(0)
    except Exception as e:
        print(f"✗ Failed to read labels from {image_list.name}: {e}")
        labels = [0] * len(images)

    # Prepare output root for this arch
    output_root = Path(output_root)
    arch_root = output_root / arch
    arch_root.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    n_failed = 0
    n_total = len(images)
    n_batches = (n_total + batch_size - 1) // batch_size

    # Ensure probe_direction is on correct device
    if probe_direction is not None:
        probe_direction = probe_direction.to(device)

    for batch_id in tqdm(range(n_batches), desc=f"  {arch}", leave=False):
        start = batch_id * batch_size
        end = min(start + batch_size, n_total)

        batch_images = images[start:end]
        batch_len = len(batch_images)

        try:
            # Load batch images
            loaded = []
            valid_idx = []

            for i, img in enumerate(batch_images):
                try:
                    loaded.append(load_image(Path(img), device))
                    valid_idx.append(i)
                except Exception:
                    n_failed += 1

            if not loaded:
                continue

            # Encode to latents
            batch_tensor = torch.cat(loaded, dim=0)  # (B, 3, H, W)
            latents = model.encode_image(batch_tensor)  # (B, C, h, w)

            # Prepare probe direction for this batch
            if epsilon > 0:
                if probe_direction is None:
                    raise ValueError("Epsilon > 0 but no probe_direction provided")

                # probe_direction shape: (B_p, C, h, w)
                B_p = probe_direction.shape[0]
                B = latents.shape[0]

                if B_p == 1 and B > 1:
                    # repeat first direction for whole batch
                    direction = probe_direction.repeat(B, *[1] * (probe_direction.ndim - 1))
                elif B_p == B:
                    direction = probe_direction
                else:
                    # try to broadcast if possible
                    try:
                        direction = probe_direction[:B]
                    except Exception:
                        raise ValueError(
                            f"Probe direction batch size ({B_p}) not compatible with data batch size ({B})"
                        )

                direction = direction.to(device)

                # Apply latent probe
                latents_probed = apply_latent_probe(latents, direction, epsilon)

                # Reconstruct probed latents (denoise / reconstruct in latent space)
                latents_recon_probed = model.reconstruct_latent_batch(
                    latents=latents_probed,
                    seeds=[seed],
                    num_steps=num_steps,
                    strength=strength,
                )

                # Compute difference: latents_recon_probed - latents
                diff_batch = latents_recon_probed - latents

                # Save per-sample as (diff, isFake)
                for j, local_idx in enumerate(valid_idx):
                    try:
                        global_idx = start + local_idx
                        img_path = Path(batch_images[local_idx])
                        isfake = labels[global_idx] if global_idx < len(labels) else 0
                        kind = "real" if int(isfake) == 0 else "fake"

                        save_path = arch_root / kind / f"{img_path.stem}.pt"
                        save_tensor_tuple(diff_batch[j], isfake, save_path)
                        n_saved += 1
                    except Exception:
                        n_failed += 1
            else:
                # epsilon == 0: nothing to probe; skip
                continue

        except Exception as e:
            print(f"  ✗ Batch {batch_id+1}/{n_batches} error: {e}")
            n_failed += batch_len - len(valid_idx)

    # Summary
    print(f"✓ {arch}: saved={n_saved}/{n_total}, failed={n_failed}")

    return {
        "image_list": str(image_list),
        "arch": arch,
        "saved": n_saved,
        "failed": n_failed,
    }


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """
    Main pipeline: batch encode, apply probe, reconstruct (latent), save diffs.
    Uses global 'args' variable for command-line arguments.
    """
    config = ReconConfig()

    # Get parameters from command-line args or config
    recon_target = args.recon_target or config.RECON_TARGET
    save_img_tensor = args.save_img_tensor or config.SAVE_IMG_TENSOR
    probe_path = args.probe_path or config.PROBE_PATH
    
    batch_size = config.BATCH_SIZE
    num_steps = config.NUM_STEPS
    seed = config.SEED
    epsilon = config.EPSILON
    strength = config.STRENGTH
    device = config.DEVICE

    # Convert to Path
    recon_target = Path(recon_target)
    save_img_tensor = Path(save_img_tensor)
    checkpoint = Path(config.CHECKPOINT)
    probe_path = Path(probe_path)

    image_root = Path(config.IMAGE_ROOT)

    # Validate paths
    if not recon_target.exists():
        raise FileNotFoundError(f"CSV root not found: {recon_target}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if epsilon > 0 and not probe_path.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    # Create output directory
    save_img_tensor.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 70)
    print("Batch Latent Probing Pipeline (save diffs)")
    print("=" * 70)
    print(f"Device:         {device}")
    print(f"Batch size:     {batch_size}")
    print(f"Steps:          {num_steps}")
    print(f"Seed:           {seed}")
    print(f"Epsilon:        {epsilon}")
    print(f"Probe path:     {probe_path}")
    print(f"Recon target:   {recon_target}")
    print(f"Save output:    {save_img_tensor}")
    print("=" * 70 + "\n")

    # Load model
    print("Loading model...")
    try:
        model = DiffusionModel(Path(checkpoint), device)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return

    # Load probe
    probe_direction = None
    if epsilon > 0:
        try:
            probe_direction = load_probe_direction(probe_path, device=device)
            print(f"✓ Probe loaded: {probe_path}\n")
        except Exception as e:
            print(f"✗ Failed to load probe: {e}")
            return

    # Get image list files
    try:
        list_files = get_list_files(recon_target)
        print(f"Found {len(list_files)} image list files\n")
    except Exception as e:
        print(f"✗ Failed to get image list files: {e}")
        return

    # Process each image list file
    stats_all = []
    total_saved = 0
    total_failed = 0

    for list in list_files:
        stats = reconstruct_from_csv(
            image_list=list,
            model=model,
            output_root=save_img_tensor,
            image_root=image_root,
            batch_size=batch_size,
            num_steps=num_steps,
            seed=seed,
            epsilon=epsilon,
            strength=strength,
            probe_direction=probe_direction,
            device=device,
        )
        stats_all.append(stats)
        total_saved += stats.get("saved", 0)
        total_failed += stats.get("failed", 0)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Saved tensors:   {total_saved}")
    if total_failed > 0:
        print(f"Failed:          {total_failed}")
    print(f"Output:          {save_img_tensor}")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    args = get_args()
    main()
