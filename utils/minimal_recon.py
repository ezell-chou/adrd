import torch
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

from core.utilities import (
    get_list_files,
    load_image_list,
    load_image,
    tensor_to_pil,
)
from core.diffusion_model_v3 import DiffusionModel

# ============================================================================
# Configuration Parameters
# ============================================================================
LIST_ROOT = "datasets/seek_probe/real"
OUTPUT_ROOT = "results/minimal"
CHECKPOINT = "ldm/models/split-sd-v1-5"
IMAGE_ROOT = r"D:\major_revision\GenImage4Probe"
NUM_STEPS = 25
SEED = 42
BATCH_SIZE = 2
STRENGTH = 0.6  # Image-to-image strength: 0.0=preserve original, 1.0=complete regeneration
DEVICE = None  # None = auto-detect (cuda or cpu)
# ============================================================================


# ============================================================================
# Image I/O utilities
# ============================================================================

def save_image(
    image_tensor: torch.Tensor,
    save_path: Path,
) -> None:
    """
    Save image tensor as PNG file.
    
    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape (1, 3, H, W) or (3, H, W), range [-1, 1].
    save_path : Path
        Output file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = tensor_to_pil(image_tensor)
    
    # May return single Image or List[Image]
    if isinstance(pil_image, list):
        pil_image = pil_image[0]

    pil_image.save(save_path)


# ============================================================================
# Batch reconstruction from CSV
# ============================================================================

@torch.no_grad()
def reconstruct_from_csv(
    image_list: Path,
    model: DiffusionModel,
    output_root: Path,
    image_root: Optional[Path] = None,
    batch_size: int = 16,
    num_steps: int = 50,
    seed: int = 42,
    strength: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, any]:
    """
    Batch reconstruct all images listed in a CSV file.
    
    Workflow:
    1. Load image paths from CSV
    2. Process in batches: encode -> reconstruct -> decode
    3. Save reconstructed images
    
    Parameters
    ----------
    image_list : Path
        Image list file containing image paths.
    model : DiffusionModel
        DiffusionModel instance.
    output_root : Path
        Root directory for outputs.
    image_root : Path, optional
        Root directory for input images.
    batch_size : int, optional
        Batch size (default: 16).
    num_steps : int, optional
        Reconstruction steps (default: 50).
    seed : int, optional
        Random seed (default: 42).
    strength : float, optional
        Image guidance strength, 0.0-1.0 (default: 1.0).
    device : torch.device, optional
        Compute device.
    
    Returns
    -------
    stats : dict
        Statistics: csv, arch, success, failed
    """
    if device is None:
        device = model.device
    
    # Load image list from CSV
    try:
        images, arch = load_image_list(image_list, image_root)
    except Exception as e:
        print(f"✗ Failed to load {image_list.name}: {e}")
        return {"image_list": str(image_list), "success": 0, "failed": 0}
    
    print(f"Processing {arch}: {len(images)} images")
    
    # Create output directory
    output_dir = output_root / arch
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare output file paths
    outputs = [output_dir / f"{Path(img).stem}.png" for img in images]
    
    # Batch processing
    n_success = 0
    n_failed = 0
    n_total = len(images)
    n_batches = (n_total + batch_size - 1) // batch_size
    
    for batch_id in tqdm(range(n_batches), desc=f"  {arch}", leave=False):
        start = batch_id * batch_size
        end = min(start + batch_size, n_total)
        
        batch_images = images[start:end]
        batch_outputs = outputs[start:end]
        batch_len = len(batch_images)
        
        try:
            # Load batch
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
            
            # Encode -> Reconstruct -> Decode
            batch_tensor = torch.cat(loaded, dim=0)
            latents = model.encode_image(batch_tensor)
            latents_recon = model.reconstruct_latent_batch(
                latents=latents,
                seeds=[seed],
                num_steps=num_steps,
                strength=strength,
            )
            images_recon = model.decode_latent(latents_recon)
            
            # Save results
            for i, idx in enumerate(valid_idx):
                try:
                    save_image(images_recon[i:i+1], batch_outputs[idx])
                    n_success += 1
                except Exception:
                    n_failed += 1
            
        except Exception as e:
            print(f"  ✗ Batch {batch_id+1}/{n_batches} error: {e}")
            n_failed += batch_len - len(valid_idx)
    
    print(f"  ✓ {arch}: {n_success}/{n_total} succeeded")
    
    return {
        "image_list": str(image_list),
        "arch": arch,
        "success": n_success,
        "failed": n_failed,
    }


# ============================================================================
# Main entry point
# ============================================================================

def main(
    list_root: str,
    output_root: str,
    checkpoint: str,
    image_root: Optional[str] = None,
    batch_size: int = 16,
    num_steps: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> None:
    """
    Main pipeline: batch reconstruct images from CSV files.
    
    Parameters
    ----------
    list_root : str
        Directory containing image list files.
    output_root : str
        Root directory for outputs.
    checkpoint : str
        Model checkpoint file.
    config : str
        Model config file.
    image_root : str, optional
        Root directory for input images.
    batch_size : int, optional
        Batch size (default: 16).
    num_steps : int, optional
        Reconstruction steps (default: 50).
    seed : int, optional
        Random seed (default: 42).
    device : str, optional
        Compute device (default: auto-detect).
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Convert to Path
    list_root = Path(list_root)
    output_root = Path(output_root)
    checkpoint = Path(checkpoint)

    if image_root is not None:
        image_root = Path(image_root)
    
    # Validate paths
    if not list_root.exists():
        raise FileNotFoundError(f"CSV root not found: {list_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("Batch Image Reconstruction Pipeline")
    print("="*70)
    print(f"Device:     {device}")
    print(f"Batch size: {batch_size}")
    print(f"Steps:      {num_steps}")
    print(f"Seed:       {seed}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    try:
        # for V1 model
        # ldm = load_ldm_model_from_ckpt(checkpoint, config, device)
        # model = DiffusionModel(ldm, device)
        # for V2 model
        model = DiffusionModel(checkpoint, device)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return
    
    # Get image list
    try:
        list_files = get_list_files(list_root)
        print(f"Found {len(list_files)} image list files\n")
    except Exception as e:
        print(f"✗ Failed to get image list files: {e}")
        return
    
    # Process each image list file
    stats_all = []
    total_success = 0
    total_failed = 0
    
    for list in list_files:
        stats = reconstruct_from_csv(
            image_list=list,
            model=model,
            output_root=output_root,
            image_root=image_root,
            batch_size=batch_size,
            num_steps=num_steps,
            seed=seed,
            strength=STRENGTH,
            device=device,
        )
        stats_all.append(stats)
        total_success += stats["success"]
        total_failed += stats["failed"]
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Total:   {total_success + total_failed} images")
    print(f"Success: {total_success}")
    print(f"Failed:  {total_failed}")
    print(f"Output:  {output_root}")
    print("="*70 + "\n")


# ============================================================================
# Script entry point
# ============================================================================

if __name__ == "__main__":
    main(
        list_root=LIST_ROOT,
        output_root=OUTPUT_ROOT,
        checkpoint=CHECKPOINT,
        image_root=IMAGE_ROOT,
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
        seed=SEED,
        device=DEVICE,
    )
