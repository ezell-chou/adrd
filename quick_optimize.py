import torch
import numpy as np
from pathlib import Path
import sys
import argparse

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from core.optimize_probe import (
    load_training_data,
    train_universal_probe,
)
from core.diffusion_model_v3 import DiffusionModel
from configs.config import OptConfig



def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Quick Start - Latent Space Probe Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python quick_optimize.py
  
  # Specify custom paths
  python quick_optimize.py --real path/to/real.csv --fake path/to/fake.csv --save_probe results/
        """
    )
    
    parser.add_argument('--real', type=str, default=None,
                       help='Path to real images CSV file')
    parser.add_argument('--fake', type=str, default=None,
                       help='Path to fake images CSV file')
    parser.add_argument('--save_probe', type=str, default=None,
                       help='Directory to save probe results')
    
    return parser.parse_args()


def verify_paths(config: OptConfig) -> bool:
    """Verify all required paths exist and parameters are valid."""
    paths_to_check = [
        ("Real CSV", Path(config.REAL_CSV)),
        ("Fake CSV", Path(config.FAKE_CSV)),
        ("Image root", Path(config.IMAGE_ROOT)),
        ("Checkpoint", Path(config.CHECKPOINT)),
    ]
    
    print("\n" + "="*70)
    print("Path and Parameter Verification")
    print("="*70)
    
    all_valid = True
    for name, path in paths_to_check:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_valid = False
    
    # Verify STRENGTH parameter
    if not (0.0 <= config.STRENGTH <= 1.0):
        print(f"✗ STRENGTH: {config.STRENGTH} (must be between 0.0 and 1.0)")
        all_valid = False
    else:
        print(f"✓ STRENGTH: {config.STRENGTH}")
    
    print("="*70 + "\n")
    return all_valid


def main():
    """Quick start training."""
    print("\n" + "="*70)
    print("QUICK START TEST - Latent Space Probe Training")
    print("="*70)
    print("This is a minimal test with 5+5 images and 10 iterations.")
    print("Expected runtime: 5-10 minutes")
    print("="*70 + "\n")
    
    # Parse command line arguments
    args = get_args()
    
    config = OptConfig()
    
    # Get parameters from command-line args or config
    real_csv = args.real or config.REAL_CSV
    fake_csv = args.fake or config.FAKE_CSV
    output_dir = args.save_probe or config.OUTPUT_DIR
    
    # Override config with command line arguments if provided
    if args.real is not None:
        config.REAL_CSV = args.real
    if args.fake is not None:
        config.FAKE_CSV = args.fake
    if args.save_probe is not None:
        config.OUTPUT_DIR = args.save_probe
    
    # Verify paths
    if not verify_paths(config):
        print("❌ ERROR: Some paths or parameters are invalid!")
        print("\nPlease update the paths in configs/config.py OptConfig class:")
        print("  - REAL_CSV: Path to CSV file listing real images")
        print("  - FAKE_CSV: Path to CSV file listing fake images")
        print("  - IMAGE_ROOT: Root directory containing images")
        print("  - CHECKPOINT: Path to model directory (e.g., models/sd_v1_5)")
        print("  - STRENGTH: Image guidance strength (0.0-1.0)")
        print("\nOr provide via command-line arguments:")
        print("  python quick_optimize.py --real <path> --fake <path> --save_probe <dir>")
        return
    
    # Setup device
    device = config.DEVICE
    print(f"Device: {device}\n")
    
    if device.type == "cpu":
        print("⚠️  WARNING: Running on CPU. This will be VERY slow.")
        print("   Recommend using GPU if available.\n")
    
    try:
        # Load model
        print("Loading diffusion model...")
        model = DiffusionModel(Path(config.CHECKPOINT), device)
        print("✓ Model loaded\n")
        
        # Load data
        real_images, fake_images = load_training_data(
            real_csv=config.REAL_CSV,
            fake_csv=config.FAKE_CSV,
            image_root=config.IMAGE_ROOT,
            n_samples=config.N_SAMPLES_REAL,
            device=device,
        )
        
       
        # Train
        print("Starting quick training...\n")
        direction, history = train_universal_probe(
            real_images=real_images,
            fake_images=fake_images,
            model=model,
            config=config,
        )
        
        # Check results
        final_gap = history[-1]["gap"]
        final_R_fake = history[-1]["response_fake"]
        final_R_real = history[-1]["response_real"]
        
        print("\n" + "="*70)
        print("Quick Test Results")
        print("="*70)
        print(f"Final gap:        {final_gap:.4f}")
        print(f"Final R_fake:     {final_R_fake:.4f}")
        print(f"Final R_real:     {final_R_real:.4f}")
        print(f"Direction norm:   {direction.view(1, -1).norm().item():.4f}")
        print("="*70 + "\n")
        
        # Interpret results
        if final_gap > 0:
            print("✅ SUCCESS: Gap is positive!")
            print("   This means fake images have higher reconstruction error.")
            print("   The method is working as expected.\n")
        else:
            print("⚠️  WARNING: Gap is negative or zero.")
            print("   This suggests the optimization may need adjustment.")
            print("   Try increasing LEARNING_RATE or EPSILON.\n")
        
        if final_gap > 0.1:
            print("🎉 EXCELLENT: Large gap observed!")
            print("   Ready for full-scale training with more samples.\n")
        
        # Save
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(direction.cpu(), output_dir / "quick_probe.pt")
        print(f"✓ Results saved to: {output_dir}")
        print("\nNext steps:")
        print("1. If results look good, run full training with train_universal_probe.py")
        print("2. Use more samples (15-30 per class)")
        print("3. Use more iterations (50-100)")
        print("4. Evaluate with evaluate_probe.py\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. All paths are correct")
        print("2. CSV files are properly formatted")
        print("3. Images can be loaded")
        print("4. Model checkpoint is compatible")


if __name__ == "__main__":
    main()
