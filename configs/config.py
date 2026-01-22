import torch

class BaseConfig:
    """Base configuration class - contains common configuration items for all projects"""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Device: GPU if available, else CPU
    CHECKPOINT = "models/sd_v1_5"                                           # Stable Diffusion model checkpoint path

    IMAGE_ROOT = "/Users/zhouyi/Documents/Codes/AI4Safe/GenImage"           # Root directory for image data

    # =========================================================================
    # Network Selection for train/test
    # =========================================================================
    # Options: 'SimpleCNN' / 'BetterCNN' / 'FusionCNN' / 'AttentionCNN'
    NETWORK = 'BetterCNN'  

class ReconConfig(BaseConfig):
    """Reconstruction configuration class - parameters for recon.py"""
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    RECON_TARGET = "datasets/seek_probe/fake"                               # Path to fake images for reconstruction
    SAVE_IMG_TENSOR = "results/img_tensor"                                  # Directory to save reconstructed image tensors
    PROBE_PATH = "data/probe/probe.pt"                                      # Path to pre-trained probe model
    
    # =========================================================================
    # Reconstruction Parameters
    # =========================================================================
    NUM_STEPS = 25                  # Number of diffusion steps for reconstruction
    SEED = 12                       # Random seed for reproducibility
    BATCH_SIZE = 1                  # Batch size for processing
    EPSILON = 30                    # Perturbation magnitude constraint
    STRENGTH = 0.6                  # Strength of the diffusion process


class OptConfig(ReconConfig):
    """Optimize configuration - for quick start testing with minimal samples"""
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    REAL_CSV = "datasets/seek_probe/real/wukong_nature.csv"   # CSV file with real image metadata
    FAKE_CSV = "datasets/seek_probe/fake/wukong_ai.csv"       # CSV file with AI-generated image metadata
    
    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_SAMPLES_REAL = 1       # Number of real images per diffusion model
    N_SAMPLES_FAKE = 1       # Number of fake images per diffusion model
    N_ITERATIONS = 20        # Number of optimization iterations
    LEARNING_RATE = 0.1      # Step size for gradient ascent
    MC_SAMPLES = 8           # Monte Carlo samples per image
    SEED_BASE = 56           # Base seed for MC sampling
    
    # =========================================================================
    # Output
    # =========================================================================
    OUTPUT_DIR = "results/probes"                           # Directory to save optimized probes
    SAVE_EVERY = 10                                         # Save checkpoint every N iterations
    



class TrainConfig(BaseConfig):
    """Training configuration class - training-specific parameters inheriting from BaseConfig"""
    
    # =========================================================================
    # Data Paths
    # =========================================================================
    TRAIN_TENSOR = 'results/img_tensor'                      # Path to training image tensors
    EVAL_TENSOR = 'results/img_tensor'                       # Path to evaluation image tensors
    SAVE_MODELS = 'results/cross_eval_models/models_GenImage'# Directory to save trained models
    RESULTS = 'results/cross_eval_res/models_GenImage'       # Directory to save training results
    
    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    SPLIT_RATIO = 0.7                                        # Train/test split ratio (0.7 = 70% train, 30% test)
    EPOCHS = 50                                              # Number of training epochs
    BATCH_SIZE = 1                                           # Batch size for training
    LEARNING_RATE = 1e-3                                     # Initial learning rate for optimizer
    N_EVAL_RUNS = 1                                          # Number of evaluation runs
    
    # =========================================================================
    # Training Control Parameters
    # =========================================================================
    FORCE_RETRAIN = False                                   # Force model retraining even if checkpoint exists
    RETRAIN_TIMES = 5                                       # Number of training repetitions per subset for statistical results



class PlotConfig(BaseConfig):
    """Plot configuration class - plotting-specific parameters inheriting from BaseConfig"""

    # =========================================================================
    # Plot Output Paths
    # =========================================================================
    RESULTS_DIR = 'results/cross_eval_res/res_genimage'     # Directory containing result files for plotting
    RESULTS_JSON = 'final_results.json'                      # JSON file with final evaluation results
