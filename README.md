# ADRD: Detecting diffusion-generated images via adversarial perturbation induced reconstruction discrepancy

## Project Overview

The rapid advancement of diffusion models has raised concerns about their misuse in generating deceptive visual content. Existing detection methods rely heavily on image semantic features, but modern diffusion models are optimized to closely match the semantic structure of real images, reducing the effectiveness of semantic-based detection. This project proposes ADRD (Adversarial Diffusion Reconstruction Distance), a detection framework that actively probes reconstruction behavior by introducing perturbations in latent space and measuring how reconstruction deviations respond under identical perturbations. Experiments show that real images typically exhibit larger and more variable reconstruction responses, while diffusion-generated images tend to display more stable reconstruction behavior. By characterizing reconstruction sensitivity rather than absolute reconstruction error, ADRD provides a complementary perspective to existing reconstruction-based detectors.

**Key Features:**
- 🎯 Dynamic Response Probing: Actively probe reconstruction behavior through latent space perturbations
- 🚀 Fast image reconstruction and evaluation pipeline
- 📊 Support for cross-validation and multi-run statistics
- 🔧 Flexible configuration system and command-line parameter support

---

## Environment Setup

### System Requirements

- **Operating System**: Linux / macOS / Windows
- **Python Version**: 3.9+
- **GPU**: Strongly recommended (NVIDIA GPU with CUDA support)
- **Memory**: At least 16GB RAM (8GB+ VRAM for GPU usage)

### Installation Steps

```bash
# Using conda (recommended)
conda create -n adrd python=3.9
conda activate adrd

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# If using GPU (CUDA 11.x), you may need to manually install PyTorch
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Model Setup

### Download Stable Diffusion v1.5 Model

The project requires downloading pre-trained model components from HuggingFace.

#### Quick Download Method (Recommended)

Use HuggingFace CLI tool:

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download the entire model locally
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --repo-type model \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

#### Manual Download Method

Visit https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

Download the following files to the `models/sd_v1_5` directory:

1. **text_encoder**
   - `config.json`
   - `pytorch_model.safetensors`

2. **tokenizer**
   - `merges.txt`
   - `special_tokens_map.json`
   - `tokenizer_config.json`
   - `vocab.json`

3. **unet**
   - `config.json`
   - `diffusion_pytorch_model.safetensors`

4. **vae**
   - `config.json`
   - `diffusion_pytorch_model.safetensors`

5. **Other files**
   - `model_index.json`
   - `scheduler/config.json`
   - etc.

#### Verify Model

```bash
# Check if model files are complete
ls -la models/sd_v1_5/
```

Expected directory structure:

```
models/sd_v1_5/
├── text_encoder/
│   ├── config.json
│   └── pytorch_model.safetensors
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── model_index.json
└── scheduler/
    └── config.json
```

---

## Quick Start

The project includes a complete workflow with 5 main steps. All scripts support running with default configuration or can be customized via command-line parameters.

### Step 0: Dataset Generation (Optional)

Use `generate_dataset.py` to sample and generate CSV index files from the GenImage dataset.

**Script Location**: `utils/generate_dataset.py`

**Features**:
- Randomly select AI-generated and real images from GenImage dataset
- Generate CSV index files
- Different random results each execution

**Running with Default Parameters**:

```bash
python utils/generate_dataset.py
```

**Running with Custom Parameters**:

```bash
python utils/generate_dataset.py \
    --img_root /path/to/GenImage \
    --save_index datasets/GenImage \
    --n 20
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--img_root` | `D:\GenImage` | GenImage dataset root directory |
| `--save_index` | `datasets/GenImage` | CSV output directory |
| `--n` | `20` | Number of samples per category (ai and nature) for each dataset |

**Output**:
- `<dataset_name>_ai.csv` - AI-generated image list
- `<dataset_name>_nature.csv` - Real image list

#### 0.2 Using Dataset Path Index

Modify the `IMAGE_ROOT` parameter in `BaseConfig` within `configs/config.py` to point to your actual dataset path. For example: `D:\GenImage`

---

### Step 1: Compute Optimized Probe

Use `quick_optimize.py` to train latent space probes. This script performs a quick test using minimal samples and iterations.

**Script Location**: `quick_optimize.py`

**Features**:
- Load real and fake images
- Train universal latent space probe
- Compute reconstruction error gap
- Save trained probe direction

**Running with Default Parameters**:

```bash
python quick_optimize.py
```

**Running with Custom Parameters**:

```bash
python quick_optimize.py \
    --real datasets/seek_probe/real/wukong_nature.csv \
    --fake datasets/seek_probe/fake/wukong_ai.csv \
    --save_probe results/probes/
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--real` | `datasets/seek_probe/real/wukong_nature.csv` | Real image CSV file path |
| `--fake` | `datasets/seek_probe/fake/wukong_ai.csv` | Fake image CSV file path |
| `--save_probe` | `results/probes` | Directory to save probe |

**Configuration Parameters** (in `configs/config.py` `OptConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | auto | Computing device (cuda or cpu) |
| `CHECKPOINT` | `models/sd_v1_5` | Model checkpoint path |
| `N_SAMPLES_REAL` | `1` | Number of real image samples per model |
| `N_SAMPLES_FAKE` | `1` | Number of fake image samples per model |
| `N_ITERATIONS` | `20` | Optimization iterations |
| `LEARNING_RATE` | `0.1` | Gradient ascent learning rate |
| `MC_SAMPLES` | `8` | Monte Carlo samples |
| `STRENGTH` | `0.6` | Image guidance strength (0.0-1.0) |

**Output**:
- `results/probes/quick_probe.pt` - Trained probe direction tensor

**Expected Results**:
- ✅ Final gap is positive, indicating larger real image reconstruction error

---

### Step 2: Image Reconstruction and Probe Overlay

Use `recon.py` to overlay probes on images and reconstruct, generating tensors for classifier training.

**Script Location**: `recon.py`

**Features**:
- Batch encode images to latent space
- Overlay trained probe
- Reconstruct in latent space
- Save reconstruction difference tensors for subsequent training

**Running with Default Parameters**:

```bash
python recon.py
```

**Running with Custom Parameters**:

```bash
python recon.py \
    --recon_target datasets/seek_probe \
    --probe_path data/probe/probe.pt \
    --save_img_tensor results/img_tensor
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--recon_target` | `datasets/seek_probe/fake` | CSV root directory or single CSV file |
| `--probe_path` | `data/probe/probe.pt` | Probe file path |
| `--save_img_tensor` | `results/img_tensor` | Output tensor directory |

**Configuration Parameters** (in `configs/config.py` `ReconConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | `1` | Batch processing size |
| `NUM_STEPS` | `25` | Diffusion reconstruction steps |
| `SEED` | `12` | Random seed |
| `EPSILON` | `30` | Probe strength |
| `STRENGTH` | `0.6` | Image guidance strength (0.0-1.0) |

**Output**:
- `results/img_tensor/<arch>/real/*.pt` - Real image reconstruction differences
- `results/img_tensor/<arch>/fake/*.pt` - Fake image reconstruction differences

**File Format**:
Each `.pt` file contains a tuple `(tensor, is_fake)`, where:
- `tensor`: Reconstruction difference tensor (4D: 1×4×64×64)
- `is_fake`: Label (0=real, 1=fake)

---

### Step 3: Train and Evaluate Classifier

Use `train_eval.py` to train classifier and perform cross-validation evaluation.

**Script Location**: `train_eval.py`

**Features**:
- Load reconstructed tensor data
- Train classifier network
- Perform cross-validation evaluation
- Compute evaluation metrics (ACC, AP, AUC)
- Save training results

**Running with Default Parameters**:

```bash
python train_eval.py
```

**Running with Custom Parameters**:

```bash
python train_eval.py \
    --train_tensor results/img_tensor \
    --eval_tensor results/img_tensor \
    --save_models results/models \
    --results results/eval_results
```

**Parameter Description**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_tensor` | `results/img_tensor` | Training data directory |
| `--eval_tensor` | `results/img_tensor` | Evaluation data directory |
| `--save_models` | `results/cross_eval_models/models_GenImage` | Model save directory |
| `--results` | `results/cross_eval_res/models_GenImage` | Results save directory |

**Configuration Parameters** (in `configs/config.py` `TrainConfig`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NETWORK` | `BetterCNN` | Network architecture (SimpleCNN/BetterCNN/FusionCNN/AttentionCNN) |
| `EPOCHS` | `50` | Training epochs |
| `BATCH_SIZE` | `1` | Batch processing size |
| `LEARNING_RATE` | `1e-3` | Learning rate |
| `SPLIT_RATIO` | `0.7` | Train/validation split ratio |
| `RETRAIN_TIMES` | `3` | Repetitions per subset |
| `FORCE_RETRAIN` | `False` | Force retraining |
| `N_EVAL_RUNS` | `1` | Evaluation runs per model |

**Output**:
- `results/cross_eval_models/models_GenImage/*.pth` - Trained models
- `results/cross_eval_res/models_GenImage/final_results.json` - Evaluation results

**Result Format** (`final_results.json`):

```json
{
  "train_subset_name": {
    "eval_subset_name": {
      "acc": {"mean": 0.75, "std": 0.02},
      "ap": {"mean": 0.68, "std": 0.01},
      "auc": {"mean": 0.66, "std": 0.02}
    }
  }
}
```

---

## Configuration Details

Project configuration is defined in `configs/config.py`. Main configuration classes:

### BaseConfig (Base Configuration)

Parent class for all other configuration classes, containing global settings:

```python
class BaseConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "models/sd_v1_5"
    IMAGE_ROOT = "/path/to/GenImage"
    NETWORK = 'BetterCNN'
```

### OptConfig (Optimization/Quick Test Configuration)

Used by `quick_optimize.py`:

- `REAL_CSV`: Real image CSV path
- `FAKE_CSV`: Fake image CSV path
- `N_SAMPLES_REAL`: Real samples count (quick test: 1)
- `N_SAMPLES_FAKE`: Fake samples count (quick test: 1)
- `N_ITERATIONS`: Optimization iterations (quick test: 20)
- `LEARNING_RATE`: Learning rate (0.1)
- `MC_SAMPLES`: Monte Carlo samples (8)
- `STRENGTH`: Image guidance strength (0.6)

### ReconConfig (Reconstruction Configuration)

Used by `recon.py`:

- `RECON_TARGET`: Reconstruction target CSV directory
- `SAVE_IMG_TENSOR`: Tensor output directory
- `PROBE_PATH`: Probe file path
- `NUM_STEPS`: Diffusion steps (25)
- `BATCH_SIZE`: Batch size (1)
- `EPSILON`: Probe strength (30)
- `STRENGTH`: Image guidance strength (0.6)

### TrainConfig (Training Configuration)

Used by `train_eval.py`:

- `TRAIN_TENSOR`: Training data directory
- `EVAL_TENSOR`: Evaluation data directory
- `SAVE_MODELS`: Model save directory
- `RESULTS`: Results save directory
- `EPOCHS`: Training epochs (50)
- `BATCH_SIZE`: Batch size (1)
- `LEARNING_RATE`: Learning rate (1e-3)
- `SPLIT_RATIO`: Train/validation split (0.7)
- `RETRAIN_TIMES`: Repetitions (3)

---

## Project Directory Structure

```
adrd_v3/
├── README.md                    # This file
├── requirements.txt             # Project dependencies
│
├── configs/
│   ├── __init__.py
│   └── config.py               # Configuration class definitions
│
├── core/
│   ├── diffusion_model_v3.py   # Diffusion model core class
│   ├── optimize_probe.py        # Probe training algorithm
│   ├── latent_probe.py          # Probe loading and application
│   ├── classifier.py            # Classifier network
│   ├── utilities.py             # Data loading utilities
│   └── utilities_train.py       # Training utilities
│
├── utils/
│   ├── generate_dataset.py      # Dataset generation tool
│   └── minimal_recon.py         # Minimal reconstruction script
│
├── analysis/                    # Analysis and visualization tools
│   ├── draw.py
│   ├── plot_img_pairs.py
│   ├── plot_matrix.py
│   ├── plot_robust.py
│   └── plot_strength.py
│
├── models/
│   └── sd_v1_5/                # Stable Diffusion v1.5 model
│
├── datasets/
│   └── seek_probe/              # Dataset CSV files
│       ├── real/
│       │   └── *.csv
│       └── fake/
│           └── *.csv
│
├── data/
│   └── probe/
│       └── probe.pt             # Trained probe
│
└── results/                     # Output results
    ├── img_tensor/              # Reconstruction tensors
    ├── models/                  # Trained classifiers
    ├── eval_results/            # Evaluation results
    └── *.json                   # Result statistics
```

---

## Complete Workflow Example

The following is a complete workflow from data preparation to model evaluation:

### 1. Prepare Environment

```bash
# Create virtual environment
conda create -n adrd python=3.9
conda activate adrd

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

```bash
# Download using huggingface-cli
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

### 3. Generate Dataset (Optional)

```bash
python utils/generate_dataset.py \
    --img_root /path/to/dataset \
    --n 20
```

### 4. Train Probe

```bash
python quick_optimize.py
```

### 5. Reconstruct Images

```bash
python recon.py
```

### 6. Train and Evaluate Classifier

```bash
python train_eval.py
```

### 7. Analyze Results

```bash
# View evaluation results
cat results/cross_eval_res/models_GenImage/final_results.json

# Visualize results (using analysis tools)
python analysis/plot_matrix.py
```

---

## Troubleshooting

### Q1: CUDA Out of Memory

**Error Message**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
# Modify in configs/config.py:
# ReconConfig.BATCH_SIZE = 1
# TrainConfig.BATCH_SIZE = 4

# Or use CPU (though it will be very slow)
# BaseConfig.DEVICE = torch.device("cpu")
```

### Q2: Model Files Not Found

**Error Message**: `FileNotFoundError: models/sd_v1_5 not found`

**Solution**:
```bash
# Ensure model is downloaded
ls -la models/sd_v1_5/

# If directory is empty, re-download
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir models/sd_v1_5 \
    --local-dir-use-symlinks False
```

### Q3: CSV File Format Error

**Error Message**: `KeyError: 'imagepath'`

**Solution**:
- Ensure CSV file has `num`, `imagepath`, `IsFake` columns
- Use `utils/generate_dataset.py` to generate standard format CSV

### Q4: Insufficient Memory

**Error Message**: `MemoryError` or program freezes

**Solution**:
```bash
# Reduce sample count
# Modify in configs/config.py:
# OptConfig.N_SAMPLES_REAL = 1
# OptConfig.N_SAMPLES_FAKE = 1

# Or reduce Monte Carlo samples
# OptConfig.MC_SAMPLES = 4
```

### Q5: Image Loading Failure

**Error Message**: `PIL.UnidentifiedImageError` or image loading error

**Solution**:
- Check image path correctness
- Ensure `IMAGE_ROOT` configuration points to correct directory
- Verify CSV relative paths are correct

### Q6: Slow Execution

**Possible Causes**:
- Using CPU instead of GPU
- Batch size too small
- Diffusion steps too many

**Solution**:
```bash
# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Increase batch size
# Modify in configs/config.py:
# ReconConfig.BATCH_SIZE = 8
# TrainConfig.BATCH_SIZE = 16

# Reduce diffusion steps
# ReconConfig.NUM_STEPS = 10
```

---

## Citation and License

For detailed license information, please refer to the `LICENSE` file.

---

## Contact and Support

If you have questions or suggestions, please submit an issue or contact the project maintainers.

---

**Last Updated**: January 2026
**Version**: v3.0
