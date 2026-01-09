# Diffusion Models: DDPM and DDIM Implementation

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) with a standard UNet architecture using residual blocks.

## Features

- **StandardUNet**: UNet architecture with residual blocks, GroupNorm, and time conditioning via FiLM
- **DDPM**: Stochastic sampling with 1000 timesteps
- **DDIM**: Fast, deterministic sampling with configurable steps (typically 50, ~20x faster)
- **Multiple Datasets**: Support for FashionMNIST, MNIST, and Oxford Flowers-102
- **Checkpoint Management**: Resume training from any epoch with flexible checkpoint loading
- **Standalone Generation**: Generate samples from trained models without training code
- **Time Measurement**: Built-in timing for sampling performance comparison

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── models.py          # StandardUNet architecture with residual blocks
├── ddpm.py            # DDPM scheduler and sampling functions
├── ddim.py            # DDIM sampling functions
├── dataset.py         # Data loading utilities (FashionMNIST, MNIST, Oxford Flowers)
├── utils.py           # Training, validation, and visualization utilities
├── main.py            # Main training script
├── generate.py        # Standalone generation script
└── requirements.txt   # Python dependencies
```

## Quick Start

### Training

**Train on FashionMNIST (default):**
```bash
python main.py --epochs 50 --batch_size 128
```

**Train on Oxford Flowers:**
```bash
python main.py --dataset oxford_flowers --image_size 64 --batch_size 32 --epochs 50
```

**Train on MNIST:**
```bash
python main.py --dataset mnist --epochs 50
```

### Resume Training

**Resume from checkpoint (continues from next epoch):**
```bash
python main.py --checkpoint ./checkpoints/best_model.pt --epochs 50
```

**Resume from specific epoch:**
```bash
python main.py --checkpoint ./checkpoints/best_model.pt --resume_from_epoch 20 --epochs 50
```

### Generate Samples (Standalone)

**Generate 16 samples (4x4 grid) from checkpoint:**
```bash
python generate.py --weight_path ./checkpoints/best_model.pt --sampler ddim --mode samples
```

**Visualize denoising process (5x2 grid):**
```bash
python generate.py --weight_path ./checkpoints/best_model.pt --sampler ddim --mode process --save_path denoising.png
```

**Using Python API:**
```python
from generate import generate_multiple_samples, visualize_denoising_process

# Generate 16 samples with timing
samples = generate_multiple_samples(
    weight_path="./checkpoints/best_model.pt",
    sampler="ddim",
    num_ddim_steps=50,
    save_path="samples.png"
)

# Visualize denoising process
visualize_denoising_process(
    weight_path="./checkpoints/best_model.pt",
    sampler="ddim",
    save_path="denoising.png"
)
```

## Command Line Arguments

### Training Arguments (`main.py`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-4)

### Model Arguments
- `--base_channels`: Base channels for UNet (default: 64)
- `--time_emb_dim`: Time embedding dimension (default: 128)
- `--num_res_blocks`: Number of residual blocks per level (default: 2)

### Diffusion Arguments
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--schedule_type`: Noise schedule type - 'linear' or 'cosine' (default: 'linear')

### DDIM Arguments
- `--ddim_steps`: Number of DDIM sampling steps (default: 50)
- `--eta`: DDIM stochasticity parameter, 0=deterministic (default: 0.0)

### Dataset Arguments
- `--dataset`: Dataset to use - 'fashion_mnist', 'mnist', or 'oxford_flowers' (default: 'fashion_mnist')
- `--image_size`: Image size for Oxford Flowers (default: 64)

### Checkpoint Arguments
- `--checkpoint`: Path to checkpoint to resume from
- `--resume_from_epoch`: Epoch to resume from (overrides checkpoint epoch if specified)
- `--save_dir`: Directory to save checkpoints (default: './checkpoints')

### Generation Arguments (`generate.py`)
- `--weight_path`: Path to checkpoint file (required)
- `--sampler`: 'ddpm' or 'ddim' (default: 'ddim')
- `--ddim_steps`: Number of DDIM steps (default: 50)
- `--mode`: 'process' (denoising visualization) or 'samples' (16 samples) (default: 'samples')
- `--save_path`: Path to save output image
- `--device`: Device to use ('auto', 'cuda', 'cpu', 'mps')

## Architecture

### StandardUNet
- **Encoder-decoder** architecture with residual blocks
- **GroupNorm** and **SiLU** activation (better for diffusion models)
- **Time conditioning** via FiLM (Feature-wise Linear Modulation)
- **Skip connections** between encoder and decoder
- Automatically adjusts channels (1 for grayscale, 3 for RGB) based on dataset

### Sampling Methods

**DDPM**: Stochastic sampling with 1000 steps
- More diverse samples
- Slower generation (~60-70 seconds for 16 samples)

**DDIM**: Deterministic sampling with fewer steps (typically 50)
- Faster generation (~3-4 seconds for 16 samples, ~20x speedup)
- Deterministic (same noise → same image)
- Uses the same trained model as DDPM

## Examples

### Complete Training Workflow

```bash
# 1. Train model
python main.py --dataset fashion_mnist --epochs 50

# 2. Resume training from epoch 20
python main.py --checkpoint ./checkpoints/best_model.pt --resume_from_epoch 20 --epochs 50

# 3. Generate samples
python generate.py --weight_path ./checkpoints/best_model.pt --sampler ddim --mode samples
```

### Custom Image Size

```bash
# Train on Oxford Flowers with 128x128 images
python main.py --dataset oxford_flowers --image_size 128 --batch_size 16 --epochs 50
```

### Sample Only Mode (No Training)

```bash
# Generate samples from existing checkpoint
python main.py --checkpoint ./checkpoints/best_model.pt --sample_only --dataset oxford_flowers --image_size 64
```

### Time Measurement

The `generate_multiple_samples` function automatically measures and reports:
- Total sampling time
- Time per sample
- Time per step

Example output:
```
Generating 16 samples using DDIM...
Sampling completed in 3.45 seconds
  Steps: 50
  Time per sample: 0.216 seconds
  Time per step: 0.0690 seconds
```

## Key Functions

### Training (`main.py`)
- Automatic dataset configuration (channels, image size)
- Checkpoint saving (best model + periodic saves)
- Sample generation every 10 epochs
- Final DDPM vs DDIM speed comparison

### Generation (`generate.py`)
- `load_model_from_checkpoint()`: Automatically infers model configuration from checkpoint
- `visualize_denoising_process()`: Shows 10-step denoising process (5x2 grid)
- `generate_multiple_samples()`: Generates 16 different samples (4x4 grid) with timing

### Utilities (`utils.py`)
- `train_one_epoch()`: Training with progress bar
- `validate_one_epoch()`: Validation
- `generate_samples()`: Sample generation with RGB/Grayscale support
- `visualize_reverse_process()`: Reverse diffusion visualization
- `save_checkpoint()` / `load_checkpoint()`: Checkpoint management

## Data Preprocessing

- **Transform**: `resize(1.5x)` → `centerCrop` for better aspect ratio handling
- **Normalization**: Images normalized to `[-1, 1]` range (standard for diffusion models)
- **Automatic sizing**: FashionMNIST/MNIST (28x28), Oxford Flowers (configurable, default 64x64)

## Notes

- Checkpoints are saved automatically (best model + every 10 epochs)
- Samples are generated every 10 epochs during training
- Visualization functions support both RGB and Grayscale images
- Model automatically adjusts channels based on dataset
- `generate.py` can infer model configuration from checkpoint weights

## License

This project is for educational purposes.
