"""
generate.py - Standalone generation functions for diffusion models

Functions:
- visualize_denoising_process: Show denoising from pure noise to final image (5x2 grid)
- generate_multiple_samples: Generate 16 different samples from different noises (4x4 grid)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from models import StandardUNet
from ddpm import DiffusionScheduler, p_sample_loop
from ddim import ddim_sample_loop


def load_model_from_checkpoint(weight_path, device='auto'):
    """
    Load model from checkpoint and infer configuration.
    
    Args:
        weight_path: Path to checkpoint file
        device: Device to load model on ('auto', 'cuda', 'cpu', 'mps')
    
    Returns:
        model: Loaded model
        scheduler: DiffusionScheduler
        config: Dictionary with model configuration
    """
    # Auto device selection
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)
    
    # Infer model configuration from checkpoint weights
    model_state = checkpoint['model_state_dict']
    
    # Infer in_channels from conv_in.weight shape: [out_ch, in_ch, k, k]
    conv_in_weight = model_state['conv_in.weight']
    in_channels = conv_in_weight.shape[1]
    
    # Infer out_channels from conv_out.weight shape: [out_ch, in_ch, k, k]
    conv_out_weight = model_state['conv_out.weight']
    out_channels = conv_out_weight.shape[0]
    
    # Infer image size from the model (estimate based on channel pattern)
    # For FashionMNIST: 28x28, for Oxford Flowers: typically 64x64
    if in_channels == 1:
        image_size = 28  # Grayscale (MNIST/FashionMNIST)
    else:
        image_size = 64  # RGB (Oxford Flowers default)
    
    # Check if config is saved in checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        image_size = config.get('image_size', image_size)
    
    config = {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'image_size': image_size,
        'device': device
    }
    
    print(f"Detected configuration: channels={in_channels}, image_size={image_size}x{image_size}")
    
    # Create model with inferred configuration
    model = StandardUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=128
    ).to(device)
    
    # Load weights
    model.load_state_dict(model_state)
    model.eval()
    
    # Create scheduler
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='linear')
    
    print(f"Model loaded from {weight_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, scheduler, config


def visualize_denoising_process(weight_path, sampler='ddim', num_ddim_steps=50, save_path=None, device='auto'):
    """
    Visualize the denoising process from pure noise to final image.
    
    Shows 10 images in a 5x2 grid: from pure noise (step T) to clean image (step 0).
    
    Args:
        weight_path: Path to checkpoint file
        sampler: 'ddpm' or 'ddim'
        num_ddim_steps: Number of DDIM steps (only used if sampler='ddim')
        save_path: Path to save the figure (optional)
        device: Device to use
    
    Returns:
        Final generated image tensor
    """
    # Load model
    model, scheduler, config = load_model_from_checkpoint(weight_path, device)
    device = config['device']
    
    # Shape for single image
    shape = (1, config['in_channels'], config['image_size'], config['image_size'])
    
    print(f"Generating denoising process using {sampler.upper()}...")
    
    # Generate samples with all intermediate steps
    with torch.no_grad():
        if sampler == 'ddpm':
            samples = p_sample_loop(model, scheduler, shape, device=device, return_all_steps=True)
            title = f"DDPM Denoising Process ({scheduler.timesteps} steps)"
        else:  # ddim
            samples = ddim_sample_loop(
                model, scheduler, shape,
                num_ddim_steps=num_ddim_steps,
                eta=0.0,
                device=device,
                return_all_steps=True
            )
            title = f"DDIM Denoising Process ({num_ddim_steps} steps)"
    
    total_steps = len(samples)
    
    # Select 10 evenly spaced steps to display
    indices = np.linspace(0, total_steps - 1, 10, dtype=int)
    
    # Create 5x2 grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]
        
        # Get the image
        img = samples[idx][0].squeeze().cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        # Handle different channel counts
        if len(img.shape) == 3 and img.shape[0] == 3:
            # RGB image: transpose from (C, H, W) to (H, W, C)
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        else:
            # Grayscale image
            if len(img.shape) == 3:
                img = img.squeeze()
            ax.imshow(img, cmap='gray')
        
        # Calculate step number (noise level)
        if sampler == 'ddpm':
            step_num = scheduler.timesteps - idx
        else:
            step_num = total_steps - 1 - idx
        
        ax.set_title(f"Step {step_num}")
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Return final image
    return samples[-1]


def generate_multiple_samples(weight_path, sampler='ddim', num_ddim_steps=50, save_path=None, device='auto'):
    """
    Generate 16 different samples from different random noises.
    
    Shows 16 images in a 4x4 grid with sampling time measurement.
    
    Args:
        weight_path: Path to checkpoint file
        sampler: 'ddpm' or 'ddim'
        num_ddim_steps: Number of DDIM steps (only used if sampler='ddim')
        save_path: Path to save the figure (optional)
        device: Device to use
    
    Returns:
        tuple: (samples tensor [16, C, H, W], elapsed_time in seconds)
    """
    # Load model
    model, scheduler, config = load_model_from_checkpoint(weight_path, device)
    device = config['device']
    
    # Shape for 16 images
    num_samples = 16
    shape = (num_samples, config['in_channels'], config['image_size'], config['image_size'])
    
    print(f"Generating {num_samples} samples using {sampler.upper()}...")
    
    # Generate samples with time measurement
    start_time = time.time()
    
    with torch.no_grad():
        if sampler == 'ddpm':
            samples = p_sample_loop(model, scheduler, shape, device=device)
        else:  # ddim
            samples = ddim_sample_loop(
                model, scheduler, shape,
                num_ddim_steps=num_ddim_steps,
                eta=0.0,
                device=device
            )
    
    elapsed_time = time.time() - start_time
    
    # Print timing info
    if sampler == 'ddpm':
        steps = scheduler.timesteps
    else:
        steps = num_ddim_steps
    
    print(f"Sampling completed in {elapsed_time:.2f} seconds")
    print(f"  Steps: {steps}")
    print(f"  Time per sample: {elapsed_time / num_samples:.3f} seconds")
    print(f"  Time per step: {elapsed_time / steps:.4f} seconds")
    
    # Convert to displayable format
    samples = samples.cpu()
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)
    
    # Create 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flatten()):
        img = samples[i].squeeze().numpy()
        
        # Handle different channel counts
        if len(img.shape) == 3 and img.shape[0] == 3:
            # RGB image: transpose from (C, H, W) to (H, W, C)
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        else:
            # Grayscale image
            if len(img.shape) == 3:
                img = img.squeeze()
            ax.imshow(img, cmap='gray')
        
        ax.axis('off')
    
    title = f"Generated Samples ({sampler.upper()}"
    if sampler == 'ddim':
        title += f", {num_ddim_steps} steps"
    title += ")"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate samples from trained diffusion model')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'], help='Sampling method')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--mode', type=str, default='samples', choices=['process', 'samples'], 
                       help='process: show denoising, samples: generate 16 samples')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save output')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    
    args = parser.parse_args()
    
    if args.mode == 'process':
        visualize_denoising_process(
            weight_path=args.weight_path,
            sampler=args.sampler,
            num_ddim_steps=args.ddim_steps,
            save_path=args.save_path,
            device=args.device
        )
    else:
        generate_multiple_samples(
            weight_path=args.weight_path,
            sampler=args.sampler,
            num_ddim_steps=args.ddim_steps,
            save_path=args.save_path,
            device=args.device
        )

