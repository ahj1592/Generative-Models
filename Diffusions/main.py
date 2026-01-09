"""
main.py - Main entry point for training and sampling with diffusion models

This script demonstrates:
- Training a StandardUNet on FashionMNIST
- Sampling with both DDPM and DDIM
- Comparison of sampling methods
"""

import argparse
import torch
import torch.optim as optim

from models import StandardUNet
from ddpm import DiffusionScheduler, p_sample_loop
from ddim import ddim_sample_loop
from dataset import get_fashion_mnist_loaders, get_mnist_loaders, get_oxford_flowers_loaders
from utils import (
    train_one_epoch, 
    validate_one_epoch, 
    visualize_reverse_process,
    generate_samples,
    save_checkpoint,
    load_checkpoint
)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and sample with diffusion models')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Model arguments
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels for UNet')
    parser.add_argument('--time_emb_dim', type=int, default=128, help='Time embedding dimension')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks per level')
    
    # Diffusion arguments
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--schedule_type', type=str, default='linear', choices=['linear', 'cosine'])
    
    # DDIM arguments
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta parameter (0=deterministic)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='fashion_mnist', 
                       choices=['fashion_mnist', 'mnist', 'oxford_flowers'],
                       help='Dataset to use')
    parser.add_argument('--image_size', type=int, default=None,
                       help='Image size (for Oxford Flowers, default: 64)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/mps/auto)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume_from_epoch', type=int, default=None,
                       help='Epoch to resume from (overrides checkpoint epoch if specified)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--sample_only', action='store_true', help='Only generate samples (no training)')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    
    return parser.parse_args()


def get_device(device_str):
    """Get the appropriate device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def main():
    args = get_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Determine dataset-specific parameters
    if args.dataset == 'oxford_flowers':
        in_channels = 3  # RGB
        out_channels = 3
        image_size = args.image_size if args.image_size else 64
        default_batch_size = 32  # Smaller batch for larger images
    elif args.dataset in ['fashion_mnist', 'mnist']:
        in_channels = 1  # Grayscale
        out_channels = 1
        image_size = 28
        default_batch_size = 128
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Adjust batch size if not specified and using Oxford Flowers
    if args.dataset == 'oxford_flowers' and args.batch_size == 128:
        print(f"Note: Using smaller default batch size for Oxford Flowers")
    
    # Create model
    model = StandardUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4),
        num_res_blocks=args.num_res_blocks,
        time_emb_dim=args.time_emb_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {args.dataset}, Image size: {image_size}x{image_size}, Channels: {in_channels}")
    
    # Create scheduler
    scheduler = DiffusionScheduler(
        timesteps=args.timesteps,
        schedule_type=args.schedule_type
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint_epoch, _ = load_checkpoint(model, optimizer, args.checkpoint, device)
        
        if args.resume_from_epoch is not None:
            if args.resume_from_epoch < checkpoint_epoch:
                print(f"Warning: resume_from_epoch ({args.resume_from_epoch}) < checkpoint epoch ({checkpoint_epoch})")
                print(f"Starting from checkpoint epoch + 1 ({checkpoint_epoch + 1}) instead")
                start_epoch = checkpoint_epoch + 1
            else:
                start_epoch = args.resume_from_epoch
                print(f"Resuming from epoch {start_epoch} (checkpoint was at epoch {checkpoint_epoch})")
        else:
            start_epoch = checkpoint_epoch + 1  # Start from next epoch (default behavior)
            print(f"Starting from epoch {start_epoch} (checkpoint was at epoch {checkpoint_epoch})")
    
    # Sample only mode
    if args.sample_only:
        print("\n=== Generating Samples ===")
        
        shape = (args.num_samples, in_channels, image_size, image_size)
        
        print("\n--- DDPM Sampling ---")
        generate_samples(
            model, scheduler, 
            num_samples=args.num_samples,
            sampler='ddpm',
            shape=shape,
            device=device,
            save_path='samples_ddpm.png'
        )
        
        print(f"\n--- DDIM Sampling ({args.ddim_steps} steps) ---")
        generate_samples(
            model, scheduler,
            num_samples=args.num_samples,
            sampler='ddim',
            num_ddim_steps=args.ddim_steps,
            eta=args.eta,
            shape=shape,
            device=device,
            save_path='samples_ddim.png'
        )
        
        return
    
    # Load data
    print("\n=== Loading Data ===")
    if args.dataset == 'fashion_mnist':
        train_loader, valid_loader, _ = get_fashion_mnist_loaders(
            batch_size=args.batch_size
        )
    elif args.dataset == 'mnist':
        train_loader, valid_loader, _ = get_mnist_loaders(
            batch_size=args.batch_size
        )
    elif args.dataset == 'oxford_flowers':
        train_loader, valid_loader, _ = get_oxford_flowers_loaders(
            batch_size=args.batch_size,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Training loop
    print("\n=== Training ===")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(
            model, scheduler, train_loader, optimizer, device, epoch
        )
        
        # Validate
        valid_loss = validate_one_epoch(
            model, scheduler, valid_loader, device, epoch
        )
        
        # Save checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_checkpoint(
                model, optimizer, epoch, valid_loss,
                f"{args.save_dir}/best_model.pt"
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, valid_loss,
                f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt"
            )
        
        # Visualize samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n--- Generating samples at epoch {epoch + 1} ---")
            
            # DDIM is faster for periodic visualization
            shape = (8, in_channels, image_size, image_size)
            generate_samples(
                model, scheduler,
                num_samples=8,
                sampler='ddim',
                num_ddim_steps=args.ddim_steps,
                shape=shape,
                device=device,
                save_path=f"{args.save_dir}/samples_epoch_{epoch+1}.png"
            )
    
    # Final sampling comparison
    print("\n=== Final Sampling Comparison ===")
    
    test_shape = (4, in_channels, image_size, image_size)
    
    print("\n--- DDPM Sampling (1000 steps) ---")
    import time
    start_time = time.time()
    _ = p_sample_loop(model, scheduler, test_shape, device=device)
    ddpm_time = time.time() - start_time
    print(f"DDPM sampling time: {ddpm_time:.2f}s")
    
    print(f"\n--- DDIM Sampling ({args.ddim_steps} steps) ---")
    start_time = time.time()
    _ = ddim_sample_loop(model, scheduler, test_shape, num_ddim_steps=args.ddim_steps, device=device)
    ddim_time = time.time() - start_time
    print(f"DDIM sampling time: {ddim_time:.2f}s")
    print(f"Speedup: {ddpm_time / ddim_time:.1f}x")
    
    # Generate final samples
    final_shape = (16, in_channels, image_size, image_size)
    generate_samples(
        model, scheduler,
        num_samples=16,
        sampler='ddpm',
        shape=final_shape,
        device=device,
        save_path=f"{args.save_dir}/final_samples_ddpm.png"
    )
    
    generate_samples(
        model, scheduler,
        num_samples=16,
        sampler='ddim',
        num_ddim_steps=args.ddim_steps,
        shape=final_shape,
        device=device,
        save_path=f"{args.save_dir}/final_samples_ddim.png"
    )
    
    # Visualize reverse process
    print("\n--- Visualizing Reverse Process ---")
    visualize_reverse_process(
        model, scheduler,
        shape=(1, in_channels, image_size, image_size),
        sampler='ddim',
        num_ddim_steps=args.ddim_steps,
        save_path=f"{args.save_dir}/reverse_process_ddim.png",
        device=device
    )
    
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

