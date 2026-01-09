"""
utils.py - Utility functions for diffusion model training and evaluation

Contains:
- train_one_epoch: Training function
- validate_one_epoch: Validation function
- visualize_reverse_process: Visualization for both DDPM and DDIM
- save_checkpoint: Save model checkpoint
- load_checkpoint: Load model checkpoint
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, scheduler, dataloader, optimizer, device, epoch, log_interval=100):
    """
    Train the model for one epoch.
    
    Args:
        model: UNet model
        scheduler: DiffusionScheduler
        dataloader: Training DataLoader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        log_interval: How often to log (in steps)
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()

        x_0 = batch[0].to(device)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()

        # Generate noise and create noisy images
        noise = torch.randn_like(x_0)
        x_noisy = scheduler.q_sample(x_start=x_0, t=t, noise=noise)

        # Predict noise
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)

        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if step % log_interval == 0 and step > 0:
            avg_loss = total_loss / (step + 1)
            tqdm.write(f"Epoch [{epoch}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f">>> Epoch {epoch} Average Train Loss: {avg_loss:.4f} <<<\n")
    return avg_loss


def validate_one_epoch(model, scheduler, dataloader, device, epoch):
    """
    Validate the model for one epoch.
    
    Args:
        model: UNet model
        scheduler: DiffusionScheduler
        dataloader: Validation DataLoader
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0

    print(f"--- Epoch {epoch} Validation ---")
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")):
            x_0 = batch[0].to(device)
            batch_size = x_0.shape[0]

            t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0)
            x_noisy = scheduler.q_sample(x_start=x_0, t=t, noise=noise)

            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(noise, predicted_noise)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f">>> Epoch {epoch} Average Validation Loss: {avg_loss:.4f} <<<\n")
    return avg_loss


@torch.no_grad()
def visualize_reverse_process(
    model, 
    scheduler, 
    shape=(1, 1, 28, 28),
    sampler='ddpm',
    num_ddim_steps=50,
    save_path=None,
    device=None
):
    """
    Visualize the reverse diffusion process.
    
    Args:
        model: UNet model
        scheduler: DiffusionScheduler
        shape: Shape of images to generate (B, C, H, W)
        sampler: 'ddpm' or 'ddim'
        num_ddim_steps: Number of DDIM steps (only used if sampler='ddim')
        save_path: Path to save the figure (optional)
        device: Device to use
    """
    from ddpm import p_sample_loop
    from ddim import ddim_sample_loop
    
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Generate samples with all intermediate steps
    if sampler == 'ddpm':
        samples = p_sample_loop(model, scheduler, shape, device=device, return_all_steps=True)
        title = f"DDPM Reverse Process ({scheduler.timesteps} steps)"
    else:  # ddim
        samples = ddim_sample_loop(
            model, scheduler, shape, 
            num_ddim_steps=num_ddim_steps, 
            eta=0.0, 
            device=device, 
            return_all_steps=True
        )
        title = f"DDIM Reverse Process ({num_ddim_steps} steps)"
    
    total_steps = len(samples)
    
    # Select 10 evenly spaced steps to display
    indices = np.linspace(0, total_steps - 1, 10, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]
        
        # Get the first image in the batch
        img = samples[idx][0].squeeze().cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        # Handle different channel counts
        if len(img.shape) == 3 and img.shape[0] == 3:
            # RGB image: transpose from (C, H, W) to (H, W, C)
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        elif len(img.shape) == 2:
            # Grayscale image: already 2D
            ax.imshow(img, cmap='gray')
        else:
            # Single channel: squeeze to 2D
            img = img.squeeze()
            ax.imshow(img, cmap='gray')
        
        # Calculate actual step number
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
        print(f"Saved visualization to {save_path}")
    
    plt.clf()
    plt.close()


@torch.no_grad()
def generate_samples(
    model, 
    scheduler, 
    num_samples=16,
    sampler='ddim',
    num_ddim_steps=50,
    eta=0.0,
    shape=None,
    device=None,
    save_path=None
):
    """
    Generate and display sample images.
    
    Args:
        model: UNet model
        scheduler: DiffusionScheduler
        num_samples: Number of samples to generate
        sampler: 'ddpm' or 'ddim'
        num_ddim_steps: Number of DDIM steps
        eta: DDIM stochasticity parameter
        shape: Shape of images (B, C, H, W). If None, infers from model or uses default
        device: Device to use
        save_path: Path to save the figure
    
    Returns:
        Generated images tensor [num_samples, C, H, W]
    """
    from ddpm import p_sample_loop
    from ddim import ddim_sample_loop
    
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Infer shape if not provided
    if shape is None:
        # Try to infer from model's expected input
        # Default to FashionMNIST shape if can't infer
        in_channels = model.in_channels if hasattr(model, 'in_channels') else 1
        shape = (num_samples, in_channels, 28, 28)
    else:
        # Ensure batch size matches num_samples
        shape = (num_samples,) + shape[1:]
    
    print(f"Generating {num_samples} samples using {sampler.upper()}...")
    
    if sampler == 'ddpm':
        samples = p_sample_loop(model, scheduler, shape, device=device)
    else:
        samples = ddim_sample_loop(
            model, scheduler, shape,
            num_ddim_steps=num_ddim_steps,
            eta=eta,
            device=device
        )
    
    # Convert to displayable format
    samples = samples.cpu()
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)
    
    # Display in a grid
    nrows = int(np.sqrt(num_samples))
    ncols = (num_samples + nrows - 1) // nrows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < num_samples:
            img = samples[i].squeeze().numpy()
            
            # Handle different channel counts
            if len(img.shape) == 3 and img.shape[0] == 3:
                # RGB image: transpose from (C, H, W) to (H, W, C)
                img = img.transpose(1, 2, 0)
                ax.imshow(img)
            elif len(img.shape) == 2:
                # Grayscale image: already 2D
                ax.imshow(img, cmap='gray')
            else:
                # Single channel: squeeze to 2D
                img = img.squeeze()
                ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    title = f"Generated Samples ({sampler.upper()}"
    if sampler == 'ddim':
        title += f", {num_ddim_steps} steps, eta={eta}"
    title += ")"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples to {save_path}")
    
    plt.clf()
    plt.close()
    
    return samples


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        path: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into (can be None)
        path: Path to the checkpoint
        device: Device to load the model to
    
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    
    return epoch, loss


if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("Available functions:")
    print("  - train_one_epoch(model, scheduler, dataloader, optimizer, device, epoch)")
    print("  - validate_one_epoch(model, scheduler, dataloader, device, epoch)")
    print("  - visualize_reverse_process(model, scheduler, shape, sampler, num_ddim_steps)")
    print("  - generate_samples(model, scheduler, num_samples, sampler)")
    print("  - save_checkpoint(model, optimizer, epoch, loss, path)")
    print("  - load_checkpoint(model, optimizer, path, device)")

