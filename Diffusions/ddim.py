"""
ddim.py - Denoising Diffusion Implicit Models

Contains:
- ddim_sample: Single DDIM sampling step (deterministic)
- ddim_sample_loop: Full DDIM sampling loop with configurable steps

DDIM enables:
- Deterministic sampling (same noise -> same image)
- Faster generation (e.g., 50 steps instead of 1000)
- Uses the same trained model as DDPM
"""

import torch
import numpy as np


@torch.no_grad()
def ddim_sample(model, scheduler, x, t, t_prev, eta=0.0):
    """
    Single step of DDIM reverse diffusion.
    
    DDIM update rule:
    x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - sigma^2) * pred_noise + sigma * noise
    
    When eta=0: Deterministic (no noise added)
    When eta=1: Equivalent to DDPM
    
    Args:
        model: Noise prediction model
        scheduler: DiffusionScheduler instance
        x: Current noisy image x_t [B, C, H, W]
        t: Current timestep tensor [B]
        t_prev: Previous timestep tensor [B] (can be -1 for final step)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)
    Returns:
        Denoised image x_{t-1} [B, C, H, W]
    """
    # Get alpha values
    alpha_cumprod_t = scheduler.extract(scheduler.alphas_cumprod, t, x.shape)
    
    # Handle t_prev = -1 case (final step)
    if t_prev[0].item() >= 0:
        alpha_cumprod_t_prev = scheduler.extract(scheduler.alphas_cumprod, t_prev, x.shape)
    else:
        alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
    
    # Predict noise
    predicted_noise = model(x, t)
    
    # Predict x_0 from x_t and predicted noise
    # x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    
    # Clip predicted x_0 to [-1, 1] for stability
    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
    
    # Compute variance
    # sigma_t = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})
    variance = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
    sigma = eta * torch.sqrt(variance)
    
    # Direction pointing to x_t
    # sqrt(1 - alpha_{t-1} - sigma^2)
    dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma ** 2)
    
    # Compute x_{t-1}
    x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt * predicted_noise
    
    # Add noise if eta > 0
    if eta > 0:
        noise = torch.randn_like(x)
        x_prev = x_prev + sigma * noise
    
    return x_prev


def get_ddim_timesteps(num_ddim_steps, num_train_steps, skip_type='uniform'):
    """
    Get timesteps for DDIM sampling.
    
    Args:
        num_ddim_steps: Number of DDIM sampling steps
        num_train_steps: Number of training timesteps (e.g., 1000)
        skip_type: 'uniform' or 'quad' (quadratic spacing)
    Returns:
        Array of timesteps to use for DDIM sampling
    """
    if skip_type == 'uniform':
        # Uniform spacing
        skip = num_train_steps // num_ddim_steps
        timesteps = np.asarray(list(range(0, num_train_steps, skip)))
    elif skip_type == 'quad':
        # Quadratic spacing (more steps near t=0)
        timesteps = (np.linspace(0, np.sqrt(num_train_steps * 0.8), num_ddim_steps) ** 2).astype(int)
    else:
        raise ValueError(f"Unknown skip type: {skip_type}")
    
    return timesteps


@torch.no_grad()
def ddim_sample_loop(
    model, 
    scheduler, 
    shape, 
    num_ddim_steps=50,
    eta=0.0,
    device=None, 
    return_all_steps=False,
    skip_type='uniform'
):
    """
    Full DDIM sampling loop: generate images from noise with fewer steps.
    
    Args:
        model: Noise prediction model
        scheduler: DiffusionScheduler instance
        shape: Shape of images to generate (B, C, H, W)
        num_ddim_steps: Number of DDIM sampling steps (default: 50)
        eta: Stochasticity (0 = deterministic, 1 = DDPM-like)
        device: Device to use
        return_all_steps: If True, return all intermediate steps
        skip_type: Timestep spacing ('uniform' or 'quad')
    Returns:
        Generated images [B, C, H, W] or list of all steps
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    
    # Get DDIM timesteps (subset of training timesteps)
    timesteps = get_ddim_timesteps(num_ddim_steps, scheduler.timesteps, skip_type)
    timesteps = np.flip(timesteps)  # Reverse: from T to 0
    
    # Start from pure noise x_T ~ N(0, I)
    img = torch.randn(shape, device=device)
    
    if return_all_steps:
        imgs = [img.cpu()]
    
    # DDIM reverse diffusion
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Get previous timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
        else:
            t_prev = -1  # Final step
        t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
        
        # DDIM step
        img = ddim_sample(model, scheduler, img, t_tensor, t_prev_tensor, eta)
        
        if return_all_steps:
            imgs.append(img.cpu())
    
    if return_all_steps:
        return imgs
    return img


if __name__ == "__main__":
    from ddpm import DiffusionScheduler
    
    # Test timestep generation
    timesteps_50 = get_ddim_timesteps(50, 1000, 'uniform')
    timesteps_quad = get_ddim_timesteps(50, 1000, 'quad')
    
    print(f"Uniform timesteps (50 steps): {timesteps_50[:10]}...")
    print(f"Quadratic timesteps (50 steps): {timesteps_quad[:10]}...")
    
    # Test scheduler compatibility
    scheduler = DiffusionScheduler(timesteps=1000)
    print(f"Scheduler timesteps: {scheduler.timesteps}")

