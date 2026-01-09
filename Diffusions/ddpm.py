"""
ddpm.py - Denoising Diffusion Probabilistic Models

Contains:
- DiffusionScheduler: Manages noise schedules and diffusion process
- p_sample: Single DDPM sampling step (stochastic)
- p_sample_loop: Full DDPM sampling loop
"""

import torch
import torch.nn.functional as F


class DiffusionScheduler:
    """
    Diffusion scheduler for DDPM and DDIM.
    
    Manages the noise schedule and provides utilities for:
    - Forward diffusion (q_sample): Adding noise to clean images
    - Coefficient extraction for reverse process
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        """
        Args:
            timesteps: Total number of diffusion timesteps
            beta_start: Starting value of beta (noise level)
            beta_end: Ending value of beta
            schedule_type: 'linear' or 'cosine'
        """
        self.timesteps = timesteps

        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule (Improved DDPM)
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Coefficients for q_sample (forward diffusion)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Coefficients for p_sample (reverse diffusion - DDPM)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance for DDPM
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t given x_0.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_start: Clean image x_0 [B, C, H, W]
            t: Timestep tensor [B]
            noise: Optional pre-generated noise
        Returns:
            Noisy image x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        """
        Extract coefficients for batch indices.
        
        Args:
            a: 1D tensor of coefficients
            t: Batch of timestep indices [B]
            x_shape: Shape of x for broadcasting
        Returns:
            Extracted coefficients with proper shape for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def get_alphas_cumprod(self, t):
        """Get alpha_bar_t for given timesteps."""
        return self.extract(self.alphas_cumprod, t, (t.shape[0], 1, 1, 1))


@torch.no_grad()
def p_sample(model, scheduler, x, t, t_index):
    """
    Single step of DDPM reverse diffusion (stochastic).
    
    Computes x_{t-1} from x_t using the model's noise prediction.
    
    Args:
        model: Noise prediction model
        scheduler: DiffusionScheduler instance
        x: Current noisy image x_t [B, C, H, W]
        t: Current timestep tensor [B]
        t_index: Current timestep index (scalar)
    Returns:
        Denoised image x_{t-1} [B, C, H, W]
    """
    # Extract coefficients
    betas_t = scheduler.extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler.extract(
        scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = scheduler.extract(
        scheduler.sqrt_recip_alphas, t, x.shape
    )

    # Model predicts noise
    predicted_noise = model(x, t)

    # Compute mean of p(x_{t-1} | x_t) using DDPM formula
    # mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        # No noise at the final step
        return model_mean
    else:
        # Add noise for stochastic sampling
        posterior_variance_t = scheduler.extract(scheduler.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, scheduler, shape, device=None, return_all_steps=False):
    """
    Full DDPM sampling loop: generate images from noise.
    
    Args:
        model: Noise prediction model
        scheduler: DiffusionScheduler instance
        shape: Shape of images to generate (B, C, H, W)
        device: Device to use (defaults to model's device)
        return_all_steps: If True, return all intermediate steps
    Returns:
        Generated images [B, C, H, W] or list of all steps
    """
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]

    # Start from pure noise x_T ~ N(0, I)
    img = torch.randn(shape, device=device)
    
    if return_all_steps:
        imgs = [img.cpu()]

    # Reverse diffusion: T -> T-1 -> ... -> 1 -> 0
    for i in reversed(range(scheduler.timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model, scheduler, img, t, i)
        
        if return_all_steps:
            imgs.append(img.cpu())

    if return_all_steps:
        return imgs
    return img


if __name__ == "__main__":
    # Test the scheduler
    scheduler = DiffusionScheduler(timesteps=1000, schedule_type='linear')
    
    # Test q_sample
    x = torch.randn(2, 1, 28, 28)
    t = torch.randint(0, 1000, (2,))
    
    x_noisy = scheduler.q_sample(x, t)
    print(f"Original shape: {x.shape}")
    print(f"Noisy shape: {x_noisy.shape}")
    print(f"Timesteps: {t}")

