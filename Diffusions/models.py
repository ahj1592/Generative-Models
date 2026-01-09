"""
models.py - Neural network architectures for diffusion models

Contains:
- SinusoidalPositionEmbeddings: Time embedding module
- ResidualBlock: ResNet-style block with time conditioning
- Downsample/Upsample: Resolution changing modules
- StandardUNet: Full UNet architecture with residual blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.
    Transforms scalar timesteps into high-dimensional embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning using FiLM (Feature-wise Linear Modulation).
    
    Structure:
    Input -> Conv -> GroupNorm -> SiLU -> Time Embedding -> Conv -> GroupNorm -> SiLU -> + Input -> Output
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
        super().__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        
        # Time embedding projection (FiLM: scale and shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)  # *2 for scale and shift
        )
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        
        # Residual connection (1x1 conv if channel dimensions differ)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        
        self.activation = nn.SiLU()

    def forward(self, x, t_emb):
        """
        Args:
            x: Input tensor [B, C, H, W]
            t_emb: Time embedding [B, time_emb_dim]
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        residual = self.residual_conv(x)
        
        # First block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)
        
        # Time conditioning via FiLM (Feature-wise Linear Modulation)
        time_emb = self.time_mlp(t_emb)
        time_emb = time_emb[:, :, None, None]  # [B, C*2, 1, 1]
        scale, shift = time_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        # Second block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + residual


class Downsample(nn.Module):
    """Downsampling layer using strided convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer using nearest neighbor interpolation + convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class StandardUNet(nn.Module):
    """
    Standard UNet architecture with residual blocks for diffusion models.
    
    Architecture:
    - Encoder: Downsampling path with residual blocks
    - Bottleneck: Residual blocks at lowest resolution
    - Decoder: Upsampling path with skip connections from encoder
    - Time embeddings integrated at each residual block
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=128,
        num_groups=8
    ):
        """
        Args:
            in_channels: Number of input image channels
            out_channels: Number of output image channels
            base_channels: Base number of channels (multiplied by channel_mults)
            channel_mults: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per resolution level
            time_emb_dim: Dimension of time embeddings
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels]
        current_channels = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Residual blocks at this level
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(current_channels, out_ch, time_emb_dim, num_groups)
                )
                current_channels = out_ch
                channels.append(current_channels)
            
            # Downsample (except at the last level)
            if level < len(channel_mults) - 1:
                self.downsample_blocks.append(Downsample(current_channels))
                channels.append(current_channels)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(current_channels, current_channels, time_emb_dim, num_groups),
            ResidualBlock(current_channels, current_channels, time_emb_dim, num_groups)
        ])
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            # Residual blocks at this level (with skip connection concatenation)
            for i in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.decoder_blocks.append(
                    ResidualBlock(current_channels + skip_channels, out_ch, time_emb_dim, num_groups)
                )
                current_channels = out_ch
            
            # Upsample (except at the last level)
            if level < len(channel_mults) - 1:
                self.upsample_blocks.append(Upsample(current_channels))
        
        # Output layers
        self.norm_out = nn.GroupNorm(num_groups, current_channels)
        self.conv_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timestep):
        """
        Args:
            x: Input tensor [B, C, H, W]
            timestep: Timestep tensor [B]
        Returns:
            Predicted noise tensor [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(timestep)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Encoder with skip connections
        skip_connections = [h]
        
        block_idx = 0
        for level in range(len(self.channel_mults)):
            # Residual blocks
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, t_emb)
                skip_connections.append(h)
                block_idx += 1
            
            # Downsample
            if level < len(self.channel_mults) - 1:
                h = self.downsample_blocks[level](h)
                skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)
        
        # Decoder with skip connections
        block_idx = 0
        for level in range(len(self.channel_mults)):
            # Residual blocks with skip connections
            for _ in range(self.num_res_blocks + 1):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            # Upsample
            if level < len(self.channel_mults) - 1:
                h = self.upsample_blocks[level](h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = StandardUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        time_emb_dim=128
    ).to(device)
    
    # Test forward pass
    x = torch.randn(2, 1, 28, 28).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

