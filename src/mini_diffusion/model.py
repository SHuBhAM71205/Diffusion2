import torch
import torch.nn as nn
import math

from mini_diffusion.config import load_config

config_path = "./configs/base.yaml"

config = load_config(config_path)




class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(),
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.conv(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[:, :, None, None]
        return h + time_emb


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        x = self.block(x, t)
        return self.pool(x), x

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        time_dim = base_channels * 4
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)

        self.init_conv = ConvBlock(in_channels, base_channels, time_dim)

        self.down1 = Down(base_channels, base_channels * 2, time_dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim)

        self.bottleneck = ConvBlock(
            base_channels * 4, base_channels * 4, time_dim)

        self.up1 = Up(256, 256, 128, time_dim)
        self.up2 = Up(128, 128, 64, time_dim)


        self.final_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t = self.time_embedding(t)

        x = self.init_conv(x, t)
        
        x1, skip1 = self.down1(x, t)
        
        x2, skip2 = self.down2(x1, t)
        
        x = self.bottleneck(x2, t)
        
        x = self.up1(x, skip2, t)
        
        x = self.up2(x, skip1, t)
        

        return self.final_conv(x)
