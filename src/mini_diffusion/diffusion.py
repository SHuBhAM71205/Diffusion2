import torch
import torch.nn as nn
from mini_diffusion.config import load_config, Config


class Diffusion(nn.Module):

    def __init__(self, config: Config, device="cpu"):

        super().__init__()
        self.config = config.diffusion
        self.device = torch.device(device)
        
        # beta = torch.linspace(
        #     self.config.beta_start,
        #     self.config.beta_end,
        #     self.config.timesteps
        # ).to(device)


        s = 0.008  # must be calculateds just for testing

        tbyT = torch.arange(start=0, end=self.config.timesteps, step=1, device=device) / self.config.timesteps
        t = (tbyT + s )/ (1+s)
        t = t * (torch.pi/2)

        cosine = torch.cos(t) ** 2

        alpha_hat = cosine / cosine[0]
        
        beta = 1 - (alpha_hat[1:] /alpha_hat[:-1])
        
        beta = torch.cat([torch.tensor([0.02], device=device),beta])
        beta = beta.clamp(0.0001, 0.9999)
        
        
        
        beta = beta.to(device)
        
        alpha = 1-beta
        
        
        
        # alpha_hat = torch.cumprod(alpha, dim=0).to(device)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

    def sample_time_stamp(self, batch_size):

        return torch.randint(0, self.config.timesteps, size=(batch_size,), device=self.device)

    def add_noise(self, x: torch.Tensor, t):
        # shape if x is = B,Color,T,T
        # shape if t = B,1
        B, C, H, W = x.shape

        # assert self.device == x.device , f"The device of the x is {x.device} and the Model is {self.device} which is different different"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]  # type: ignore

        one_minus_alpha = torch.sqrt(
            1-self.alpha_hat[t])[:, None, None, None]  # type: ignore

        eps = torch.randn_like(x)

        return x * sqrt_alpha_hat + one_minus_alpha * eps, eps
