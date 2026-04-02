import torch
import math

import argparse
from mini_diffusion.config import load_config, Config
from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion
from mini_diffusion.device import resolve_device

import numpy as np
from PIL import Image
import io

from Logger.logger import setup_logger

# import matplotlib.pyplot as plt


def _format_step_stats(
    step: int,
    x_t: torch.Tensor,
    eps: torch.Tensor,
    x0_hat: torch.Tensor,
) -> str:
    x_std = x_t.std().item()
    eps_std = eps.std().item()
    x0_std = x0_hat.std().item()
    eps_to_state = eps_std / max(x_std, 1e-8)
    return (
        f"step={step:4d} "
        f"x_mean={x_t.mean().item():8.4f} x_std={x_std:8.4f} "
        f"eps_std={eps_std:8.4f} eps/x={eps_to_state:8.5f} "
        f"x0_mean={x0_hat.mean().item():8.4f} x0_std={x0_std:8.4f}"
    )


def sample(config:Config| None = None):

    if config is None:
        config = load_config("./configs/base.yaml")
        
    model_path=config.inference.model_path
    device = resolve_device(config.inference.device, strict=True)
    
    
    img_dim = config.model.im_size
    logger = setup_logger(config.inference.logs)
    print(f"Using device: {device}")

    unet = UNet(config.model)
    try:
        chkpt = torch.load(model_path, map_location=device)

        state_dict_key = "unet" if "unet" in chkpt else "ema_unet"
        print(f"state_dct_key{state_dict_key}")
        unet.load_state_dict(chkpt[state_dict_key])
        unet = unet.to(device)
        logger.info(f"Loaded checkpoint from {model_path} (weights: {state_dict_key})")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint at {model_path}. "
            "Sampling with random weights will not denoise correctly."
        ) from e
    
    diffusion = Diffusion(config=config, device=device)
    
    cnt = 0
    
    for parameter in unet.parameters():
        cnt += parameter.numel()
        
    print(f"Total parameters in UNET: {cnt}")
    alpha_hat_from_alpha = torch.cumprod(diffusion.alpha, dim=0)
    schedule_consistency = (alpha_hat_from_alpha - diffusion.alpha_hat).abs().max().item()
    logger.info(
        f"Schedule consistency max|cumprod(alpha)-alpha_hat|={schedule_consistency:.6e}"
    )


    unet.eval()
    with torch.inference_mode() , torch.no_grad():
        x_t = torch.randn(size=(1,3,img_dim,img_dim)).to(device)
        
        # n_plots = 10 
        # steps_to_plot = torch.linspace(config.diffusion.timesteps - 1, 0, n_plots, dtype=torch.long)
        
        # cols = 5
        # rows = math.ceil(n_plots / cols)
        # fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        # axes = axes.flatten()
        # plot_idx = 0
        
        for i in reversed(range(config.diffusion.timesteps)):
            t = torch.full((x_t.size(0),), i, device=device, dtype=torch.long)

            v = unet(x_t, t)
            alpha_t = diffusion.alpha[i]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            eps = sqrt_alpha_t * v + sqrt_one_minus_alpha_t * x_t
            
            alpha_t = diffusion.alpha[i]
            alpha_hat_t = diffusion.alpha_hat[i]
            beta_t = diffusion.beta[i]

            # compute coefficient
            coef = beta_t / torch.sqrt(1 - alpha_hat_t)

            # mean of reverse distribution
            mean = (x_t - coef * eps) / torch.sqrt(alpha_t)

            if i > 0:
                alpha_hat_prev = diffusion.alpha_hat[i-1]

                posterior_var = beta_t * (1 - alpha_hat_prev) / (1 - alpha_hat_t)

                noise = torch.randn_like(x_t)

                x_t = mean + torch.sqrt(posterior_var) * noise
            else:
                x_t = (x_t - torch.sqrt(1 - alpha_hat_t) * eps) / torch.sqrt(alpha_hat_t)

            x0_hat = (x_t - torch.sqrt(1 - alpha_hat_t) * eps) / torch.sqrt(alpha_hat_t)
            if i % 100 == 0 or i < 10:
                logger.info(_format_step_stats(i, x_t, eps, x0_hat))
            
            
            # if i in steps_to_plot:
            #     img = x_t[0].permute(1,2,0).cpu().numpy()
        
            #     img = (img * 0.5 ) + 0.5
                
            #     img = (img * 255).astype(np.uint8)    
            #     ax = axes[plot_idx]
            #     ax.imshow(img)
            #     ax.set_title(f"T = {i}")
            #     ax.axis('off')
            #     plot_idx += 1
                
            # print(
            #     f"Step {i}: mean={x_t.mean():.4f}, std={x_t.std():.4f} "
            #     f"min={x_t.min()} max = {x_t.max()}"
            # )

        img = x_t[0].permute(1, 2, 0).cpu().numpy()

        logger.info(f"{img.shape} {img.mean()} {img.std()}")
        
        # plt.show()
        img = x_t[0].permute(1,2,0).cpu().numpy()
        
        img = (img * 0.5 ) + 0.5
        
        img = (img * 255).astype(np.uint8)
        logger.info(f"{img.shape} {img.mean()} {img.std()}")
        pil_image = Image.fromarray(img)

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        
        return buf.getvalue()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--config", type=str, default="./configs/base.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    sample(config)
    
