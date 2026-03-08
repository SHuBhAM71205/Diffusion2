import torch
from torchvision.transforms import v2
import math
import torch.nn as nn

import argparse
from mini_diffusion.config import load_config, Config
from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion

import numpy as np
from PIL import Image
import io

from Logger.logger import setup_logger

def sample(config:Config| None = None):

    if config is None:
        config = load_config("./configs/base.yaml")
        
    model_path=config.inference.model_path
    device = config.inference.device if torch.cuda.is_available() else "cpu"
    img_dim = config.model.image_size
    logger = setup_logger(config.inference.logs)
    print(f"Using device: {device}")

    unet = UNet(in_channels=config.model.in_channels,base_channels=config.model.base_channels)
    try:
        chkpt = torch.load(model_path)
        
        if isinstance(chkpt,nn.Module):
            unet = chkpt.to(device)
        elif isinstance(chkpt, dict):
            unet.load_state_dict(chkpt["ema_unet"])
            unet.to(device)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(chkpt)}")
        
    except Exception as e:
        print(f"Error loading model: {e} \n")
        unet.to(device)
    
    diffusion = Diffusion(config=config, device=device)

    trans = v2.Compose(
        [
            v2.ToImage(),
            v2.Lambda(lambda x: x[:3] if x.shape[0] == 3 else x[:3].repeat(3//x.shape[0],1,1)),
            v2.Resize((img_dim,img_dim)),
            v2.CenterCrop(img_dim),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]
    )


    #inference loop::>

    unet.eval()
    with torch.inference_mode() , torch.no_grad():
        x_t = torch.randn(size=(1,3,img_dim,img_dim)).to(device)
        for i in reversed(range(config.diffusion.timesteps)):
            t = torch.full((x_t.size(0),), i, device=device, dtype=torch.long)

            eps = unet(x_t, t)
            # print("eps std:", eps.std().item())
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
                x_t = mean
            
            print(f"Step {i}: mean={x_t.mean():.4f}, std={x_t.std():.4f}")

        img = x_t[0].permute(1, 2, 0).cpu().numpy()

        logger.info(f"{img.shape} {img.mean()} {img.std()}")

        # convert to image space
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
    path_group.add_argument("--config", type=str, default="./config/base.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    sample(config)
    
