import torch
from torchvision.transforms import v2
import math
import torch.nn as nn
from mini_diffusion.config import load_config, Config
from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def sample(): 
    model_path="./saves/a.pth"

    config = load_config("./configs/base.yaml")

    config = load_config("./config/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    unet = UNet(in_channels=3)
    try:
        chkpt = torch.load(model_path, map_location=device)
        
        if isinstance(chkpt,nn.Module):
            unet = chkpt.to(device)
        elif isinstance(chkpt, dict):
            unet.load_state_dict(chkpt)
            unet.to(device)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(chkpt)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        unet.to(device)
    
    diffusion = Diffusion(config=config, device=device)

    trans = v2.Compose(
        [
            v2.ToImage(),
            v2.Lambda(lambda x: x[:3] if x.shape[0] == 3 else x[:3].repeat(3//x.shape[0],1,1)),
            v2.Resize((256,256)),
            v2.CenterCrop(256),
            v2.ToDtype(torch.float32,scale=True),
            v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]
    )


    #inference loop::>

    unet.eval()
    with torch.inference_mode() , torch.no_grad():
        x_t = torch.randn(size=(1,3,256,256)).to(device)
        for i in reversed(range(config.diffusion.timesteps)):
            x_t = 1/math.sqrt(diffusion.alpha[i]) * (x_t - ((1-diffusion.alpha[i])/torch.sqrt(1-diffusion.alpha_hat[i]))* unet(x_t,torch.Tensor([i,]).to(device))) #type:ignore
            
        img = x_t[0].permute(1,2,0).cpu().numpy()

        img = np.clip(img , -1.0,1.0)/2.0
        img = (img * 255.0).astype(np.uint8)
        
        pil_image = Image.fromarray(img)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        
        return buf.getvalue()
        
if __name__ == "__main__":
    sample()