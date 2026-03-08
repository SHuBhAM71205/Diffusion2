import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transform
from mini_diffusion.config import Config, load_config, TrainingConfig

from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion

import argparse

import sys
print(sys.path)
sys.path.append('/home/shubham/Desktop/Diffusion2')
print(sys.path)
from torch.utils.data import DataLoader

from Dataset.plane import Plane

parser = argparse.ArgumentParser(description="config")

path_group = parser.add_argument_group("Paths")
path_group.add_argument("--config", type=str, default="./config/base.yaml", help="Path to config file")
args = parser.parse_args()

config = load_config(args.config)


train_cfg: TrainingConfig = config.training
    # seeing up the config constants
device = train_cfg.device if torch.cuda.is_available() else "cpu"
epochs = train_cfg.epochs
batch_size = train_cfg.batch_size

# init models
unet = UNet(in_channels=config.model.in_channels).to(device)
ema_unet = UNet(in_channels=config.model.in_channels).to(device)
ema_unet.eval()
diffusion = Diffusion(config=config, device=device)
# init transform
trans = v2.Compose(
    [
        
        v2.RandomHorizontalFlip(),v2.RandomRotation([-10,10]),v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        v2.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)), # type: ignore
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)


dataset = Plane(
                    train_cfg.data_dir,
                    channel=3,
                    width=config.model.image_size,
                    height=config.model.image_size,
                    transform=trans
                )

    
dataloader = DataLoader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=train_cfg.num_workers,
    pin_memory=True,
    drop_last=True
)


try:
    chkpt_dict = torch.load(
        f"{train_cfg.save_path}/a.pth")
    unet.load_state_dict(chkpt_dict["unet"])
    ema_unet.load_state_dict(chkpt_dict["ema_unet"])
    
except Exception as e:
    print(f"Error loading model: {e} \n")    


for batch in dataloader:
    
    x0 = batch[0]
    x0 = x0[None, :, :, :]
    x0=x0.to(device)
    print(x0.shape)
    print(x0.min(), x0.max())

    x_t,eps = diffusion.add_noise(x0, t=torch.tensor([900,],device=device))
    
    # print(x_t.mean(), x_t.std())
    
    # print((torch.sqrt(diffusion.alpha_hat[900]) * x0).mean())
    
    # print((1-diffusion.alpha_hat[900]) + diffusion.alpha_hat[900] * x_t.var())
    
    
    # UNET TEST
    
    eps1 = unet(x_t, torch.tensor([900]).cuda())
    
    print(eps1.mean(), eps1.std())
    print((eps1- x_t).abs().mean())
    break