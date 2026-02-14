import torch


from mini_diffusion.config import load_config
from mini_diffusion.model import UNet


config = load_config("./config/base.yaml")


model = UNet()

x = torch.randn(4, 1, 28, 28)
t = torch.randint(0, 1000, (4,))

out = model(x, t)
print(out.shape)