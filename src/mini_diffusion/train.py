import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset,DataLoader
import matplotlib.pyplot as plt


from mini_diffusion.config import load_config
from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion


def train():
    
    # constants
    config = load_config("./config/base.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 2
    batch_size = 10

    print(f"Using device: {device}")

    unet = UNet(in_channels=3).to(device)
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
    # DataLoader
    dataset = datasets.Caltech101(root="./data",transform=trans,download=True)
    airplane_indices = [i for i, (_, label) in enumerate(dataset) if label == 5] #type:ignore
    airplane_dataset = Subset(dataset, airplane_indices)

    dataloader = iter(DataLoader(
        airplane_dataset,
        batch_size,
        shuffle=True,
        num_workers=2,   
    ))


    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # axes = axes.flatten()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3)

    losses = []

    for epoch in range(epochs):
        for i,batch in enumerate(dataloader):
            optimizer.zero_grad()
            x,_ = batch[0].to(device),batch[1]
            
            timestamp = diffusion.sample_time_stamp(batch_size)
            
            x_t,eps = diffusion.add_noise(x,timestamp)
            
            predict_eps = unet(x,timestamp)
            
            loss = loss_fn(eps,predict_eps)
            
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            if i%10 == 0:
                print(f"Loss: {sum(losses[-10:])/10}")

    torch.save(unet,f="./saves/a.pth")
    
    
if __name__ == "__mian__":
    train()