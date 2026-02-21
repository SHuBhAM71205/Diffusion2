import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset, DataLoader

from mini_diffusion.config import Config, load_config, TrainingConfig

from mini_diffusion.model import UNet
from mini_diffusion.diffusion import Diffusion

import argparse

from Logger.logger import setup_logger


def train(config: Config):

    # loading train_config
    train_cfg: TrainingConfig = config.training
    # seeing up the config constants
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    epochs = train_cfg.epochs
    batch_size = train_cfg.batch_size
    logger = setup_logger(train_cfg.logs)

    logger.info(f"Config used:\n{train_cfg}")

    logger.info(f"Using device: {device}")
    # init models
    unet = UNet(in_channels=3).to(device)

    diffusion = Diffusion(config=config, device=device)
    # init transform
    trans = v2.Compose(
        [
            v2.ToImage(),
            v2.Lambda(lambda x: x[:3] if x.shape[0] ==
                      3 else x[:3].repeat(3//x.shape[0], 1, 1)),
            v2.Resize((256, 256)),
            v2.CenterCrop(256),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    # DataLoader
    dataset = datasets.Caltech101(
        root=f"{train_cfg.data_dir}", transform=trans, download=True)
    airplane_indices = [i for i, (_, label) in enumerate(
        dataset) if label == 5]  # type:ignore
    airplane_dataset = Subset(dataset, airplane_indices)

    dataloader = DataLoader(
        airplane_dataset,
        batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # axes = axes.flatten()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=train_cfg.learning_rate)
    epoch_start=0
    try:
        chkpt_dict = torch.load(
            f"{train_cfg.save_path}/a.pth")
        
        unet.load_state_dict(chkpt_dict["unet"])
        optimizer.load_state_dict(chkpt_dict["optimizer"])
        epoch_start = chkpt_dict["epochs"]
    except Exception as e:
        logger.error(e)
        logger.info(f"Defaulting to reinit random UNET model")
    
    losses = []
    print(epochs)
    i=0
    for epoch in range(epoch_start,epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                x, _ = batch[0].to(device), batch[1]

                timestamp = diffusion.sample_time_stamp(batch_size)

                x_t, eps = diffusion.add_noise(x, timestamp)

            predict_eps = unet(x_t, timestamp)

            loss = loss_fn(eps, predict_eps)

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            i+=1
            
            if i % 10 == 0:
                logger.info(f"epoch: {epoch} Loss: {sum(losses[-10:])/10}")
        torch.save(
                    {
                        "unet" : unet.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "epochs":epochs
                    },
                    f=f"{train_cfg.save_path}/a.pth"
                )
        logger.info(f"Model save at the path: {train_cfg.save_path}/a.pth \n\n")
    torch.save(
                {
                    "unet" : unet.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "epochs":epochs
                },
                f=f"{train_cfg.save_path}/a.pth"
            )
    logger.info(f"Model save at the path: {train_cfg.save_path}/a.pth \n\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")

    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--config", type=str, default="./config/base.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    train(config)