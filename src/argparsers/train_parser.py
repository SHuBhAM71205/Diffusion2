import argparse

def parse_train_args():
    parser = argparse.ArgumentParser(description="Diffusion Model Training Configuration")

    # Grouping arguments makes --help much more readable
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--config", type=str, default="./config/base.yaml", help="Path to config file")
    path_group.add_argument("--data_dir", type=str, default="./data", help="Dataset root")
    path_group.add_argument("--save_path", type=str, default="./saves/model.pth", help="Save destination")

    train_group = parser.add_argument_group("Hyperparameters")
    train_group.add_argument("--epochs", type=int, default=2)
    train_group.add_argument("--batch_size", type=int, default=10)
    train_group.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    system_group = parser.add_argument_group("System")
    system_group.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    system_group.add_argument("--workers", type=int, default=2, help="DataLoader workers")

    return parser.parse_args()
