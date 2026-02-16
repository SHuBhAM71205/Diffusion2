import argparse

from mini_diffusion.train import train
from mini_diffusion.sample import sample

from mini_diffusion.config import load_config, Config
import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample", "serve"])
    
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--config", type=str, default="./configs/base.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config:Config = load_config(args.config)
    if args.mode == "train":
        train(config)

    elif args.mode == "sample":
        bytes = sample(config)
        with open("sample.png","wb") as f:
            f.write(bytes)

    elif args.mode == "serve":
        uvicorn.run("app.app:app",
                    host="localhost",
                    port=8000,
                    reload=True)

if __name__ == "__main__":
    main()
