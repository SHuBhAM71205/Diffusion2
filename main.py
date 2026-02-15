import argparse

from mini_diffusion.train import train
from mini_diffusion.sample import sample

import uvicorn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "sample", "serve"])
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "sample":
        sample()

    elif args.mode == "serve":
        uvicorn.run("mini_diffusion.api:app",
                    host="0.0.0.0",
                    port=8000)



if __name__ == "__main__":
    main()
