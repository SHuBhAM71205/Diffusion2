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

import matplotlib.pyplot as plt


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


def sample(config: Config | None = None):

    if config is None:
        config = load_config("./configs/base.yaml")

    model_path = config.inference.model_path
    device = resolve_device(config.inference.device, strict=True)

    img_dim = config.model.im_size
    logger = setup_logger(config.inference.logs)

    print(f"Using device: {device}")

    unet = UNet(config.model)

    try:
        chkpt = torch.load(model_path, map_location=device)

        state_dict_key = "ema_unet"
        print(f"state_dct_key {state_dict_key}")

        unet.load_state_dict(chkpt[state_dict_key])
        unet = unet.to(device)

        logger.info(
            f"Loaded checkpoint from {model_path} "
            f"(weights: {state_dict_key})"
        )

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

    alpha_hat_from_alpha = torch.cumprod(
        diffusion.alpha, dim=0
    )

    schedule_consistency = (
        alpha_hat_from_alpha - diffusion.alpha_hat
    ).abs().max().item()

    logger.info(
        f"Schedule consistency "
        f"max|cumprod(alpha)-alpha_hat|="
        f"{schedule_consistency:.6e}"
    )

    unet.eval()

    # ============================================================
    # LIVE VISUALIZATION SETUP
    # ============================================================

    plt.ion()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.axis("off")

    image_display = None

    # ============================================================

    with torch.inference_mode(), torch.no_grad():

        x_t = torch.randn(
            size=(
                1,
                config.model.im_channels,
                img_dim,
                img_dim
            )
        ).to(device)

        # ========================================================
        # DIFFUSION SAMPLING
        # ========================================================

        for i in reversed(range(config.diffusion.timesteps)):

            t = torch.full(
                (x_t.size(0),),
                i,
                device=device,
                dtype=torch.long
            )

            eps = unet(x_t, t)

            alpha_hat_t = diffusion.alpha_hat[i]
            alpha_t = diffusion.alpha[i]
            beta_t = diffusion.beta[i]

            # v → eps
            # eps = torch.sqrt(alpha_hat_t) * v + \
            #       torch.sqrt(1 - alpha_hat_t) * x_t

            coef = beta_t / torch.sqrt(1 - alpha_hat_t)

            mean = (
                x_t - coef * eps
            ) / torch.sqrt(alpha_t)

            if i > 0:

                alpha_hat_prev = diffusion.alpha_hat[i - 1]

                posterior_var = beta_t * (
                    (1 - alpha_hat_prev)
                    / (1 - alpha_hat_t)
                )

                noise = torch.randn_like(x_t)

                x_t = (
                    mean
                    + torch.sqrt(posterior_var) * noise
                )

            else:

                x_t = (
                    x_t
                    - torch.sqrt(1 - alpha_hat_t) * eps
                ) / torch.sqrt(alpha_hat_t)

            # ====================================================
            # LIVE IMAGE UPDATE
            # ====================================================

            img = (
                x_t[0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                if config.model.im_channels == 3
                else
                x_t[0, 0]
                .detach()
                .cpu()
                .numpy()
            )

            # Convert [-1, 1] -> [0, 1]
            display_img = (img * 0.5) + 0.5

            # Clamp because intermediate diffusion values
            # can occasionally go outside the expected range.
            display_img = np.clip(
                display_img,
                0.0,
                1.0
            )

            # ====================================================
            # FIRST FRAME
            # ====================================================

            if image_display is None:

                if config.model.im_channels == 1:

                    image_display = ax.imshow(
                        display_img,
                        cmap="gray",
                        vmin=0,
                        vmax=1
                    )

                else:

                    image_display = ax.imshow(
                        display_img,
                        vmin=0,
                        vmax=1
                    )

            # ====================================================
            # SUBSEQUENT FRAMES
            # ====================================================

            else:

                image_display.set_data(display_img)

            # ====================================================
            # UPDATE TITLE
            # ====================================================

            ax.set_title(
                f"Diffusion Sampling\n"
                f"T = {i}",
                fontsize=18
            )

            # Draw the new frame
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Controls how fast the visualization runs.
            #
            # 0.01 = very fast
            # 0.03 = fast
            # 0.05 = comfortable
            # 0.10 = slow
            #
            plt.pause(0.03)

        # ========================================================
        # FINAL IMAGE
        # ========================================================

        plt.ioff()

        plt.show(block=False)

        img = (
            x_t[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            if config.model.im_channels == 3
            else
            x_t[0, 0]
            .cpu()
            .numpy()
        )

        logger.info(
            f"{img.shape} "
            f"{img.mean()} "
            f"{img.std()}"
        )

        img = (img * 0.5) + 0.5

        img = np.clip(img, 0.0, 1.0)

        img = (img * 255).astype(np.uint8)

        logger.info(
            f"{img.shape} "
            f"{img.mean()} "
            f"{img.std()}"
        )

        pil_image = Image.fromarray(img)

        buf = io.BytesIO()

        pil_image.save(
            buf,
            format="PNG"
        )

        buf.seek(0)

        return buf.getvalue()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    path_group = parser.add_argument_group("Paths")

    path_group.add_argument(
        "--config",
        type=str,
        default="./configs/base.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    sample(config)