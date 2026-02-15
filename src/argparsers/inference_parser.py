import argparse

def parse_inference_args():
    parser = argparse.ArgumentParser(description="Diffusion Inference")
    
    path_group = parser.add_argument_group("Paths")
    path_group.add_argument("--model_path", type=str, default="./saves/a.pth", help="Path to .pth model")
    path_group.add_argument("--config", type=str, default="./config/base.yaml")
    
    gen_group = parser.add_argument_group("Generation Params")
    gen_group.add_argument("--img_size", type=int, default=256)
    gen_group.add_argument("--batch_size", type=int, default=1, help="Number of images to generate")
    
    system_group = parser.add_argument_group("System")
    system_group.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    return parser.parse_args()