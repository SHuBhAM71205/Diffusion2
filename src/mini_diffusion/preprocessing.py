
import torchvision.transforms.v2 as v2
import torch
from argparse import ArgumentParser
import os
import array
from PIL import Image
import tqdm
from mini_diffusion.config import Config,load_config

def setup_config():
    parser = ArgumentParser(description="The preprocessing of data")
    
    path_group = parser.add_argument_group("Paths")
    
    path_group.add_argument("--config",type=str,default="./config/base.yaml",help="Pass the path to the Config file")
    
    args = parser.parse_args()
    
    return args

def preprocessing(config: Config):
    
    data_dir,save_dir = config.preprocessing.data_dir,config.preprocessing.save_dir
    
    img_lst = sorted(os.listdir(data_dir))
    
    offset = array.array('Q')
    
    offset.append(0)
    
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.Resize((config.model.image_size,config.model.image_size)),
            v2.CenterCrop((config.model.image_size,config.model.image_size)),
            v2.ToDtype(torch.uint8,scale=False)
        ]
    )
    
    with open(f"{save_dir}/data.bin","wb") as f_bin,\
        open(f"{save_dir}/offset.bin","wb") as f_off:
        
        for add in tqdm.tqdm(img_lst,desc="Processing Images"):

            im  =  Image.open(f"{data_dir}/{add}")
            
            tensor = transform(im).contiguous()
            bytes_data = tensor.numpy().tobytes()
            
            f_bin.write(bytes_data)
            length = len(bytes_data)            
            
            offset.append(offset[-1] + length)

        f_off.write(offset)
    
    

if __name__ == "__main__":
    
    args = setup_config()
    
    config = load_config(args.config)
    
    preprocessing(config)
    
    with open(f"./data/offset.bin","rb") as f,\
        open(f"./data/data.bin","rb") as img_bin:
        
        bin = f.read()
        im = img_bin.read()
        
        arr = array.array('Q',bin)
        
        print(len(arr),arr[:10])