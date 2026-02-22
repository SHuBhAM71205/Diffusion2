import os
import mmap
import array
import numpy as np
import torch
from torch.utils.data import Dataset
from mini_diffusion.config import Config

class Plane(Dataset):
    def __init__(self, data_dir: str,channel:int = 3,width:int = 32,height:int = 32,transform=None):
        super().__init__()

        self.transform = transform

        self.data_path = os.path.join(data_dir, "data.bin")
        self.offset_path = os.path.join(data_dir, "offset.bin")

        assert os.path.exists(self.data_path), "data.bin not found"
        assert os.path.exists(self.offset_path), "offset.bin not found"

        # Load offsets
        with open(self.offset_path, "rb") as f:
            self.offsets = array.array('Q')
            self.offsets.frombytes(f.read())

        self.num_images = len(self.offsets) - 1

        # Memory map data file
        self.data_file = open(self.data_path, "rb")
        self.mm = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_COPY)

        # Fixed image specs
        self.channels = channel
        self.height = height
        self.width = width
        self.image_size = self.channels * self.height * self.width

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        start = self.offsets[index]
        end = self.offsets[index + 1]
        length = end - start

        # Safety check
        assert length == self.image_size, "Unexpected image size"

        tensor = torch.frombuffer(
            self.mm,
            dtype=torch.uint8,
            count=length,
            offset=start
        )

        tensor_writable = torch.clone(tensor).reshape((self.channels, self.height, self.width))

        if self.transform is not None:
            tensor_writable = self.transform(tensor_writable)

        return tensor_writable

    def __del__(self):
        if hasattr(self, "mm"):
            self.mm.close()
        if hasattr(self, "data_file"):
            self.data_file.close()
            

