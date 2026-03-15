import yaml
from pydantic import BaseModel # type: ignore
from typing import List
# class ModelConfig(BaseModel):
    
class ModelConfig(BaseModel):
    im_channels : int
    im_size : int
    down_channels : List[int]
    mid_channels : List[int]
    down_sample : List[bool]
    time_emb_dim : int
    num_down_layers : int
    num_mid_layers : int
    num_up_layers : int
    num_heads : int


class DiffusionConfig(BaseModel):
    timesteps: int
    beta_start: float
    beta_end: float


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    device: str
    data_dir:str
    save_path:str
    num_workers:int
    logs:str

class InferenceConfig(BaseModel):
    model_path: str
    device: str
    logs:str

class Preprocessing(BaseModel):
    data_dir:str
    save_dir:str


class Config(BaseModel):
    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig
    inference: InferenceConfig
    preprocessing: Preprocessing


def load_config(path:str)->Config:
    
    with open(path,"r") as f:
        data = yaml.safe_load(f)
    return Config(** data)
    
