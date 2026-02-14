import yaml
from pydantic import BaseModel # type: ignore


# class ModelConfig(BaseModel):
    
class ModelConfig(BaseModel):
    image_size: int
    in_channels: int
    base_channels: int


class DiffusionConfig(BaseModel):
    timesteps: int
    beta_start: float
    beta_end: float


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    device: str


    
class Config(BaseModel):
    model: ModelConfig
    diffusion: DiffusionConfig
    training: TrainingConfig


def load_config(path:str)->Config:
    
    with open("./configs/base.yaml","r") as f:
        data = yaml.safe_load(f)
    return Config(** data)
    
