from models.diffusion import GaussianDiffusion
import yaml
from easydict import EasyDict
import torch as th

from utils import instantiate_from_config, get_device

device = get_device()

def __main__():
    schedule_kwargs={
        "schedule": "linear",
        "n_timesteps": 1000,
        "linear_start": 1.e-4,
        "linear_end": 2.e-2,
        "ddim_S": 20,
        "ddim_eta": 0.0,
    }
    ddim_model = GaussianDiffusion(schedule_kwargs=schedule_kwargs, device=device)

    gen32_args_path = "config/gen32/chair.yaml"
    gen32_ckpt_path = "results/gen32/chair.pth"
    sr64_args_path = "config/sr32_64/chair.yaml"
    sr64_ckpt_path = "results/sr32_64/chair.pth"

    with open(gen32_args_path) as f:
        args1 = EasyDict(yaml.safe_load(f))
    with open(sr64_args_path) as f:
        args2 = EasyDict(yaml.safe_load(f))

    model1 = instantiate_from_config(args1.model)
    ckpt = th.load(gen32_ckpt_path, map_location=device)
    model1.load_state_dict(ckpt["model_ema"])
    model1 = model1.to(device)
    model1.eval()
    model1.training

    ddim_model.sample_ddim(model1, shape=(1, 1, 32, 32, 32), device=device)


    if __name__ == "__main__":
        __main__()