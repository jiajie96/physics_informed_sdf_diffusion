import torch as th 
from diffusers import DDIMInverseScheduler

ddim_inverse_scheduler = DDIMInverseScheduler(
    num_train_timesteps=1000,
    prediction_type="sample",
    set_alpha_to_one=True
)
ddim_inverse_scheduler.set_timesteps(num_inference_steps=50)
print(f"timesteps: {ddim_inverse_scheduler.timesteps}")

with th.no_grad():
    pred = th.randn((1, 1, 2, 2, 2))
    samples = th.randn((1, 1, 2, 2, 2))
    t = ddim_inverse_scheduler.timesteps[0]
    #t = 1 # Same issue
    samples = ddim_inverse_scheduler.step(pred, t, samples).prev_sample
    assert not th.isinf(samples).any(), f"samples contain inf values at t={t}"