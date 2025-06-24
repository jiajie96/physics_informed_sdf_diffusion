import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.utils.indexing import unsqueeze_as
from .utils import identity, rand_uniform

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = th.arange(n_timestep + 1, dtype=th.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}")
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


# gaussian diffusion trainer class


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(self, loss_type="l1", model_mean_type="eps", schedule_kwargs=None, device="cpu"):
        super().__init__()
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.set_new_noise_schedule(schedule_kwargs, device=device)
        self.set_loss("cpu")

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="mean").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="mean").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(th.tensor, dtype=th.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt.get("linear_start", 1e-4),
            linear_end=schedule_opt.get("linear_end", 2e-2),
            cosine_s=schedule_opt.get("cosine_s", 8e-3),
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, th.Tensor) else betas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        self.register_buffer("sqrt_alphas_cumprod_prev", to_torch(sqrt_alphas_cumprod_prev))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer(
            "posterior_mean_coef2", to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        )

        if "ddim_S" in schedule_opt and "ddim_eta" in schedule_opt:
            self.set_ddim_schedule(schedule_opt["ddim_S"], schedule_opt["ddim_eta"])

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t
            - unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], noise) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
        # x_0 = A x_t - B e
        # e = A/B x_t - 1/B x_0
        recip = 1 / unsqueeze_as(self.sqrt_recipm1_alphas_cumprod[t], x_t)
        return (unsqueeze_as(self.sqrt_recip_alphas_cumprod[t], x_t) * x_t - x_0) * recip

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            unsqueeze_as(self.posterior_mean_coef1[t], x_start) * x_start
            + unsqueeze_as(self.posterior_mean_coef2[t], x_t) * x_t
        )
        posterior_log_variance_clipped = unsqueeze_as(self.posterior_log_variance_clipped[t], x_t)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, x, t, clip_denoised: bool, denoise_kwargs={}, post_fn=identity):
        # noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(b, 1)
        # noise_level = th.tensor([self.sqrt_alphas_cumprod_prev[t + 1]], dtype=th.float, device=x.device).repeat(b, 1)
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1]
        noise_pred = post_fn(denoise_fn(x, noise_level, **denoise_kwargs))
        if self.model_mean_type == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        else:
            x_recon = noise_pred

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @th.no_grad()
    def p_sample(self, denoise_fn, x, t, clip_denoised=True, denoise_kwargs={}, post_fn=identity):
        model_mean, model_log_variance = self.p_mean_variance(
            denoise_fn, x, t, clip_denoised=clip_denoised, denoise_kwargs=denoise_kwargs, post_fn=post_fn
        )
        # noise = th.randn_like(x) if t > 0 else th.zeros_like(x)
        noise = th.randn_like(x)
        noise[t == 0] = 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @th.no_grad()
    def sample(
        self,
        denoise_fn,
        shape,
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        show_pbar=False,
        pbar_kwargs={},
    ):
        b = shape[0]
        rankzero = not dist.is_initialized() or dist.get_rank() == 0
        tqdm_kwargs = dict(
            desc="Sample DDPM",
            total=self.num_timesteps,
            ncols=128,
            disable=not (show_pbar and rankzero),
        )
        tqdm_kwargs.update(pbar_kwargs)
        pbar = tqdm(reversed(range(0, self.num_timesteps)), **tqdm_kwargs)

        device = self.betas.device
        sample_inter = 1 | (self.num_timesteps // 10)

        img = th.randn(shape, device=device)
        ret_img = [img]
        for i in pbar:
            t = img.new_full((b,), i, dtype=th.long)
            img = self.p_sample(denoise_fn, img, t, clip_denoised=clip_denoised, denoise_kwargs=denoise_kwargs, post_fn=post_fn)
            if i % sample_inter == 0:
                ret_img += [img]

        if return_intermediates:
            return ret_img
        else:
            return ret_img[-1]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: th.randn_like(x_start))

        # random gamma
        return (
            unsqueeze_as(continuous_sqrt_alpha_cumprod, x_start) * x_start
            + unsqueeze_as(1 - continuous_sqrt_alpha_cumprod**2, noise).sqrt() * noise
        )

    def p_losses(self, denoise_fn, x_0, noise=None, denoise_kwargs={}, post_fn=identity):
        b = x_0.size(0)
        dev = x_0.device

        t = th.randint(1, self.num_timesteps + 1, (b,), device=dev)
        v1 = self.sqrt_alphas_cumprod_prev[t - 1]
        v2 = self.sqrt_alphas_cumprod_prev[t]
        continuous_sqrt_alpha_cumprod = (v2 - v1) * th.rand(b, device=dev) + v1  # b

        noise = default(noise, lambda: th.randn_like(x_0))
        x_noisy = self.q_sample(x_0, continuous_sqrt_alpha_cumprod, noise)
        x_recon = post_fn(denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, **denoise_kwargs))

        if self.model_mean_type == "eps":
            loss = self.loss_func(noise, x_recon)
        else:
            loss = self.loss_func(x_0, x_recon)
        return loss

    def forward(self, denoise_fn, x, denoise_kwargs={}, post_fn=identity, *args, **kwargs):
        return self.p_losses(denoise_fn, x, denoise_kwargs=denoise_kwargs, post_fn=post_fn, *args, **kwargs)

    def set_ddim_schedule(self, S=None, eta=0., ddim_timesteps=None):
        to_torch = partial(th.tensor, dtype=th.float32, device="cpu")

        assert (S is not None) != (ddim_timesteps is not None), "Either S or ddim_timesteps must be provided, but not both."

        # make ddim schedule
        if ddim_timesteps is None:
            self.ddim_timesteps = make_ddim_timesteps(
                ddim_discr_method="uniform",
                num_ddim_timesteps=S,
                num_ddpm_timesteps=self.num_timesteps,
                verbose=False,
            )
        else:
            self.ddim_timesteps = ddim_timesteps
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.cpu().numpy(), ddim_timesteps=self.ddim_timesteps, eta=eta, verbose=False
        )
        ddim_sqrt_one_minus_alphas = np.sqrt(1.0 - ddim_alphas)

        ddim_sigmas = to_torch(ddim_sigmas)
        ddim_alphas = to_torch(ddim_alphas)
        ddim_alphas_prev = to_torch(ddim_alphas_prev)
        ddim_sqrt_one_minus_alphas = to_torch(ddim_sqrt_one_minus_alphas)

        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", ddim_sqrt_one_minus_alphas)


    def sample_ddim(
        self,
        denoise_fn,
        shape=None,
        x_t=None,
        from_t_idx=0,
        to_t_idx=None,
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
        requires_grad=False,
        debug_plot=False,
        ddpm_indexing=True
    ):
        assert (shape is not None) != (x_t is not None), "Either shape or x_t must be provided, but not both." 
        assert (not shape is not None) or (from_t_idx==0), "If shape is provided, then the sampling must start from the last timestep (pure noise)."

        assert hasattr(self, "ddim_timesteps"), "ddim parameters are not initialized"
        rankzero = not dist.is_initialized() or dist.get_rank() == 0
        dev = self.betas.device
        shape = x_t.shape if x_t is not None else shape
        b = shape[0]
        timesteps = self.ddim_timesteps

        x_t = th.randn(shape[:1] + shape[2:], device=dev).unsqueeze(1) if x_t is None else x_t
        if to_t_idx == 0: 
            to_t_idx = None # Until last value
        time_range = np.flip(timesteps)[from_t_idx:to_t_idx] 
        #print("sampling:", time_range)
        total_steps = timesteps.shape[0]
        tqdm_kwargs = dict(
            total=time_range.shape[0],
            desc="Sample DDIM",
            ncols=128,
            disable=not (show_pbar and rankzero),
        )
        tqdm_kwargs.update(pbar_kwargs)
        pbar = tqdm(time_range, **tqdm_kwargs)

        intermediates = [x_t]
        #noise_levels = []
        #ddim_alphas_cumprod = self.alphas_cumprod[timesteps] 
        #sqrt_alphas_cumprod_prev = th.sqrt(th.cat((ddim_alphas_cumprod[[0]], ddim_alphas_cumprod[:-1])))

        alphas_cumprod = self.ddim_alphas[from_t_idx:to_t_idx] 
        alphas_cumprod_prev = th.cat((self.ddim_alphas[[from_t_idx-1]], alphas_cumprod[:-1])) if not (from_t_idx % total_steps == 0) else th.cat((self.alphas_cumprod[[0]], alphas_cumprod[:-1]))
        sigmas = self.ddim_sigmas[from_t_idx:to_t_idx]
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas[from_t_idx:to_t_idx]

        if from_t_idx is not None: 
            idx_shift = from_t_idx if from_t_idx >= 0 else (total_steps + from_t_idx)
        else:
            idx_shift = 0
        if debug_plot:
            from ..utils.vis import plot_sdfs
        with th.enable_grad() if requires_grad else th.no_grad():
            for i, step in enumerate(pbar):
                #index = total_steps - i - 1 - idx_shift 
                index = - i - 1
                index = th.full((b,), index, device=dev, dtype=th.long)
                ts = th.full((b,), step, device=dev, dtype=th.long)
                #noise_level = self.sqrt_alphas_cumprod_prev[ts] # Even tho seems wrong, the model works with this 
                #print(f"self.sqrt_alphas_cumprod_prev[ts]:{self.sqrt_alphas_cumprod_prev[ts]} vs th.sqrt(self.ddim_alphas_prev[[index]]): {th.sqrt(self.ddim_alphas_prev[[index]])}")
                #noise_levels.append(noise_level)
                noise_level = self.sqrt_alphas_cumprod_prev[ts] if ddpm_indexing else alphas_cumprod_prev[index].sqrt()

                # try to correct 
                assert (not from_t_idx > 0) or index == 0, "Index should never reach 0 when from_t_idx is greater than 0."
                #noise_level = th.sqrt(self.ddim_alphas_prev[[index]]) # TODO: not sure it is correct when from_t_idx !=0 (it should be since self.ddim_alphas_prev[0] should never be accessed)

                e_t = post_fn(denoise_fn(x_t, noise_level, **denoise_kwargs)) 
                if self.model_mean_type == "x_0":
                    e_t = self.predict_noise_from_start(x_t, ts, e_t) 

                index = index[0]
                a_t = unsqueeze_as(th.full((b,), alphas_cumprod[index], device=dev), x_t)
                a_prev = unsqueeze_as(th.full((b,), alphas_cumprod_prev[index], device=dev), x_t)
                sigma_t = unsqueeze_as(th.full((b,), sigmas[index], device=dev), x_t)
                sqrt_one_minus_at = unsqueeze_as(th.full((b,), sqrt_one_minus_alphas[index], device=dev), x_t)

                # current prediction for x_0
                pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
                if clip_denoised:
                    pred_x0.clamp_(-1.0, 1.0)
                if i % log_every_t == 0 or i == len(pbar)-1:
                    intermediates.append(pred_x0)

                # direction pointing to x_t
                dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * th.randn_like(x_t)
                x_t = a_prev.sqrt() * pred_x0 + dir_xt + noise

                if to_t_idx and i == len(pbar) - 1 and not return_intermediates:
                    return x_t

                if debug_plot and (i % log_every_t == 0 or i == len(pbar) - 1):
                    print(f'{"ddpm_indexing" if ddpm_indexing else "ddim_indexing"} -- a_t: {a_t[0].item()}, a_prev: {a_prev[0].item()}, sqrt_one_minus_at: {sqrt_one_minus_at[0].item()}, noise_level: {noise_level[0].item()}')
                    plot_sdfs(x_t, titles=f"t={step}/{self.num_timesteps} (DDIM: t={index+1}/{len(self.ddim_timesteps)})")

        if return_intermediates:
            return intermediates
        else:
            return x_t
            #return intermediates[-1]

    def invert_ddim(
        self,
        denoise_fn,
        x_t,
        from_t_idx=0,
        to_t_idx=None,
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
        debug_plot=False,
        requires_grad=False,
        ddpm_indexing=True,
    ):
        assert hasattr(self, "ddim_timesteps"), "ddim parameters are not initialized"
        rankzero = not dist.is_initialized() or dist.get_rank() == 0
        dev = self.betas.device
        shape = x_t.shape # if x_t is not None else shape
        b = shape[0]
        timesteps = self.ddim_timesteps
        if to_t_idx == 0: 
            to_t_idx = None # Until last value
        time_range = timesteps[from_t_idx:to_t_idx]
        #print("inverting:",  time_range)
        total_steps = timesteps.shape[0]
        tqdm_kwargs = dict(
            total=time_range.shape[0],
            desc="Inverse DDIM",
            ncols=128,
            disable=not (show_pbar and rankzero),
        )
        tqdm_kwargs.update(pbar_kwargs)
        pbar = tqdm(time_range, **tqdm_kwargs)

        #sqrt_alphas_cumprod_next = th.sqrt(th.cat((self.alphas_cumprod[1:], self.alphas_cumprod[[-1]])))[timesteps]
        #sqrt_alphas_cumprod_next = th.sqrt(th.cat((self.alphas_cumprod[timesteps][1:], self.alphas_cumprod[timesteps][[-1]])))
        alphas_cumprod = self.ddim_alphas[from_t_idx:to_t_idx]
        alphas_cumprod_next = th.cat((self.ddim_alphas[from_t_idx+1:], self.ddim_alphas[[-1]])) if (to_t_idx is None or to_t_idx % total_steps == 0) else self.ddim_alphas[from_t_idx+1:to_t_idx+1] 
        sqrt_alphas_cumprod_next = th.sqrt(alphas_cumprod_next)

        sqrt_alphas_cumprod_next_ddpm = th.sqrt(th.cat((self.alphas_cumprod[1:], self.alphas_cumprod[[-1]])))
        #sqrt_alphas_cumprod_next_ddpm = sqrt_alphas_cumprod_next_ddpm[timesteps]

        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas[from_t_idx:to_t_idx]

        # # all with ddim indexing (WRONG)
        # alphas_cumprod_ddim = self.alphas_cumprod[timesteps]
        # alphas_cumprod_next = alphas_cumprod_ddim # [a_0, a_1, ..., a_T]
        # sqrt_alphas_cumprod_next = th.sqrt(alphas_cumprod_next)  # [√(a_0), √(a_1), ..., √(a_T)]
        # alphas_cumprod = th.cat((alphas_cumprod_ddim[1:], alphas_cumprod_ddim[[-1]])) # [a_1, a_2, ..., a_T, a_T]

        # sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod - 1) # [√(1/a_1 - 1), √(1/a_2 - 1), ..., √(1/a_T - 1), √(1/a_T - 1)]
        # sqrt_recip_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod) # [√(1/a_1), √(1/a_2), ..., √(1/a_T), √(1/a_T)]

        # ddim_sqrt_one_minus_alphas = th.sqrt(1.0 - alphas_cumprod) # [√(1 - a_1), √(1 - a_2), ..., √(1 - a_T), √(1 - a_T)]
        intermediates = [x_t]
        if from_t_idx is not None: 
            idx_shift = from_t_idx if from_t_idx >= 0 else (total_steps + from_t_idx)
        else:
            idx_shift = 0

        if debug_plot:
            from ..utils.vis import plot_sdfs

        with th.enable_grad() if requires_grad else th.no_grad():
            for i, step in enumerate(pbar):
                #index = i + idx_shift
                index = i
                index = th.full((b,), index, device=dev, dtype=th.long)
                ts = th.full((b,), step, device=dev, dtype=th.long)
                #noise_level = ddim_alphas_next[[index]].sqrt()
                #noise_level = th.sqrt(self.ddim_alphas[[index]])
                #noise_level = sqrt_alphas_cumprod_next[ts]
                #noise_level = sqrt_alphas_cumprod_next[[index]] # √(a_{t})
                # Try DDPM indexing  (AFTER RESULTS: similar (undesired) noising of when using sqrt_alphas_cumprod_next[[index]])
                #assert (not from_t_idx > 0) or index == len(pbar)-1, f"Index should never reach {len(pbar)-1} when from_t_idx is greater than 0."
                
                noise_level = sqrt_alphas_cumprod_next_ddpm[ts] if ddpm_indexing else sqrt_alphas_cumprod_next[index]

                e_t = post_fn(denoise_fn(x_t, noise_level, **denoise_kwargs)) # x_theta(x_{t+1}) or  x_theta(x_{t}) ?? 
                if self.model_mean_type == "x_0":
                    e_t = self.predict_noise_from_start(x_t, ts, e_t)
                    # TODO: check if it is the same -- why on diffusers is not possible (division by 0)
                    #recip = 1 / unsqueeze_as(sqrt_recipm1_alphas_cumprod[index], x_t) # 1 / √(1/a_{t+1} - 1)
                    #e_t = (unsqueeze_as(sqrt_recip_alphas_cumprod[index], x_t) * x_t - e_t) * recip # eps_theta(x_{t+1}) = ( √(1/a_{t+1})*x_{t+1} - x_theta(x_{t+1}) ) / √(1/a_{t+1} - 1)

                index = index[0]
                a_t = unsqueeze_as(th.full((b,), alphas_cumprod[index], device=dev), x_t) # a_{t}
                a_next = unsqueeze_as(th.full((b,), alphas_cumprod_next[index], device=dev), x_t) # a_{t+1}
                sqrt_one_minus_at = unsqueeze_as(th.full((b,), sqrt_one_minus_alphas[index], device=dev), x_t) # √(1-a_{t})
 
                # current prediction for x_0
                pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()  
                if clip_denoised:
                    pred_x0.clamp_(-1.0, 1.0)
                # if i % log_every_t == 0 or i == len(pbar)-1:
                #     intermediates.append(pred_x0)

                # direction pointing to x_t
                dir_xt = (1.0 - a_next).sqrt() * e_t
                x_t = a_next.sqrt() * pred_x0 + dir_xt 

                # if to_t_idx and i == len(pbar)-1 and not return_intermediates:
                #     return x_t

                if debug_plot and (i % log_every_t == 0 or i == len(pbar) - 1):
                    print(f'{"ddpm_indexing" if ddpm_indexing else "ddim_indexing"} -- a_t: {a_t[0].item()}, a_next: {a_next[0].item()}, sqrt_one_minus_at: {sqrt_one_minus_at[0].item()}, noise_level: {noise_level[0].item()}')
                    plot_sdfs(x_t, titles=f"t={step}/{self.num_timesteps} (DDIM: t={index+1}/{len(self.ddim_timesteps)})")

        return x_t        
        # if return_intermediates:
        #     return intermediates
        # else:
        #     return intermediates[-1]
        
        
    # TODO: maybe move this to another file
    def ddim_sample_latent_optimization(
        self,
        denoise_fn,
        x_0,
        t_optim_idx,
        obj_fn,
        obj_fn_args={},
        tgt_noise_level = "t_optim",
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
        opt_kwargs={"lr":.5e-3}, 
        loss_threshold=1e-4, 
        max_opt_iters=1500, 
        early_stopping_iters=200,
        grad_clip_value=None,
    ):        
        ddim_args = {
            "clip_denoised": clip_denoised,
            "denoise_kwargs": denoise_kwargs,
            "post_fn": post_fn,
            "log_every_t": log_every_t,
            "show_pbar": show_pbar,
            "pbar_kwargs": pbar_kwargs,
            "return_intermediates": return_intermediates,
        }
        ddim_args_no_intermediates = ddim_args.copy()
        ddim_args_no_intermediates.pop("return_intermediates", None)

        # Get the latent at the defined noise level 
        x_t = self.invert_ddim(denoise_fn, x_0, to_t_idx=t_optim_idx, requires_grad=False, **ddim_args)
        #print(f"x_t range: {x_t.min().item(), x_t.max().item()} and mean (±std): {x_t.mean().item()} (±{th.var(x_t).sqrt().item()})")

        # Optimize it
        with th.enable_grad(): 
            x_t_optim = x_t.clone()
            x_t_optim.requires_grad = True
            optimizer = th.optim.AdamW([x_t_optim], **opt_kwargs)
            x_t_optim.grad = th.zeros_like(x_t_optim)
            best_state_dict = {"x_t_optim": x_t_optim.clone(), "loss": obj_fn(x_t_optim, **obj_fn_args)}
            with tqdm(range(max_opt_iters), desc="Latent optimization") as pbar:
                iters_without_improvement = 0 
                for i in pbar:
                    if tgt_noise_level == "t_optim": 
                        tgt_pred = x_t_optim
                    elif tgt_noise_level == "zero": 
                        tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t_optim, from_t_idx=-t_optim_idx, requires_grad=True, **ddim_args)
                    elif tgt_noise_level == "zero_pred": 
                        tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t_optim, from_t_idx=-(t_optim_idx), to_t_idx=-(t_optim_idx-1), requires_grad=True, return_intermediates=True, **ddim_args_no_intermediates)[-1]
                        #tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-(t_optim_idx), to_t_idx=None, requires_grad=True, return_intermediates=True, **ddim_args)[1] # slow, unnecessary (tested to be the equal)
                    else: 
                        raise ValueError("Invalid noise level: available levels are " + ["t_optim", "zero"] + ".")
                    tgt_pred.grad = x_t_optim.grad
                    loss_i = obj_fn(tgt_pred, **obj_fn_args)
                    optimizer.zero_grad(set_to_none=True)
                    loss_i.backward()
                    
                    #print(f"x_t_optim range: {x_t_optim.min().item(), x_t_optim.max().item()} and mean (±std): {x_t_optim.mean().item()} (±{th.var(x_t_optim).sqrt().item()})")
                    #print(f"grad range: {x_t_optim.grad.min().item(), x_t_optim.grad.max().item()} and mean (±std): {x_t_optim.grad.mean().item()} (±{th.var(x_t_optim.grad).sqrt().item()})")

                    if grad_clip_value is not None:
                        th.nn.utils.clip_grad_value_(x_t_optim, grad_clip_value)

                    optimizer.step()

                    pbar.set_postfix({"Loss (mean)": th.mean(loss_i).item(), "Best loss (mean)": th.mean(best_state_dict["loss"]).item()})

                    # Early stopping
                    improved_samples_idxs = loss_i < best_state_dict["loss"]
                    if th.sum(improved_samples_idxs) > 0: 
                        best_state_dict["x_t_optim"] = x_t_optim.clone()
                        best_state_dict["loss"] = loss_i
                        iters_without_improvement = 0
                    else: 
                        iters_without_improvement += 1
                        if iters_without_improvement >= early_stopping_iters:
                            print(f"Early stopping after {early_stopping_iters} iterations without improvement.")
                            # Restore the best state
                            print(f"Restoring the best state with loss {best_state_dict['loss']}.")
                            x_t_optim = best_state_dict["x_t_optim"].detach()
                            break

                    # Loss threshold
                    if th.all(loss_i < loss_threshold):
                        print(f"Loss threshold (set to {loss_threshold}) reached after {i} iterations.")
                        break

        # Denoise the optimized latent 
        x_edited = self.sample_ddim(denoise_fn, x_t=x_t_optim, from_t_idx=-t_optim_idx, requires_grad=False, **ddim_args)
        
        return {
            "x_edited": x_edited,
            "x_t": x_t,
            "x_t_optim": x_t_optim
        }
        
    # TODO: maybe move this to another file
    def ddim_sample_noise_guidance(
        self,
        denoise_fn,
        x_0,
        from_t_optim_idx,
        obj_fn,
        obj_fn_args={},
        tgt_noise_level = "t_optim",
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        #return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
        opt_kwargs={"lr":1e-2, "decay_fn": identity}, 
        grad_clip_value=None,
        plot_debug=False,
    ):        
        ddim_args = {
            "clip_denoised": clip_denoised,
            "denoise_kwargs": denoise_kwargs, 
            "post_fn": post_fn, 
            #"return_intermediates": return_intermediates, 
            "log_every_t": log_every_t, 
            "show_pbar": show_pbar, 
            "pbar_kwargs": pbar_kwargs
        }

        # Get the latent at the defined noise level
        x_t = self.invert_ddim(denoise_fn, x_0, to_t_idx=from_t_optim_idx, requires_grad=False, **ddim_args)
        if plot_debug:  
            from src.utils.vis import plot_sdfs
            plot_sdfs(x_t, title=f"x_t inverted to timestep {self.ddim_timesteps[from_t_optim_idx]}")

        # Denoise it with guidance
        # if from_t_optim_idx < 0:
        #     idxs = range(from_t_optim_idx, 0)
        # else:
        #     idxs = range(from_t_optim_idx, len(self.ddim_timesteps))
        idxs = range(from_t_optim_idx, 0, -1)
        
        #x_t_optim = x_t.clone()
        #x_t_optim.requires_grad = True
        #optimizer = th.optim.SGD([x_t_optim], **opt_kwargs)
        with tqdm(idxs, desc="Latent optimization") as pbar_idxs:
            for i, t_idx in enumerate(pbar_idxs):
                x_t.requires_grad_(True)
                if tgt_noise_level == "t_optim":
                    tgt_pred = x_t
                elif tgt_noise_level == "zero":
                    tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-t_idx, requires_grad=True, **ddim_args) # Too slow and computationally and memory intensive 
                elif tgt_noise_level == "zero_pred": 
                    tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-(t_idx), to_t_idx=-(t_idx-1), requires_grad=True, return_intermediates=True, **ddim_args)[-1]
                    #tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-(t_optim_idx), to_t_idx=None, requires_grad=True, return_intermediates=True, **ddim_args)[1] # slow, unnecessary (tested to be the equal)
                else:
                    raise ValueError("Invalid noise level: available levels are " + ["t_optim", "zero"] + ".")
                # optimizer.zero_grad(set_to_none=True)
                # loss_i = obj_fn(tgt_pred, **obj_fn_args)
                # loss_i.backward()
                # if grad_clip_value is not None:
                #     th.nn.utils.clip_grad_value_(x_t, grad_clip_value)
                # optimizer.step()

                #tgt_pred.requires_grad_(True)
                with th.enable_grad():
                    loss_i = obj_fn(tgt_pred, **obj_fn_args) # tgt_pred is a function of x_t
                grad_t = th.autograd.grad(loss_i, x_t, retain_graph=False)[0] # grad of loss wrt to x_t

                x_t = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-(t_idx), to_t_idx=-(t_idx-1), requires_grad=False, **ddim_args) # x_{t-1}
                decay_fn = opt_kwargs.get("decay_fn", identity)
                x_t = x_t - decay_fn(i) * opt_kwargs["lr"] * grad_t
                x_t.grad = None
                loss_i.grad = None
                tgt_pred.grad = None

                pbar_idxs.set_postfix({"Loss (mean)": th.mean(loss_i).item()})

                if plot_debug:
                    plot_sdfs([tgt_pred, x_t], title=f"Optimization step {i} at timestep {self.ddim_timesteps[t_idx]}", titles=[f"Target shape (target type: \"{tgt_noise_level}\")", "Optimized shape"])

        return x_t
            
    def ddim_sample_noise_guidance(
        self,
        denoise_fn,
        x_0,
        from_t_optim_idx,
        obj_fn,
        obj_fn_args={},
        tgt_noise_level = "t_optim",
        clip_denoised=True,
        denoise_kwargs={},
        post_fn=identity,
        #return_intermediates=False,
        log_every_t=5,
        show_pbar=False,
        pbar_kwargs={},
        opt_kwargs={"lr":1e-2, "decay_fn": identity}, 
        grad_clip_value=None,
        plot_debug=False,
    ):        
        ddim_args = {
            "clip_denoised": clip_denoised,
            "denoise_kwargs": denoise_kwargs, 
            "post_fn": post_fn, 
            #"return_intermediates": return_intermediates, 
            "log_every_t": log_every_t, 
            "show_pbar": show_pbar, 
            "pbar_kwargs": pbar_kwargs
        }

        # Get the latent at the defined noise level
        x_t = self.invert_ddim(denoise_fn, x_t=x_0, to_t_idx=from_t_optim_idx, requires_grad=False, clip_denoised=clip_denoised)
        if plot_debug:  
            from src.utils.vis import plot_sdfs
            plot_sdfs(x_t, title=f"x_t inverted to timestep {self.ddim_timesteps[from_t_optim_idx]}")

        # Denoise it with guidance
        # if from_t_optim_idx < 0:
        #     idxs = range(from_t_optim_idx, 0)
        # else:
        #     idxs = range(from_t_optim_idx, len(self.ddim_timesteps))
        idxs = range(from_t_optim_idx, 0, -1)
        
        # x_t_optim = x_t.clone()
        # x_t_optim.requires_grad = True
        #optimizer = th.optim.AdamW([x_t_optim], **opt_kwargs)
        with tqdm(idxs, desc="Latent optimization") as pbar_idxs:
            for i, t_idx in enumerate(pbar_idxs):
                #x_t_optim = x_t.clone().detach()
                x_t_optim = x_t.detach()
                x_t_optim.requires_grad_(True)
                if tgt_noise_level == "t_optim":
                    tgt_pred = x_t_optim
                elif tgt_noise_level == "zero":
                    tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t_optim, from_t_idx=-t_idx, requires_grad=True, clip_denoised=clip_denoised) 
                elif tgt_noise_level == "zero_pred": 
                    tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t_optim, from_t_idx=-t_idx, to_t_idx=-(t_idx-1), requires_grad=True, return_intermediates=True, clip_denoised=clip_denoised)[-1]
                    #tgt_pred = self.sample_ddim(denoise_fn, x_t=x_t, from_t_idx=-(t_optim_idx), to_t_idx=None, requires_grad=True, return_intermediates=True, **ddim_args)[1] # slow, unnecessary (tested to be the equal)
                else:
                    raise ValueError("Invalid noise level: available levels are " + ["t_optim", "zero"] + ".")
                
                # loss_i = obj_fn(tgt_pred, **obj_fn_args)
                # #tgt_pred.requires_grad_(True)
                # optimizer.zero_grad()
                # loss_i.backward()
                # # if grad_clip_value is not None:
                # #     th.nn.utils.clip_grad_value_(x_t, grad_clip_value)
                # optimizer.step()

                
                with th.enable_grad():
                    loss_i = obj_fn(tgt_pred, **obj_fn_args) # tgt_pred is a function of x_t
                    grad_t = th.autograd.grad(loss_i, x_t_optim, retain_graph=False)[0] # grad of loss wrt to x_t
                print("Grad. norm: ", th.norm(grad_t).item(), " vs x_t norm: ", th.norm(x_t).item())

                # x_{t} optimized 
                decay_fn = opt_kwargs.get("decay_fn", identity)
                x_t = x_t - decay_fn(i) * opt_kwargs["lr"] * grad_t
                x_t.grad = x_t_optim.grad = loss_i.grad = tgt_pred.grad = None

                pbar_idxs.set_postfix({"Loss (mean)": th.mean(loss_i).item()})

                if plot_debug:
                    plot_sdfs([tgt_pred, x_t], title=f"Optimization step {i} at timestep {self.ddim_timesteps[t_idx]}", titles=[f"Target shape (target type: \"{tgt_noise_level}\")", f"Optimized shape (V={th.sum(x_t<0, dim=list(range(1, len(x_t.shape)))).item()})"])

                # x_{t-1} 
                x_t = self.sample_ddim(denoise_fn, x_t=x_t.detach(), from_t_idx=-t_idx, to_t_idx=-(t_idx-1), requires_grad=False, clip_denoised=clip_denoised)

        return x_t