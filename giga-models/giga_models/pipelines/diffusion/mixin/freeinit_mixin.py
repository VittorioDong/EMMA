import math

import torch
import torch.fft as fft


class FreeInitMixin:
    def freeinit_sampling(
        self,
        cur_step,
        latents,
        filter_params=None,
        generator=None,
    ):
        device = self._execution_device
        if cur_step == 0:
            if filter_params is None:
                filter_params = dict(
                    method='butterworth',
                    n=4,
                    d_s=0.25,
                    d_t=0.25,
                )
            self.freeinit_noise = latents.detach().clone()
            self.freeinit_freq_filter = get_freq_filter(latents.shape, device, **filter_params)
        else:
            # 1. DDPM Forward with initial noise, get noisy latents z_T
            diffuse_timestep = self.scheduler.config.num_train_timesteps - 1  # diffuse to t=999 noise level
            diffuse_timesteps = torch.full((1,), diffuse_timestep, dtype=torch.long, device=device)
            z_T = self.scheduler.add_noise(
                original_samples=latents,
                noise=self.freeinit_noise,
                timesteps=diffuse_timesteps,
            )
            # 2. create random noise z_rand for high-frequency
            z_rand = torch.randn(latents.shape, device=device, generator=generator)
            # 3. Roise Reinitialization
            latents = freq_mix_3d(z_T.to(dtype=torch.float32), z_rand, LPF=self.freeinit_freq_filter)
            latents = latents.to(self.dtype)
        return latents


def freq_mix_3d(x, noise, LPF):
    """Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    dim = (0, 2, 3)
    x_freq = fft.fftn(x, dim=dim)
    x_freq = fft.fftshift(x_freq, dim=dim)
    noise_freq = fft.fftn(noise, dim=dim)
    noise_freq = fft.fftshift(noise_freq, dim=dim)
    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high  # mix in freq domain
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=dim)
    x_mixed = fft.ifftn(x_freq_mixed, dim=dim).real
    return x_mixed


def get_freq_filter(shape, device, method, n=4, d_s=0.25, d_t=0.25):
    """Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        params: filter parameters
    """
    if method == 'gaussian':
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif method == 'ideal':
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif method == 'box':
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif method == 'butterworth':
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[0], shape[2], shape[3]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[t, :, h, w] = math.exp(-1 / (2 * d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[0], shape[2], shape[3]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[t, :, h, w] = 1 / (1 + (d_square / d_s**2) ** n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[0], shape[2], shape[3]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = ((d_s / d_t) * (2 * t / T - 1)) ** 2 + (2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2
                mask[t, :, h, w] = 1 if d_square <= d_s * 2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[0], shape[2], shape[3]
    mask = torch.zeros(shape)
    if d_s == 0 or d_t == 0:
        return mask
    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)
    cframe, crow, ccol = T // 2, H // 2, W // 2
    mask[
        cframe - threshold_t : cframe + threshold_t,
        :,
        crow - threshold_s : crow + threshold_s,
        ccol - threshold_s : ccol + threshold_s,
    ] = 1.0
    return mask
