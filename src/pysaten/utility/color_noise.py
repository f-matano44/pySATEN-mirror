import torch


def white(length: int, seed: int, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.rand(length, generator=gen, device=device) * 2.0 - 1.0


def blue(length: int, sr: float, seed: int, device: str = "cpu") -> torch.Tensor:
    offset = int(length / 2)
    # white noise
    wh = white(length + (offset * 2), seed, device)
    # fft
    WH_f = torch.fft.rfft(wh)
    freqs = torch.fft.rfftfreq(len(wh), 1 / sr).to(WH_f.device)
    # white -> blue
    BL_f = WH_f * torch.sqrt(freqs)
    # irfft
    bl = torch.fft.irfft(BL_f)
    # normalize
    bl /= bl.abs().max()

    return bl[offset : length + offset]


def pink(length: int, sr: float, seed: int, device: str = "cpu") -> torch.Tensor:
    offset = int(length / 2)
    # white noise
    wh = white(length + (offset * 2), seed, device)
    # fft
    WH_f = torch.fft.rfft(wh)
    freqs = torch.fft.rfftfreq(len(wh), 1 / sr).to(WH_f.device)
    # white -> pink
    mask = freqs > 20.0
    PK_f = torch.zeros_like(WH_f)
    PK_f[mask] = WH_f[mask] / torch.sqrt(freqs[mask])
    # irfft
    pk = torch.fft.irfft(PK_f)
    # normalize
    pk /= pk.abs().max()

    return pk[offset : length + offset]
