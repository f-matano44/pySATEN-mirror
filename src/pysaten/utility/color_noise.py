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
    freqs = torch.fft.rfftfreq(len(wh), 1 / sr)
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
    freqs = torch.fft.rfftfreq(len(wh), 1 / sr)
    # white -> pink
    PK_f = WH_f.clone()
    for i in range(len(WH_f)):
        PK_f[i] = WH_f[i] / torch.sqrt(freqs[i]) if 20 < freqs[i] else 0
    # irfft
    pk = torch.fft.irfft(PK_f)
    # normalize
    pk /= pk.abs().max()

    return pk[offset : length + offset]
