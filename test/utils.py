import numpy as np


def load_answer(filename: str):
    ans = []
    with open(filename, "r") as file:
        for line in file:
            ans.append([x for x in line.split(" ")])
    return float(ans[0][1]) / 1e7, float(ans[-1][0]) / 1e7


def gen_noise_signal(
    x: np.ndarray,
    fs: int,
    snr: int,
    is_white: bool,
    rand: np.random.Generator,
    ans_s_sec: float,
    ans_e_sec: float,
):
    speech_start_idx: int = int(ans_s_sec * fs)
    speech_end_idx: int = int(ans_e_sec * fs)
    # generate noise (white or pink)
    noise = (
        rand.uniform(low=-1.0, high=1.0, size=len(x))
        if is_white
        else _gen_pink_noise(len(x), fs, rand)
    )
    # mix stationary noise and signal (in specified snr)
    noise_scale = _determine_noise_scale(x[speech_start_idx:speech_end_idx], noise, snr)
    noise_x = x + noise * noise_scale
    # add pulse noise
    # generate pulse
    pulse = rand.random(2) - 0.5 * 2
    # determine index adding pulse noise
    start_pulse_index = np.random.randint(0, speech_start_idx)
    end_pulse = np.random.randint(speech_end_idx, len(x) - 1)
    # add pulse noise
    noise_x[start_pulse_index] = pulse[0]
    noise_x[end_pulse] = pulse[1]
    return noise_x


def _gen_pink_noise(length: int, fs: int, rand: np.random.Generator) -> np.array:
    length2 = length + 1000

    # white noise
    wh = rand.uniform(low=-1.0, high=1.0, size=length2)
    # fft
    WH = np.fft.rfft(wh)
    WH_f = np.fft.rfftfreq(len(wh), 1 / fs)
    # white -> pink
    PK = WH.copy()
    for i in range(len(WH)):
        PK[i] = WH[i] / np.sqrt(WH_f[i]) if WH_f[i] > 20 else 0
    # ifft
    pk = np.fft.irfft(PK)
    # normalize
    pk /= np.max(np.abs(pk))

    return pk[:length]


def _determine_noise_scale(
    signal: np.array, noise: np.array, desired_snr_db: int
) -> float:
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    return float(scaling_factor)
