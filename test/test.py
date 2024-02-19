import soundfile as sf
from tqdm import tqdm
from pysaten import pysaten
from rvad import rvad_fast
from rvad import speechproc
import scipy.stats as stats
import numpy as np

def _main():
    np.random.seed(0)
    for snr in [20, 15, 10, 5, 0, -5]:
        saten = []
        rvad = []
        print(f"SNR: {snr}")
        for i in tqdm(range(1,101)):
            ans_s, ans_e = load_answer(f"./jsut-lab/labels/basic5000/BASIC5000_{i:04}.lab")
            x, fs = sf.read(f"./jsut_ver1.1/basic5000/wav/BASIC5000_{i:04}.wav")
            x = gen_noise_signal(x, snr, True)
            S, E = pysaten.vsed(x, fs)
            E = 0 if S == 0 and 0.01 >= abs(E - len(x) / fs) else E
            saten.append(abs(S - ans_s))
            saten.append(abs(E - ans_e))
            S, E = rvad_fast.vad(x, fs)
            rvad.append(abs(S - ans_s))
            rvad.append(abs(E - ans_e))
        print(f"saten error mean: {np.mean(saten):.3f} [s]")
        print(f"rvad error mean:  {np.mean(rvad):.3f} [s]")
        print(f"saten error 95%:  {_bootstrap_confidence_interval(saten):.3f} [s]")
        print(f"rvad error 95%:   {_bootstrap_confidence_interval(rvad):.3f} [s]")
        print(f"diff: {_wilcoxon_test(saten, rvad)}")
        print("")


def load_answer(filename):
    ans = []
    with open(filename, 'r') as file:
        for line in file:
            ans.append([x for x in line.split(" ")])
    return float(ans[0][1]) / 1e7, float(ans[-1][0]) / 1e7


def _bootstrap_confidence_interval(sample, confidence_level=0.95, n_iterations=1000):
    bootstrapped_means = []
    for _ in range(n_iterations):
        bootstrapped_sample = np.random.choice(sample, size=len(sample), replace=True)
        bootstrapped_means.append(np.mean(bootstrapped_sample))
    lower_bound = np.percentile(bootstrapped_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + confidence_level) / 2 * 100)
    return (upper_bound - lower_bound) / 2


def _wilcoxon_test(list1, list2, alpha=0.05):
    stat, p_value = stats.wilcoxon(list1, list2)
    return p_value < alpha


def gen_noise_signal(x, snr, is_white):
    # ノイズを生成（ホワイトノイズまたはピンクノイズ）
    noise = np.random.normal(0, 1, len(x)) \
        if is_white else _pink_noise(len(x))
    # ノイズの長さを信号の長さに合わせる
    noise_padded = _repeat_to_length(noise, len(x))
    # 合成信号の生成
    return _add_noise_to_signal(x, noise_padded, snr)


def _pink_noise(length):
    """ ピンクノイズを生成する """
    # ホワイトノイズを生成
    white_noise = np.random.normal(0, 1, length)

    # 1/fノイズに変換
    from scipy.signal import lfilter
    num_taps = 64  # フィルタのタップ数
    f = np.array([1.0 / i for i in range(1, num_taps + 1)])
    pink_noise = lfilter(f, [1.0], white_noise)
    return pink_noise


def _repeat_to_length(signal, target_length):
    """ シグナルを繰り返して目標の長さにする """
    repeat_times = np.ceil(target_length / len(signal)).astype(int)
    return np.tile(signal, repeat_times)[:target_length]


def _add_noise_to_signal(signal, noise, desired_snr_db):
    """ 指定されたSNRでノイズを信号に追加する """
    desired_snr_linear = 10 ** (desired_snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(signal_power / (desired_snr_linear * noise_power))
    noise_adjusted = noise * scaling_factor
    return signal + noise_adjusted


def _calculate_snr_db(signal, noise):
    """ SNRをデシベル単位で計算する """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


if __name__ == "__main__":
    _main()
