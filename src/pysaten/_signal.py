from math import ceil

from numpy import diff, isnan, sign, zeros


def zcr(y, win_length, hop_length):
    zc = zeros(int(ceil(len(y) / hop_length)))
    for i in range(len(zc)):
        # get target array
        idx = i * hop_length
        zc_start = int(max(0, idx - (win_length / 2)))
        zc_end = int(min(idx + (win_length / 2), len(y) - 1))
        target = y[zc_start:zc_end]
        # calc zcr
        sign_arr = sign(target)[target != 0 & ~isnan(target)]
        zc[i] = sum(abs(diff(sign_arr)) != 0) / hop_length
    return zc
