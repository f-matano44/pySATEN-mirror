from __future__ import division
import numpy as np
from scipy.signal import lfilter
import rvad.speechproc
from copy import deepcopy


# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, vol. 59, pp. 1-21, 2020. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# Version: 2.0
# 02 Dec 2017, Achintya Kumar Sarkar and Zheng-Hua Tan

def vad(data, fs):
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5; vadThres = 0.4
    opts = 1

    ft, flen, fsh10, nfr10 = rvad.speechproc.sflux(
        data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(nfr10)
    pv01[np.less_equal(ft, ftThres)] = 1 
    pitch = deepcopy(ft)

    pvblk = rvad.speechproc.pitchblockdetect(
        pv01, pitch, nfr10, opts)


    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata=lfilter(b, a, data, axis=0)


    #--pass 1--
    noise_samp, noise_seg, n_noise_samp = rvad.speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

    #sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j,0]), int(noise_samp[j,1]) +1)] = 0 


    vad_seg = rvad.speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)


    start = np.min(np.where(vad_seg != 0)) * ovrlen if np.any(vad_seg != 0) else 0
    end = np.max(np.where(vad_seg != 0)) * ovrlen if np.any(vad_seg != 0) else 0

    return start, end
