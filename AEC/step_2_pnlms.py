import numpy as np
import os
import scipy.io.wavfile as wavfile
from os.path import join
import matplotlib.pyplot as plt
import matplotlib, librosa
from tqdm import tqdm
import soundfile as sf


"""
H[n+1] = H[n] + prop * mu * E[n-1] * conj(X[n-1]) / ema_X2
with pre-emphsize & notch dc & proportional mu
"""


class aec_state:
    filter_length = 1024 * 5
    nframe = 128
    nfft = nframe * 2
    nfreq = nfft // 2 + 1

    nblocks = (filter_length + nframe - 1) // nframe

    X_fifo = np.zeros([nblocks + 1, nfreq], dtype=np.complex128)
    H_hat = np.zeros([nblocks, nfreq], dtype=np.complex128)
    X = np.zeros([nfreq], dtype=np.complex128)
    Y = np.zeros([nfreq], dtype=np.complex128)
    Y_hat = np.zeros([nfreq], dtype=np.complex128)
    Error = np.zeros([nfreq], dtype=np.complex128)

    y = np.zeros([nfft], dtype=np.float64)
    x = np.zeros([nfft], dtype=np.float64)
    y_hat = np.zeros([nfft], dtype=np.float64)
    error = np.zeros([nfft], dtype=np.float64)

    X_energy = np.zeros([nfreq], dtype=np.float64)
    inv_X_energy = np.ones([nfreq], dtype=np.float64)
    prop = np.zeros([nblocks], dtype=np.float64)

    notch_mem = np.zeros([2], dtype=np.float64)
    mem_x = np.zeros([1], dtype=np.float64)
    mem_y = np.zeros([1], dtype=np.float64)
    mem_o = np.zeros([1], dtype=np.float64)
    preemph = 0.9
    notch_radius = 0.994

    Sxx = 0.
    See = 0.
    ss = 0.35 / nblocks

    def __init__(self):
        M = self.nblocks
        # Ratio of ~10 between adaptation rate of first and last block */
        decay = np.exp(-2.4 / M)
        self.prop[0] = 0.7
        sum_ = self.prop[0]
        for i in range(1, M, 1):
            self.prop[i] = self.prop[i - 1] * decay
            sum_ = sum_ + self.prop[i]
        for i in range(M - 1, -1, -1):
            self.prop[i] = 0.8 * self.prop[i] / sum_


def preemph(x, coef, mem):
    for i in range(x.shape[0]):
        tmp = x[i] - coef * mem[0]
        mem[0] = x[i]
        x[i] = tmp


def mdf_adjust_prop(st):
    # compute block energy
    tmp = 1 + np.sum(st.H_hat * np.conj(st.H_hat), axis=1).real
    st.prop[:] = np.sqrt(tmp)

    # add small portion of max_sum to every entry
    max_sum = np.max(st.prop) + 1.0
    st.prop[:] += 0.1 * max_sum

    # do normalizing by prop_sum
    prop_sum = np.sum(st.prop) + 1.0
    st.prop[:] = (0.99 / prop_sum) * st.prop


def filter_dc_notch16(memNotch, in_audio, radius):
    frame_size = in_audio.shape[0]
    den2 = radius * radius + 0.7 * (1 - radius) * (1 - radius)
    for i in range(frame_size):
        vin = in_audio[i]
        vout = memNotch[0] + vin
        memNotch[0] = memNotch[1] + 2 * (-vin + radius * vout)
        memNotch[1] = vin - den2 * vout
        in_audio[i] = radius * vout


def aec_process_frame(st, out, y_data, x_data):
    N, M, F = st.nfft, st.nblocks, st.nframe

    filter_dc_notch16(st.notch_mem, y_data, st.notch_radius)
    preemph(y_data, st.preemph, st.mem_y)
    preemph(x_data, st.preemph, st.mem_x)

    # moving update x and X-queue
    st.x[:F] = st.x[F:]
    st.x[F:] = x_data
    st.X_fifo[1:, :] = st.X_fifo[:-1, :]
    st.X_fifo[0, :] = np.fft.rfft(st.x)

    # compute y_hat and error
    st.Y_hat[:] = np.sum(st.X_fifo[:-1, :] * st.H_hat, axis=0)
    st.y_hat[:] = np.fft.irfft(st.Y_hat).real
    st.error[F:] = y_data.astype(np.float64) - st.y_hat[F:]

    # compute gradients and linearize filter
    st.H_hat[:, :] += st.prop[..., np.newaxis] * (st.inv_X_energy * (np.conj(st.X_fifo[1:, :]) * st.Error))
    wtmp = np.fft.irfft(st.H_hat, axis=1).real
    wtmp[:, F:] = 0.
    st.H_hat[:, :] = np.fft.rfft(wtmp, axis=1)

    if 1:
        mdf_adjust_prop(st)

    # compute output
    for i in range(F):
        tmp = st.error[F + i]
        tmp += st.preemph * st.mem_o
        st.mem_o = tmp
        out[i] = np.int16(min(max(tmp, -32768), 32767))

    # compute frequency domain error
    st.error[:F] = 0.
    st.Error[:] = np.fft.rfft(st.error)

    # compute frequency domain recursive average energy of x
    X2 = (st.X_fifo[0, :] * np.conj(st.X_fifo[0, :])).real
    st.X_energy[:] = (1. - st.ss) * st.X_energy + st.ss * X2

    # compute learning rate
    Sxx = np.dot(x_data, x_data)
    See = max(np.dot(st.error[F:], st.error[F:]), 100*N)
    lr = 0.0
    # if far-end signal nearly silent, don't adapt
    if Sxx > 1000 * N:
        lr = 0.25 * min(See, Sxx) / See
    st.inv_X_energy[:] = lr / (10 + st.X_energy)


st = aec_state()
F = st.nframe

outdir = os.path.basename(__file__).split('.')[0] + "_out"
os.makedirs(outdir, exist_ok=True)

far_file = join("../test_files/farend_SpDTs247_double.wav")
mic_file = join(f"../test_files/mic_SpDTs247_double.wav")
_, mic_audio = wavfile.read(mic_file)
_, far_audio = wavfile.read(far_file)
subdir = os.path.basename(os.path.dirname(mic_file))
out_path = join(outdir, f"{outdir}_{subdir}_{os.path.basename(mic_file)[:-4]}_9.wav")


num_frames = (mic_audio.shape[0] - st.nfft) // F

main_out = np.zeros(mic_audio.shape[0], dtype=np.int16)
shadow_out = np.zeros(mic_audio.shape[0], dtype=np.float64)
frame_mic = np.zeros(F, dtype=np.float64)
frame_far = np.zeros(F, dtype=np.float64)

for i in tqdm(range(num_frames)):
    frame_mic[:] = mic_audio[i * F: (i + 1) * F]
    frame_far[:] = far_audio[i * F: (i + 1) * F]
    frame_out = main_out[i * F: (i + 1) * F]

    aec_process_frame(st, frame_out, frame_mic, frame_far)

sf.write(out_path, main_out, 16000)
