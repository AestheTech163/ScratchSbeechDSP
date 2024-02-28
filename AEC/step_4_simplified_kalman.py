import numpy as np
import os
import scipy.io.wavfile as wavfile
from os.path import join
import matplotlib.pyplot as plt
import matplotlib, librosa
import scipy.signal
from tqdm import tqdm
import soundfile as sf


########################################################
# frequency domain diagonal kalman
########################################################


class aec_state:
    A = 0.999
    A2 = A * A
    A2_1 = 1.0 - A2

    filter_length = 1024 * 5
    nframe = 256
    nfft = nframe * 2
    nfreq = nfft // 2 + 1

    nblocks = (filter_length + nframe - 1) // nframe

    X_fifo = np.zeros([nblocks + 1, nfreq], dtype=np.complex128)
    H_hat = np.zeros([nblocks, nfreq], dtype=np.complex128)
    H_hat_tmp = np.zeros([nblocks, nfreq], dtype=np.complex128)
    W2 = np.zeros([nblocks, nfreq], dtype=np.float64)
    H_hat_plus = np.zeros([nblocks, nfreq], dtype=np.complex128)
    X = np.zeros([nfreq], dtype=np.complex128)
    Y = np.zeros([nfreq], dtype=np.complex128)
    Y_hat = np.zeros([nfreq], dtype=np.complex128)
    Error = np.zeros([nfreq], dtype=np.complex128)
    E2_fifo = np.zeros([nblocks, nfreq], dtype=np.float64)

    P = np.ones([nblocks, nfreq], dtype=np.float64)
    X2_fifo = np.zeros([nblocks + 1, nfreq], dtype=np.float64)
    PHIss = np.zeros([nblocks, nfreq], dtype=np.float64)

    y = np.zeros([nfft], dtype=np.float64)

    tmp_out = np.zeros([nframe], dtype=np.float64)
    out_buf = np.zeros([nframe], dtype=np.float64)
    x = np.zeros([nfft], dtype=np.float64)
    y_hat = np.zeros([nfft], dtype=np.float64)
    error = np.zeros([nfft], dtype=np.float64)

    X_energy = np.zeros([nfreq], dtype=np.float64)
    inv_X_energy = np.ones([nfreq], dtype=np.float64)

    notch_mem = np.zeros([2], dtype=np.float64)
    prop = np.zeros([nblocks], dtype=np.float64)
    mem_x = np.zeros([1], dtype=np.float64)
    mem_y = np.zeros([1], dtype=np.float64)
    mem_o = np.zeros([1], dtype=np.float64)
    preemph = 0.9
    notch_radius = 0.964
    Sxx = 0.
    See = 0.
    rho = 0.2

    def __init__(self):
        M = self.nblocks
        self.window = scipy.signal.get_window("hamming", 32)[:16]
        self.window_inv = scipy.signal.get_window("hamming", 32)[16:]
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


def filter_dc_notch(memNotch, in_audio, radius):
    frame_size = in_audio.shape[0]
    den2 = radius * radius + 0.7 * (1 - radius) * (1 - radius)
    for i in range(frame_size):
        vin = in_audio[i]
        vout = memNotch[0] + vin
        memNotch[0] = memNotch[1] + 2 * (-vin + radius * vout)
        memNotch[1] = vin - den2 * vout
        in_audio[i] = radius * vout


def compute_output(st, out, de_emph=True):
    F = st.nframe
    for i in range(F):
        tmp = st.error[F + i]
        if de_emph:
            tmp += st.preemph * st.mem_o
            st.mem_o = tmp
        out[i] = np.int16(min(max(tmp, -32768), 32767))

    # out[:16] = st.tmp_out[:16] * st.window + st.out_buf[-16:] * st.window_inv
    # out[16:] = st.tmp_out[16:]
    # st.out_buf[:] = st.tmp_out


def aec_process_frame(st, out, y_data, x_data):
    N, M, F = st.nfft, st.nblocks, st.nframe
    filter_dc_notch(st.notch_mem, y_data, st.notch_radius)
    # preemph(y_data, st.preemph, st.mem_y)
    # preemph(x_data, st.preemph, st.mem_x)

    # update x, X-queue, X2-queue
    st.x[:F] = st.x[F:]
    st.x[F:] = x_data
    st.X_fifo[1:, :] = st.X_fifo[:-1, :]
    st.X_fifo[0, :] = np.fft.rfft(st.x)
    st.X2_fifo[1:, :] = st.X2_fifo[:-1, :]
    st.X2_fifo[0, :] = (st.X_fifo[0, :] * np.conj(st.X_fifo[0, :])).real

    # compute y_hat and error
    st.Y_hat[:] = np.sum(st.X_fifo[:-1, :] * st.H_hat, axis=0)
    st.y_hat[:] = np.fft.irfft(st.Y_hat).real
    st.error[F:] = y_data - st.y_hat[F:]
    st.error[:F] = 0.
    st.Error[:] = np.fft.rfft(st.error)

    # update fg filter
    st.W2[:, :] = (st.H_hat[:, :] * np.conj(st.H_hat[:, :])).real
    st.PHIss[0, :] = (np.conj(st.Error) * st.Error).real
    PHIee = st.P[:, :] * np.sum(st.X2_fifo[:-1, :], axis=0)
    mu = st.P[:, :] / (PHIee + 2 * st.PHIss[0, :])
    st.H_hat[:, :] = st.H_hat[:, :] + mu * np.conj(st.X_fifo[:-1, :]) * st.Error[:]
    st.P[:, :] = st.A2 * st.P[:, :] * (1.0 - 0.5 * mu * st.X2_fifo[:-1, :]) + st.A2_1 * st.W2[:, :]

    wtmp = np.fft.irfft(st.H_hat, axis=1).real
    wtmp[:, F:] = 0.
    st.H_hat[:, :] = np.fft.rfft(wtmp, axis=1)

    # compute output
    compute_output(st, out, de_emph=False)


st = aec_state()
F = st.nframe

outdir = os.path.basename(__file__).split('.')[0] + "_out"
os.makedirs(outdir, exist_ok=True)

far_file = join("../test_files/farend_SpDTs247_double.wav")
mic_file = join(f"../test_files/mic_SpDTs247_double.wav")
_, mic_audio = wavfile.read(mic_file)
_, far_audio = wavfile.read(far_file)
subdir = os.path.basename(os.path.dirname(mic_file))
out_path = join(outdir, f"{outdir}_{subdir}_{os.path.basename(mic_file)[:-4]}_1.wav")

num_frames = (mic_audio.shape[0] - st.nfft) // F

main_out = np.zeros(mic_audio.shape[0], dtype=np.int16)
frame_mic = np.zeros(F, dtype=np.float64)
frame_far = np.zeros(F, dtype=np.float64)

for i in tqdm(range(num_frames)):
    frame_mic[:] = mic_audio[i * F: (i + 1) * F]
    frame_far[:] = far_audio[i * F: (i + 1) * F]
    frame_out = main_out[i * F: (i + 1) * F]

    aec_process_frame(st, frame_out, frame_mic, frame_far)

sf.write(out_path, main_out, 16000)
