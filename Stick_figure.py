import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from scipy import signal

from scipy.signal import butter, lfilter, freqz


def get_data(file_name):
  samplerate_, data_ = wavfile.read(file_name)
  data_og_ = data_
  data_ = data_.T

  return data_, data_og_, samplerate_


def remove_high_freq(wave_, lc):
  nyq = 0.5 * fs
  low = lc/nyq
  sos = signal.cheby1(2, low, 2, 'hp', fs=fs, output='sos')
  filtered = signal.sosfilt(sos, wave_)

  return filtered


def butter_lowpass(cutoff, fs, order=2):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  [b, a] = butter(order, normal_cutoff, btype='low')
  return b, a


def butter_lowpass_filter(data, cutoff, fs, order=2):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y


def t_value(xx, fs_):
  T_ = len(xx) / fs_
  nsamples = int(T_ * fs_)
  tt = np.linspace(0, T_, nsamples, endpoint=False)
  return tt, T_


print(os.getcwd())
dir_name = "temp/634_sqa_1_0.wav"
sig, sig_og, fs = get_data(dir_name)

sig_a = butter_lowpass_filter(sig[0:fs*6], 2, fs)
x, T = t_value(sig_a, fs)

plt.plot(x, sig_a)
plt.show()

