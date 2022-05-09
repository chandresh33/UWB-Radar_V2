import os
import pickle
import shutil
from pathlib import Path

import pywt

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy import signal
from scipy.io import wavfile
from tqdm.notebook import tqdm
# import librosa

def get_data(file_name):
    samplerate_, data_ = wavfile.read(file_name)
    data_og_ = data_
    data_ = data_.T

    return data_, data_og_, samplerate_


def split_files(aa, bb):
    a_files = []
    b_files = []
    n_files = []

    wav_ref = ".wav"

    for i in os.listdir('Tori_data'):
        if i.endswith(wav_ref):
            ii = i.replace(wav_ref, "")
            if ii.endswith(aa):
                a_files.append(i)
            elif ii.endswith(bb):
                b_files.append(i)
            else:
                n_files.append(i)

    return a_files, b_files, n_files


def remove_high_freq(wave_, fs):
    sos = signal.cheby1(4, 1, 5, 'hp', fs=fs, output='sos')
    sos2 = signal.cheby1(3, 3, 0.1, 'lp', fs=fs, output='sos')

    filtered = signal.sosfilt(sos, wave_)
    filtered = signal.sosfilt(sos2, filtered)

    return filtered


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    [b_, a_] = signal.butter(order, [low, high], btype='band')
    return b_, a_


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b_, a_ = butter_bandpass(lowcut, highcut, fs, order=order)
    yy = signal.lfilter(b_, a_, data)
    return yy


def split_hand_waves(data_, s_rate):
    ch_1_og, ch_2_og = data_[:]
    # ch_1 = remove_high_freq(ch_1_og[110000:])
    # ch_2 = remove_high_freq(ch_2_og[110000:])

    ch_1 = butter_bandpass_filter(ch_1_og, 10, 40, s_rate)
    ch_2 = butter_bandpass_filter(ch_2_og, 10, 40, s_rate)

    # ch_1_full = [ch_1[], ch_1[], ch_1[], ch_1[], ch_1[]]

    plt.subplot(211)
    plt.plot(np.arange(len(ch_1_og)), ch_1_og, 'r', label='Original Signal')

    plt.legend()
    plt.xlabel("Samples, N")
    plt.ylabel("Signal Amplitude")

    plt.subplot(212)
    plt.plot(np.arange(len(ch_1)), ch_1, 'g', label="Filtered Signal")
    plt.vlines(len(ch_1_og)/2, 0, 500, linestyles="solid", colors="k")

    plt.xlabel("Samples, N")
    plt.ylabel("Signal Amplitude")
    plt.legend()

    plt.show()

    return ch_1, ch_2


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def t_value(x, fs_):
    T = len(x) / fs_
    nsamples = int(T * fs_)
    tt = np.linspace(0, T, nsamples, endpoint=False)
    return tt, T


def signal_segmentation(f_name, sig):
    def plot_function():
        plt.subplot(211)
        # plt.title("I")
        plt.plot(t_a[0], sig_a, label='original', zorder=30)
        plt.plot(t_a[0], y_a, label='filtered', zorder=50)
        plt.ylabel("Signal amplitude")
        plt.xlabel("time (s)")

        for a in range(len(tp_a)):
            plt.vlines(tp_a[a], min(sig_a), max(sig_a), linestyles="dashed", colors="r", linewidth=2, zorder=100)
            plt.vlines(tp_mid_a[a], min(sig_a), max(sig_a), linestyles="dashed", colors="g", linewidth=2, zorder=100)

        plt.legend()

        plt.subplot(212)
        plt.title("Q")
        plt.plot(t_b[0], sig_b, label='original', zorder=30)
        plt.plot(t_b[0], y_b, label='filtered', zorder=50)
        plt.ylabel("Signal amplitude")
        plt.xlabel("time (s)")


        for b in range(len(tp_b)):
            plt.vlines(tp_b[b], min(sig_b), max(sig_b), linestyles="dashed", colors="r", linewidth=2, zorder=100)
            plt.vlines(tp_mid_b[b], min(sig_b), max(sig_b), linestyles="dashed", colors="g", linewidth=2, zorder=100)

        plt.legend()
        plt.tight_layout()
        plt.show()

    test_sig, test_sig_og, test_rate = get_data("Tori_data/" + sig)
    sig_a, sig_b = test_sig[0], test_sig[1]

    def plot_freq_time(sig_, sp):
        fourier = np.fft.fft(sig_)

        n = len(sig_)
        # getting length of channel
        ll = np.int(np.ceil(n / 2))

        # scale the signal and align for magnitude
        fourier = fourier[0:ll - 1]
        fourier = fourier / float(n)
        freqArray = np.arange(0, ll - 1, 1.0) * (test_rate * 1.0 / n)

        plt.figure(2, figsize=(7, 7))
        plt.subplot(sp)
        Pxx, freqs, bins, im = plt.specgram(sig_, Fs=test_rate, NFFT=2048, cmap=plt.get_cmap('RdPu'))
        cbar = plt.colorbar(im)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        cbar.set_label('Intensity (dB)')

    # plot_freq_time(sig_a, 211)
    # plot_freq_time(sig_b, 212)
    # plt.show()

    def save_obj_func():
        if os.path.isdir(path):
            data_dict = load_obj(path + '\\test_points')
            # data_dict[f_name] = {sig: {"CH:1": [tp_a, tp_mid_a], "CH:2": [tp_b, tp_mid_b]}}
            data_dict[f_name][sig] = {"CH:1": [tp_a, tp_mid_a], "CH:2": [tp_b, tp_mid_b]}
            save_obj(data_dict, path + '\\test_points')
        else:
            data_dict = {sig: {"CH:1": [tp_a, tp_mid_a], "CH:2": [tp_b, tp_mid_b]}}
            os.makedirs(path)
            save_obj(data_dict, path + '\\test_points')

        print(data_dict.keys())
        print("\n")
        print(data_dict)

    t_a, t_b = t_value(sig_a, test_rate),  t_value(sig_b, test_rate)

    cutoff_range = [1, 10, 30, 40, 60, 80, 100, 200, 400, 600]

    tp_a = [[2.84, 4.46], [4.46, 6.19], [6.19, 7.97], [7.97, 9.68], [9.68, 11.47]]
    tp_mid_a = [3.64, 5.36, 7.16, 8.87, 10.52]

    tp_b = [[2.84, 4.50], [4.50, 6.19], [6.19, 7.97], [7.97, 9.69], [9.69, 11.547]]
    tp_mid_b = [3.65, 5.37, 7.14, 8.81, 10.52]

    dir_name = "\Tori_data"
    parent_dir = os.getcwd()
    path = parent_dir + dir_name

    # To save or load obj
    # save_obj_func()
    #
    y_a = butter_bandpass_filter(sig_a, 10, 50, test_rate)
    y_b = butter_bandpass_filter(sig_b, 10, 50, test_rate)
    plot_function()
    # load_dict = load_obj(path + '\\test_points')
    # print(load_dict)


def break_signals(f_path, data_dict):

    for activity in data_dict.keys():
        activity_dict = data_dict[activity]
        active_name = activity[0:3]
        for file in activity_dict.keys():
            file_ext = "_" + file.split("_")[1]
            full_sig, sig_og, fs = get_data("Tori_data/" + file)

            for ch in activity_dict[file].keys():
                idx = int(ch[-1:]) - 1
                sig = full_sig[idx]
                t = t_value(sig, fs)[0]

                ch_ext = "_" + ch.replace(":", "")
                file_name = f_path+(active_name+file_ext+ch_ext).lower()

                channel_dict = activity_dict[file][ch]

                sig_points = channel_dict[0]
                mid_points = channel_dict[1]

                for i in range(len(sig_points)):
                    sig_pts = [round(ii * fs) for ii in sig_points[i]]
                    s_pts, m_pts, e_pts = sig_pts[0], round(mid_points[i] * fs), sig_pts[1]
                    motion_exts = ["_sl", "_sm", "_ml"]
                    f_name_sub = file_name + "_x" + str(i)
                    f_names = [f_name_sub+ext for ext in motion_exts]

                    full_signal = (f_names[0], sig[s_pts:e_pts])
                    mid_sig_start = (f_names[1], sig[s_pts:m_pts])
                    mid_sig_end = (f_names[2], sig[m_pts:e_pts])

                    wavfile.write(full_signal[0]+".wav", fs, full_signal[1])
                    wavfile.write(mid_sig_start[0]+".wav", fs, mid_sig_start[1])
                    wavfile.write(mid_sig_end[0]+".wav", fs, mid_sig_end[1])

    print("Finished Saving Files")


def test_train_split(data_dir):
    files = os.listdir(data_dir)
    file_list = []
    train_ext, test_ext = r"\Train_set\\", r"\Test_set\\"

    if not os.path.isdir(data_dir+train_ext):
        os.makedirs(data_dir+train_ext)

    if not os.path.isdir(data_dir+test_ext):
        os.makedirs(data_dir+test_ext)

    for f in files:
        if f.endswith(".wav"):
            file_list.append(f)

    split_array = np.arange(len(file_list))
    np.random.shuffle(split_array)
    train_split_val = int(len(file_list)*0.7)

    train_split, test_split = split_array[:train_split_val], split_array[train_split_val:]

    print(data_dir+train_ext)
    for tr in train_split:
        shutil.move(data_dir+"\\"+file_list[tr], data_dir+train_ext)

    for te in test_split:
        shutil.move(data_dir+"\\"+file_list[te], data_dir+test_ext)

    print("Finished moving files into Test_set and Train_set")

def normalise(sig_1):  # this function accepts two signals and returns the normalised versions for each
    sig_1 = (sig_1 - np.min(sig_1)) / (np.max(sig_1) - np.min(sig_1))  # normalising of sig_1
    return sig_1


def mel_spec(sig, fs):
  mel_spec = librosa.feature.melspectrogram(y=sig, sr=fs, n_mels=128, fmax=2000)
  sdb = librosa.power_to_db(mel_spec, ref=np.max)
  return mel_spec, sdb


squat_f, star_f, still_f = split_files("a", "b")

fs = 44100
print(still_f)

# test_sig, test_sig_og, test_rate = get_data("Processed_data/Test_set/squ_1_ch1_x0_ml" + still_f[2])
test_sig, test_sig_og, test_rate = get_data("temp/811_sta_1_1.wav")

# sig_a, sig_b = test_sig[0][40857:], test_sig[1]
sig_a, sig_b = test_sig[0], test_sig[1]
# sig_a = remove_high_freq(sig_a, test_rate)
# spec = mel_spec(sig_a, test_rate)
t, T = t_value(sig_a, test_rate)

# plt.imshow(spec)3483

# plt.show()

# plt.plot(t, normalise(sig_a))
plt.plot(t, normalise(sig_a))
plt.ylabel('Signal Amplitude')
plt.xlabel('Time (s)')
plt.title('Heart Rate Signal Filtered')
plt.show()
#
# (ca, cd) = pywt.dwt(sig_a,'bior3.7')
#
# cat = pywt.threshold(ca, np.std(ca)/0.5, 'soft')
# cdt = pywt.threshold(cd, np.std(cd)/0.5, 'soft')
#
# ts_rec = pywt.idwt(cat, cdt, 'bior3.7')

# plt.close('all')
# plt.subplot(311)
# # Original coefficients
# plt.plot(ca, '--*b')
# plt.plot(cd, '--*r')
# # Thresholded coefficients
# plt.plot(cat, '--*c')
# plt.plot(cdt, '--*m')
# plt.legend(['ca','cd','ca_thresh', 'cd_thresh'], loc=0)
# plt.grid('on')
#
# plt.subplot(312)
# plt.plot(sig_a)
#
# plt.subplot(313)
# plt.plot(t, ts_rec[:-1], 'r')
#
# plt.legend(['original signal', 'reconstructed signal'])
# plt.grid('on')
# plt.show()

# signal_segmentation("Star", star_f[4])  # completed (onl to be used for labelling the signals
#
# dir_name_ = "\Tori_data\\"
# parent_dir_ = os.getcwd()
# path_ = parent_dir_ + dir_name_
# data_dict_ = load_obj(path_ + '\\test_points')

# break_signals(path_, data_dict_)  # this is for generating the different files and svaing them
# test_train_split(path_)  # This method is to generate two folders, Test_set and Train_set and populate them randomly
