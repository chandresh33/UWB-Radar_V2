import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import torch
from scipy import signal
from scipy.io import wavfile


def open_wav(full_path):
    return wavfile.read(full_path)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_maxlen(path_type):
    counter = 0
    glob_max, glob_min = 0, 0
    glob_max_val, glob_min_val = 0, 0

    global fs

    for file in os.listdir(path_type):
        fs, data_ = open_wav(path_type+file)
        len_prev = len(data_)
        val_min, val_max = min(data_), max(data_)

        if counter == 0:
            glob_min = len_prev
            glob_min_val = val_min
            glob_max_val = val_max

        if len_prev == 0:
            print("Empty file: ", file)

        if counter > 0:
            if len_prev > glob_max:
                glob_max = len_prev

            if val_max > glob_max_val:
                glob_max_val = val_max

            if val_min < glob_min_val:
                glob_min_val = val_min

            if len_prev < glob_min:
                glob_min = len_prev

        counter += 1
    return (glob_min, glob_max), (glob_min_val, glob_max_val)


def global_minmax(mm_a, mm_b):
    global_minimium, global_maximum = min([mm_a[0], mm_b[0]]), max([mm_a[1], mm_b[1]])
    return global_minimium, global_maximum


def t_value(x, fs_):
    TT = len(x) / fs_
    nsamples = int(TT * fs_)
    tt = np.linspace(0, TT, nsamples, endpoint=False)
    return tt, TT


def normalise(x, glob_min, glob_max):
    x_bar = (2 * ((x-glob_min)/(glob_max-glob_min))) - 1
    return x_bar


def pad_signals(file_pth, max_len_, g_min, g_max):
    for filename in os.listdir(file_pth):
        filename = file_pth + filename
        fs_, data = open_wav(filename)
        data = normalise(data, g_min, g_max)
        data_mean = np.mean(data)

        if len(data) < max_len_:
            silence = max_len_ - len(data)
            s_1 = int(silence / 2)
            s_2 = silence - s_1

            new_data = np.append(np.ones(s_1) * data_mean, data)
            new_data = np.append(new_data, np.ones(s_2) * data_mean)

            wavfile.write(filename, fs_, new_data)

    print("Finished Padding signals")


fs = 44100  # is redefined from get_maxlen() method
path = "Processed_data\\"
train_path, test_path = path+"Train_set\\", path+"Test_set\\"

# min_len, max_len = global_minmax(get_maxlen(train_path)[0], get_maxlen(test_path)[0])
# min_val, max_val = global_minmax(get_maxlen(train_path)[1], get_maxlen(test_path)[1])
#
# print("Max T:", max_len/fs)
# print("Min Val:", min_val, "Max Val: ", max_val)
# pad_signals(test_path, max_val, min_val, max_val)  # used to pad all signals to make them the same length

