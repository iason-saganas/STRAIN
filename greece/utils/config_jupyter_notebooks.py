import matplotlib.pyplot as plt
import numpy as np
import nifty8 as ift
import pickle
from gwpy.timeseries import TimeSeries


def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def pickle_me_this(filename: str, data_to_pickle: object):
    path = filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


# -- Turn on interactive plotting
# plt.ion()

# -- Set a GPS time:
t0 = 1126259462.4    # -- GW150914
#-- Choose detector as H1, L1, or V1

detector = 'H1'
strain = unpickle_me_this("data/GW150914_strain.pickle")
data = 1e19 * strain.value

zero_time = 1126259446  # I got this zero time by looking at the caption of the figure produced by strain.plot().
time = np.array(strain.times) - zero_time  # in seconds
onset = t0 - zero_time

length_of_windows = 2

time_domain_strip, k_lengths, power_spectrum = unpickle_me_this("data/results_from_welch_averaging_data.pickle")

signal_strip_idcs = np.where( (time > onset - length_of_windows/2) & ( time < onset + length_of_windows/2) )
signal_strip_time = time[signal_strip_idcs][:-1]  # im taking away the very last element to have a compatible shape with the power spectrum, shouldn't matter
signal_strip_strain = data[signal_strip_idcs][:-1]

approximate_duration_of_merger = 0.04

from scipy.signal.windows import tukey

n_dtps = time_domain_strip.shape[0]
window_function = tukey(M=n_dtps, alpha=0.1, sym=True)

signal_strip_strain_tapered = signal_strip_strain * window_function

from quickview.greece.utils.inference_class import *

N = NoiseOperatorFromPowerSpectrum(power_spectrum=power_spectrum, real_space=time_domain_strip, t0=signal_strip_time[0])

print_auxiliary = False
if print_auxiliary:
    print("Important variables: ")
    print("\t\tsignal_strip_time, signal_strip_strain ")
    print("\t\tsignal_strip_strain_tapered")
    print("\t\tstrain")
    print("\t\ttime_domain_strip")
    print("\t\tN\n\n")