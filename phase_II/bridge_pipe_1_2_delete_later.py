import numpy as np
import matplotlib.pyplot as plt
import nifty8 as ift
from utils.helpers import generative_gaussian_comb

# this file demonstrates the found peak positions from stress in the harmonic stress

f = np.loadtxt("spike_frequencies.txt")
spike_amps = np.loadtxt("spike_amplitudes.txt")
spike_amps = spike_amps  # remove threshhold since will be later added
random_baseline_amplitude_spectrum = np.loadtxt("some_posterior_amp_spec_flexible.txt")

assert np.allclose(np.diff(f), np.diff(f)[0], rtol=0, atol=1e-10)

df = np.diff(f)[0]

# (index of frequency at peak, amplitude of peak, half-width of the peak in INDEX counts)
# => Convert idx_center and idx_width to freq_center and freq_width later on
# information_idx = [(30, 131, 12), (60, 0.5, 25), (36*4, 4.2, 3), (143, 0.38, 12), (995*4, 256/2 / 28.75, 13), (1084*4, 8.3/21, 2), (1480*4, 208/2 * 0.02, 10)] #(3983, 1.64, 18), (5883, 1.1, 30)]
information_idx = [(30, 131, 12), (60, 0.5, 25), (36*4, 4.2, 3), (143, 0.38, 12), (995*4, 256, 13), (1084*4, 8.3, 2), (1470*4, 208, 30)]
# information_idx = [(30, 131, 12), (60, 0.5, 25), (36*4, 4.2, 3), (143, 0.38, 12), (995*4, 256 *1e-1 , 13), (1084*4, 8.3 *1e-1 , 2), (1470*4, 208*1e-1, 30)]

f_minus = 45 # 30
f_plus = 650 # 860
additional_peaks_idcs = np.where((spike_amps > 0) & (f_minus < f) & (f < f_plus))[0]

print("Adding additionally ", len(additional_peaks_idcs), " peaks to the collection of self-selected ones")

information_idx = information_idx + [(idx, spike_amps[idx], 3) for idx in additional_peaks_idcs]  #

information_idx = [(t[0], max(t[1], 0), t[2]) for t in information_idx]  # subtract threshhold in amplitudes

information_idx = [t for t in information_idx if t[1] != 0] # get rid of amplitudes that were 0 after subtracting the threshhold

np.savetxt("information_idcs.txt", information_idx)

print("\n Saved information_idcs.txt")

information_freq = [( f[t[0]],  t[1], t[2] * df) for t in information_idx]
frequency_centers = [t[0] for t in information_freq]
peak_amplitudes = [t[1] for t in information_freq]
frequency_half_widths = [t[2] for t in information_freq]

import pickle
def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

_, k_lengths, power_spectrum = unpickle_me_this(
    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle", absolute_path=True)
k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
amp_spectrum_welch = np.sqrt(power_spectrum.val[1:])



plot_1 = True
plot_2 = True
if plot_1:
    fig, axs = plt.subplots(4, 1, figsize=(8, 6))

    axs[3].sharex(axs[2])
    axs[2].sharex(axs[1])

    # linear scale
    axs[0].plot(f, spike_amps, "g.")

    # log-log scale
    axs[1].plot(f, spike_amps, "g.")


    baseline = np.min(spike_amps[np.nonzero(spike_amps)])

    axs[1].hlines(baseline, np.min(f), np.max(f), label="Added baseline")
    # Here I add baseline since otherwise the Gaussians don't drop fast enough to 0 leading to the log-log plot
    # blowing up to values like 1e-235. Note that later on, I will add the Gaussian comb to another field effectively when
    # drawing a realization so we will not have to worry about this effect in the real use case

    # Overlay Gaussians
    gaussians = []
    for freq_center, amp, freq_width in information_freq:
        gaussian = amp * np.exp(-0.5 * ((f - freq_center)/freq_width)**2)
        gaussians.append(gaussian)

    gaussian_comb = np.sum(gaussians, axis=0)

    axs[0].plot(f, gaussian_comb+baseline, 'r--', alpha=0.6)
    axs[1].plot(f, gaussian_comb+baseline, 'r--', alpha=0.6)
    axs[3].plot(f, spike_amps*max(amp_spectrum_welch)/max(spike_amps), "g.")

    comb_with_amp_spec_baseline = gaussian_comb+random_baseline_amplitude_spectrum
    comb_with_amp_spec_baseline = comb_with_amp_spec_baseline * max(amp_spectrum_welch)/max(comb_with_amp_spec_baseline)
    axs[2].plot(f, comb_with_amp_spec_baseline, 'b-', alpha=0.6, label="Adding amp. spec. as baseline and normalized")
    axs[3].plot(k_lengths, amp_spectrum_welch, label="Empirical amplitude spectrum")


    axs[1].set_yscale("log")
    axs[1].set_xscale("log")

    axs[2].set_yscale("log")
    axs[2].set_xscale("log")

    axs[3].set_yscale("log")
    axs[3].set_xscale("log")

    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    axs[3].set_xlabel("Unique frequencies")
    axs[3].set_ylabel("Power")
    axs[2].set_ylabel("Power")

    axs[1].set_ylabel("Power")
    axs[0].set_ylabel("Power")

    axs[0].set_title("Linear scale amplitude peaks with fitted gaussians")
    axs[1].set_title("Same with log scale")
    axs[2].set_title("Adding peaks to random amplitude spectrum")


    plt.tight_layout()
    plt.show()

harmonic_pixel_num = 16383
helper_domain = ift.RGSpace(harmonic=True, shape=(harmonic_pixel_num, ), distances=df)
p_space = ift.PowerSpace(helper_domain)
frequency_field = ift.Field(ift.DomainTuple.make(p_space, ), val=f)
gaussian_comb_op = generative_gaussian_comb(x_field=frequency_field, position_of_peaks=frequency_centers,
                                            amplitude_of_peaks=peak_amplitudes, half_width_of_peaks=frequency_half_widths)

F = ift.FFTOperator(domain=helper_domain.get_default_codomain())
PD = ift.PowerDistributor(target=helper_domain)

if plot_2:

    ns = 3
    comb_samples = [gaussian_comb_op(ift.from_random(gaussian_comb_op.domain)).val for _ in range(ns)]
    samples_baseline_amplitude_spectrum = [random_baseline_amplitude_spectrum + sl for sl in comb_samples]

    for idx in range(len(samples_baseline_amplitude_spectrum)):

        fig,axs = plt.subplots(2,1, figsize=(12,8))

        amp_spec_sample = samples_baseline_amplitude_spectrum[idx]
        amp_spec_sample_field = ift.Field(ift.DomainTuple.make(p_space, ), val=amp_spec_sample)

        amp_spec_sample_field_full = PD(amp_spec_sample_field)
        amp_spec_sample_field_full_diag = ift.DiagonalOperator(amp_spec_sample_field_full)

        xi = ift.from_random(helper_domain)
        real_space_sample = F.inverse(amp_spec_sample_field_full_diag(xi)).val.real

        axs[0].plot(f, amp_spec_sample, "-", label="Amp spec sample")
        axs[0].set_xlabel("Unique frequencies")
        axs[0].set_ylabel("Power")

        axs[0].loglog()

        axs[1].plot(real_space_sample, "-", label="Real space sample")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Field value")

        for ax in axs:
            ax.legend()
        plt.tight_layout()
        plt.show()