import numpy as np
import matplotlib.pyplot as plt
import nifty8 as ift
from utils.helpers import generative_gaussian_comb

f = np.loadtxt("spike_frequencies.txt")
spike_amps = np.loadtxt("spike_amplitudes.txt")
random_baseline_amplitude_spectrum = np.sqrt(np.loadtxt("random_prior_power_spectrum.txt"))

assert np.allclose(np.diff(f), np.diff(f)[0], rtol=0, atol=1e-10)

df = np.diff(f)[0]

# (index of frequency at peak, amplitude of peak, half-width of the peak in INDEX counts)
# => Convert idx_center and idx_width to freq_center and freq_width later on
# information_idx = [(30, 15, 8), (3983, 6, 18), (5883, 4.3, 30), (10465, 5.3, 63), (12400, 11, 16), (16357, 207, 5)]  # old
information_idx = [(30, 15, 8), (3983, 1.64, 18), (5883, 1.1, 30)]

information_freq = [( f[t[0]],  t[1], t[2] * df) for t in information_idx]
frequency_centers = [t[0] for t in information_freq]
peak_amplitudes = [t[1] for t in information_freq]
frequency_half_widths = [t[2] for t in information_freq]


plot_1 = True
plot_2 = False
if plot_1:
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

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
    # axs[2].plot(f, gaussian_comb+random_baseline_amplitude_spectrum, 'b-', alpha=0.6, label="Adding amp. spec. as baseline")

    axs[1].set_yscale("log")
    axs[1].set_xscale("log")

    axs[2].set_yscale("log")
    axs[2].set_xscale("log")

    axs[1].legend()
    axs[2].legend()

    axs[2].set_xlabel("Unique frequencies")
    axs[2].set_ylabel("Power")

    axs[1].set_ylabel("Power")
    axs[0].set_ylabel("Power")

    axs[0].set_title("Linear scale amplitude peaks with fitted gaussians")
    axs[1].set_title("Same with log scale")
    axs[2].set_title("Adding peaks to random amplitude spectrum")


    plt.tight_layout()
    plt.show()

helper_domain = ift.DomainTuple.make(ift.RGSpace(harmonic=True, shape=(32764, ), distances=df))
p_space = ift.PowerSpace(helper_domain[0])
frequency_field = ift.Field(ift.DomainTuple.make(p_space, ), val=f)
gaussian_comb_op = generative_gaussian_comb(x_field=frequency_field, position_of_peaks=frequency_centers,
                                            amplitude_of_peaks=peak_amplitudes, half_width_of_peaks=frequency_half_widths)

if plot_2:

    ns = 3
    comb_samples = [gaussian_comb_op(ift.from_random(gaussian_comb_op.domain)).val for _ in range(ns)]
    samples_baseline_amplitude_spectrum = [random_baseline_amplitude_spectrum + sl for sl in comb_samples]

    for sample in samples_baseline_amplitude_spectrum:
        plt.plot(f, sample, "-")

        plt.xlabel("Unique frequencies")
        plt.ylabel("Power")
        plt.loglog()
        plt.tight_layout()
        plt.show()