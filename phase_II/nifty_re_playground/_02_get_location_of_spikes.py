from _01_get_smooth_baseline_ps import *
from nifty.nifty.re.conjugate_gradient import static_cg
from useful.calculate_pseudoinverse import find_penrose_moore_solution, sample_from_ps


data_grid = pipe_1.d_dom_real
harmonic_signal_grid = pipe_1.s_dom_harmonic
signal_distributor = harmonic_signal_grid.power_distributor

# d_tilde = hartley(strain_tapered, signal_grid=data_grid)

posterior_pipe_1_ps_mean_std, _, _ = pipe_1.get_posterior_statistics()
posterior_pipe_1_ps_mean = (posterior_pipe_1_ps_mean_std[0])[signal_distributor]
N = pipe_1.n_ds
M = pipe_1.n_ss

# pipe_3_xi_s = d_tilde / np.sqrt(posterior_pipe_1_ps_mean)

# plt.plot(pipe_1.k_data_full, pipe_3_xi_s)
# plt.show()


penrose_xi = find_penrose_moore_solution(pipe=pipe_1)

peaks_k, norm_amplitudes_k = get_peaks_from_cache(sigma_thresh=3)

# Plot of penrose_xi and its 2 sigma peaks
plt.plot(pipe_1.k_signal_full, penrose_xi.real)
plt.plot(peaks_k, [200]*len(peaks_k), "r.", markersize=5)
plt.show()


# Get Welch average for plot
_, k_lengths, power_spectrum = unpickle_me_this(
                    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                    absolute_path=True)
k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
spectrum_welch = power_spectrum.val[1:]

# Plot of smooth background together with found peaks
where_positive = np.where(pipe_1.k_signal_full>0)
plt.plot(pipe_1.k_signal_full[where_positive], posterior_pipe_1_ps_mean[where_positive], label=r"Smooth background $p_s(k)$")
plt.plot(peaks_k, norm_amplitudes_k, "b.", markersize=5, label=r"Normalized amplitudes of peaks in penrose $\xi$")
plt.plot(k_lengths, spectrum_welch, label=r"Empirical estimate of $p(k)$", color="orange")
plt.legend()
plt.loglog()
plt.show()

# Plot in data space
ps_mean_std, _, _ = pipe_1.get_posterior_statistics()
ps_mean = (ps_mean_std[0])[pipe_1.s_h_dom_expander]
posterior_penrose_data = sample_from_ps(penrose_xi, N=pipe_1.n_ds, inverse_h_trafo=lambda p: jnp.fft.ifft(p, norm="ortho"),
                                        ps=ps_mean)

plt.plot(time, posterior_penrose_data.real , label="Tapered Penrose-Moore")
plt.plot(time, pipe_1.d, label="Data")
plt.legend()
plt.show()


