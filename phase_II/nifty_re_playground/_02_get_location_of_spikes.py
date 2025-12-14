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


penrose_xi = find_penrose_moore_solution(pipe=pipe_1, itr=10_000)

peaks_k, norm_amplitudes_k = get_peaks_from_cache(sigma_thresh=4, power_spectrum=posterior_pipe_1_ps_mean)

# Plot of penrose_xi and its 2 sigma peaks
plt.plot(pipe_1.k_signal_full, penrose_xi.real, color=red)
# plt.plot(peaks_k, [200]*len(peaks_k), "r.", markersize=5)
usual_plot(xl="Frequency $f$", yl="Arbitrary units", title=r"$\tilde{\xi}_d^{\ast}$")

# Plot of smooth background together with found peaks
where_positive = np.where(pipe_1.k_signal_full>0)
plt.plot(pipe_1.k_signal_full[where_positive], posterior_pipe_1_ps_mean[where_positive],
         color=blue, label=r"Smooth background $p_n(f)$")
plt.plot(peaks_k, norm_amplitudes_k, color=red, marker="v", markersize=5, linewidth=0,
         label=r"Peaks found in $\tilde{\xi}_d^{\ast}$")
plot_welch_averaged_ps()
plt.loglog()
usual_plot(xl="Frequency $f$", yl="Power")

# Plot in data space
ps_mean_std, _, _ = pipe_1.get_posterior_statistics()
ps_mean = (ps_mean_std[0])[pipe_1.s_h_dom_expander]
posterior_penrose_data = sample_from_ps(penrose_xi, N=pipe_1.n_ds, inverse_h_trafo=lambda p: jnp.fft.ifft(p, norm="ortho"),
                                        ps=ps_mean)

tmp = np.where((time>15.1) & (time<16.75))
plt.plot(time[tmp], pipe_1.d[tmp], label=r"$d_{\mathrm{obs}}$", color="orange")
plt.plot(time[tmp], posterior_penrose_data.real[tmp], color=blue, label=r"Data from smooth $p_n(f)$ and $\tilde{\xi}_d^{\ast}$")
usual_plot(save_fig=True)


