from _01_get_smooth_baseline_ps import *
from useful.calculate_pseudoinverse import find_penrose_moore_solution

if __name__ == "__main__":
    pipe_2_called_as_import = False
else:
    pipe_2_called_as_import = True

# 1. Get pipe 1 results to build template operator
prior_ps_template_mean_std, _, _ = pipe_1.get_posterior_statistics()
prior_ps_template_mean = prior_ps_template_mean_std[0]
template_operator = PowerSpectrumTemplate(ps_template=prior_ps_template_mean, scale=(1, .1))

# 2. Get amplitudes and positions of peaks in penrose xi to build line model operator
peak_pos, peak_amps = get_peaks_from_cache(sigma_thresh=3)

gaussian_comb_op = GaussianComb(unique_k_lengths=pipe_1.k_signal, list_of_peaks=peak_pos, list_of_amplitudes=peak_amps,
                                a_priori_width_of_peaks=1, rel_sigma_widths=.1, rel_sigma_amp=.1)

# 3. Initialize inference scheme
pipe_2 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key, plotting_callback=analyze_kl_callback)


# 4. Build correlated field model with custom operators
pipe_2.add_cfm_signal_model(add_custom_power_op=[gaussian_comb_op, template_operator],
                            fluct=(1, 2), llslope=(-2, 1), flex=(2, 1))

# 5. Add noise level
pipe_2.add_noise_op(noise_var_level=1e-16)

if not pipe_2_called_as_import:

    # plt.plot(peak_pos, peak_amps, "b.", markersize=3, label="Input peaks")
    # plt.plot(pipe_1.k_signal, prior_ps_template_mean, label="Prior template")
    # pipe_2.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True, rolling=True)
    #
    # plt.plot(time, strain, label="Actual data")
    # pipe_2.plot_prior_samples(mode="signal", num=5, rolling=True, plot_welch_average=False)
    #
    # pipe_2.plot_prior(mode="power spectrum")
    # pipe_2.plot_prior_samples(mode="signal & power spectrum", num=0)
    pass

latent_samples = pipe_2.run_inference(kl_iterations=15, use_strict_minimizers=False, out_name="re_pipe_2", resume=True)
key = pipe_2.get_current_key()

if not pipe_2_called_as_import:

    pipe_2.plot_posterior_signal()
    # pipe_2.plot_posterior_power_spectrum(print_posterior_parameters=True)
    # pipe_2.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
    # pipe_2.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)

    # xi_s_harmonic = pipe_2.plot_posterior_harmonic_xi_s(show=False)
    # S, t_dual, f_dual = Stress_re(xi_s_harmonic, time=time, downsample=True)
    # visualize_stress(S, f_dual, t_dual, smooth=False, detect_outliers=False)

    penrose_xi = pipe_2.calculate_and_plot_penrose_xi()

    S, t_dual, f_dual = Stress_re(penrose_xi, time=time, downsample=True)
    visualize_stress(S, f_dual, t_dual, smooth=True, detect_outliers=True)

