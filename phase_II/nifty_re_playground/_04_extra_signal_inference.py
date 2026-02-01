from _03_baseline_plus_line_model_inference import *
from scipy.signal.windows import tukey


# In this file, I take the Welch average for the noise statistics, model the signal as a CFM and initiate
# it at the waveform found inverting the Wigner function

# -- Set a GPS time:
t0 = 1126259462.4    # -- GW150914
#-- Choose detector as H1, L1, or V1

strain = unpickle_me_this("/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle", absolute_path=True)
data_ar = 1e19 * strain.value

zero_time = 1126259446  # I got this zero time by looking at the caption of the figure produced by strain.plot().
time = np.array(strain.times) - zero_time  # in seconds
onset = t0 - zero_time

length_of_windows = 2

signal_strip_idcs = np.where( (time > onset - length_of_windows/2) & ( time < onset + length_of_windows/2) )
signal_strip_time = time[signal_strip_idcs][:-1]  # im taking away the very last element to have a compatible shape with the power spectrum, shouldn't matter
signal_strip_strain = data_ar[signal_strip_idcs][:-1]

# 1. Initialize inference scheme
# strain = strain * tukey(len(strain))
strain = signal_strip_strain * tukey(len(signal_strip_strain))
time = signal_strip_time
print("data length: ", len(strain), " time length: ", len(time))

pipe_3 = InferenceSchemeRe(t=time, d=strain, e_fac=1, r_fac=1, key=key, plotting_callback=analyze_kl_callback)

# 2. Build correlated field for signal model
# pipe_3.add_cfm_signal_model(fluct=(5, 2), llslope=(-2, 2), #flex=(1, 1)
#                             )

pipe_3.add_custom_signal_model(BrokenPowerLaw(
    signal_grid=pipe_3.s_dom_real,
    # pl_slope_left=11,
    pl_slope_left=(1, 1),
    # peak_power=(1000, 100),
    peak_power=1e3,
    # sigmoid_width=1.8,
    sigmoid_width=30,
    # pl_slope_right=(-10, 2),
    pl_slope_right=-10,
    k_break=(120, 150),
    # k_break=120,
    # fluctuations=1e-1,
    fluctuations=(1, 1),
    envelope_fluctuations=(1, 1),
    envelope_loglogavgslope=(-4, 1),
    # flexibility=(.2, .1),
                              ))

# pipe_3.add_matern_signal_model(scale=(1e-1,1e-1), llslope=(-20, 1), cutoff=(100, 20), add_cfm_env=True)
# pipe_3.add_cfm_signal_model(fluct=(1e-1,1e-1), llslope=(-1,1), flex=None, add_cfm_env=False)

# pipe_3.plot_prior_samples(mode="power spectrum", num=6, rolling=False, plot_welch_average=False, plot_data=False)
# pipe_3.plot_prior_samples(mode="signal response", num=6, rolling=True, plot_welch_average=False, plot_data=True)
# pipe_3.plot_prior_samples("signal & power spectrum", num=2, rolling=True, plot_welch_average=False, plot_data=True)

raise_warning("Using welch averaged power spectrum for inference!!! ")

# 3. Add custom noise operator based on welch averaged
welch_k, welch_pow_spec = get_welch_averaged_ps()

N_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1),
                                     data_grid=pipe_3.d_dom_real)

N_sqrt_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1/2),
                                          data_grid=pipe_3.d_dom_real)
N_sqrt = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(1/2),
                                          data_grid=pipe_3.d_dom_real)

print("Variance of data: " , np.var(pipe_3.d))
# xi_111225 = np.loadtxt("xi_111225.txt")
#
# CL_N_inv_applied_111225 = np.loadtxt("CL_N_inv_applied_111225.txt")
# CL_N_inv_sqrt_applied_111225 = np.loadtxt("CL_N_inv_sqrt_applied_111225.txt")
#
# RE_N_inv_applied_111225 = N_inv(xi_111225)
# RE_N_inv_sqrt_applied_111225 = N_sqrt_inv(xi_111225)
#
# plt.plot(CL_N_inv_applied_111225/RE_N_inv_applied_111225, label="CL N inv")
# # plt.plot(RE_N_inv_applied_111225, label="RE N inv")
# usual_plot()
#
# plt.plot(CL_N_inv_sqrt_applied_111225/RE_N_inv_sqrt_applied_111225, label="CL N sqrt inv")
# # plt.plot(RE_N_inv_sqrt_applied_111225, label="RE N sqrt inv")
# usual_plot()
#
#
# stopWelchAverageTest

pipe_3.add_noise_op(inverse_noise_op=N_inv, sqrt_inverse_noise_op=N_sqrt_inv, sqrt_noise_op=N_sqrt)

# 4. Get some noise and signal samples
wigner_xi_waveform = interpolate_waveform_from_inverted_wigner(pipe_3.t_ss)
wigner_xi_waveform /= max(wigner_xi_waveform)  # The inverted wigner misses some normalization factors.
# these can be recovered and should then give the correct amplitude. I set it manually here

norm = pipe_3.k_signal_full ** (-2)
norm[0]=1
harmonic_wigner_xi_waveform = fw_hartley(wigner_xi_waveform, norm=None) / norm / 1e6

# pipe_3.plot_noise_sample_with_data(num=2, rolling=False)

# pipe_3.set_init_pos(init_pos={"s_xi": jnp.array(harmonic_wigner_xi_waveform),
#                               # "s_flexibility": -1e3,
#                               "s_fluctuations": 1.,
#                               "s_loglogavgslope": -1.},
#                     plot=True, plot_welch_average=False)

latent_samples = pipe_3.run_inference(kl_iterations=10, use_strict_minimizers=False, out_name="re_extra_pipe_3_151225_8",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)
key = pipe_3.get_current_key()

pipe_3.plot_posterior_signal(plot_nrt=True, print_posterior_parameters=True, over_full_signal_space=False,
                             plot_data=False,
                             # xlim=(16.35, 16.45),
                             save_fig=False
                             )
pipe_3.plot_posterior_power_spectrum(mode="mean", plot_welch_average=False)
