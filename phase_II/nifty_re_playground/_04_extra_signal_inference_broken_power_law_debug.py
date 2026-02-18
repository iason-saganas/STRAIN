
"""
This is a=0.1 reconstruction with no variance-matching, a consistent taper across welch, data and model, and analyzing
a 2-second strip which is what was assumed in the Welch-average.

Setting alpha=0.5 works so well because the data variance is reduced by a lot, but the Welch average predicted variance
stays constant, making the SNR lower, apparently allowing there to be more space for a higher amplitude signal.

Wait but I was variance matching lol ???
"""

from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from strain_tools import *
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)
import matplotlib.pyplot as plt

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

num_of_windows_to_include = 1
signal_strip_idcs = np.where( (time >= onset - num_of_windows_to_include*length_of_windows/2) & ( time <= onset + num_of_windows_to_include*length_of_windows/2) )
signal_strip_time = time[signal_strip_idcs]  # im taking away the very last element to have a compatible shape with the power spectrum, shouldn't matter
signal_strip_strain = data_ar[signal_strip_idcs]

print("length of sig strip: ", )

# plt.plot(signal_strip_time, signal_strip_strain)
# plt.show()

# Or: use full
# signal_strip_strain = data_ar
# signal_strip_time = time

# 1. Initialize inference scheme
# strain = strain * tukey(len(strain))
alpha_taper_on_data = 0.1
if alpha_taper_on_data == 0.5:
    raise_warning("Tukey-windowing data with a shape parameter of 0.5!")
elif alpha_taper_on_data == .1:
    raise_warning("Tukey-windowing data with a shape parameter of 0.1!")
elif alpha_taper_on_data == .01:
    raise_warning("Tukey-windowing data with a shape parameter of 1e-2!")
elif alpha_taper_on_data == 0.:
    print("Not tukey-windowing data")
else:
    raise ValueError("For debugging purposes not allowed")
strain = signal_strip_strain - jnp.mean(signal_strip_strain)
strain = strain * tukey(len(strain), alpha=alpha_taper_on_data)
time = signal_strip_time
print("data length: ", len(strain), " time length: ", len(time), " and variance: ", np.var(strain))
print("mean of the data:", np.mean(strain))
# plt.plot(time, strain)
# plt.show()

pipe_3 = InferenceSchemeRe(t=time, d=strain, e_fac=1, r_fac=1, key=key, plotting_callback=analyze_kl_callback)

# 2. Build correlated field for signal model
# pipe_3.add_cfm_signal_model(fluct=(5, 2), llslope=(-2, 2), #flex=(1, 1)
#                             )

pipe_3.add_custom_signal_model(
    custom_signal_model=BrokenPowerLaw(
                            signal_grid=pipe_3.s_dom_real,
                            # pl_slope_left=11,
                            pl_slope_left=(1, .5),
                            # peak_power=(1000, 100),
                            peak_power=1e3,
                            # sigmoid_width=1.8,
                            sigmoid_width=30,
                            pl_slope_right=(-1, .5),
                            # pl_slope_right=-10,
                            k_break=(30, 200),
                            # k_break=(120, 150),
                            # k_break=120,
                            # fluctuations=1e-1,
                            fluctuations=(1, 1),
                            envelope_fluctuations=(1, 1),
                            envelope_loglogavgslope=(-4, 1),
                            # flexibility=(.2, .1),
                            ),
    alpha=alpha_taper_on_data,
)

# pipe_3.add_matern_signal_model(scale=(1e-1,1e-1), llslope=(-20, 1), cutoff=(100, 20), add_cfm_env=True)
# pipe_3.add_cfm_signal_model(fluct=(1e-1,1e-1), llslope=(-1,1), flex=None, add_cfm_env=False)

# pipe_3.plot_prior_samples(mode="power spectrum", num=6, rolling=False, plot_welch_average=False, plot_data=False)
# pipe_3.plot_prior_samples(mode="signal response", num=6, rolling=True, plot_welch_average=False, plot_data=True)
# pipe_3.plot_prior_samples("signal & power spectrum", num=2, rolling=True, plot_welch_average=False, plot_data=True)

raise_warning("Using welch averaged power spectrum for inference!!! ")

# 3. Add custom noise operator based on welch averaged
# welch_k_2, welch_pow_spec_2 = get_welch_averaged_ps()
strain_object_old = unpickle_me_this("/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle", absolute_path=True)
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
strain_object = TimeSeries(strain_object_old.value * 1e19, t0=strain_object_old.t0, dt=strain_object_old.dt, name=strain_object_old.name, channel=strain_object_old.channel)

# welch_pow_spec_object_old = strain_object_old.psd(fftlength=2, window="tukey", overlap=0)
# welch_pow_spec_object = strain_object.psd(fftlength=2, window="tukey", overlap=0)
# welch_k = welch_pow_spec_object.frequencies
# welch_pow_spec = jnp.array(welch_pow_spec_object.value)/2  # because of scipy convention when one_sided arg is True!

# Calculate scipy welch

from scipy.signal import welch

t = np.array(strain_object.times)
x = np.array(strain_object.value)
# t = signal_strip_time
# x = signal_strip_strain
x = x - np.mean(x)
dt = t[1] - t[0]
fs = 1.0 / dt
welch_k, welch_pow_spec = welch(
        x=x,
        fs=fs,
        window=("tukey", alpha_taper_on_data),
        nperseg=int(length_of_windows/dt),
        noverlap=None,
        detrend=False,
        scaling="density",
        return_onesided=True,
    )

welch_pow_spec /= 2

# welch_k_3, welch_pow_spec_3 = welch(
#         x=x,
#         fs=fs,
#         window="tukey",
#         nperseg=int(1/dt),
#         noverlap=None,
#         detrend=False,
#         scaling="density",
#         return_onesided=True,
#     )
# welch_pow_spec_3 /= 2
#
#
# plt.plot(welch_k, welch_pow_spec, label="2 sec long")
# plt.plot(welch_k_3, welch_pow_spec_3, label="1 sec long")
# plt.loglog()
# plt.legend()
# plt.show()
print("freq var: ", np.trapezoid(y=welch_pow_spec, x=welch_k)*2)

interpolate_k = np.fft.rfftfreq(len(time), time[1]-time[0])

interp_psd = interp1d(welch_k, welch_pow_spec, kind='linear', fill_value='extrapolate')
welch_pow_spec = interp_psd(interpolate_k)  # now PSD matches new times' frequency grid


# correction_factor = 0.8370667368758037
correction_factor = 1
N_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1),
                                     data_grid=pipe_3.d_dom_real, apply_correction_factor=True,
                              correction_factor_dont_change_default_for_legacy_reasons=correction_factor)

N_sqrt_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1/2),
                                          data_grid=pipe_3.d_dom_real, apply_correction_factor=True,
                                   correction_factor_dont_change_default_for_legacy_reasons=correction_factor)

N_sqrt = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(1/2),
                                          data_grid=pipe_3.d_dom_real, apply_correction_factor=True,
                               correction_factor_dont_change_default_for_legacy_reasons=correction_factor)

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
# wigner_xi_waveform = interpolate_waveform_from_inverted_wigner(pipe_3.t_ss)
# wigner_xi_waveform /= max(wigner_xi_waveform)  # The inverted wigner misses some normalization factors.
# these can be recovered and should then give the correct amplitude. I set it manually here

# norm = pipe_3.k_signal_full ** (-2)
# norm[0]=1
# harmonic_wigner_xi_waveform = fw_hartley(wigner_xi_waveform, norm=None) / norm / 1e6

# pipe_3.plot_noise_sample_with_data(num=2, rolling=False)

# pipe_3.set_init_pos(init_pos={"s_xi": jnp.array(harmonic_wigner_xi_waveform),
#                               # "s_flexibility": -1e3,
#                               "s_fluctuations": 1.,
#                               "s_loglogavgslope": -1.},
#                     plot=True, plot_welch_average=False)

latent_samples = pipe_3.run_inference(kl_iterations=20, use_strict_minimizers=True, out_name="re_pipe_3_broken_power_law_debug",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)
key = pipe_3.get_current_key()

pipe_3.plot_posterior_signal(plot_default_nrt=True, print_posterior_parameters=True, over_full_signal_space=True,
                             plot_data=False,
                             # xlim=(16.35, 16.45),
                             save_fig=False
                             )
pipe_3.plot_posterior_power_spectrum(mode="mean", plot_welch_average=False)
