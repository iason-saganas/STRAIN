from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from strain_tools import *
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)
from gwpy.timeseries import TimeSeries

# In this file, I take the Welch average for the noise statistics, model the signal as a CFM and initiate
# it at the waveform found inverting the Wigner function

length_of_windows = 2
alpha_taper_on_data = 0.1

strain_obj = get_strain_from_disc(event_name="GW150914", detector="H1", data_duration="4096sec", desired_duration=32,
                              add_whitened_data=True, L=length_of_windows)

data_ar = strain_obj.event_strain
time = strain_obj.event_time

signal_strip_strain = data_ar
signal_strip_time = time

# 1. Initialize inference scheme

strain = signal_strip_strain - jnp.mean(signal_strip_strain)
strain = strain * tukey(len(strain), alpha=alpha_taper_on_data)
time = signal_strip_time
print("data length: ", len(strain), " time length: ", len(time), " and variance: ", np.var(strain))
print("mean of the data:", np.mean(strain))


pipe_3 = InferenceSchemeRe(t=time, d=strain, e_fac=1, r_fac=1, key=key, plotting_callback=analyze_kl_callback)

# 2. Build broken power law for signal model
pipe_3.add_custom_signal_model(
    custom_signal_model=BrokenPowerLaw(
                            signal_grid=pipe_3.s_dom_real,
                            pl_slope_left=(1, .5),
                            peak_power=1e3,
                            sigmoid_width=30,
                            pl_slope_right=(-1, .5),
                            k_break=(30, 200),
                            fluctuations=(1, 1),
                            envelope_fluctuations=(1, 1),
                            envelope_loglogavgslope=(-4, 1),
                            ),
    alpha=alpha_taper_on_data,
)

raise_warning("Using welch averaged power spectrum for inference!!! ")

# 3. Add custom noise operator based on welch averaged
strain_object_old = unpickle_me_this("/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle", absolute_path=True)
strain_object = TimeSeries(strain_object_old.value * 1e19, t0=strain_object_old.t0, dt=strain_object_old.dt, name=strain_object_old.name, channel=strain_object_old.channel)


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
        detrend='constant',
        scaling="density",
        return_onesided=True,
    )

welch_pow_spec /= 2

print("freq var: ", np.trapezoid(y=welch_pow_spec, x=welch_k)*2)
interpolate_k = np.fft.rfftfreq(len(time), time[1]-time[0])

interp_psd = interp1d(welch_k, welch_pow_spec, kind='linear', fill_value='extrapolate')
welch_pow_spec = interp_psd(interpolate_k)  # now PSD matches new times' frequency grid

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

pipe_3.add_noise_op(inverse_noise_op=N_inv, sqrt_inverse_noise_op=N_sqrt_inv, sqrt_noise_op=N_sqrt)


latent_samples = pipe_3.run_inference(kl_iterations=10, use_strict_minimizers=False, out_name="re_pipe_3_broken_power_law_debug_debug",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)
key = pipe_3.get_current_key()

NR = get_waveform_template(event_name="GW150914", detector="H1", gps_center=strain_obj.gps,
                                   silent=True, plot=False, force_online_fetch=False, model_approximant="IMRPhenomXPHM")

pipe_3.plot_posterior_signal(plot_default_nrt=False, print_posterior_parameters=True, over_full_signal_space=True,
                             plot_data=False,
                             save_fig=False,
                             xlim=(time.min(), time.max()),
                             maxL_template_xy=(NR.time, NR.strain),
                             )

pipe_3.plot_posterior_power_spectrum(mode="mean", plot_welch_average=False)
