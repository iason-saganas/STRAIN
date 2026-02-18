"""
Using consistent tapers across everything, no variance matching, equal length segments in
welch and data input into likelihood; comparing with correct NR template.

# Good run: fluct=(1,1), llslope=(-4,1) envelope on the wavelet and the stochastic oscillator prior

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (50, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-4, 1)},
    "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1e3, 1e3), "loglogavgslope": (0, 1e-16)},
    "global_amplitude": (1, 1),
    "init_conditions": (.0, 0.)
}

with

oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior,
                                tukey_window_alpha=alpha_taper_on_data,
                                normalize=True,
                                add_global_amp=False,
                                add_cfm_envelope=True)

Even better reconstruction:


oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (50, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-4, 1)},  # LOG FLUCTUATIONS!!
    "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (0, 1e-16)},
    "global_amplitude": (1e-2, 1e-2),
    "init_conditions": (.0, 0.)
}

oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior,
                                tukey_window_alpha=alpha_taper_on_data,
                                normalize=True,
                                add_global_amp=True,
                                cfm_envelope={"fluct": (1,1e-16), "llslope": (-4,1e-16)})


In general, the problem is, I think, that I allow omega to be large at all times. Actually, it must be near 0
almost always since there is no GW at most times. We then have to hope that the data is strong enough to pull
omega upwards where the GW is and we should then not normalize in order for the omega evolution to dictate the
injected energy.

You see the waveform in xi like this:

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (10, 10), "loglogavgslope": (-4, 1e-16)},  # LOG FLUCTUATIONS!!
    "damping": {"offset_mean": 100, "offset_std": (1e-16, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-4, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (500, 1e-16), "loglogavgslope": (0, 1e-16)},
    "global_amplitude": (1e1, 1e1),
    "init_conditions": (0., 0.)
}
use_driving_force = True

oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior,
                                tukey_window_alpha=alpha_taper_on_data,
                                normalize=False,
                                add_global_amp=True,
                                cfm_envelope=None
                                # cfm_envelope={"fluct": (1,1e-16), "llslope": (-4,1e-16)}
                                )


omega = add_peak(omega)


"""
from scipy.interpolate import interp1d
from phase_II.nifty_re_playground.strain_tools import *
from phase_III.strain import HarmonicOscillator, StochasticOscillatorPrior
from phase_III.useful.helpers import *
from phase_III.useful.diff_equ_solver import *
import numpy as np
jax.config.update("jax_enable_x64", True)
key = jax.random.key(34)
from gwpy.timeseries import TimeSeries


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

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (10, 10), "loglogavgslope": (-4, 1e-16)},  # LOG FLUCTUATIONS!!
    "damping": {"offset_mean": 100, "offset_std": (100, 1e-16), "fluctuations": (10, 10), "loglogavgslope": (-4, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (500, 1e-16), "loglogavgslope": (0, 1e-16)},
    "global_amplitude": (1e1, 1e1),
    "init_conditions": (.1, .1)
}
use_driving_force = False

signal_domain = pipe_3.t_ss
target_domain = pipe_3.t_ds

signal_prior = StochasticOscillatorPrior(oscillator_prior_dct, signal_time_domain=signal_domain, forceless=not use_driving_force)

oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior,
                                tukey_window_alpha=alpha_taper_on_data,
                                normalize=False,
                                add_global_amp=True,
                                cfm_envelope=None
                                # cfm_envelope={"fluct": (1,1e-16), "llslope": (-4,1e-16)}
                                )


pipe_3.add_custom_signal_model(custom_signal_model=oscillator, alpha=alpha_taper_on_data)
# pipe_3.add_custom_signal_model(
#     custom_signal_model=BrokenPowerLaw(
#                             signal_grid=pipe_3.s_dom_real,
#                             pl_slope_left=(1, .5),
#                             peak_power=1e3,
#                             sigmoid_width=30,
#                             pl_slope_right=(-1, .5),
#                             k_break=(30, 200),
#                             fluctuations=(1, 1),
#                             envelope_fluctuations=(1, 1),
#                             envelope_loglogavgslope=(-4, 1),
#                             ),
#     alpha=alpha_taper_on_data,
# )

# oscillator.plot_samples(20, key)
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

print("DC frequency: " , welch_pow_spec[0], " at f=", interpolate_k[0])

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

latent_samples, other_stuff = pipe_3.run_inference(kl_iterations=10, use_strict_minimizers=True, out_name="signal_reconstruction_sde_DEBUG_DEBUG",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)
key = pipe_3.get_current_key()

NR = get_waveform_template(event_name="GW150914", detector="H1", gps_center=strain_obj.gps,
                                   silent=True, plot=False, force_online_fetch=False, model_approximant="IMRPhenomXPHM")

pipe_3.plot_posterior_signal(plot_default_nrt=False, print_posterior_parameters=True, over_full_signal_space=False,
                             plot_data=False,
                             xlim=(time.min(), time.max()),
                             save_fig=False,
                             maxL_template_xy=(NR.time, NR.strain),
                             yl=r"$h(t)$ $\mathrm{[10^{-19}]}$")


key = plot_posterior(key, times=time, operator_list=[oscillator.omega, oscillator.gamma, oscillator.xi_force, oscillator], latent_samples=latent_samples,
                     label_list=["omega", "gamma res", "xi", "waveform"], save_fig=False)