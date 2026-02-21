import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from phase_III.strain import HarmonicOscillator, StochasticOscillatorPrior
from phase_III.useful.helpers import *
from gwpy.timeseries import TimeSeries
import numpy as np
jax.config.update("jax_enable_x64", True)
key = jax.random.key(34)


length_of_windows = 2
alpha_taper_on_data = 0.1

strain_object = get_strain_from_disc(event_name="GW150914", detector="H1", data_duration="4096sec", desired_duration=32,
                              add_whitened_data=True, L=length_of_windows)

strain_strip = strain_object.event_strain
time_strip = strain_object.event_time


# 1. Initialize inference scheme
strain_strip = strain_strip - jnp.mean(strain_strip)  # detrend
strain_strip = strain_strip * tukey(len(strain_strip), alpha=alpha_taper_on_data)  # taper
print("data length: ", len(strain_strip), " time length: ", len(time_strip), " and variance: ", np.var(strain_strip))
print("mean of the data:", np.mean(strain_strip))

pipe_3 = InferenceSchemeRe(t=time_strip, d=strain_strip, e_fac=1, r_fac=1, key=key, plotting_callback=analyze_kl_callback)

# oscillator_prior_dct = {
#     "frequency": {"offset_mean": 1000, "offset_std": (250, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-2, 1e-16)},  # LOG FLUCTUATIONS!!
#     "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
#     "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1, .1), "loglogavgslope": (0, 1e-16)}, # "flex":(2,2) flex helps sometimes
#     "global_amplitude": (1e2, 1e1),
#     "init_conditions": (0., 0.)
# }
# with use_driving_force = True and xi_force = multiply_op_2_to_op_1(xi_force, omega) NOT normed

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-2, 1e-16)},  # LOG FLUCTUATIONS!!
    "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (0, 1e-16), }, #"flex":(2,2)},
    "global_amplitude": (1, 1),
    "init_conditions": (0, 0),
}

use_driving_force = True

signal_domain = pipe_3.t_ss
target_domain = pipe_3.t_ds

signal_prior = StochasticOscillatorPrior(oscillator_prior_dct, signal_time_domain=signal_domain,
                                         forceless=not use_driving_force)

oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior,
                                tukey_window_alpha=alpha_taper_on_data * 0,  # no taper on the oscillator model itself
                                normalize=False,
                                add_global_amp=True,
                                cfm_envelope=None,
                                # cfm_envelope={"fluct": (1,1e-16), "llslope": (-4,1e-16)}
                                sample_initial_time=False
                                )
# oscillator.plot_samples(20, key)

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


# 3. Add custom noise operator based on welch averaged
# strain_object_old = unpickle_me_this("/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle", absolute_path=True)
# strain_object = TimeSeries(strain_object_old.value * 1e19, t0=strain_object_old.t0, dt=strain_object_old.dt, name=strain_object_old.name, channel=strain_object_old.channel)


# Calculate scipy welch
from scipy.signal import welch
t = np.array(strain_object.time)
x = np.array(strain_object.strain)
x = x - np.mean(x)
dt = t[1] - t[0]
fs = 1.0 / dt
welch_k, welch_pow_spec = welch(
        x=x,
        fs=fs,
        window=("tukey", alpha_taper_on_data),
        nperseg=len(time_strip),
        noverlap=None,
        detrend='constant',
        scaling="density",
        return_onesided=True,
    )
welch_pow_spec /= 2

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
# pipe_3.plot_noise_sample_with_data(1)


noise_variance_check = True
if noise_variance_check:
    samples = [N_sqrt(np.random.standard_normal(pipe_3.n_ds)) for _ in range(10000)]
    print("(DC frequency: ", welch_pow_spec[0], ")")
    print("freq var: ", np.trapezoid(y=welch_pow_spec, x=welch_k) * 2)
    print("Mean variance of samples: ", np.mean(np.var(samples, axis=0)))


pipe_3.set_init_pos(init_pos=dict(t0=jnp.float64(0.)), plot=False)

latent_samples, other_stuff = pipe_3.run_inference(kl_iterations=12, use_strict_minimizers=True, out_name="signal_reconstruction_sde_DEBUG_DEBUG",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)

t0 = jft.NormalPrior(mean=0, std=0.5, name="t0")
t0_mean = jft.mean(jnp.array([t0(sl) for sl in latent_samples]))
print("Center placed at: ", t0_mean)

key = pipe_3.get_current_key()

NR = get_waveform_template(event_name="GW150914", detector="H1", gps_center=strain_object.gps_center,
                                   silent=True, plot=False, force_online_fetch=False, model_approximant="IMRPhenomXPHM")

if __name__ == "__main__":
    plot_results = True
else:
    plot_results = False

if plot_results:

    pipe_3.plot_posterior_signal(plot_default_nrt=False, print_posterior_parameters=True, over_full_signal_space=False,
                                 plot_data=False,
                                 save_fig=False,
                                 maxL_template_xy=(NR.time, NR.strain),
                                 yl=r"$h(t)$ $\mathrm{[10^{-19}]}$", show=False)
    ax = plt.gca()
    two_sided_welch_amp = np.sqrt(welch_pow_spec[pipe_3.s_h_dom_expander])
    whitened_data = whiten(y=strain_strip, amp=two_sided_welch_amp)
    whitened_data /= whitened_data.max() / NR.strain.max()
    whitened_data_bp = bandpass(x=time_strip, y=whitened_data, bp=(30, 200))
    # ax.plot(time_strip, whitened_data, label="Whitened and bandpassed data", ls="-", color="black")
    thesis_plot(mode="longer", xlim=(time_strip.min(), time_strip.max()))


    key = plot_posterior(key, times=time_strip, operator_list=[oscillator.omega, oscillator.gamma, oscillator.xi_force, oscillator], latent_samples=latent_samples,
                         label_list=["omega", "gamma res", "xi", "waveform"], save_fig=False)