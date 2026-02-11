from phase_II.nifty_re_playground.strain_tools import *
from phase_III.useful.helpers import *
from phase_III.useful.diff_equ_solver import *
import numpy as np
jax.config.update("jax_enable_x64", True)
key = jax.random.key(34)

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


r_fac = 2
gw_dt = time[1]-time[0]
cfm_dt = gw_dt/r_fac
cfm_times  = np.arange(len(time)*r_fac) * cfm_dt + time.min()


log_omega_cfm, log_gamma_cfm, xi_cfm = [create_cfm(time_domain=cfm_times, prefix=p, offset_std=ofs_std, offset_mean=ofs_m, fluct=flu, llslope=ll, flex=fle) for
                                p,              ofs_std,        ofs_m,      flu,        ll,             fle in [
                                ("omega_cfm_",  (.5, 1e-16),      9.2,        (1,1),  (-2, 1e-16),    None),  # just 100Hz and then everything in the same o.o.m.
                                # ("gamma_cfm_",  (2, 1e-16),     3.9,         (.5,.5),  (-4, 1e-16),   None),  # vary gamma significantly => Leads to nan curvature and probably too large gammas ,
                                ("gamma_cfm_",  (1, 1e-16),     3.9,         (1e-16,1e-16),  (-10, 1e-16),   None),  # fix gamma
                                ("xi_cfm_",     (1e-16, 1e-16),     0,         (1e3, 1e3),  (0, 1),    None),  # xi and gamma on equal footing. although sign of xi is ambiguous and depends on local
        # properties => set 0.
                                ]
                                ]
scaling_constant = (1e1, 1e1)

omega_cfm = lambda p: jnp.sqrt(jnp.exp(log_omega_cfm(p)))
omega_cfm.domain = log_omega_cfm.domain

gamma_cfm = lambda p: jnp.exp(log_gamma_cfm(p))
gamma_cfm.domain = log_gamma_cfm.domain

# log_envelope = create_cfm(time_domain=cfm_times, prefix="cfm_env_", offset_std=(1e-16,1e-16), offset_mean=0,fluct=(1,1), llslope=(-4,1e-16), flex=None)
# envelope = lambda p: jnp.exp(log_envelope(p))
# envelope.domain = log_envelope.domain

generative_wavelet = AutoDiffEquationSolver(
    prefix="stochastic_diff_equ_",
    reconstruction_times=time,
    cfm_sampling_times=cfm_times,
    omega_cfm=omega_cfm,
    gamma_cfm=gamma_cfm,
    xi_cfm=xi_cfm,
    scaling_constant=scaling_constant,
    tukey_window=True,
)

mask = DomainCheckAndMask(domain_time=cfm_times, target_time=time)
s_prime = lambda p: mask(generative_wavelet(p))
s_prime.domain = generative_wavelet.domain
s_prime.get_model_components = generative_wavelet.get_model_components

pipe_3.add_custom_signal_model(custom_signal_model=s_prime)

# for idx in range(5):
#     print("idx+1: ", idx+1)
#     sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet,
#                                                             omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
#                                                             key=key)

# plt.plot(time, strain, label="Numerical relativity template synth data")
# plt.plot(cfm_times, sample_waveform, label="Fw model evaluation")
# usual_plot()

raise_warning("Using welch averaged power spectrum for inference!!! ")

# 3. Add custom noise operator based on welch averaged
welch_k, welch_pow_spec = get_welch_averaged_ps()

N_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1),
                                     data_grid=pipe_3.d_dom_real)

N_sqrt_inv = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(-1/2),
                                          data_grid=pipe_3.d_dom_real)
N_sqrt = NoiseCovarianceFromPs(one_sided_noise_ps=welch_pow_spec, callable_to_apply=lambda x: x**(1/2),
                                          data_grid=pipe_3.d_dom_real)

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

latent_samples, other_stuff = pipe_3.run_inference(kl_iterations=20, use_strict_minimizers=True, out_name="signal_reconstruction_sde",
                                      resume=True, choose_low_kl_starting_pos=False, geoVi=True)
key = pipe_3.get_current_key()

pipe_3.plot_posterior_signal(plot_nrt=True, print_posterior_parameters=True, over_full_signal_space=False,
                             plot_data=False,
                             xlim=(16.3, 16.46),
                             save_fig=False,
                             yl=r"$h(t)$ $\mathrm{[10^{-19}]}$")

# signal_samples = [generative_wavelet(xi) for xi in latent_samples]
# s_prime_samples = [s_prime(xi) for xi in latent_samples]
# signal_mean, signal_std = jft.mean_and_std(signal_samples)
# s_prime_mean, s_prime_std = jft.mean_and_std(s_prime_samples)
#
#
# plt.errorbar(cfm_times, signal_mean, yerr=signal_std,label="SDE model", ecolor="lightblue")
# nrt_strain_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_strain_values.txt") * 1e19
# nrt_time_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_time_values.txt")
# nrt_time_values = nrt_time_values - nrt_time_values[0] + 15
# go_until = np.max(np.where(nrt_time_values<max(time)))
#
# plt.xlim(16.2, 16.5)
# plt.plot(nrt_time_values[:go_until], nrt_strain_values[:go_until], label="LIGO Template",
#          color=red)
#
# plt.plot(time, s_prime_mean, "--", color="red", label="Predicted data")
# usual_plot()

key = plot_posterior(key, times=cfm_times, operator_list=[omega_cfm, gamma_cfm, xi_cfm, generative_wavelet], latent_samples=latent_samples,
                     label_list=["omega", "gamma res", "xi", "waveform"], save_fig=True)