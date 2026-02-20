"""
Best one yet, almost perfectly gets it.

Interesting:

- Very good fit very soon.
- Nan curvature error is thrown only after third iteration. Do not understand why.
- omega has a peak where its supposed to have one!
- xi looks white, except where the waveform is, where it LOOKS like it models a CORRECTION! Like model stress..

"""

from phase_II.nifty_re_playground.useful.helpers import *
from phase_III.useful.diff_equ_solver import *
from phase_III.useful.helpers import *
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(34)

# Try to use another mean for gamma. like 0. and let it vary a bit more but not too much

path1 = "/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_strain_values.txt"
path2 = "/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_time_values.txt"
gw_waveform = np.loadtxt(path1)
gw_waveform /= max(gw_waveform)
gw_times = np.loadtxt(path2)
gw_times = gw_times-gw_times[0]
# sub_idcs = np.where((1.25 < gw_times) & (gw_times < 1.45))
# gw_times = gw_times[sub_idcs]
# gw_waveform = gw_waveform[sub_idcs]

noise_fac = 1e-2
noise_lvl = noise_fac * jnp.max(gw_waveform)
grount_truth_noise = noise_lvl * np.random.standard_normal(len(gw_times))
print("using a noise level of ", noise_lvl)

# print("Adding noise to waveform to make synthetic data")
# data = gw_waveform + grount_truth_noise
data = gw_waveform


r_fac = 2
gw_dt = gw_times[1]-gw_times[0]
cfm_dt = gw_dt/r_fac
cfm_times  = np.arange(len(gw_times)*r_fac) * cfm_dt + gw_times.min()


log_omega_cfm, log_gamma_cfm, xi_cfm = [create_cfm(time_domain=cfm_times, prefix=p, offset_std=ofs_std, offset_mean=ofs_m, fluct=flu, llslope=ll, flex=fle) for
                                p,              ofs_std,        ofs_m,      flu,        ll,             fle in [
                                ("omega_cfm_",  (.5, 1e-16),      9.2,        (1,1),  (-3, 1),    None),  # idk just 100 and then everything in the same o.o.m.
                                # ("gamma_cfm_",  (2, 1e-16),     3.9,         (.5,.5),  (-4, 1e-16),   None),  # vary gamma significantly => Leads to nan curvature and probably too large gammas ,
                                ("gamma_cfm_",  (.5, 1e-16),     3.9,         (1,1),  (-10, 1e-16),   None),  # fix gamma
                                ("xi_cfm_",     (50, 1e-16),     0,         (1e3, 1e3),  (0, 1),    None),  # xi and gamma on equal footing. although sign of xi is ambiguous and depends on local
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
    reconstruction_times=gw_times,
    cfm_sampling_times=cfm_times,
    omega_cfm=omega_cfm,
    gamma_cfm=gamma_cfm,
    xi_cfm=xi_cfm,
    scaling_constant=scaling_constant
)

mask = DomainCheckAndMask(domain_time=cfm_times, target_time=gw_times)
s_prime = lambda p: mask(generative_wavelet(p))
s_prime.domain = generative_wavelet.domain
s_prime.get_model_components = generative_wavelet.get_model_components

for idx in range(18):
    print("idx+1: ", idx+1)
    sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet,
                                                            omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
                                                            key=key)

plt.plot(gw_times, data, label="Numerical relativity template synth data")
plt.plot(cfm_times, sample_waveform, label="Fw model evaluation")
usual_plot()

inference = InferenceSchemeRe(t=gw_times, d=data, key=key, e_fac=1, r_fac=1, plotting_callback=analyze_kl_callback)
inference.add_noise_op(noise_var_level=noise_lvl)
inference.add_custom_signal_model(custom_signal_model=s_prime)

latent_samples, other_stuff = inference.run_inference(kl_iterations=10, n_samples=15, use_strict_minimizers=True,
                                                      out_name="sde7", resume=True, choose_low_kl_starting_pos=False, geoVi=True)

signal_samples = [generative_wavelet(xi) for xi in latent_samples]
s_prime_samples = [s_prime(xi) for xi in latent_samples]
signal_mean, signal_std = jft.mean_and_std(signal_samples)
s_prime_mean, s_prime_std = jft.mean_and_std(s_prime_samples)


plt.errorbar(cfm_times, signal_mean, yerr=signal_std,label="SDE model", ecolor="lightblue")
plt.plot(gw_times, data, ".-", markersize=4, color="orange", label="Synthetic data")
plt.plot(gw_times, s_prime_mean, "--", color="red", label="Predicted data")
usual_plot()

key = plot_posterior(key, times=cfm_times, operator_list=[omega_cfm, gamma_cfm, xi_cfm, generative_wavelet], latent_samples=latent_samples,
                     label_list=["omega", "gamma res", "xi", "waveform"])