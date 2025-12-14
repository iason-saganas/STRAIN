import numpy
import numpy as np

from sub_utils.generative_models import *
from sub_utils.helpers import *
from data.style_components.matplotlib_style import *

ift.random.push_sseq_from_seed(311)

signal_strip_strain_tapered = signal_strip_strain * tukey(len(signal_strip_strain))  # do not taper and see what happens

n_dtps = len(signal_strip_time)
n_pix = 2*n_dtps-1  # Important! see file `nrt_signal_power_spectrum_inferring_nrt_template.py`
L = np.max(signal_strip_time)-np.min(signal_strip_time)

data_domain = ift.RGSpace(shape=(n_dtps,), distances=L/n_dtps)
signal_domain = ift.RGSpace(shape=(n_pix,), distances=L/n_dtps)
h_domain = signal_domain.get_default_codomain()

time_data_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_dtps)
time_signal_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_pix)  # this is not correct.

M = SimpleMask(domain=dt(signal_domain), target=dt(data_domain), keep_th=2)

# plot_histogram(mean=4, sigma=None, mode="Exponential", n_samples=2000, color="blue")
# plot_histogram(mean=1, sigma=None, mode="Exponential", n_samples=2000)
# plt.show()
# stop

# NOTE: You can recover the good reconstruction by just uncommenting the line s = generative_model_continuous_double_power_law

s = generative_model_continuous_double_power_law(h_domain)
# s_cfm = ift.SimpleCorrelatedField(target=signal_domain, offset_mean=None, offset_std=None,
#                               fluctuations=(1,1), flexibility=(1,1), loglogavgslope=(0, 1), asperity=None, use_uniform_prior_on_fluctuations=False,
#                                   prefix="s_wavelet_")
#
# cf_env = ift.SimpleCorrelatedField(target=signal_domain, fluctuations=(4,2), loglogavgslope=(-4,1),
#                                            offset_mean=None, offset_std=None, flexibility=None, asperity=None,
#                                             prefix="cfm_envelope_", use_uniform_prior_on_fluctuations=False).ptw("exp")
#
# s = s_cfm * cf_env

# # Plot signal samples
# plt.title("Data and two signal realizations (dashed)")
# colors = ["black", "red"]
# for i in range(5):
#     xi = ift.from_random(s.domain)
#     sl = s(xi)
#     # if max(sl.val) < 0.1:
#     #     plt.plot(time_signal_domain_values, sl.val, ls="--", color=colors[i])  # with color
#     plt.plot(time_signal_domain_values, sl.val, ls="-")  # without color
#
# plt.plot(signal_strip_time, signal_strip_strain_tapered, label="Data", color="green")
# plt.plot(nrt_time_values, nrt_strain_values, label="nr template", ls="--" )
# usual_plot(xl=r"Time in $\mathrm{sec}$", yl=r"Strain $\mathrm{[10^{-19}]}$")

# Plot power spectrum samples

s_test = generative_model_continuous_double_power_law(h_domain,
                                                      exact_values_dict={
                                            "k0": 50,
                                             "p0": 1000,
                                            "c": 100,
                                            "alpha": .1,
                                            "beta": -5,
                                            "wavelet_fluct": 1,
                                            "cfm_envelope_fluctuations": 4,
                                            "cfm_envelope_loglogavgslope": -4,
                                            }
                                                      )

# k_lengths_other_fwm_131225 = np.loadtxt("x_131215.txt")
# ps_other_fwm_131225 = np.loadtxt("y_131215.txt")
#
# ps_op = s_test.ps
# k_lengths_sl = h_domain.get_unique_k_lengths()
# for i in range(1):
#     xi = ift.from_random(domain=ps_op.domain)
#     sl = ps_op(xi)
#     # print("picked xi values -> ", xi.val)
#     # print("corresponding real space values -> ", get_real_space_values([xi]))
#     # print("sl.val: ", sl.val)
#     # stop
#     plt.plot(k_lengths_sl, sl.val, color="blue", ls="--")
#
# # plt.plot(k_lengths_other_fwm_131225, ps_other_fwm_131225, color="red", ls="--")
# #
# # plt.loglog()
# # plt.show()

# Now: Are the classes I use to store the welch averaged power spectrum with identical between my nifty8 and nifty.re
# implementation?

# N_inv_of_interest_111225 = N.inverse
# N_inv_sqrt_of_interest_111225 = N_inv_of_interest_111225.get_sqrt()
#
# xi_111225 = ift.from_random(N_inv_sqrt_of_interest_111225.domain)
#
# N_inv_applied_111225 = N_inv_of_interest_111225(xi_111225)
# N_inv_sqrt_applied_111225 = N_inv_sqrt_of_interest_111225(xi_111225)
#
# np.savetxt("xi_111225.txt", xi_111225.val)
# np.savetxt("CL_N_inv_applied_111225.txt", N_inv_applied_111225.val)
# np.savetxt("CL_N_inv_sqrt_applied_111225.txt", N_inv_sqrt_applied_111225.val)
#
# stop

# Plot noise samples
sample_operator = N.get_sqrt()

plt.title("Noise samples")
plt.plot(signal_strip_time, signal_strip_strain_tapered, label="Data", color="green")
for _ in range(3):
    sl = sample_operator(ift.from_random(sample_operator.domain)).val
    plt.plot(signal_strip_time, sl)

plt.show()

d = ift.Field(dt(data_domain), val=signal_strip_strain_tapered)
s_prime = M(s)

energy = ift.GaussianEnergy(d, N.inverse, sampling_dtype=np.float64) @ s_prime

# new_sampling_rate = lambda x: 1

print("length of data: ", len(d.val))

print("Type of minimizers used: ", descent_finder, geoVI_sampling_minimizer)

posterior_samples = ift.optimize_kl(
            likelihood_energy=energy,
            total_iterations=5,
            n_samples=kl_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory="outs/inference_with_continuous_double_power_law_111225_without_L_BFGS_different_priors_dlt_later",
            return_final_position=False,
            resume=True,
            # inspect_callback=inspect_sample,
    )

# os.system('say "Skript ausgeführt"')


s_mean, s_var = posterior_samples.sample_stat(s)

latent_sl = posterior_samples.sample_stat()[0]

print("Latent sl domain: ", latent_sl.domain)

ps_posterior_xi_values = latent_sl.extract_by_keys(["alpha ", "beta ", "p0 ", "c ", "k0 ", "wavelet_fluct "])
# ps_posterior_xi_values = latent_sl.extract_by_keys(["s_wavelet_flexibility", "s_wavelet_fluctuations", "s_wavelet_loglogavgslope", "s_wavelet_spectrum"])
ps = s.ps
# ps = s_cfm.power_spectrum

posterior_pow_spec = ps(ps_posterior_xi_values)
plt.plot(h_domain.get_unique_k_lengths(),posterior_pow_spec.val)
plt.title("posterior power spectrum")
plt.loglog()
plt.show()

blue = (0, 0.37, 0.99, 1)
light_blue = (0.42, 0.8, 0.93, 0.4)
lighter_blue = (0.42, 0.8, 0.93, 0.3)
lightest_blue = (0.42, 0.8, 0.93, 0.1)

fig, ax = plt.subplots()

# plt.plot(time_signal_domain_values, s_mean.val, label="Mean reconstruction")
ax.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val), ecolor=lightest_blue)#label=r"Mean reconstructed field with 1$\sigma$ bands")
ax.set_title("Reconstruction")
to_keep = np.where(nrt_time_values > np.min(signal_strip_time))
nrt_time_values_short = nrt_time_values[to_keep]
nrt_strain_values_short = nrt_strain_values[to_keep]

ax.plot(nrt_time_values_short, nrt_strain_values_short,)# label=r"Suggested numerical relativity template")
ax.plot(signal_strip_time, signal_strip_strain_tapered, label="Data")
ax.set_ylim(-12, 5)

inset_ax = ax.inset_axes([0.1, 0.1, 0.8, 0.3])
# inset_ax.plot(time_signal_domain_values, s_mean.val, label="Mean reconstruction")
inset_ax.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val), label=r"Mean reconstructed field with 1$\sigma$ bands", ecolor=lightest_blue, color=blue)
inset_ax.plot(nrt_time_values_short, nrt_strain_values_short, color="orange",label=r"Suggested numerical relativity template", lw=2)
inset_ax.set_title("Zoomed in", fontsize=15)
inset_ax.set_xlim(16.2, 16.55)
inset_ax.legend(fontsize=10)

usual_plot(xl=r"Time in $\mathrm{sec}$", yl=r"Strain $\mathrm{[10^{-19}]}$")

plot_parameter_evolution()


