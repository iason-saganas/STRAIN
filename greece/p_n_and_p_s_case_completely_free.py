import matplotlib.pyplot as plt
import numpy as np
from utils.generative_models import *
from utils.helpers import *
import os

n_dtps = len(signal_strip_time)
n_pix = 2*n_dtps-1  # Important! see file `nrt_signal_power_spectrum_inferring_nrt_template.py`
L = np.max(signal_strip_time)-np.min(signal_strip_time)

data_domain = ift.RGSpace(shape=(n_dtps,), distances=L/n_dtps)
signal_domain = ift.RGSpace(shape=(n_pix,), distances=L/n_dtps)
h_domain = signal_domain.get_default_codomain()

time_data_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_dtps)
time_signal_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_pix)

M = SimpleMask(domain=dt(signal_domain), target=dt(data_domain), keep_th=2)

s = generative_model_continuous_double_power_law(h_domain)

# Plot signal samples
# plt.title("Data and two signal realizations (dashed)")
# colors = ["black", "red"]
# for i in range(2):
#     xi = ift.from_random(s.domain)
#     sl = s(xi)
#     plt.plot(time_signal_domain_values, sl.val, ls="--", color=colors[i])
#
# plt.plot(signal_strip_time, signal_strip_strain_tapered, label="Data", color="green")
# plt.plot(nrt_time_values, nrt_strain_values, label="nr template", )
# usual_plot(xl=r"Time in $\mathrm{sec}$", yl=r"Strain $\mathrm{[10^{-19}]}$")

# stop

# Plot power spectrum samples

s_test = generative_model_continuous_double_power_law(h_domain,
                                                      exact_values_dict={
                                            "k0": 1692,
                                            "p0": 1000,
                                            "c": 100,
                                            "alpha": 40.4717,
                                            "beta": -113.998,
                                            "cfm_envelope_fluctuations": 5,
                                            "cfm_envelope_loglogavgslope": -6}
                                                      )
ps_op = s_test.ps
k_lengths_sl = h_domain.get_unique_k_lengths()
for i in range(1):
    xi = ift.from_random(domain=ps_op.domain)
    # print("Chosen random xi --> ", get_real_space_values([xi]))
    sl = ps_op(xi)
    print("sl val: ",sl.val)
    stop
    plt.plot(k_lengths_sl, sl.val)

plt.loglog()
plt.show()
stop

d = ift.Field(dt(data_domain), val=signal_strip_strain_tapered)
s_prime = M(s)

energy = ift.GaussianEnergy(d, N.inverse, sampling_dtype=np.float64) @ s_prime

new_sampling_rate = lambda x: 1

posterior_samples = ift.optimize_kl(
            likelihood_energy=energy,
            total_iterations=3,
            n_samples=new_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory="outs/inference_with_continuous_double_power_law",
            return_final_position=False,
            resume=True,
            inspect_callback=inspect_sample,)

# os.system('say "Skript ausgeführt"')


s_mean, s_var = posterior_samples.sample_stat(s)

latent_sl = posterior_samples.sample_stat()[0]

ps_domain = latent_sl.extract_by_keys(["alpha ", "beta ", "p0 ", "c ", "k0 "])
ps = s.ps

posterior_pow_spec = ps(ps_domain)
plt.plot(h_domain.get_unique_k_lengths(),posterior_pow_spec.val)
plt.loglog()
plt.show()

fig, ax = plt.subplots()

# plt.plot(time_signal_domain_values, s_mean.val, label="Mean reconstruction")
ax.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val), )#label=r"Mean reconstructed field with 1$\sigma$ bands")
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


