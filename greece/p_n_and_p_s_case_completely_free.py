import numpy as np
from utils.generative_models import *
import os

n_dtps = len(signal_strip_time)
n_pix = 2*n_dtps-1  # Important! see file `nrt_signal_power_spectrum_inferring_nrt_template.py`
L = np.max(signal_strip_time)-np.min(signal_strip_time)

data_domain = ift.RGSpace(shape=(n_dtps,), distances=L/n_dtps)
signal_domain = ift.RGSpace(shape=(n_pix,), distances=L/n_pix)
h_domain = signal_domain.get_default_codomain()

time_data_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_dtps)
time_signal_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_pix)

M = SimpleMask(domain=dt(signal_domain), target=dt(data_domain), keep_th=2)

s = generative_model_broken_power_law(h_domain, apply_envelope=True, add_extra_amplitude=False)
# Plot samples
# plt.title("Data and two signal realizations (dashed)")
# colors = ["black", "red"]
# for i in range(2):
#     xi = ift.from_random(s.domain)
#     sl = s(xi)
#     plt.plot(time_signal_domain_values, sl.val, ls="--", color=colors[i])
#
# plt.plot(signal_strip_time, signal_strip_strain_tapered, label="Data", color="green")
# # plt.plot(nrt_time_values, nrt_strain_values, label="nr template", )
# usual_plot(xl=r"Time in $\mathrm{sec}$", yl=r"Strain $\mathrm{[10^{-19}]}$")
#
# stop

d = ift.Field(dt(data_domain), val=signal_strip_strain_tapered)
s_prime = M(s)

energy = ift.GaussianEnergy(d, N.inverse, sampling_dtype=np.float64) @ s_prime


posterior_samples = ift.optimize_kl(
            likelihood_energy=energy,
            total_iterations=30,
            # total_iterations=9,
            n_samples=kl_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory="outs/GOOD_applying_learned_nrt_template_full_freedom_high_fluct_cf",
            return_final_position=False,
            resume=True)

# os.system('say "Skript ausgeführt"')

s_mean, s_var = posterior_samples.sample_stat(s)

latent_sl = posterior_samples.sample_stat()[0]
print("latent_sl.val ", latent_sl.val)

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
inset_ax.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val), label=r"Mean reconstructed field with 1$\sigma$ bands", ecolor=lightest_blue, color=blue)
inset_ax.plot(nrt_time_values_short, nrt_strain_values_short, color="orange",label=r"Suggested numerical relativity template", lw=2)
inset_ax.set_title("Zoomed in", fontsize=15)
inset_ax.set_xlim(16.2, 16.55)
inset_ax.legend(fontsize=10)

usual_plot(xl=r"Time in $\mathrm{sec}$", yl=r"Strain $\mathrm{[10^{-19}]}$")



conclusions = """

    We tested a completely free broken power law with parameter alpha, beta and cf envelope loglogavgslope and 
    fluctuations. 
    
    We did not add an extra multiplicative amplitude degree of freedom. Maybe we should so env = A*e^cfm.
    Think about it with what this amplitude could be degenerate.
    
    Prior choices on broken power law:
    
    # --- Prior choices
    prior_choices = {
        "alpha ": (15,2),
        "beta ": (-6,3),
        "cfm envelope fluctuations": (1, 0.5),
        "cfm envelope loglogavgslope": (-4, 1)
    }
    
    => Quite good results! Out folder: GOOD_applying_learned_nrt_template_full_freedom 
    and model: 
    
        s = generative_model_broken_power_law(h_domain, apply_envelope=True, add_extra_amplitude=FALSE)
    
    If we do add an extra amplitude degree of freedom, 
    
        s = generative_model_broken_power_law(h_domain, apply_envelope=True, add_extra_amplitude=TRUE),
    
    with amp = ift.NormalTransform(mean=500, sigma=500, key="amplitude "), we see a WORSE solution. 
    Because the amplitude is applied  globally over the whole field and for most of the domain the amplitude SHOULD 
    be zero. This checks out with our  observation that the waveform has really damped oscillation where it should 
    have high ones. Folder: applying_learned_nrt_template_full_freedom_and_amplitude.
    
    This we should try once again with really large correlated field fluctuations in order to simulate a pixel-variable
    amplitude operator. I guess. We use 
    
        fluctuations = (100,100)
         
    for the fluct of the correlated field.
    
    => Indeed, works! out folder: GOOD_applying_learned_nrt_template_full_freedom_high_fluct_cf
    
    Sorry I accidently put a uniform prior on fluctuations in (0,1), my bad. 
    Now it really is lognormally distributed in (4, 2). Folder still stays the same, if results are good. 
    Are results good? Yeah I'd even say there a bit better...
    
"""

print(conclusions)