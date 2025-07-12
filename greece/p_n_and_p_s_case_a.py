import numpy as np
from utils.generative_models import *


n_dtps = len(signal_strip_time)
n_pix = 2*n_dtps-1  # Important! see file `nrt_signal_power_spectrum_inferring_nrt_template.py`
L = np.max(signal_strip_time)-np.min(signal_strip_time)

data_domain = ift.RGSpace(shape=(n_dtps,), distances=L/n_dtps)
signal_domain = ift.RGSpace(shape=(n_pix,), distances=L/n_pix)
h_domain = signal_domain.get_default_codomain()

time_data_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_dtps)
time_signal_domain_values = np.linspace(np.min(signal_strip_time), np.max(signal_strip_time), n_pix)


M = SimpleMask(domain=dt(signal_domain), target=dt(data_domain), keep_th=2)


def generative_model_fixed_envelope(harmonic_space):
    # a broken power law generative model with free modulation flexibility but fixed (to the optimal field) value
    # of the correlated field envelope, as well as an extra amplitude degree of freedom
    # note that i am not normalizing any of the fourier transforms since I assume this can be fixed via the
    # amplitude degree of freedom..


    n_pix_tmp = 2 * len(nrt_time_values) - 1  # so actually every second support point in the signal domain is equal to that in that data domain!
    L_tmp = np.max(nrt_time_values) - np.min(nrt_time_values)
    nrt_signal_time_values = np.linspace(min(nrt_time_values), max(nrt_time_values), n_pix_tmp)
    nrt_signal_domain = ift.RGSpace(shape=(n_pix_tmp,), distances=L_tmp / n_pix_tmp) # ee file `nrt_signal_power_spectrum_inferring_nrt_template.py`

    ht_tmp = ift.HartleyOperator(nrt_signal_domain)

    optimal_harmonic_xi_cf = np.loadtxt("data/xi_s_cfm_envelope.txt")
    optimal_real_xi_cf = ht_tmp.adjoint(ift.Field(dt(ht_tmp.target),val=optimal_harmonic_xi_cf))
    optimal_real_xi_cf_before_manipulation = optimal_real_xi_cf  # for testing
    optimal_real_xi_cf_values = optimal_real_xi_cf.val * 680  # also ich hab's ausgetestet im file `nrt_signal_power_spectrum_inferring_nrt_template`:
    # das tatsächlich optimale cf part hat einen peak bei ein bisschen über 200. Da ich hier nicht normierte harmonic
    # transforms mach is dieser Peak nicht, wenn du mit dieser Zahl multiplizierst dann passt es wieder ca.

    # cut away zero values of the template to the left where there is no data
    indcs_to_keep = np.where(nrt_signal_time_values >= np.min(signal_strip_time))
    time_domain_values_signal_space = nrt_signal_time_values[indcs_to_keep]

    optimal_real_xi_cf_values = optimal_real_xi_cf_values[indcs_to_keep]

    # append 0's to the end
    optimal_real_xi_cf_values = np.concatenate((optimal_real_xi_cf_values, np.zeros(len(time_signal_domain_values)-len(time_domain_values_signal_space))))

    # now see e.g. the following plot:
    # plt.plot(time_signal_domain_values, optimal_real_xi_cf_values, label="after manipulating")
    # plt.plot(nrt_signal_time_values, optimal_real_xi_cf_before_manipulation.val, label="before")
    # usual_plot()

    ht_on_the_larger_domain = ift.HartleyOperator(signal_domain)
    xi_harmonic_manipulated = ht_on_the_larger_domain(ift.Field(dt(ht_on_the_larger_domain.domain),val=optimal_real_xi_cf_values))

    fixed_cf_envelope = ift.VerySimpleCorrelatedField(target=signal_domain,
                                                      fluctuations=5.5207253465,
                                                      loglogavgslope=-2.48558889,
                                                      prefix="fixed_cf_envelope ",
                                                      override_with_exact_values=True,
                                                      custom_harmonic_xi_s=xi_harmonic_manipulated.val
                                                      ).ptw("exp")

    fixed_cf_envelope_op = ift.DiagonalOperator(ift.Field(dt(signal_domain), val=fixed_cf_envelope.val))


    amp = ift.NormalTransform(mean=50, sigma=50, key="amplitude ")
    expander = ift.ContractionOperator(domain=signal_domain, spaces=0).adjoint
    amp = expander @ amp

    wavelet = generative_model_broken_power_law(harmonic_space, apply_envelope=False)

    op = amp*fixed_cf_envelope_op(wavelet)

    return op

s = generative_model_fixed_envelope(h_domain)
# Plot samples
# for _ in range(3):
#     xi = ift.from_random(s.domain)
#     sl = s(xi)
#     plt.plot(time_signal_domain_values, sl.val, label="sample")
#
# plt.plot(signal_strip_time, signal_strip_strain_tapered, label="data")
# plt.plot(nrt_time_values, nrt_strain_values, label="nr template")
# usual_plot()


d = ift.Field(dt(data_domain), val=signal_strip_strain_tapered)
s_prime = M(s)

energy = ift.GaussianEnergy(d, N.inverse, sampling_dtype=np.float64) @ s_prime

posterior_samples = ift.optimize_kl(
            likelihood_energy=energy,
            total_iterations=10,
            # total_iterations=9,
            n_samples=kl_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory="outs/applying_learned_nrt_template_3 ",
            return_final_position=False,
            resume=True)

s_mean, s_var = posterior_samples.sample_stat(s)

latent_sl = posterior_samples.sample_stat()[0]
print("latent_sl.val ", latent_sl.val)

plt.plot(time_signal_domain_values, s_mean.val, label="Mean reconstruction")
plt.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val),label="Mean reconstruction with errorbars")
plt.plot(nrt_time_values, nrt_strain_values, label=r"Actual nrt template")
plt.plot(signal_strip_time, signal_strip_strain_tapered, label="actual data")
usual_plot()

conclusions = """

    Basically here, we are testing whether it can get the right modulation as well as amplitude, given a hint 
    as to the right position o the peak...
    
    But by the way the samples did not looked super peaked at where the NR template is. 
    
    Result with THIS prior (basing towards high alpha values is important, this should be our prior knowledge or 
    our assumption that it is super unsmooth and then if the data demands it it may smooth it out...)
    
    # --- Prior choices
    prior_choices = {
        "alpha ": (15,2),
        "beta ": (-6,3),
        "cfm envelope fluctuations": 5.5207253465,
        "cfm envelope loglogavgslope": -2.4855888
    }
    
    Let's just do the completely free model and see what happens...
    
"""

print(conclusions)