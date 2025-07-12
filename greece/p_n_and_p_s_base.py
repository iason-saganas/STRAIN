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


def generative_model_nrt():
    nrt_signal_time_values = np.linspace(min(nrt_time_values), max(nrt_time_values), 2*len(nrt_time_values)-1)
    optimal_nrt_field = unpickle_me_this("data/best_nrt_field.pickle")
    optimal_nrt_field_values = optimal_nrt_field.val

    # cut away zero values of the template to the left where there is no data
    indcs_to_keep = np.where(nrt_signal_time_values >= np.min(signal_strip_time))
    time_domain_values_signal_space = nrt_signal_time_values[indcs_to_keep]
    optimal_nrt_field_values = optimal_nrt_field_values[indcs_to_keep]

    # append 0's to the end
    optimal_nrt_field_values = np.concatenate((optimal_nrt_field_values, np.zeros(len(time_signal_domain_values)-len(time_domain_values_signal_space))))

    # now see e.g. the following plot:
    # plt.plot(signal_strip_time, signal_strip_strain_tapered)
    # plt.plot(time_signal_domain_values, optimal_nrt_field_values)
    # usual_plot()

    nrt_op = ift.DiagonalOperator(ift.Field(dt(signal_domain), val=optimal_nrt_field_values))
    amp = ift.NormalTransform(mean=500, sigma=500, key="amplitude of nrt ")
    expander = ift.ContractionOperator(domain=signal_domain, spaces=0).adjoint
    amp = expander @ amp

    op = nrt_op @ amp

    return op

s = generative_model_nrt()

# Plot samples
for _ in range(10):
    xi = ift.from_random(s.domain)
    sl = s(xi)
    plt.plot(time_signal_domain_values, sl.val, label="sample..")

plt.plot(signal_strip_time, signal_strip_strain_tapered, label="data")
plt.plot(nrt_time_values, nrt_strain_values, label="nr template")
usual_plot()

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
            output_directory="outs/applying_learned_nrt_template ",
            return_final_position=False,
            resume=True)

s_mean, s_var = posterior_samples.sample_stat(s)

plt.plot(time_signal_domain_values, s_mean.val, label="Mean reconstruction")
plt.errorbar(time_signal_domain_values, s_mean.val, yerr=np.sqrt(s_var.val),label="Mean reconstruction with errorbars")
plt.plot(nrt_time_values, nrt_strain_values, label="Actual nrt template")
plt.plot(signal_strip_time, signal_strip_strain_tapered, label="actual data")
usual_plot()

conclusions = """

    As we can see, when the broken power law model is fixed to its optimal values und just an amplitude degree 
    of freedom remains, the posterior field settles in at the amplitude also suggested by the numerical relativity 
    template. 
    
    This was expected. 
    
    Now, we want to re-introduce the model degrees of freedom: Assuming the optimal form of 
    
    a.) the envelope => Will the model find a wave modulation similar to the numerical relativity template ? 
    b.) the wave form => Will the model find the correct envelope i.e. position of the excess power signal?  
    
"""

print(conclusions)