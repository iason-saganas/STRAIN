import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
from phase_I.utils.config_jupyter_notebooks import *
from phase_II.fast_wigner_function.utils import thin_out
from phase_II.utils.helpers import hartley_to_fftshift
from utils.helpers import *

data = 1e19 * strain.value
# data = np.loadtxt("tmp_rnd_data.txt")

def model_wrap(real_dom_ext, real_dom_ext_values):
    h = real_dom_ext.get_default_codomain()
    s_broken_power_law = generative_model_continuous_double_power_law(h, apply_envelope=False)
    return s_broken_power_law

# plt.plot(time, data)
# usual_plot()

inference_scheme = ExecuteRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_d/",
    fluct=(5,2),
    llslope=(0,1),
    flex=(1e-16,1e-16),
    custom_generative_model=model_wrap
)

# response = inference_scheme._R_full
# rnd_data = response(ift.from_random(response.domain))
# np.savetxt("tmp_rnd_data.txt", rnd_data.val)

# inference_scheme.plot_power_spectrum_prior_samples(num=10)
# plt.plot(time, data)
# inference_scheme.plot_prior_samples(num=4)

inference_scheme.run()

inference_scheme.plot_posterior()

k_domain_lengths, mean_pow_spec = inference_scheme.plot_posterior_pow_spec(show=False)
plt.show()


p_sls = inference_scheme.posterior_samples
post_mean, post_var = p_sls.sample_stat()
post_s_xi = post_mean.val["s_xi"]  # length == length of extended signal domain

mdl = inference_scheme.model

prior_xi_s = []
num = 10
xi_subdomain = mdl.domain["s_xi"]

h_svol = xi_subdomain[0]._dvol
print("h_svol ", h_svol)

for _ in range(num):
    rnd = ift.from_random(xi_subdomain)
    prior_xi_s.append(rnd.val)

mean_prior_xi = hartley_to_fftshift(np.mean(prior_xi_s, axis=0))
std_prior_xi = hartley_to_fftshift(np.std(prior_xi_s, axis=0))
post_s_xi = hartley_to_fftshift(post_s_xi)
freqs = hartley_to_fftshift(xi_subdomain[0].get_k_length_array().val, flip_negatives=True)

# mean_prior_xi, std_prior_xi, stand_in_x_range, post_s_xi = [thin_out(arr, num=5) for arr in (mean_prior_xi, std_prior_xi, stand_in_x_range, post_s_xi)]
# norm = np.sqrt(len(mean_prior_xi)) * inference_scheme.domain_ext.distances[0]

plt.errorbar(freqs, mean_prior_xi, yerr=std_prior_xi, color="blue", ecolor="lightblue", label=r"Prior harmonic $\xi_s$", zorder=1)
plt.plot(freqs, post_s_xi, "r-", label=r"Posterior harmonic $\xi_s$", zorder=2)
usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")
