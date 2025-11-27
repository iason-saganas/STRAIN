from _03_baseline_plus_line_model_inference import *

# 1. Initialize inference scheme
pipe_3 = InferenceSchemeRe(t=time, d=strain, e_fac=1, r_fac=1, key=key, plotting_callback=analyze_kl_callback)
pipe_2_ps = pipe_2.get_posterior_statistics(moment="mean", quantity="power spectrum full")

# 2. Build correlated field for signal model
pipe_3.add_cfm_signal_model(fluct=(1e-2, 1e-2), llslope=(-2, 2), flex=(1, 1))

raise_warning("Using welch averaged power spectrum for inference!!! ")

# 3. Add custom noise operator based on data power spectrum from previous inference
tmp_k, tmp_pow_spec = get_welch_averaged_ps(interpolate_to=len(strain))
N_inv_pipe_2 = InvNoiseCovFromPs(noise_ps=tmp_pow_spec, data_grid=pipe_2.d_dom_real, e_fac=1,
                                 n_dtps=len(strain), custom_norm=1)  #  1e-3/4 to get into the data order of mag.

N_sqrt_inv_pipe_2 = InvNoiseCovFromPs(noise_ps=np.sqrt(tmp_pow_spec), data_grid=pipe_2.d_dom_real, e_fac=1,
                                 n_dtps=len(strain), custom_norm=1)

tmp_xi = np.random.standard_normal(len(strain))
a = N_inv_pipe_2(tmp_xi)
b = N_sqrt_inv_pipe_2(N_sqrt_inv_pipe_2(tmp_xi))

import jax.numpy as jnp
import numpy as np

p = tmp_pow_spec
x = np.random.randn(len(tmp_pow_spec))

F  = lambda y: jnp.fft.fft(y, norm="ortho")
iF = lambda y: jnp.fft.ifft(y, norm="ortho")

A = lambda xi: iF(p**-1 * F(xi))
B = lambda xi: iF(p**-0.5 * F(xi))

print(np.allclose(A(x), B(B(x))))  # should be True
stop

plt.plot(a, label="Inverse Noise Cov")
plt.plot(b, label="Sqrt Noise Cov applied twice")
plt.legend()
plt.show()

pipe_3.add_noise_op(inverse_noise_op=N_inv_pipe_2, sqrt_inverse_noise_op=N_sqrt_inv_pipe_2)

# 4. Get some noise and signal samples
pipe_3.plot_noise_sample_with_data(num=1, rolling=False)

# pipe_3.plot_prior_samples(num=3, mode="signal")

latent_samples = pipe_3.run_inference(kl_iterations=15, use_strict_minimizers=True, out_name="re_pipe_3", resume=True,
                                      choose_low_kl_starting_pos=True, geoVi=True)
key = pipe_3.get_current_key()

pipe_3.plot_posterior_signal()
pipe_3.plot_posterior_power_spectrum(print_posterior_parameters=True)
pipe_3.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
pipe_3.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)
