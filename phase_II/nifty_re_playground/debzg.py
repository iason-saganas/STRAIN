"""

Ok so the pseudo-inversion ended up working really well for the last power spectrum iteration

"""

import matplotlib.pyplot as plt

from strain_tools import *
from _03_baseline_plus_line_model_inference import *
import jax.numpy as jnp

pipe_2.plot_posterior_power_spectrum(mode="mean")
penrose_xi = find_penrose_moore_solution(pipe=pipe_2, itr=100_000, absdelta=1e-10, reload_from_cache=False, filename="AAAH.txt")

plt.plot(pipe_2.k_signal_full, penrose_xi.real)
plt.show()

ps_mean_std, _, _ = pipe_1.get_posterior_statistics()
ps_mean = (ps_mean_std[0])[pipe_1.s_h_dom_expander]
posterior_penrose_data = sample_from_ps(penrose_xi, N=pipe_1.n_ds, inverse_h_trafo=lambda p: jnp.fft.ifft(p, norm="ortho"),
                                        ps=ps_mean)

plt.plot(pipe_2.t_ds, pipe_2.d, color="orange", label=r"OG data")
plt.plot(pipe_2.t_ds, posterior_penrose_data.real, color=blue, label=r"Data from smooth $p_n(f)$ and $\tilde{\xi}_d^{\ast}$")
plt.show()

S_penrose, t_dual, f_dual = Stress_jft(penrose_xi, time=pipe_2.t_ss, downsample=False)
visualize_stress(S_penrose, f_dual, t_dual, smooth=True)