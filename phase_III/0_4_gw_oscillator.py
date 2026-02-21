# Updated version of `phase_III/signal_reconstruction.py`
import numpy as np
from scipy.signal.windows import tukey
from functools import partial
import os
import nifty.nifty.re as jft
import matplotlib.pyplot as plt
from phase_II.nifty_re_playground.strain_tools import *
from phase_III.strain import *
import jax
from phase_III.strain.helpers import jft_model_vjp_jvp_stability
from phase_III.useful.helpers import plot_posterior

jax.config.update("jax_enable_x64", True)

key = jax.random.key(34)

GW150914 = StrainSignalInference(
    key=key,
    event_name="GW150914",
    detector="H1",
    data_duration_of_hdf5_file="4096sec",
    stationarity_time_scale=32,
    e_fac=1,
    r_fac=1,
    alpha_taper_on_data=.1
)

signal_domain = GW150914.machinery.t_ss
target_domain = GW150914.machinery.t_ds

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-2, 1e-16)},  # log fluctuations...
    "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (0, 1e-16), },
    "global_amplitude": (1, 1),
    "init_conditions": (0, 0),
}

signal_prior = StochasticOscillatorPrior(oscillator_prior_dct, signal_time_domain=signal_domain)
oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior, tukey_window=True,
                                normalize=False, add_global_amp=True)

# oscillator.plot_samples(15, key)
inference.add_signal_model(s_model=oscillator)

one_sided_welch_ps = inference.ps_welch[inference.f_welch >= 0]
noise_cov_args = dict(one_sided_noise_ps=one_sided_welch_ps, data_grid=inference.pipe.d_dom_real,
                      apply_correction_factor=True,
                      correction_factor_dont_change_default_for_legacy_reasons=1.8268888821199445)

N_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1), **noise_cov_args)
N_sqrt = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(1/2), **noise_cov_args)
N_sqrt_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1/2), **noise_cov_args)

inference.add_noise_model(N_inv=N_inv, N_sqrt_inv=N_sqrt_inv, N_sqrt=N_sqrt)

# nan_xi_state = unpickle_me_this("STRAIN_GW150914_H1/errors/nan_state_2026-02-14 20:51:54.888990.pkl")
# s_prime = inference.pipe.signal_response()
# oscillator.plot_samples(num=1, key=key, custom_latent_position=nan_xi_state)
# plt.show()

try:
    posterior_latent_samples, vi_info, key = inference.run(kl_iterations=20, use_strict_minimizers=True)
    os.system('afplay /System/Library/Sounds/Glass.aiff')  # or Basso.aiff
except Exception as e:
    print("script stopped: ", e)
    os.system('afplay /System/Library/Sounds/Basso.aiff')  # or Basso.aiff

inference.pipe.plot_posterior_signal(print_posterior_parameters=True,
                                     over_full_signal_space=False,
                                     plot_data=False,
                                     # xlim=(inference.strain.event_time.min(), inference.strain.event_time.max()),
                                     xlim=(-0.2, 0.2),
                                     save_fig=False,
                                     maxL_template_xy=(inference.NR.time, inference.NR.strain),
                                     yl=r"$h(t)$ $\mathrm{[10^{-19}]}$")

# s_prime = inference.pipe.signal_response()
key = plot_posterior(key, times=inference.pipe.t_ss, operator_list=[oscillator.omega, oscillator.gamma, oscillator.xi_force, oscillator],
                     latent_samples=posterior_latent_samples,
                     label_list=["omega", "gamma res", "xi", "waveform"], save_fig=False)