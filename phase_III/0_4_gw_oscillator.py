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
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

key = jax.random.key(34)

GW150914 = StrainSignalInference(
    key=key,
    event_name="GW150914",
    detector="L1",
    data_duration_of_hdf5_file="4096sec",
    stationarity_time_scale=32,
    e_fac=1,
    r_fac=1,
    alpha_taper_on_data=.1,
    out_name='osc'
)

signal_domain = GW150914.machinery.t_ss
target_domain = GW150914.machinery.t_ds

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-2, 1e-16)},  # log fluctuations...
    "damping": {"offset_mean": 500, "offset_std": (250, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (-2, 1e-16)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (0, 1e-16), },
    "global_amplitude": (1, 1),
    "init_condition": (0., 0.),
}

signal_prior = StochasticOscillatorPrior(oscillator_prior_dct, signal_time_domain=signal_domain)
oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior)

# oscillator.plot_samples(20, key)
GW150914.add_signal_model(s_model=oscillator)

# pipe_3.add_custom_signal_model(
# broken_power_law = BrokenPowerLaw(
#                             signal_grid=GW150914.machinery.s_dom_real,
#                             pl_slope_left=(1, .5),
#                             peak_power=1e3,
#                             sigmoid_width=30,
#                             pl_slope_right=(-1, .5),
#                             k_break=(10, 2000),
#                             fluctuations=(1, 1),
#                             envelope_fluctuations=(1, 1e-16),
#                             envelope_loglogavgslope=(-4, 1),
#                             )
# GW150914.add_signal_model(s_model=broken_power_law)

noise_cov_args = dict(one_sided_noise_ps=GW150914.ps_welch, data_grid=GW150914.machinery.d_dom_real)

N_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1), **noise_cov_args)
N_sqrt = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(1/2), **noise_cov_args)
N_sqrt_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1/2), **noise_cov_args)

GW150914.add_noise_model(N_inv=N_inv, N_sqrt_inv=N_sqrt_inv, N_sqrt=N_sqrt)

GW150914.machinery.set_init_pos(init_pos=dict(t0=jnp.float64(0.),), plot=False)

posterior_latent_samples, vi_info, key = GW150914.run(kl_iterations=12, use_strict_minimizers=True)

GW150914.visualize_results(xlim=(-0.1,0.1))