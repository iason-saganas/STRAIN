from phase_III.strain import *

# Updated version of `phase_III/signal_reconstruction.py`
from phase_II.nifty_re_playground.strain_tools import *
from phase_III.strain import *
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import nifty.re as jft

key = jax.random.key(34)

GW150914 = StrainSignalInference(
    key=key,
    event_name="GW150914",
    detector="H1",
    data_duration_of_hdf5_file="4096sec",
    stationarity_time_scale=32,
    e_fac=1,
    r_fac=1,
    alpha_taper_on_data=.1,
    out_name='osc_dlt_later'
)

signal_domain = GW150914.machinery.t_ss
target_domain = GW150914.machinery.t_ds

oscillator_prior_dct = {
    "frequency": {"offset_mean": 1000, "offset_std": (500, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-4, 1)},  # log fluctuations...
    "damping": {"offset_mean": 0, "offset_std": (5, 1e-16), "fluctuations": (10, 10), "loglogavgslope": (-4, 1)},
    "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1e-16, 1e-16), "loglogavgslope": (0, 1e-16), },
    "global_amplitude": (1, 1),
    "init_condition": (0., 1),
}

signal_prior = StochasticOscillatorPrior(oscillator_prior_dct, signal_time_domain=signal_domain, localize_force=None, couple_force_to_frequency=False)
oscillator = HarmonicOscillator(signal_domain_times=signal_domain, signal_prior=signal_prior)

# oscillator.plot_samples(20, key, show_spectrogram=True)

GW150914.add_signal_model(s_model=oscillator)

# pipe_3.add_custom_signal_model(

noise_cov_args = dict(one_sided_noise_ps=GW150914.ps_welch, data_grid=GW150914.machinery.d_dom_real)

N_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1), **noise_cov_args)
N_sqrt = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(1/2), **noise_cov_args)
N_sqrt_inv = NoiseCovarianceFromPs(callable_to_apply=lambda x: x**(-1/2), **noise_cov_args)

GW150914.add_noise_model(N_inv=N_inv, N_sqrt_inv=N_sqrt_inv, N_sqrt=N_sqrt)

lh = GW150914.machinery.build_lh()

key, key_i = jax.random.split(key)
init_pos = lh.init(key_i)

posterior_latent_samples, final_state_and_params = jft.blackjax_nuts(
    likelihood=GW150914.machinery.build_lh(),
    position=init_pos,
    key=key,
    n_warmup_steps=3,
    n_samples=3,
)

# to_save = [posterior_latent_samples, final_state_and_params]
# pickle_me_this("hmc_res", to_save)

GW150914.machinery.posterior_xi_samples = posterior_latent_samples
GW150914.visualize_results(xlim=(-0.1,0.1))

# posterior_latent_samples, vi_info, key = GW150914.run(kl_iterations=5, use_strict_minimizers=False)
# GW150914.visualize_results(xlim=(-0.1,0.1))

