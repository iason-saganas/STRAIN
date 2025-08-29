import nifty.nifty.re as jft
from jax import random
from jax import numpy as jnp
import jax
from phase_I.utils.config_jupyter_notebooks import *
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

data = jnp.array(data)

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(8, 3),
    loglogavgslope=(-4, 2),
    flexibility=(2, 2),
    asperity=(1e-5, 1e-15),
)

n_pix = len(time)
dt = time[1] - time[0]

cfm_model = jft.CorrelatedFieldMaker("s_cfm_")
cfm_model.set_amplitude_total_offset(**cf_zm)
cfm_model.add_fluctuations(shape=(n_pix,), distances=dt, **cf_fl, non_parametric_kind="power", prefix="s_cfm_params")

correlated_field = cfm_model.finalize()

class Signal(jft.Model):
    def __init__(self, cf):
        self.cf = cf
        super().__init__(init=self.cf.init)

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return self.cf(x)

# plt.plot(time, data)
# plt.show()

s_prime = Signal(cf=correlated_field)
inv_noise_cov = lambda x: 10**(-2) + 0*x

model_output_shape = s_prime.target.shape  # or s_prime.target.shape

# Create zero data
data_zero = jnp.zeros(model_output_shape)

lh = jft.Gaussian(data_zero, inv_noise_cov).amend(s_prime)

n_vi_iterations = 6
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="results_basic_inference",
    resume=True,
)