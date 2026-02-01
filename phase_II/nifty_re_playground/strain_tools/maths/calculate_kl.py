import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map
from nifty.nifty.re.evi import Samples
from nifty.nifty.re.tree_math import get_map, vdot
from dataclasses import field
from nifty.nifty.re.model import LazyModel
from nifty.nifty.re.likelihood import Likelihood
from jax.experimental.shard_map import shard_map
import nifty.nifty.re as jft


__all__ = ["calculate_kl_val_and_grad", "get_beneficial_position"]


_reduce = partial(tree_map, partial(jnp.mean, axis=0))
class _StandardHamiltonian(LazyModel):
    """Joined object storage composed of a user-defined likelihood and a
    standard normal prior.
    """

    likelihood: Likelihood = field(metadata=dict(static=False))

    def __init__(self, likelihood: Likelihood, /):
        self.likelihood = likelihood

    def __call__(self, primals, **primals_kw):
        return self.energy(primals, **primals_kw)

    def energy(self, primals, **primals_kw):
        return self.likelihood(primals, **primals_kw) + 0.5 * vdot(primals, primals)

    def metric(self, primals, tangents, **primals_kw):
        return self.likelihood.metric(primals, tangents, **primals_kw) + tangents


def calculate_kl_val_and_grad(
    likelihood,
    primals,
    primals_samples=Samples(samples=jnp.array([])),
    *,
    map=jax.vmap,
    reduce=_reduce,
    named_sharding=None,
    kl_device_map="shard_map",
    full_output=False
):
    assert isinstance(primals_samples, Samples)
    map = get_map(map)
    ham = _StandardHamiltonian(likelihood)

    if len(primals_samples) == 0:
        if full_output:
            return jax.value_and_grad(ham)(primals)
        else:
            return jnp.float64((jax.value_and_grad(ham)(primals))[0])


    if named_sharding is None:
        vvg = map(jax.value_and_grad(ham))
    else:
        if kl_device_map == "shard_map":
            vvg = map(jax.value_and_grad(ham))
            spec_tree = tree_map(lambda x: named_sharding.spec, primals)
            out_spec = (named_sharding.spec, spec_tree)
            in_spec = (spec_tree,)
            vvg = shard_map(
                vvg, mesh=named_sharding.mesh, in_specs=in_spec, out_specs=out_spec
            )
        elif kl_device_map == "jit":
            vvg = map(jax.value_and_grad(ham))
            sharding_tree = tree_map(lambda x: named_sharding, primals)
            out_sharding = (named_sharding, sharding_tree)
            in_sharding = (sharding_tree,)
            vvg = jax.jit(vvg, in_shardings=in_sharding, out_shardings=out_sharding)
        elif kl_device_map == "pmap":
            vvg = jax.pmap(jax.value_and_grad(ham))
        else:
            ve = f"`kl_device_map` need to be `pmap`, `shard_map`, or `jit`, not {kl_device_map}"
            raise ValueError(ve)

    s = vvg(primals_samples.at(primals).samples)

    if full_output:
        res = reduce(s)
    else:
        res = jnp.float64(reduce(s)[0])

    return res


def get_beneficial_position(key, lh, samples_to_draw=50):
    # :FIXME: Here, I break the convention that jax.random.split should be used as key, subkey = jax.random.split(key)

    print(f"\tChoosing initial position with lowest KL energy amongst {samples_to_draw} random samples...")

    kl_val = lambda p: calculate_kl_val_and_grad(likelihood=lh, primals=p)

    def sample_once(_key):
        _, subkey = jax.random.split(_key)
        primal = jft.Vector(lh.init(subkey))
        return primal

    keys = jax.random.split(key, samples_to_draw)
    primals = jax.vmap(lh.init)(keys)
    kl_energies = jax.vmap(kl_val)(primals)

    min_init_kl_idx = jnp.argmin(kl_energies-jnp.mean(kl_energies))
    best_primals = jft.Vector({k: primals[k][min_init_kl_idx] for k in primals})
    print("... Found. KL energy of initial position: ", kl_energies[min_init_kl_idx])

    return best_primals

