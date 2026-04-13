import pickle
import warnings
import jax.numpy as jnp
import nifty.re as jft
import jax
import matplotlib.pyplot as plt
import numpy as np

from .plotting import *

__all__ = ["unpickle_me_this", "pickle_me_this", "raise_warning", "fw_hartley", "bw_hartley",
                 "plot_histogram", "Logger"]


def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def pickle_me_this(filename: str, data_to_pickle: object):
    path = filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


def raise_warning(msg):
    print("\n")
    warnings.warn(msg, category=UserWarning, stacklevel=2)
    print("\n")


def fw_hartley(x, norm="ortho"):
    r"""
    If ortho, preserves scaling of input. I.e.

        np.var(fw_hartley(\xi)) = np.var(\xi) if e.g. \xi is iid.

    :param x:
    :param norm:
    :return:
    """
    N = len(x)
    Xf = jnp.fft.fft(x)  # Accumulates √N of intrinsic scaling
    Hx = Xf.real - Xf.imag  # standard Hartley: cos+sin → real - imag
    if norm == "ortho":
        Hx /= jnp.sqrt(N)  #  scales with 1/√N
    return Hx  # total scale: 1 if ortho, else √N


def bw_hartley(Hx, norm="ortho"):
    r"""
    This is unitary if ortho norm i.e.

            xi = np.random.standard_normal(8193)
            v = bw_hartley(xi)

            v.T @ v ==  xi.T @ xi

    :param Hx:
    :param norm:
    :return:
    """
    # Hartley is its own inverse! Note: H(H(x)) = N for not-normalized Hartley H.
    # Further, Hx = fw_ortho_hartley(x) ~ 1.
    N = len(Hx)
    x = fw_hartley(Hx, norm=None)  # ~ √N if input scales with 1 (which it does if it comes from ortho fw hartley)
    if norm == "ortho":
        x /= jnp.sqrt(N)  # scales with 1/√N
    return x  # total scale: 1 if ortho AND input is from ortho fw_hartley.
    # if instead input is not from non-ortho fw_hartley: ~ √N I think.

# xi = np.random.standard_normal(8193)
# v = bw_hartley_inv(xi)
# print("v^T v: ", v.T @ v)
# print("xi^T xi: ", xi.T @ xi)

# uH_xi_list = []
# for _ in range(10):
#     xi = np.random.standard_normal(8193)
#     uH_xi = fw_hartley(xi)
#     print("uH_xi var: ", np.var(bw_hartley_inv(uH_xi)))
#     plt.plot(uH_xi)
# plt.show()

# print(np.var(fw_hartley(np.random.standard_normal(1000))))



def plot_histogram(key, mean: float, sigma: float, n_samples: int, bins=200, mode="Lognormal", apply_func=None,
                   apply_func_descriptive_string=None):
    """
    Plots a histogram visualizing the moment-matched lognormal transform.
    If `vlines` is provided, vertical lines will be drawn at the specified x-locations.
    Usage:

    plot_lognormal_histogram(mean=.06, sigma=0.03, n_samples=10000, vlines=[0.023, 0.05], save=True, show=True)

    :param mean:            The mean from which logmean is calculated with logsigma's help.
    :param sigma:           The sigma from which logsigma is calculated.
    :param n_samples:       How many samples to plot
    :param apply_func:      Optional function to apply to each sample.
    :param apply_func_descriptive_string:   If not None, will be added to the title
    :return:
    """
    fig = plt.figure(figsize=(8., 4.))
    if mode == "Normal":
        print("Normal distrubution")
        op = jft.NormalPrior(mean=mean, std=sigma, name="Normal for Histogram")
    elif mode == "Lognormal":
        op = jft.LogNormalPrior(mean=mean, std=sigma, name='Lognormal for Histogram')
    else:
        raise ValueError("Unknown mode '{}'".format(mode))

    rnd_states = []
    for _ in range(n_samples):
        key, key_i = jax.random.split(key)
        rnd_states.append(jft.random_like(key=key_i, primals=op.domain))

    op_samples = np.array([op(state) for state in rnd_states])

    tl = rf"{mode} with $(\mu, \sigma)\approx$" + f"$({np.round(mean,2)}, {np.round(sigma,2)})$" if not (mode=="Uniform") else rf"{mode} in " + r"$\mathrm{[0,1]}$"
    if apply_func is not None:
        op_samples = apply_func(op_samples)
        if not apply_func_descriptive_string:
            tl += "; custom function applied sample-wise!"
        else:
            tl += "; " + apply_func_descriptive_string
    print("Mean and variance of samples:", np.mean(op_samples), np.var(op_samples))
    print("Interval containing ~68% of prob. mass: ", _hdi(op_samples))
    plt.hist(op_samples, bins=bins,
             histtype='step', facecolor='white', color="black")
    thesis_plot(mode="longer", xl="Value", yl="Number", title=tl)
    return key

def _hdi(samples, mass=0.6827):
    # x = np.sort(samples)
    # N = len(x)
    # m = int(np.floor(mass * N))
    # widths = x[m:] - x[:N-m]
    # i = np.argmin(widths)
    # return x[i], x[i+m]
    low, high = np.quantile(samples, [0.1587, 0.8413])
    return low, high


class Logger:
    def __init__(self, silent=False):
        self.silent = silent

    def print(self, *args):
        if self.silent:
            pass
        else:
            print(*args)