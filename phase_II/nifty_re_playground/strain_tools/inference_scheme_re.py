import jax.numpy as jnp
import nifty.re as jft
from scipy.signal.windows import tukey
from typing import Optional, Callable, Literal
from scipy.interpolate import interp1d
import jax
from time import time
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import re

from .basics.common_utils import unpickle_me_this, pickle_me_this, bw_hartley, raise_warning
from .basics.plotting import *

from .maths.calculate_kl import get_beneficial_position
from .maths.calculate_pseudoinverse import find_penrose_moore_solution, sample_from_ps

from .models.custom_correlated_field import CustomCorrelatedFieldMaker


__all__ = ["InferenceSchemeRe", "analyze_kl_callback", "plot_welch_averaged_ps", "get_welch_averaged_ps",
           "kl_sampling_rate"]

def join_k_arrays(harmonic_grid):
    """
    Expands unique k's into a full, signed k-array.
    :param harmonic_grid:
    :return:
    """
    k_lengths = harmonic_grid.mode_lengths
    expander = harmonic_grid.power_distributor
    full_k_lengths = k_lengths[expander]
    arr1 = full_k_lengths[:int(len(full_k_lengths) / 2) + 1]
    arr2 = -1 * full_k_lengths[int(len(full_k_lengths) / 2) + 1:]
    joint_k = np.concatenate((arr1, arr2))
    return joint_k


def get_welch_averaged_ps(interpolate_over_k_grid=None):
    """

    :param interpolate_over_k_grid:   If an array, interpolates the power spectrum over that new k grid.
    :return:
    """
    welch = "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle"
    _, k_lengths, power_spectrum = unpickle_me_this(welch, absolute_path=True)

    if interpolate_over_k_grid is not None:
        power_spectrum = power_spectrum.val
        interpolator = interp1d(x=k_lengths, y=power_spectrum, kind="linear")
        new_power_spectrum = interpolator(interpolate_over_k_grid)
        return jnp.array(interpolate_over_k_grid), jnp.array(new_power_spectrum)

    return jnp.array(k_lengths), jnp.array(power_spectrum.val)


def plot_welch_averaged_ps(ax=None, lb="Empirical estimate", **kwargs):
    k_lengths, power_spectrum = get_welch_averaged_ps()
    k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
    spectrum_welch = power_spectrum[1:]
    if ax is None:
        ax = plt.gca()
    ax.plot(k_lengths, spectrum_welch, label=lb, color="black", **kwargs)


def mean_red_chi2(data, d_th_samples, N_inv_op):
    """
    :param data:            The data array
    :param d_th_samples:    A list of arrays representing forward model calls of posterior latent samples.
    :param N_inv_op:        The inverse noise operator
    :return:
    """
    d = data
    N = len(d)
    res_list = [d - d_th for d_th in d_th_samples]
    red_chi2_samples = [res.T @ N_inv_op(res) / N for res in res_list]
    return np.mean(red_chi2_samples)


def kl_sampling_rate(index: int):
    """
    Callable for sampling of KL. KL minimization can be performed on its samples instead of computing it directly.
    First, get a ballpark, in later iteration increase sampling rate.
    :param index:
    :return:
    """
    itrs = 7
    if index > 10:
        itrs = 30
    return itrs

def get_last_iteration(out_name):
    path = Path(out_name) / "minisanity.txt"
    pattern = re.compile(r"OPTIMIZE_KL: Iteration\s+(\d+)")

    last_iter = 1
    try:
        with open(path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    last_iter = int(m.group(1))
    except FileNotFoundError:
        pass

    return last_iter



class InferenceSchemeRe:
    def __init__(self, t, d, key, e_fac=2, r_fac=2, plotting_callback=None):
        """
        Uses an RG for the data space.

        Usage:

                pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key)
                pipe_1.add_cfm_signal_model(fluct=(5,2), llslope=(-4,1))
                pipe_1.add_noise_op(noise_var_level=1)

                latent_samples = pipe_1.run_inference(kl_iterations=5, use_strict_minimizers=False, out_name="re_pipe_1", resume=True)
                key = pipe_1.get_current_key()

                pipe_1.plot_posterior_power_spectrum()

        :param t:                   The time at which the data were sampled.
        :param d:                   The data.
        :param key:                 The jax PRNG key.
        :param e_fac:               The factor by which to extend the length of the domain to ensure periodicity.
                                    Default: 2.
        :param r_fac:               The factor by which to increase the resolution of the signal space. Default 2.
        :param plotting_callback:   Callable with signature 'out_name', 'max_kl_iterations', 'lh', 'samples' and
                                    'vi_state'.
                                    Executed after each global iteration. Default: None.
        """
        assert (lambda dt: (dt > 0).all() and np.allclose(dt, dt[0]))(np.diff(t))  # check: t array is monotoneous and
        # equally spaced

        # abbreviations: ds := 'data_space' ; ss := 'signal_space'
        self.d = d
        self.e_fac = e_fac
        self.r_fac = r_fac
        self.key = key

        self.L_ds = t[-1]-t[0]
        self.L_ss = self.L_ds * self.e_fac

        self.dist_ds = t[1] - t[0]
        self.dist_ss = self.dist_ds / r_fac

        self.n_ds = len(d)
        assert self.n_ds == int(self.L_ds/self.dist_ds) + 1  # 3 intervals correspond to four edges; self.L_ds/self.dist_ds
        # are the intervals, whereas each datapoint is supported by one edge

        if not isinstance(e_fac, int) or not isinstance(r_fac, int):
            raise ValueError("e_fac and r_fac should be int for simplicity")
        self.n_ss = int(self.L_ss / self.dist_ss) # = e_fac*L_ds / (dist_ds/r_fac)
        # = e_fac*r_fac * L_ds/dist_ds = e_fac*r_fac * (n_ds-1) = e_fac*r_fac * n_ds - e_fac*r_fac
        self.n_ss += e_fac * r_fac

        self.t_ds = t
        self.t_ss = jnp.arange(self.n_ss)*self.dist_ss + t[0]

        def adjoint_zp(arr, ext_fact, res_fact):
            # Takes an extended and more resolved array arr, cuts and downsamples
            n_large = len(arr)
            n_small = int(n_large / (ext_fact * res_fact))
            return arr[::res_fact][:n_small]
        self.adjoint_zp = adjoint_zp

        assert self.t_ds[0] == self.t_ss[0]  # same beginning?
        assert jnp.all(self.t_ds == self.adjoint_zp(self.t_ss, e_fac, r_fac))  # if you cut the extended
        # array up to the max of t_ds and then take make it coarser, do you get the same support points?


        self.d_dom_real = jft.correlated_field.make_grid(shape=(self.n_ds,), distances=(self.dist_ds,),
                                                     harmonic_type="Fourier")
        self.s_dom_real = jft.correlated_field.make_grid(shape=(self.n_ss,), distances=(self.dist_ss,),
                                                     harmonic_type="Fourier")

        self.d_dom_harmonic = self.d_dom_real.harmonic_grid
        self.s_dom_harmonic = self.s_dom_real.harmonic_grid

        self.s_h_dom_expander = self.s_dom_harmonic.power_distributor
        self.d_h_dom_expander = self.d_dom_harmonic.power_distributor

        self.k_data = self.d_dom_harmonic.mode_lengths
        self.k_signal = self.s_dom_harmonic.mode_lengths

        self.k_data_full = join_k_arrays(self.d_dom_harmonic)
        self.k_signal_full = join_k_arrays(self.s_dom_harmonic)

        self.plotting_callback = plotting_callback
        self.amplitude_op = None  # undo calls to amplitude_op and use self.ps instead
        self.ps = None
        self.s_model = None
        self.data_model_taper = .0

        self.inv_N_cov = None  # to build residuals
        self.sqrt_inv_N_cov = None  # the metric
        self.sqrt_noise_op = None  # for drawing samples visually, no role in the inference

        self.kl_kwargs = None
        self.nonlinearly_update_kwargs = None
        self.draw_linear_kwargs = None
        self.posterior_xi_samples = None
        self.parameter_choices = None
        self.model_prefix = None
        self.init_pos = None


    def add_custom_signal_model(self, custom_signal_model: jft.Model, alpha=0.):
        """

        custom_signal_model: jft.Model      Needs to have implemented __init__, __call__ and get_model_components

        model_prefix needs to include an underscore as suffix.
        parameter_choices needs to look like
            parameter_choices = {
            f'{model_prefix}fluctuations': lambda xi: np.exp(fluct[0] + xi*fluct[1]),
            f'{model_prefix}loglogavgslope': lambda xi: llslope[0] + xi*llslope[1],
            f'{model_prefix}flexibility': lambda xi: np.exp(flex[0] + xi*flex[1]),
            f'{model_prefix}asperity': lambda xi: np.exp(asper[0] + xi*asper[1]),
        }

        Be sure to implement an .init method/property to be fed into jft.Model __init__,
        like init=cfm_maker.finalize().init

        Models should be defined over the extended domain.

        Implement a .get_model_components() method that returns the tuples listed down below.

        :param custom_signal_model:  A jft.Model representing the signal.
        :param alpha:                If a float, data model will downstream be tapered by a tukey window with this
                                     shape parameter.
        :return:
        """

        try:
            amplitude_op, parameter_choices, model_prefix = custom_signal_model.get_model_components()
        except AttributeError:
            raise ValueError("Custom signal model must include method `get_model_components`, containing "
                             "\namplitude_op, parameter_choices and model_prefix")

        self.s_model = custom_signal_model
        self.amplitude_op = amplitude_op
        self.parameter_choices = parameter_choices
        self.model_prefix = model_prefix
        self.data_model_taper = alpha


    def add_cfm_signal_model(self, fluct:tuple, llslope:tuple, flex:tuple | None = None, asper:tuple | None=None,
                             offset_mean:float = 0, offset_std:tuple = (1e-16, 1e-16), model_prefix="s_",
                             add_power_spectrum_template=None, add_peak_model=None, square_iwp=False,
                             add_cfm_env=False, alpha=0.):
        """

        :param fluct:
        :param llslope:
        :param flex:
        :param asper:
        :param offset_mean:
        :param offset_std:
        :param model_prefix:
        :param add_power_spectrum_template:         An array over unique frequency values to add as a baseline to the
                                                    power spectrum of the correlated field. DON'T USE IN NEW CODE, ONLY
                                                    FOR BACKWARDS-COMPATIBILITY; INSTEAD PASS A LIST OF OPERATORS VIA
                                                    add_custom_power_ops.
        :param add_peak_model:                      A peak model in power space to be added to the model.
        :param alpha:                               If a float, data model will downstream be tapered by a tukey window
                                                    with this shape parameter.

        :return:
        """

        cfm_maker = CustomCorrelatedFieldMaker(prefix=model_prefix)
        # cfm_maker = jft.CorrelatedFieldMaker(prefix=model_prefix)

        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=fluct,
                                   loglogavgslope=llslope, flexibility=flex, asperity=asper, harmonic_type="fourier",
                                   non_parametric_kind="power", power_spectrum_template=add_power_spectrum_template,
                                   peak_model=add_peak_model,  make_iwp_pos_definite=square_iwp)

        parameter_choices = {
            f"{model_prefix}fluctuations": lambda xi: np.exp(fluct[0] + xi*fluct[1]),
            f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
            f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
            f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
        #     #"offset_mean": (offset_mean, "fix"),
        #     "offset_std": (offset_std, "lognormal"),
        }


        # gaussian_win = False
        # if gaussian_win:
        #     s_model_cfm = cfm_maker.finalize()
        #     dom = s_model_cfm.domain
        #
        #     x = self.t_ss
        #     std_dev = .1
        #     mean = 16.4
        #     gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        #
        #     s_model = lambda p: s_model_cfm(p)*gaussian
        #     s_model.domain = dom
        #
        #     raise_warning("Using fix gaussian window signal model!! Add instead a generative gaussian envelope or similar")
        if add_cfm_env:
            env_fluct = (1,1)
            env_llslope = (-4,2)

            env_maker = jft.CorrelatedFieldMaker(prefix=model_prefix+"env_")
            env_maker.set_amplitude_total_offset(1e-16, (1e-16, 1e-16))
            env_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=env_fluct,
                                       loglogavgslope=env_llslope, flexibility=None, asperity=None,
                                       harmonic_type="fourier",
                                       non_parametric_kind="power",
                                       hack_add_power_spectrum_template=None,
                                       hack_custom_amplitude_operators=None,
                                       hack_make_iwp_pos_definite=False)

            s_wavelet = cfm_maker.finalize()
            log_s_env = env_maker.finalize()
            s_env = lambda p: jnp.exp(log_s_env(p))
            s_model = lambda p: s_wavelet(p) * s_env(p) * tukey(self.n_ds, alpha=0.3)

            s_model.domain = s_wavelet.domain | log_s_env.domain

        else:
            s_model_fin = cfm_maker.finalize()
            s_model = lambda p: s_model_fin(p) * tukey(self.n_ss, alpha=alpha)
            s_model.domain = s_model_fin.domain



        self.s_model = s_model
        self.amplitude_op = cfm_maker.amplitude
        self.parameter_choices = parameter_choices
        self.model_prefix = model_prefix


    def add_matern_signal_model(self, scale:tuple, llslope:tuple, cutoff: tuple,
                             offset_mean:float = 0, offset_std:tuple = (1e-16, 1e-16), model_prefix="s_",
                                add_cfm_env=False):
        """

        :param llslope:
        :param offset_mean:
        :param offset_std:
        :param model_prefix:

        :return:
        """

        cfm_maker = jft.CorrelatedFieldMaker(prefix=model_prefix)
        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations_matern(shape=(self.n_ss,), distances=self.dist_ss, scale=scale,
                                   loglogslope=llslope, cutoff=cutoff, renormalize_amplitude=True, harmonic_type="fourier",
                                   non_parametric_kind="power")

        # parameter_choices = {
        #     f"{model_prefix}fluctuations": lambda xi: np.exp(fluct[0] + xi*fluct[1]),
        #     f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
        #     f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
        #     f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
        #     #"offset_mean": (offset_mean, "fix"),
        #     #"offset_std": (offset_std, "lognormal"),
        # }
        parameter_choices = {}

        if add_cfm_env:
            env_fluct = (1, 1)
            env_llslope = (-4, 2)

            env_maker = jft.CorrelatedFieldMaker(prefix=model_prefix + "env_")
            env_maker.set_amplitude_total_offset(1e-16, (1e-16, 1e-16))
            env_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=env_fluct,
                                       loglogavgslope=env_llslope, flexibility=None, asperity=None,
                                       harmonic_type="fourier",
                                       non_parametric_kind="power",
                                       hack_add_power_spectrum_template=None,
                                       hack_custom_amplitude_operators=None,
                                       hack_make_iwp_pos_definite=False)

            s_wavelet = cfm_maker.finalize()
            log_s_env = env_maker.finalize()
            s_env = lambda p: jnp.exp(log_s_env(p))
            s_model = lambda p: s_wavelet(p) * s_env(p) * tukey(self.n_ds, alpha=0.3)

            s_model.domain = s_wavelet.domain | log_s_env.domain

        else:
            s_model_fin = cfm_maker.finalize()
            s_model = lambda p: s_model_fin(p) * tukey(self.n_ss, alpha=0.3)
            s_model.domain = s_model_fin.domain

        self.s_model = s_model
        self.amplitude_op = cfm_maker.amplitude
        self.parameter_choices = parameter_choices
        self.model_prefix = model_prefix


    def signal_response(self):
        # if self.r_fac != 1 and self.r_fac != 2:
        #     raise ValueError("Non-Unit responses (masks) except for res_fac == 2 not implemented yet.")
        if self.s_model is None:
            raise ValueError("No signal model implemented yet, call 'add_cfm_signal_model' or add_custom_signal_model.")

        class Response(jft.Model):
            def __init__(self, signal_model, zp_adj, ext_fac, res_fac, alpha, M):
                self.sm = signal_model
                self.zp_adj = zp_adj
                self.ext_fac = ext_fac
                self.res_fac = res_fac
                # alpha: taper for tukey window; if 0. not tapered.
                # M: the number of datapoints
                self.taper = tukey(M, alpha=alpha)
                super().__init__(domain=signal_model.domain)

            def __call__(self, xi):
                model_values_on_long_domain = self.sm(xi)
                downsampled_and_cut = self.zp_adj(model_values_on_long_domain, self.ext_fac, self.res_fac)
                return self.taper * downsampled_and_cut

        print("Using alpha shape parameter of ", self.data_model_taper, " for signal response")
        s_prime = Response(signal_model=self.s_model, zp_adj=self.adjoint_zp, ext_fac=self.e_fac, res_fac=self.r_fac,
                           alpha=self.data_model_taper, M=self.n_ds)

        return s_prime


    def add_noise_op(self, noise_var_level=1e-10,
                     inverse_noise_op:Optional[Callable[[jnp.array], jnp.array]]=None,
                     sqrt_inverse_noise_op:Optional[Callable[[jnp.array], jnp.array]]=None,
                     sqrt_noise_op:Optional[Callable[[jnp.array], jnp.array]]=None):

        self.sqrt_noise_op = sqrt_noise_op  # just to draw samples, plays no role in the inference.

        if (inverse_noise_op is None) ^ (sqrt_inverse_noise_op is None):  # ^ = xOr operator!
            raise ValueError("One of inverse_noise_op or sqrt_inverse_noise_op was provided, but the other not, "
                             "likely want to provide both, eitherwise wrong metric in Gaussian likelihood.")

        if inverse_noise_op is None:
            print("\nUsing DIAGONAL noise operator.")
            # Both operators were not provided => Diagonal noise covariance
            self.inv_N_cov = lambda x: x/noise_var_level
            self.sqrt_inv_N_cov = lambda x: x/jnp.sqrt(noise_var_level)
        else:
            print("\nUpdating inference noise model with provided N**(-1) and N**(-1/2)!")
            # Both operators were correctly provided.
            self.inv_N_cov = inverse_noise_op
            self.sqrt_inv_N_cov = sqrt_inverse_noise_op


    def set_init_pos(self, init_pos:jft.Vector | dict, plot=False, plot_power_spectrum=False, plot_welch_average=True):
        """

        :param init_pos:    A jft vector or a dictionary containing latent space values to use as initial positions
                            in the inference. The dictionary does not have to contain all the keys of the likelihood
                            domain. If a key is missing, a random value is drawn.
        :param plot:        Whether to plot the resulting initial postion.
        :param plot_welch_average:  If plot and plot_welch_average, plots welch average. Useful for noise comparison.
        :return:
        """
        print("Type of init pos: ", type(init_pos))
        if type(init_pos) is jft.Vector:
            init_pos = init_pos._tree
            print("Type of init pos now: ", type(init_pos))

        lh = self.build_lh(supress_print=True)

        print("You are trying to set an initial positions. Please note that the likelihood domain keys are: \n\t",
              "<", *lh.domain.keys(),">")
        print("Out of these keys, the input initial position seems to not contain \n\t", *[k for k in lh.domain.keys() if k not in init_pos.keys()],
              " and thus these values will be drawn randomly.")


        self.key, key_i = jax.random.split(self.key)
        base_initial_position = lh.init(key_i)

        for key in init_pos.keys():
            if key in base_initial_position.keys():
                base_initial_position[key] = init_pos[key]

        self.init_pos = jft.Vector(base_initial_position)

        if plot:
            ps = lambda xi: self.amplitude_op(xi) ** 2
            s_prime = self.signal_response()

            init_ps = ps(self.init_pos)
            init_s_prime = s_prime(self.init_pos)

            nrows = 2 if plot_power_spectrum else 1
            fig, axs = plt.subplots(nrows=nrows, ncols=1, squeeze=False)
            axs = axs.flatten()

            if plot_power_spectrum:
                ax_ps = axs[0]

                if plot_welch_average:
                    plot_welch_averaged_ps(ax_ps)

                ax_ps.loglog(self.k_signal, init_ps, label="initial power spectrum")
                ax_ps.set_xlabel("Frequency $f$")
                ax_ps.set_ylabel("Power")
                ax_ps.legend()

                ax_ts = axs[1]
            else:
                ax_ts = axs[0]

            ax_ts.plot(self.t_ds, init_s_prime, label="initial position in data space")
            ax_ts.set_xlabel("Time $t$")
            ax_ts.set_ylabel("Strain")
            ax_ts.legend()

            fig.tight_layout()
            plt.show()

    def add_minimizers(self, linear_loose=(0.02, 100), linear_strict=(0.02, 150), non_linear_loose=(0.5, 20),
                       non_linear_strict=(0.3, 30), kl_loose=(0.1, 35), kl_strict=(0.01, 50), use_strict=False):

        if use_strict:
            linear_energy, linear_iter = linear_strict
            nonlinear_energy, nonlinear_iter = non_linear_strict
            kl_energy, kl_iter = kl_strict
        else:
            linear_energy, linear_iter = linear_loose
            nonlinear_energy, nonlinear_iter = non_linear_loose
            kl_energy, kl_iter = kl_loose

        draw_linear_kwargs = dict(
            cg_name="CG, linear sampling",
            cg_kwargs=dict(absdelta=linear_energy, maxiter=linear_iter),
        )

        # Arguments for the minimizer in the nonlinear updating of the samples
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="Nonlinear sampling NCG",
                absdelta=nonlinear_energy,
                xtol=1e-10,  #  so for geoVI, it looks like xtol and absdelta are xOr thresholds, which is I think
                # not the behaviour of kl_kwargs... so I supply a small xtol here in order to just use the
                # energy threshold
                cg_kwargs=dict(name="\tCG, nonlinear sampling",),
                maxiter=nonlinear_iter,
            )
        )

        # Arguments for the minimizer of the KL-divergence cost potential
        kl_kwargs = dict(
            minimize_kwargs=dict(
                name="KL minim NCG", absdelta=kl_energy, cg_kwargs=dict(name="\tCG: KL minim"), maxiter=kl_iter
            )
        )

        self.draw_linear_kwargs = draw_linear_kwargs
        self.nonlinearly_update_kwargs = nonlinearly_update_kwargs
        self.kl_kwargs = kl_kwargs


    def build_lh(self, supress_print=False):
        s_prime = self.signal_response()

        if self.inv_N_cov is None:
            level = 1e-10
            if not supress_print:
                raise_warning(f"self.add_noise_op() was not called by the user, using Gaussian noise with default "
                              f"variance level {level}.")
            self.add_noise_op(noise_var_level=level)

        # self.inv_N_cov = jft.LogNormalPrior(mean=1, std=.1, name="inv noise level", dtype=np.float64, shape=())
        if isinstance(self.inv_N_cov, jft.Model):
            print("Using variable Gaussian covariance energy")
            # std_inv_model = jft.Model(call= lambda x: jnp.sqrt(self.inv_N_cov(x)), domain=self.inv_N_cov.domain)
            input_restore = jft.Model(call=lambda xi: (s_prime(xi), self.sqrt_inv_N_cov(xi)*jnp.ones_like(self.d)),
                                      domain=s_prime.domain | self.inv_N_cov.domain)
            lh = jft.VariableCovarianceGaussian(data=self.d).amend(input_restore, domain=input_restore.domain)
            lh.noise_cov_inv = self.inv_N_cov  # using this in the plotting callback
        else:
            lh = jft.Gaussian(data=self.d, noise_cov_inv=self.inv_N_cov, noise_std_inv=self.sqrt_inv_N_cov).amend(s_prime)
        return lh

    def run_inference(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False, out_name="out",
                     resume=True, choose_low_kl_starting_pos=False, geoVi=True, chi2_threshold=jnp.inf, max_kl_iter=None,
                      **kwargs):
        """

        :param kl_iterations:               The number of kl iterations to run at least. If max_kl_iter is None,
                                            exactly the number of kl iterations.
        :param n_samples:                   How many samples to draw from the posterior distribution during each kl
                                            iteration.
        :param use_strict_minimizers:       Whether to decrease step size in all CG.
        :param out_name:                    Where results will be saved.
        :param resume:                      Whether to use stored results
        :param choose_low_kl_starting_pos:  If self.init_pos was not set explicitly, tries to find a minimum kl starting
                                            position.
        :param geoVi:                       If true, use non-linear resampling mode.
        :param chi2_threshold:              If not jnp.inf, as many kl iterations will be run (at most max_kl_iter)
                                            until chi^2 falls under this treshold.
        :param max_kl_iter:                 Alternate termination reason for iterative optimization of KL.
        :param kwargs:                      Kwargs to be passed to optimize_kl.
        :return:
        """
        lh = self.build_lh()

        min_kl_iter = kl_iterations
        if chi2_threshold != jnp.inf and ((min_kl_iter is None) or max_kl_iter is None):
            raise ValueError("Please provide min_kl_iter and max_kl_iter if you want to iteratively compute "
                             "kl until chi2_threshold=", chi2_threshold, " is hit.")


        if self.draw_linear_kwargs is None:
            self.add_minimizers(use_strict=use_strict_minimizers)

        self.key, key_sampler, key_i = jax.random.split(self.key, 3)

        if self.init_pos is None:
            if choose_low_kl_starting_pos:
                initial_position = get_beneficial_position(key=key_i, lh=lh, samples_to_draw=2000)
            else:
                print("\tChoosing random initial position in parameter space...")
                initial_position = jft.Vector(lh.init(key_i))
        else:
            print("\tSetting user-defined initial position in parameter space...")
            initial_position = self.init_pos

        if geoVi:
            sample_mode="nonlinear_resample"
        else:
            sample_mode="linear_resample"

        starting_time = time()

        if chi2_threshold == jnp.inf:

            if self.plotting_callback is not None:
                plotting_callback = lambda samples, vi_state: (
                    self.plotting_callback(out_name, kl_iterations, lh, samples, vi_state,
                                           ps_op=lambda p: self.amplitude_op(p) ** 2)
                )
            else:
                plotting_callback = None

            print("Running inference vanilla mode...")
            # Ignore the chi2 thresholding
            starting_time = time()

            samples, vi_info = jft.optimize_kl(
                likelihood=lh,
                position_or_samples=initial_position,
                key=key_sampler,
                n_total_iterations=kl_iterations,
                n_samples=n_samples,
                draw_linear_kwargs=self.draw_linear_kwargs,
                nonlinearly_update_kwargs=self.nonlinearly_update_kwargs,
                kl_kwargs=self.kl_kwargs,
                sample_mode=sample_mode,
                resume=resume,
                odir=out_name,
                callback=plotting_callback,
                **kwargs,
            )
        else:
            print("Running inference chi2 thresholding mode...")
            chi2 = jnp.inf
            vi_iterations = get_last_iteration(out_name) - 1  # e.g.: if last_iteration = 1, subtract 1 to get 1 inside
            while (chi2 > chi2_threshold) or (vi_iterations <= min_kl_iter) or np.isnan(chi2):
                vi_iterations += 1

                if vi_iterations > max_kl_iter:
                    print("Max kl iterations reached at chi square ", chi2)
                    break

                if self.plotting_callback is not None:
                    plotting_callback = lambda samples, vi_state: (
                        self.plotting_callback(out_name, vi_iterations, lh, samples, vi_state,
                                               ps_op=lambda p: self.amplitude_op(p) ** 2)
                    )
                else:
                    plotting_callback = None

                samples, vi_info = jft.optimize_kl(
                    likelihood=lh,
                    position_or_samples=initial_position,
                    key=key_sampler,
                    n_total_iterations=vi_iterations,
                    n_samples=n_samples,
                    draw_linear_kwargs=self.draw_linear_kwargs,
                    nonlinearly_update_kwargs=self.nonlinearly_update_kwargs,
                    kl_kwargs=self.kl_kwargs,
                    sample_mode=sample_mode,
                    resume=resume,
                    odir=out_name,
                    callback=plotting_callback,
                    **kwargs,
                )

                d_th_samples = [self.signal_response()(xi) for xi in samples]
                chi2 = mean_red_chi2(data=self.d, d_th_samples=d_th_samples, N_inv_op=self.inv_N_cov)


        ending_time = time()
        duration = ending_time - starting_time
        if duration > 60:
            duration = (np.round(duration/60,2), "minute(s)")
        else:
            duration = (duration, "seconds")

        self.posterior_xi_samples = samples
        print("\nSaved posterior latent samples as self.posterior_xi_samples. Finished execution in ", *duration)
        print("Please ensure to run get_current_key().")
        return samples, vi_info


    def get_current_key(self):
        """
        Should always be called after calling .run_inference().
        """
        return self.key


    def get_posterior_statistics(self, print_posterior_parameters=False,
                                 moment:Literal["mean", "mean and std"]="mean and std",
                                 quantity:Literal["power spectrum unique", "power spectrum full","signal",
                                 "hyperparameters", "all"]="all"):
        if self.posterior_xi_samples is None:
            raise ValueError("Call 'run_inference()' before reporting statistics.")
        if self.amplitude_op is None:
            self.amplitude_op = lambda x: np.nan
            print("No amplitude operator found, returning nan for power spectrum")

        post_xi_samples = self.posterior_xi_samples
        ps = lambda x: self.amplitude_op(x)**2
        ps_samples = [ps(xi) for xi in post_xi_samples]
        s_samples = [self.s_model(xi) for xi in post_xi_samples]

        ps_mean_std = jft.mean_and_std(ps_samples)
        signal_mean_std = jft.mean_and_std(s_samples)

        prior_distributions = self.parameter_choices
        parameter_names_1 = list(prior_distributions.keys())
        parameter_names_2 = [key for key in post_xi_samples[0]]

        parameter_names = [el for el in parameter_names_1 if el in parameter_names_2]

        # posterior_parameters = dict.fromkeys(prior_parameter_choices)
        try:
            posterior_parameters_samples = {
                key: [prior_distributions[key](xi[key]) for xi in post_xi_samples]
                for key in parameter_names
            }
        except TypeError:
            # primals are needed
            posterior_parameters_samples = {
                key: [prior_distributions[key]({key: xi[key]}) for xi in post_xi_samples]
                for key in parameter_names
            }

        posterior_parameters_mean_std = {
            k: (np.mean(v), np.std(v)) for k, v in posterior_parameters_samples.items()
        }

        if print_posterior_parameters:
            print("Posterior statistics:")
            for k, (mean, std) in posterior_parameters_mean_std.items():
                print(f"{k:20s}  mean = {mean:10.6f},  std = {std:10.6f}")

        return_list = [ps_mean_std, signal_mean_std, posterior_parameters_mean_std]
        if moment == "mean":
            return_ps, return_signal = [v[0] for v in return_list[:-1]]
            return_posterior_parameters = {k: v[0] for k, v in return_list[-1].items()}
        else:
            return_ps, return_signal, return_posterior_parameters = return_list

        # Add parameters that don't need a direct transform, e.g. \xi_s or \xi_spectrum
        posterior_latent_mean = jft.mean_and_std(post_xi_samples)[0]

        if quantity == "hyperparameters":
            all_keys = [*posterior_latent_mean]
            for k in all_keys:
                if k not in return_posterior_parameters.keys():
                    # Note: if "spectrum" in k:
                    # Just the first column is the actual IWP + WP, the second column is its derivative
                    # needed to create a Markov generative model.
                    return_posterior_parameters[k] = posterior_latent_mean[k]

        if quantity == "power spectrum unique":
            return return_ps
        elif quantity == "power spectrum full":
            return return_ps[self.s_h_dom_expander]
        elif quantity == "signal":
            return return_signal
        elif quantity == "hyperparameters":
            return return_posterior_parameters


        return return_ps, return_signal, return_posterior_parameters

    def get_prior_samples(self, mode="signal", num=500):
        """

        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'power spectrum'.
        :return:
        """
        if self.s_model is None:
            raise ValueError("Add signal model before plotting prior.")

        latent_samples = []
        for _ in range(num):
            self.key, sample_key = jax.random.split(self.key)
            xi_sl = jft.random_like(key=sample_key, primals=self.s_model.domain)

            latent_samples.append(xi_sl)

        latent_samples = np.array(latent_samples)

        if mode == "signal":
            op = self.s_model
        elif mode == "signal response":
            op = self.signal_response()
        elif mode == "power spectrum":
            op = lambda k: self.amplitude_op(k)**2
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        samples = jnp.array([op(l_sl) for l_sl in latent_samples])
        return samples

    def plot_prior(self, mode:Literal["signal", "signal response", "power spectrum"]="signal", num=500, plot=True):
        """

        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'power spectrum'.
        :return:
        """
        samples = self.get_prior_samples(mode=mode, num=num)
        mean, std = jft.mean_and_std(samples)

        if not plot:
            return mean, std

        xl = "Time"
        yl = "Strain"
        if mode == "signal":
            x = self.t_ss
        elif mode == "signal response":
            x = self.t_ds
        elif mode == "power spectrum":
            plot_welch_averaged_ps()
            x = self.k_signal
            xl = r"Unique $f$"
            yl = r"$\mathrm{Power}$"
            plt.loglog()
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        plt.plot(x, mean, color=blue, label="Mean")
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=light_blue,
            alpha=0.4,
            label=r"1$\sigma$ region",
        )

        thesis_plot(xl=xl, yl=yl, title=f"Prior: {mode}")


    def plot_prior_samples(self, mode:Literal["signal", "signal response", "power spectrum",
    "signal & power spectrum"]="signal",
                           num=5, plot=True, show=True, plot_welch_average=False, plot_data=True, rolling=False,
                           custom_ax=None, ):
        """

        :param plot_welch_average:
        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
        :param rolling: Whether to plot the samples one after the other.
        :return:
        """

        if not custom_ax:
            _ = plt.figure(figsize=(8,4))
            ax = plt.gca()
        else:
            ax = custom_ax

        if mode == "signal & power spectrum":
            print("Not plotting 0-mode for visual purposes")
            for _ in range(num):
                self._plot_power_spectrum_and_sample(plot_welch_average)
            return

        samples = self.get_prior_samples(mode=mode, num=num)

        if not plot:
            return samples[0]  # peel away outer bracket

        xl = "Time"
        yl = "Strain"
        if mode == "signal":
            x = self.t_ss

            samples_for_statistics = self.get_prior_samples(mode="signal", num=500)
            mean_cross_variance = np.mean(np.var(samples_for_statistics, axis=0))
            print("Mean std across samples:", np.sqrt(mean_cross_variance))

        elif mode == "signal response":
            x = self.t_ds
        elif mode == "power spectrum":
            x = self.k_signal
            xl = r"Unique $f$"
            yl = r"$\mathrm{Power}$"

            # remove 0-mode for plotting
            print("Not plotting 0-mode for visual purposes")
            x = x[1:]
            samples = [sl[1:] for sl in samples]
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        for idx, sl in enumerate(samples):
            lb = ""
            if idx == 0:
                lb = "Prior samples (various colors)"
            ax.plot(x, sl, label=lb, alpha=0.7)
            if rolling:
                _ = plt.figure(figsize=(8, 4))
                ax = plt.gca()
                if plot_welch_average:
                    plot_welch_averaged_ps(ax=ax)
                    ax.loglog()
                if plot_data:
                    ax.plot(self.t_ds, self.d, label="data", color="black")
                if show:
                    thesis_plot(mode="longer", xl=xl, yl=yl, title=f"Prior samples: {mode}")


        if not rolling:
            if plot_welch_average:
                plot_welch_averaged_ps(ax=ax)
            if mode == "power spectrum":
                ax.loglog()
            if plot_data:
                ax.plot(self.t_ds, self.d, label="data", color="orange")
            if show:
                thesis_plot(mode="longer", xl=xl, yl=yl, title=f"Prior samples: {mode}", show=True, close=True)


    def _plot_power_spectrum_and_sample(self, plot_welch_average):
        pow_spec_samples = self.get_prior_samples(mode="power spectrum", num=1)
        xi = np.random.standard_normal(self.n_ss)
        expander = self.s_dom_harmonic.power_distributor
        dk = self.k_signal[1] - self.k_signal[0]
        h_vol = self.n_ss * dk
        signal_space_samples = [bw_hartley(h_vol * ps[expander] * xi, norm="ortho") for ps in pow_spec_samples]

        x_ps = self.k_signal
        x_s = self.t_ss
        xl = r"Unique $f$"
        yl = r"$\mathrm{Power}$"

        # remove 0-mode for plotting
        x_ps = x_ps[1:]
        pow_spec_samples = [sl[1:] for sl in pow_spec_samples]

        fig, axs = plt.subplots(nrows=2, ncols=1)
        if plot_welch_average:
            plot_welch_averaged_ps(axs[0])
        for sl in pow_spec_samples:
            axs[0].plot(x_ps, sl)

        for sl in signal_space_samples:
            axs[1].plot(x_s, sl, label="normalization probably wrong see bw_hartley")

        axs[1].plot(self.t_ds, self.d, label="Actual data", color="orange")

        axs[0].loglog()
        axs[0].set_xlabel(xl)
        axs[0].set_ylabel(yl)
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Strain")
        axs[0].legend()
        axs[1].legend()

        plt.show()


    def plot_posterior_signal(self, print_posterior_parameters=False, over_full_signal_space=False,
                              plot_default_nrt=False, custom_ax=None, maxL_template_xy=None, whitened_data=None,
                              shade_1sigma=True,
                              shade_2sigma=False, plot_data=False, show=True, **kwargs):
        _, signal_mean_std_ss, _ = (
            self.get_posterior_statistics(print_posterior_parameters))

        if custom_ax is None:
            _ = plt.figure(figsize=(8,4))
            ax = plt.gca()
        else:
            ax = custom_ax

        signal_mean = signal_mean_std_ss[0]
        signal_std = signal_mean_std_ss[1]
        if over_full_signal_space:
            time = self.t_ss
        else:
            time = self.t_ds

            xi_samples = self.posterior_xi_samples
            data_model = self.signal_response()
            data_model_samples = [data_model(xi) for xi in xi_samples]
            data_model_mean = np.mean(np.array(data_model_samples), axis=0)
            data_model_std = np.std(np.array(data_model_samples), axis=0)

            # signal_mean = self.adjoint_zp(signal_mean, ext_fact=self.e_fac, res_fact=self.r_fac)
            signal_mean = data_model_mean
            # signal_std = self.adjoint_zp(signal_std, ext_fact=self.e_fac, res_fact=self.r_fac)
            signal_std = data_model_std

        # if not over_full_signal_space:
        #     N = len(self.t_ds)
        #     tmp1 = signal_mean_std_ss[0]
        #     tmp2 = signal_mean_std_ss[1]
        #
        #     signal_mean = tmp1[:N]
        #     signal_std = tmp2[:N]
        #     time = self.t_ds
        #
        #     res = np.mean(np.abs(signal_mean - self.d))
        #     print("<s_mean - d>=", res)
        #
        # else:
        #     signal_mean = signal_mean_std_ss[0]
        #     signal_std = signal_mean_std_ss[1]
        #     time = self.t_ss

        if plot_data:
            ax.plot(self.t_ds, self.d, label="Data", color="orange")

        if shade_1sigma:
            # shaded 1-sigma region
            ax.fill_between(time,
                             signal_mean - signal_std,
                             signal_mean + signal_std,
                             color=light_blue,
                             alpha=0.7)  # transparency

        if shade_2sigma:
            # shaded 2-sigma region
            ax.fill_between(time,
                             signal_mean - 2*signal_std,
                             signal_mean + 2*signal_std,
                             color=light_blue,
                             alpha=0.4)  # transparency

        if whitened_data is not None:
            wh, lb_wh = whitened_data
            ax.plot(time, wh, label=lb_wh, color="black", lw=1)

        # plot the mean line on top
        ax.plot(time, signal_mean, color=blue, label=r"Reconstructed signal", lw=2)

        if plot_default_nrt:
            nrt_strain_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_strain_values.txt") * 1e19
            nrt_time_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_time_values.txt")
            nrt_time_values = nrt_time_values - nrt_time_values[0] + 15

            go_until = np.max(np.where(nrt_time_values<max(self.t_ds)))

            ax.plot(nrt_time_values[:go_until], nrt_strain_values[:go_until], label="LIGO Template",
                     color=red)

        if maxL_template_xy is not None:
            maxL_template_x, maxL_template_y = maxL_template_xy
            ax.plot(maxL_template_x, maxL_template_y, "-", label="Maximum likelihood template", color=red)

        if show:
            thesis_plot(**kwargs, mode="longer")



    def plot_posterior_power_spectrum(self, mode:Literal["median", "mean"], plot_welch_average=True,
                                      custom_ax=None, **kwargs):
        """

        Note: If we plot only np.std(power spectrum samples) this may be larger than the mean and thus the errorbars
        are negative and don't look good on the log-log plot.

        For a Gaussian distribution, ±1σ containts ~68% of the probability mass. So 16% are lower than -1σ and higher
        than +1σ. So the first confidence interval starts at the 16th percentile and goes up to 16+68 = 84th percentile.

        So instead of Gaussian errorbars we can plot these percentiles.

        :param mode:                            If "median", plots the mentioned percentiles around the median.
                                                If "mean", plots just the mean.
        :param print_posterior_parameters
        :param plot_welch_average
        :return:
        """
        post_xi_samples = self.posterior_xi_samples
        ps = lambda x: self.amplitude_op(x) ** 2
        ps_samples = jnp.array([ps(xi) for xi in post_xi_samples])

        ps_mean = jnp.mean(ps_samples, axis=0)

        percentile_16 = jnp.percentile(ps_samples, 16, axis=0)
        percentile_84 = jnp.percentile(ps_samples, 84, axis=0)
        ps_median = jnp.percentile(ps_samples, 50, axis=0)  # percentile_50

        if not custom_ax:
            _ = plt.figure(figsize=(8,4))
            ax = plt.gca()
        else:
            ax = custom_ax

        if mode == "mean":
            ax.plot(self.k_signal[1:], ps_mean[1:], label=r"Posterior mean of power spectrum", color=blue, lw=2)
        elif mode == "median":
            ax.plot(self.k_signal[1:], ps_mean[1:], label=r"Posterior mean of power spectrum", color=blue, ls="--")
            ax.plot(self.k_signal[1:], ps_median[1:],
                     label=r"Reconstructed median power spectrum (with $1\sigma$ contour)", color=blue)
            ax.fill_between(self.k_signal[1:], percentile_16[1:], percentile_84[1:], color=light_blue)
        else:
            raise ValueError("Mode must be either 'mean' or 'median'.")

        if plot_welch_average:
            plot_welch_averaged_ps(ax=ax)

        print("Zeromode P_s(k=0) excluded in plot")  # because ~0 and then changes the y limits such that the majority
        # of the power spectrum lies in just the upper half of the coordinate system


        # plt.ylim(-3e-9, 1.4e-7)
        ax.loglog()
        if custom_ax is None:
            thesis_plot(xl="Frequency $f$", yl="Power", mode="longer", custom_ax=ax, **kwargs)


    def plot_posterior_harmonic_xi_s(self, multiply_with_posterior_amp_spec=False,
                                     multiply_with_posterior_amp_spec_v2=False,
                                     only_return=False, custom_ax=None, custom_xi=None,
                                     **kwargs):
        """
        multiply_with_posterior_amp_spec_v2:

            Let the prior covariance S be <s s^dag> ~ p_s_prior, I believe that the diagonal of the posterior covariance is
            D = (F^-1) <|xi_s|^2> p_s_posterior (F)

            Allowing to define an updated posterior power spectrum as p_D = <|xi_s|^2> p_s_posterior.

        :param multiply_with_posterior_amp_spec:
        :param multiply_with_posterior_amp_spec_v2:
        :return:
        """
        if custom_xi is None:
            posterior_latent_sl = self.posterior_xi_samples
            posterior_latent_mean_std = jft.mean_and_std(posterior_latent_sl)
            posterior_latent_mean, _ = posterior_latent_mean_std

            posterior_xi_s_mean = posterior_latent_mean[f"s_xi"]
        else:
            posterior_xi_s_mean = custom_xi

        if only_return:
            return posterior_xi_s_mean

        expander = self.s_dom_harmonic.power_distributor

        if multiply_with_posterior_amp_spec:

            ps_mean_std, _, _ =  self.get_posterior_statistics()
            amp = np.sqrt(ps_mean_std[0])
            amp_exp = amp[expander]

            _, k_lengths, power_spectrum = unpickle_me_this(
                "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                absolute_path=True)
            k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
            spectrum_welch = power_spectrum.val[1:]

            plt.plot(k_lengths, np.sqrt(spectrum_welch), label=r"Empirical estimate of $\sqrt{p(k)}$", color="orange")
            plt.plot(self.k_signal_full, np.abs(posterior_xi_s_mean)*amp_exp, "r.", label=r"$\vert\xi_s\vert\cdot \sqrt{p_s}$", markersize=4)
            plt.plot(self.k_signal_full, amp_exp, ".",label=r"Posterior $\sqrt{p(k)}$", markersize=4)
            plt.loglog()

            thesis_plot(xl="Frequency $f$", yl=r"$\sqrt{\mathrm{power}}$")

        elif multiply_with_posterior_amp_spec_v2:

            xi_s_samples_squared = jnp.array([jnp.abs(xl["s_xi"])**2 for xl in self.posterior_xi_samples])
            mean_squared_xi_s = jnp.mean(xi_s_samples_squared, axis=0)

            ps_mean_std, _, _ = self.get_posterior_statistics()
            ps_mean = ps_mean_std[0][expander]


            p_D = ps_mean * mean_squared_xi_s

            _, k_lengths, power_spectrum = unpickle_me_this(
                "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                absolute_path=True)
            k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
            spectrum_welch = power_spectrum.val[1:]

            plt.plot(k_lengths, spectrum_welch, label=r"Welch average", color="orange")
            plt.plot(self.k_signal_full, p_D, "r.",
                     label=r"$p_D$", markersize=4)
            plt.plot(self.k_signal_full, ps_mean, ".", label=r"Posterior $p(k)$", markersize=4)
            plt.loglog()

            thesis_plot(xl="Frequency $f$", yl=r"$\mathrm{power}$")

        else:
            if custom_ax is None:
                _ = plt.figure(figsize=(8, 4))
                ax = plt.gca()
            else:
                ax = custom_ax
            ax.plot(self.k_signal_full, posterior_xi_s_mean, color=blue, **kwargs)
            if not custom_ax:
                thesis_plot(xl="Frequency $f$", yl=r"$\xi_s$", mode="longer")


    def plot_noise_sample_with_data(self, num, rolling=False, show=True):
        N_sqrt = self.sqrt_noise_op
        if N_sqrt is None:
            raise ValueError("self.sqrt_noise_op must not be None to draw samples from N.")

        _ = plt.figure(figsize=(8, 2))

        noise_samples = []
        for _ in range(1000):
            xi = np.random.standard_normal(self.n_ds)
            sl = N_sqrt(xi)
            noise_samples.append(sl)

        print("\nMean cross variance of noise samples: Var(sl) = ", np.mean(np.var(noise_samples, axis=0)), " (over 1000 samples)")
        if show:
            for sl in noise_samples[:num]:
                plt.plot(self.t_ds, sl)
                if rolling:
                    plt.plot(self.t_ds, self.d, label="Actual data")
                    thesis_plot(close=True, title="Noise samples from covariance operator")

        if not rolling and show:
            plt.plot(self.t_ds, self.d, label="Actual data", color="black")
            thesis_plot("basic")


    def calculate_and_plot_penrose_xi(self, itr=10_000, plot=True, reload_from_cache=False, fn="my_penrose_xi.txt"):
        # Only use reload_from_cache if you know what you are doing!!!
        penrose_xi = find_penrose_moore_solution(itr, pipe=self, reload_from_cache=reload_from_cache, filename=fn)
        if plot:

            mean_ps = self.get_posterior_statistics(moment="mean", quantity="power spectrum full")
            iFFT = lambda p: jnp.fft.ifft(p, len(penrose_xi), norm="ortho")
            data_from_penrose_xi = sample_from_ps(xi=penrose_xi, N=len(self.d), ps=mean_ps, inverse_h_trafo=iFFT)

            fig, axs = plt.subplots(2,1)

            axs[0].plot(self.k_signal_full, penrose_xi.real, label="Harmonic penrose xi")

            axs[1].plot(self.t_ds, data_from_penrose_xi, label="data from penrose xi")
            axs[1].plot(self.t_ds, self.d, label="actual data")

            axs[0].set_xlabel("Frequencies")
            axs[1].set_xlabel("Time")
            axs[1].set_ylabel("Strain")
            axs[0].legend()
            axs[1].legend()

            plt.show()

        return penrose_xi


def analyze_kl_callback(out_name, max_kl_iterations, lh, samples, vi_state, ps_op=None):

    ### KL ENERGY CALCULATION

    p = out_name+"/custom_callback/"  # prefix
    os.makedirs(p, exist_ok=True)  # create folder if it doesn't exist
    kl_energy_file = p + "kl_energies.txt"

    kl_energy = np.float64(vi_state.minimization_state.fun)
    kl_iteration_number = vi_state.nit

    with open(kl_energy_file, "a") as f:
        f.write(str(kl_energy))
        f.write("\n")
        f.close()

    if kl_iteration_number == max_kl_iterations:
        # load kl_energy_file data and plot and save in low quality in folder
        kl_energies = np.loadtxt(kl_energy_file)
        plt.plot(kl_energies)
        usual_plot(xl="KL Iteration", yl="KL Energy", title="Evolution of KL Energy", show=False, close=False)
        plt.savefig(p+"kl_energies.png", dpi=100)
        plt.close()

    # for sl in samples:
    #     kl_val = calculate_kl_val_and_grad(likelihood=lh, primals=sl, full_output=False)
    #     print("kl_val:", kl_val)

    ### REDUCED CHI2 CALCULATION

    red_chi2_file = p + "red_chi2.txt"

    gs = lh.likelihood
    d = gs.data
    fw_model = lh.forward
    try:
        # Gaussian energy
        N_inv = gs.noise_cov_inv
        d_th_samples = jnp.array([fw_model(xi) for xi in samples])
    except AttributeError:
        # Variable Gaussian covariance energy => I attach the noise_cov_inv myself
        N_inv = lh.noise_cov_inv
        mean_inv_noise_level = jft.mean([N_inv(xi) for xi in samples])
        N_inv = lambda x: x*mean_inv_noise_level
        d_th_samples = jnp.array([fw_model(xi)[0] for xi in samples])  # fw_model(xi) is a tuple with [0] being s_prime(x) and
        # the second entry I think an array consisting of sqrt_inv_noise_cov

    red_chi2 = mean_red_chi2(data=d, d_th_samples=d_th_samples, N_inv_op=N_inv)
    d_th_mean = jnp.mean(d_th_samples, axis=0)

    with open(red_chi2_file, "a") as f:
        f.write(str(red_chi2))
        f.write("\n")
        f.close()

    if kl_iteration_number == max_kl_iterations:
        # load kl_energy_file data and plot and save in low quality in folder
        red_chi_squared = np.loadtxt(red_chi2_file)
        plt.plot(red_chi_squared)
        usual_plot(xl="KL Iteration", yl=r"$\chi^2$", title=r"Evolution of reduced $\chi^2$", show=False, close=False)
        plt.savefig(p+"red_chi2.png", dpi=100)
        plt.close()


    ### FW MODEL CALCULATION AND POWER SPECTRUM CALCULATION

    fw_model_folder = p + "/fw_model/"
    os.makedirs(fw_model_folder, exist_ok=True)

    invalid_ps = False

    if ps_op is None:
        invalid_ps = True
    else:
        try:
            test_eval = ps_op(samples[0])  # use valid input
            invalid_ps = jnp.any(jnp.isnan(test_eval))
        except Exception:
            invalid_ps = True

    if invalid_ps:
        plt.plot(d_th_mean)
        usual_plot(xl="Index", yl="Value", title="Mean forward model evaluation", show=False, close=False)
        plt.savefig(fw_model_folder+f"iter_{kl_iteration_number}.png", dpi=100)
        plt.close()
    else:
        ps_mean, ps_std = jft.mean_and_std(
            tuple(ps_op(xi) for xi in samples)
        )

        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.suptitle("Mean forward model evaluation")

        axs[0].plot(d_th_mean, label="Mean data")
        axs[0].set_xlabel("Index")
        axs[0].set_ylabel("Data value")

        axs[1].plot(
            ps_mean,
            label="Mean power spectrum",
        )
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Power")
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")

        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(fw_model_folder+f"iter_{kl_iteration_number}.png", dpi=100)
        plt.close()


    ### MEAN LATENT VARIABLE CALCULATION

    posterior_mean = jft.mean(samples)
    latent_posterior_mean_folder = p+"latent_posterior_means/"
    os.makedirs(latent_posterior_mean_folder, exist_ok=True)
    pickle_me_this(f"{latent_posterior_mean_folder}{kl_iteration_number}", posterior_mean)