from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from data.style_components.matplotlib_style import *
import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import nifty.nifty.re as jft
import jax
import warnings
from typing import Literal, Optional, Callable
from time import time
from jax import vmap
from scipy.ndimage import gaussian_filter
import os

from .calculate_kl import calculate_kl_val_and_grad, get_beneficial_position
from .calculate_pseudoinverse import find_penrose_moore_solution, sample_from_ps
import datetime
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.restoration import denoise_tv_chambolle
import cv2


def raise_warning(msg):
    print("\n")
    warnings.warn(msg, category=UserWarning, stacklevel=2)
    print("\n")

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


def usual_plot(xl=r"Time $t$ $\mathrm{[sec]}$", yl=r"Strain $h$ $\mathrm{[10^{-19}]}$", title=None, xlim=None, ylim=None,
               show=True, close=False, save_fig=False):
    plt.xlabel(xl, fontsize=20)
    plt.ylabel(yl, fontsize=20)
    plt.title(title, fontsize=25)
    ax = plt.gca()
    labels = ax.get_legend_handles_labels()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if labels != ([], []):
        plt.legend()
    if save_fig:
        plt.tight_layout()
        current_date = datetime.datetime.now()
        plt.savefig(f"{current_date}.png")
    if show:
        plt.show()
    if close:
        plt.close()


def get_sample_data(norm=1e19, time_window=(15,17), end_points_small=False, taper=False):
    """
    Gets some exemplary data from first detected GravWave event.
    :param norm:                    Scaling the data for visual purposes.
    :param time_window:             Which data to pick out.
    :param end_points_small:        Whether the left and right bound should be chosen such that the first data point should
                                    be approximately 0.
    :param taper:                   Whether to apply a Tukey window.
    :return:
    """
    strain = unpickle_me_this(
        "/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle",
        absolute_path=True)

    zero_time = 1126259446  # I got this zero time by looking at the caption of the figure produced by strain.plot().
    time = np.array(strain.times) - zero_time  # in seconds

    full_data = norm * strain.value
    full_time = time.copy()

    t_min, t_max = time_window

    if end_points_small:
        eps1 = 0.1  # how small should the starting point be
        eps2 = 1e-6  # how near the endpoint be to the start point

        close_to_zero_idcs = np.where(np.abs(full_data) < eps1)
        close_to_zero_times = time[close_to_zero_idcs]
        t_min = np.max(close_to_zero_times[close_to_zero_times<t_min])

        d0 = full_data[np.where(time == t_min)]

        where_similar_to_d0 = np.where(np.isclose(full_data, d0, rtol=0, atol=0.001))

        time_similar_to_d0 = time[where_similar_to_d0]
        res_list = time_similar_to_d0 - t_max
        most_similar_to_t_max = np.min(np.abs(time_similar_to_d0 - t_max))

        t_max = most_similar_to_t_max + t_max

    indcs = np.where((t_min <= time) & (time <= t_max))
    data = full_data[indcs]
    time = time[indcs]

    if taper:
        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)
        data = tapering_function(data)*data

    return jnp.array(time), jnp.array(data)


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


class InferenceSchemeRe():
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

        # abbreviations: ds := 'data_space' ; ss := 'signal_space'
        self.d = d
        self.e_fac = e_fac
        self.r_fac = r_fac
        self.key = key

        self.L_ds = jnp.max(t)-jnp.min(t)
        self.L_ss = self.L_ds * self.e_fac

        self.dist_ds = t[1] - t[0]
        self.dist_ss = self.dist_ds / r_fac

        self.n_ds = len(d)
        self.n_ss = int(self.L_ss / self.dist_ss) + 1  # e.g.: 3 pixels corresponds to 3+1 edges. This is the edges.

        self.t_ds = t
        self.t_ss = jnp.arange(self.n_ss)*self.dist_ss + t[0]

        if self.e_fac != 1:
            # self.adjoint_zp = lambda arr, ext_fact: arr[:int(len(arr)/ext_fact+1)]  # TODO: correct cutting?
            self.adjoint_zp = lambda arr, ext_fact: arr[:self.n_ds]
        else:
            self.adjoint_zp = lambda arr, ext_fact: arr  # unit, no extension

        assert self.t_ds[0] == self.t_ss[0]  # same beginning?
        assert jnp.all(self.t_ds == self.adjoint_zp(self.t_ss, e_fac)[::r_fac])  # if you cut the extended
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
        self.amplitude_op = None
        self.s_model = None

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


    def add_custom_signal_model(self, custom_signal_model: jft.Model):
        """

        custom_signal_model: jft.Model      Needs to have implemented __init__, __call__ and get_model_components

        model_prefix needs to include an underscore as suffix.
        parameter_choices needs to look like
            parameter_choices = {
            f"{model_prefix}fluctuations": lambda xi: np.exp(fluct[0] + xi*fluct[1]),
            f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
            f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
            f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
        }

        Be sure to implement an .init method/property to be fed into jft.Model __init__,
        like init=cfm_maker.finalize().init

        Models should be defined over the extended domain.

        :param custom_signal_model:  A tuple of s_model, amplitude_op, parameter_choices, model_prefix.
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


    def add_cfm_signal_model(self, fluct:tuple, llslope:tuple, flex:tuple | None = None, asper:tuple | None=None,
                             offset_mean:float = 0, offset_std:tuple = (1e-16, 1e-16), model_prefix="s_",
                             add_power_spectrum_template=None, add_custom_power_op=(None,), square_iwp=False,):
        """

        :param fluct:
        :param llslope:
        :param flex:
        :param asper:
        :param offset_mean:
        :param offset_std:
        :param model_prefix:
        :param add_power_spectrum_template:         An array over unique frequency values to add as a baseline to the
                                                    power spectrum of the correlated field.
        :param add_custom_power_op:                 A list of custom operators in power space to be added to the model,
                                                    e.g. a line model for spectral lines in the power spectrum.

        :return:
        """

        cfm_maker = jft.CorrelatedFieldMaker(prefix=model_prefix)
        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=fluct,
                                   loglogavgslope=llslope, flexibility=flex, asperity=asper, harmonic_type="fourier",
                                   non_parametric_kind="power", hack_add_power_spectrum_template=add_power_spectrum_template,
                                   hack_custom_amplitude_operators=add_custom_power_op,
                                   hack_make_iwp_pos_definite=square_iwp)

        parameter_choices = {
            f"{model_prefix}fluctuations": lambda xi: np.exp(fluct[0] + xi*fluct[1]),
            f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
            f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
            f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
            #"offset_mean": (offset_mean, "fix"),
            #"offset_std": (offset_std, "lognormal"),
        }


        gaussian_win = False
        if gaussian_win:
            s_model_cfm = cfm_maker.finalize()
            dom = s_model_cfm.domain

            x = self.t_ss
            std_dev = .1
            mean = 16.4
            gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

            s_model = lambda p: s_model_cfm(p)*gaussian
            s_model.domain = dom

            raise_warning("Using fix gaussian window signal model!! Add instead a generative gaussian envelope or similar")
        else:
            s_model = cfm_maker.finalize()


        self.s_model = s_model
        self.amplitude_op = cfm_maker.amplitude
        self.parameter_choices = parameter_choices
        self.model_prefix = model_prefix


    def signal_response(self):
        if self.r_fac != 1:
            raise ValueError("Non-Unit responses (masks) not implemented yet.")
        if self.s_model is None:
            raise ValueError("No signal model implemented yet, call 'add_cfm_signal_model' or add_custom_signal_model.")

        class Response(jft.Model):
            def __init__(self, signal_model, zp_adj, ext_fac):
                self.sm = signal_model
                self.zp_adj = zp_adj
                self.ext_fac = ext_fac
                super().__init__(domain=signal_model.domain)

            def __call__(self, xi):
                model_values_on_long_domain = self.sm(xi)
                return self.zp_adj(model_values_on_long_domain, self.ext_fac)

        s_prime = Response(signal_model=self.s_model, zp_adj=self.adjoint_zp, ext_fac=self.e_fac)

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
            print("Using DIAGONAL noise operators.")
            # Both operators were not provided => Diagonal noise covariance
            self.inv_N_cov = lambda x: x/noise_var_level
            self.sqrt_inv_N_cov = lambda x: x/jnp.sqrt(noise_var_level)
        else:
            print("Using provided inv_N_cov and sqrt_inv_N_cov.")
            # Both operators were correctly provided.
            self.inv_N_cov = inverse_noise_op
            self.sqrt_inv_N_cov = sqrt_inverse_noise_op


    def set_init_pos(self, init_pos:jft.Vector | dict, plot=False, plot_welch_average=True):
        """

        :param init_pos:    A jft vector or a dictionary containing latent space values to use as initial positions
                            in the inference. The dictionary does not have to contain all the keys of the likelihood
                            domain. If a key is missing, a random value is drawn.
        :param plot:        Whether to plot the resulting initial postion.
        :param plot_welch_average:  If plot and plot_welch_average, plots welch average. Useful for noise comparison.
        :return:
        """
        if type(init_pos) is jft.Vector:
            init_pos = init_pos._tree

        lh = self.build_lh(supress_print=True)

        print("You are trying to set an initial positions. Please note that the likelihood domain keys are: \n\t",
              "<", *lh.domain.keys(),">")


        self.key, key_i = jax.random.split(self.key)
        base_initial_position = lh.init(key_i)

        for key in init_pos.keys():
            if key in base_initial_position.keys():
                base_initial_position[key] = init_pos[key]

        self.init_pos = jft.Vector(base_initial_position)

        if plot:
            ps = lambda xi: self.amplitude_op(xi)**2
            s_prime = self.signal_response()

            init_ps = ps(self.init_pos)
            init_s_prime = s_prime(self.init_pos)

            fig, axs = plt.subplots(nrows=2, ncols=1)

            if plot_welch_average:
                plot_welch_averaged_ps(axs[0])
            axs[0].loglog(self.k_signal, init_ps)
            axs[0].set_xlabel("Frequency $f$")
            axs[0].set_ylabel("Power")

            axs[1].plot(self.t_ds, init_s_prime)
            axs[1].set_xlabel("Time $t$")
            axs[1].set_ylabel("Strain")

            plt.tight_layout()
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
            cg_name="CG: linear sampling.",
            cg_kwargs=dict(absdelta=linear_energy, maxiter=linear_iter),
        )

        # Arguments for the minimizer in the nonlinear updating of the samples
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="Nonlinear sampling NCG",
                absdelta=nonlinear_energy,
                cg_kwargs=dict(name="\tCG: nonlinear sampling.",),
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

        lh = jft.Gaussian(data=self.d, noise_cov_inv=self.inv_N_cov, noise_std_inv=self.sqrt_inv_N_cov).amend(s_prime)
        return lh

    def run_inference(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False, out_name="out",
                     resume=True, choose_low_kl_starting_pos=False, geoVi=True):
        """

        :param kl_iterations:
        :param n_samples:
        :param use_strict_minimizers:
        :param out_name:
        :param resume:
        :param choose_low_kl_starting_pos:  If self.init_pos was not set explicitly, tries to find a minimum kl starting
                                            position.
        :param geoVi:
        :return:
        """
        lh = self.build_lh()

        if self.draw_linear_kwargs is None:
            self.add_minimizers(use_strict=use_strict_minimizers)

        if self.plotting_callback is not None:
            plotting_callback = lambda samples, vi_state: (
                self.plotting_callback(out_name, kl_iterations, lh, samples, vi_state))

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

        samples, _ = jft.optimize_kl(
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
            callback=plotting_callback

        )

        ending_time = time()


        duration = ending_time - starting_time
        if duration > 60:
            duration = (np.round(duration/60,2), "minute(s)")
        else:
            duration = (duration, "seconds")

        self.posterior_xi_samples = samples
        print("\nSaved posterior latent samples as self.posterior_xi_samples. Finished execution in ", *duration)
        return samples


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
            raise ValueError("amplitude operator must not be None")

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

        usual_plot(xl=xl, yl=yl, title=f"Prior: {mode}")


    def plot_prior_samples(self, mode:Literal["signal", "signal response", "power spectrum",
    "signal & power spectrum"]="signal",
                           num=5, plot=True, plot_welch_average=False, plot_data=True, rolling=False):
        """

        :param plot_welch_average:
        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
        :param rolling: Whether to plot the samples one after the other.
        :return:
        """

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

        for sl in samples:
            np.savetxt("x_131215.txt", x)
            np.savetxt("y_131215.txt", sl)
            plt.plot(x, sl)
            if rolling:
                if plot_welch_average:
                    plot_welch_averaged_ps()
                    plt.loglog()
                if plot_data:
                    plt.plot(self.t_ds, self.d, label="data", color="orange")
                usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}")


        if not rolling:
            if plot_welch_average:
                plot_welch_averaged_ps()
            if mode == "power spectrum":
                plt.loglog()
            if plot_data:
                plt.plot(self.t_ds, self.d, label="data", color="orange")
            usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}", show=True, close=True)


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
            axs[1].plot(x_s, sl)

        axs[1].plot(self.t_ds, self.d, label="Actual data", color="orange")

        axs[0].loglog()
        axs[0].set_xlabel(xl)
        axs[0].set_ylabel(yl)
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Strain")
        axs[0].legend()
        axs[1].legend()

        plt.show()


    def plot_posterior_signal(self, print_posterior_parameters=False, over_full_signal_space=False, plot_nrt=False, **kwargs):
        print(": print_posterior_parameters , ", print_posterior_parameters)
        _, signal_mean_std_ss, _ = (
            self.get_posterior_statistics(print_posterior_parameters))

        if not over_full_signal_space:
            N = len(self.t_ds)
            tmp1 = signal_mean_std_ss[0]
            tmp2 = signal_mean_std_ss[1]

            signal_mean = tmp1[:N]
            signal_std = tmp2[:N]
            time = self.t_ds

            res = np.mean(np.abs(signal_mean - self.d))
            print("<s_mean - d>=", res)

        else:
            signal_mean = signal_mean_std_ss[0]
            signal_std = signal_mean_std_ss[1]
            time = self.t_ss


        plt.errorbar(time, signal_mean, yerr=signal_std, label=r"Reconstructed signal (with $1\sigma$ contour)",
                     ecolor=light_blue, color=blue)
        plt.plot(self.t_ds, self.d, label="Data", color="orange")

        if plot_nrt:
            nrt_strain_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_strain_values.txt") * 1e19
            nrt_time_values = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/num_rel_template_time_values.txt")
            nrt_time_values = nrt_time_values - nrt_time_values[0] + 15

            go_until = np.max(np.where(nrt_time_values<max(self.t_ds)))

            plt.plot(nrt_time_values[:go_until], nrt_strain_values[:go_until], label="Numerical relativity template (matched filter)",
                     color=red)

        usual_plot(**kwargs)



    def plot_posterior_power_spectrum(self, mode:Literal["median", "mean"], plot_welch_average=True, **kwargs):
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


        if plot_welch_average:
            plot_welch_averaged_ps()

        if mode == "mean":
            plt.plot(self.k_signal[1:], ps_mean[1:], label=r"Posterior mean of power spectrum", color=blue)
        elif mode == "median":
            plt.plot(self.k_signal[1:], ps_mean[1:], label=r"Posterior mean of power spectrum", color=blue, ls="--")
            plt.plot(self.k_signal[1:], ps_median[1:],
                     label=r"Reconstructed median power spectrum (with $1\sigma$ contour)", color=blue)
            plt.fill_between(self.k_signal[1:], percentile_16[1:], percentile_84[1:], color=light_blue)
        else:
            raise ValueError("Mode must be either 'mean' or 'median'.")

        print("Zeromode P_s(k=0) excluded in plot")  # because ~0 and then changes the y limits such that the majority
        # of the power spectrum lies in just the upper half of the coordinate system


        # plt.ylim(-3e-9, 1.4e-7)
        plt.loglog()
        usual_plot(xl="Frequency $f$", yl="Power", **kwargs)


    def plot_posterior_harmonic_xi_s(self, multiply_with_posterior_amp_spec=False, show=True):
        posterior_latent_sl = self.posterior_xi_samples
        posterior_latent_mean_std = jft.mean_and_std(posterior_latent_sl)
        posterior_latent_mean, _ = posterior_latent_mean_std

        posterior_xi_s_mean = posterior_latent_mean[f"s_xi"]
        if not show:
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

            usual_plot(xl="Frequency $f$", yl=r"$\sqrt{\mathrm{power}}$")

        else:
            plt.plot(self.k_signal_full, posterior_xi_s_mean, "r-", label=r"Posterior mean")
            usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")


    def plot_noise_sample_with_data(self, num, rolling=False, show=True):
        N_sqrt = self.sqrt_noise_op
        if N_sqrt is None:
            raise ValueError("self.sqrt_noise_op must not be None to draw samples from N.")

        noise_samples = []
        for _ in range(500):
            xi = np.random.standard_normal(self.n_ds)
            sl = N_sqrt(xi)
            noise_samples.append(sl)

        print("Mean cross variance of samples: Var(sl) = ", np.mean(np.var(noise_samples, axis=0)), " (over 500 samples)")
        if show:
            for sl in noise_samples[:num]:
                plt.plot(self.t_ds, sl)
                if rolling:
                    plt.plot(self.t_ds, self.d, label="Actual data")
                    usual_plot(close=True, title="Noise samples from covariance operator")

        if not rolling and show:
            plt.plot(self.t_ds, self.d, label="Actual data", color="black")
            usual_plot()


    def calculate_and_plot_penrose_xi(self, itr=10_000, plot=True):
        penrose_xi = find_penrose_moore_solution(itr, pipe=self, reload_from_cache=True, filename="my_penrose_xi.txt")
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
        Hx = Hx / jnp.sqrt(N)  #  scales with 1/√N
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
        x = x / jnp.sqrt(N) # scales with 1/√N
    return x  # total scale: 1 if ortho AND input is from ortho fw_hartley.
    # if instead input is not from non-ortho fw_hartley: ~ √N I think.


#
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

# stop

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


def interpolate_waveform_from_inverted_wigner(new_times):
    waveform = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/waveform_from_inverted_wigner.txt")
    waveform_times = np.loadtxt("/Users/iason/PycharmProjects/STRAIN/data/data_txt/times_from_inverted_wigner.txt")

    time_shift = waveform_times[0] - new_times[0]

    new_grid = np.linspace(waveform_times.min(), waveform_times.max(), len(new_times))
    interpolator = interp1d(x=waveform_times, y=waveform, kind="linear", fill_value="extrapolate")
    new_values = interpolator(new_grid)

    dt = new_times[1] - new_times[0]
    shift = int((0.1+0.136) /dt)  # please don't use this roll
    new_values = np.roll(new_values, -shift)

    # plt.plot(waveform_times, waveform, label="old")
    # plt.plot(new_times, new_values, label="new")
    # usual_plot()

    return new_values


def plot_welch_averaged_ps(ax=None):
    k_lengths, power_spectrum = get_welch_averaged_ps()
    k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
    spectrum_welch = power_spectrum[1:]
    if ax is None:
        plt.plot(k_lengths, spectrum_welch, label="Empirical estimate", color="orange")
    else:
        ax.plot(k_lengths, spectrum_welch, label="Empirical estimate", color="orange")


class SignalModelCfmAsPowerSpectrum(jft.Model):
    def __init__(self, N_ss, dist_ss, scale:tuple, llslope:tuple, flex: tuple | None = None,
                 asper:tuple | None=None, offset_mean:float = 0,
                 offset_std:tuple = (1e-16, 1e-16), model_prefix="s_hyper_"):

        """

        :param N_ss:
        :param dist_ss:
        :param scale:           Not called fluctuations, because the transformations I do here do not exactly
                                reproduce the correlated field; compare power spectra of use_fix_ps_for_debug = True
                                and the standard CFM.
        :param llslope:
        :param flex:
        :param asper:
        :param offset_mean:
        :param offset_std:
        :param model_prefix:
        """

        signal_grid = jft.correlated_field.make_grid(shape=(N_ss,), distances=dist_ss, harmonic_type="fourier")
        harmonic_grid = signal_grid.harmonic_grid
        fourier_modes = harmonic_grid.mode_lengths
        k_dist = fourier_modes[1]-fourier_modes[0]

        self.k = fourier_modes
        self.k_ext_factor = 2
        self.M = len(fourier_modes)
        self.N_ss = N_ss
        self.dist_ss = dist_ss
        self.s_grid = signal_grid

        cfm_maker = jft.CorrelatedFieldMaker(prefix=model_prefix)
        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations(shape=((self.M-1)*self.k_ext_factor,), distances=k_dist, fluctuations=(0.9, 0.1),
                                   loglogavgslope=llslope, flexibility=flex, asperity=asper, harmonic_type="fourier",
                                   non_parametric_kind="power")  # scale not used here for fluctuations, since fluctuations impact
        # will get integrated out later anyway; new scale parameter needs to be defined

        parameter_choices = {
            f"{model_prefix}fluctuations": lambda xi: np.exp(scale[0] + xi * scale[1]),
            f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
            f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
            f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
            #"offset_mean": (offset_mean, "fix"),
            #"offset_std": (offset_std, "lognormal"),
        }

        dl_mean_hyper_prior_power_spectrum = tmp2(self.k, [-10, 1e3], self.N_ss)
        dl_s0_hyper = np.sqrt(dl_mean_hyper_prior_power_spectrum)

        self.parameter_choices = parameter_choices
        self.cfm = cfm_maker.finalize()
        self.power_spectrum = lambda xi: jnp.exp(self.cfm(xi)[:self.M])*1e-7
        # self.power_spectrum = lambda xi: ((jnp.exp(tmp1(xi=xi, amp=dl_s0_hyper, custom_norm_for_your_convenience=1))[:self.M])*1e-4)  # delete later and uncomment previous!

        use_fix_ps_for_debug = False
        if use_fix_ps_for_debug:
            self.power_spectrum_2 = lambda xi: jnp.exp(self.cfm(xi)[:self.M])  # cut out zero-padded region
            test_function = lambda k: (k+1e-10)**(-10)
            test_values = test_function(self.k)
            self.power_spectrum = lambda xi: self.power_spectrum_2(xi)*1e-300 + test_values

        self.power_spectrum.domain = self.cfm.domain

        self.zm = 1e-32  # fluctuations of real space field
        self.zm_mask = jnp.ones(shape=(self.M,))
        self.zm_mask = self.zm_mask.at[0].set(self.zm)
        self.scale = jft.LogNormalPrior(scale[0], scale[1], name=f"{model_prefix}scale")


        def mock_cfm():
            # to replace the actual cfm call and be able to use log(k) for increased flexibility on log-log scale
            pass

        def make_norm_amp(xi):
            power_spectrum_realization = self.power_spectrum(xi)
            power_spectrum_realization = self.zm_mask * power_spectrum_realization

            # U_sqrt = jnp.sqrt(jnp.trapezoid(power_spectrum_realization[1:], x=self.k[1:]))  # exempt k=0 mode
            # fluct = self.scale(xi)
            fluct = 1
            U_sqrt = 1

            return fluct * jnp.sqrt(power_spectrum_realization) / U_sqrt

        # self.amplitude_spectrum_normalized = lambda xi: jnp.sqrt(self.power_spectrum(xi))
        self.amplitude_spectrum_normalized = lambda xi: make_norm_amp(xi)


        self.hyper_power_spectrum = lambda xi: cfm_maker.amplitude(xi)**2
        self.model_prefix = model_prefix
        self.power_distributor = harmonic_grid.power_distributor

        xi_s_mean = jnp.zeros(shape=(N_ss,))
        xi_s_std = jnp.ones(shape=(N_ss,))
        self.xi_s = jft.NormalPrior(mean=0, std=1, name="s_xi", shape=jft.ShapeWithDtype(shape=(N_ss,), dtype=jnp.float64),)  # introduce new xi to be multiplied onto

        # self.zm = jft.LogNormalPrior(...) :TODO Implement!! And add this to the power spectrum operator directly, not during each call

        super().__init__(domain = self.power_spectrum.domain | self.xi_s.domain | self.scale.domain, white_init=True)

    def __call__(self, xi):
        amplitude_realization = self.amplitude_spectrum_normalized(xi)
        xi_s_realization = self.xi_s(xi)

        return fw_hartley(amplitude_realization[self.power_distributor] * xi_s_realization, self.s_grid) * 1e2

    def get_model_components(self):
        amplitude_op = self.amplitude_spectrum_normalized
        parameter_choices = self.parameter_choices
        model_prefix = self.model_prefix

        return amplitude_op, parameter_choices, model_prefix


def tmp1(xi, amp, custom_norm_for_your_convenience=1):
    # spits out a realization given a power spectrum
    # power spectrum should have length len(data), i.e. be already distributed
    return jnp.fft.ifft(custom_norm_for_your_convenience * amp * xi["s_hyper_xi"], norm="ortho").real


def expand_rfft(f_unique, N):
    # work with unique k's and broadcast to full k's using this function.
    return np.concatenate([f_unique, f_unique[-2:0:-1].conj()]) if N % 2 == 0 else np.concatenate([f_unique, f_unique[-1:0:-1].conj()])


def tmp2(k, p, length_of_data):
    # p = params, k fourier modes
    # assumes k[0] = 0 and np ordering of k

    slope = p[0]
    amplitude = p[1]

    tmp = np.abs(k.copy())  # negative modes are just the positive ones mirrored. If you remove abs you will get an error for slope =-1 for example, makes sense.
    tmp[0] = 1  # mask zeromode

    tmp = tmp**slope

    sorter = np.argsort(k)

    tmp = tmp / (np.trapz(tmp[sorter][1:], k[sorter][1:]))
    tmp = amplitude * tmp
    tmp[0] = 1e-30  # fix zeromode

    if not np.all(tmp >=0 ):

        raise ValueError("Power spectrum cannot be negative, p_s(k) = ", tmp)

    return expand_rfft(tmp, length_of_data)


def Stress_re(xi, time, supress_print=False, downsample=False):
    """
    See also nifty8 `Stress` function.

    :param xi: jnp.array        A field to calculate the wigner function for. Either of complex or real data type.
                                If complex, assumed to be in DFT standard order (DC first, then positives then negatives).
    :param time: jnp.array      The real-space time array at which xi (or its iFFT if complex) was sampled at.
    :param supress_print: bool, Print imaginary part diagonstics (Wigner function should be real).
    :return:
    """

    FFT = lambda x, ax=-1: jnp.fft.fft(x, norm="ortho", axis=ax)
    iFFT = lambda x, ax=-1: jnp.fft.ifft(x, norm="ortho", axis=ax)

    if jnp.iscomplexobj(xi):
        xi = iFFT(xi)  # go to real space

    if downsample:
        step = 2
        xi = xi[::step]
        time = time[::step]

    t0 = time[0]
    dt = time[1]-time[0]
    N = len(xi)
    f = jnp.fft.fftfreq(N, d=dt)
    k = f.copy()
    df = f[1] - f[0]
    t = jnp.arange(N) / (N*df)  # dual time, equal to input time - time[0].


    if not supress_print:
        print("\nCalculating stress...")

    t_c = t[:, None]  # time cast
    k_c = k[None, :]  # shift frequencies cast
    xi_c = xi[:, None]  # xi values cast as rows

    if not supress_print:
        print("\t Calculating zeta plus")
    zeta_plus = jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta minus")
    zeta_minus = jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT(zeta_plus, ax=0)

    if not supress_print:
        print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT(zeta_minus, ax=0)

    if not supress_print:
        print("\t Calculating Phi matrix")
    Phi = tilde_zeta_plus * tilde_zeta_minus.conj()  # domain = (h_space, h_space)

    if not supress_print:
        print("\t Inverse Fourier-Transforming columns of Phi matrix")
    S = iFFT(Phi, ax=1)
    S.block_until_ready()

    if not supress_print:
        print("\t ... Done")
    if not supress_print:
        diagnostic = jnp.abs(jnp.mean(S.imag))
        tmp = float(diagnostic)
        if diagnostic < 1e-10:
            print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
        else:
            raise_warning(
                f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")
    return S, t+t0, f



import nifty8 as ift
def Stress(xi_field: ift.Field, supress_print=False):
    """

    DFT conventions:
        F[m] = sum_{n=0}^{N-1} f[n] * exp(2πi * m n / N)
        f[n] = (1/N) * sum_{m=0}^{N-1} F[m] * exp(-2πi * m n / N)

    :param xi_field:  ift.Field    The Field over real space to analyze. If harmonic, assumed to be in DFT standard order and is mapped to its real space counterpart.
    :param supress_print:          Whether to print imaginary part diagnostic.
    :return: (S_mat, t_dual, f)    S_mat is the calculated wigner matrix, t_dual is the dual time from fourier transforming
                                   the columns of an intermediate matrix in ascending, monotonic order and f are the
                                   frequencies of the setup in standard DFT order (i.e. 0 as the first element).

    """

    if xi_field.domain[0].harmonic:
        helper_fft = ift.FFTOperator(xi_field.domain, space=0)
        xi_time_field = helper_fft(xi_field)
    else:
        xi_time_field = xi_field


    time_space = xi_time_field.domain[0]
    h_space = time_space.get_default_codomain()

    N = time_space.size
    t_vol = time_space.scalar_dvol
    h_vol = h_space.scalar_dvol * N

    dt_step = time_space.distances[0]
    xi_time = xi_time_field.val.astype(np.complex128)
    N = len(xi_time)
    f = np.fft.fftfreq(N, d=dt_step)
    k = f.copy()
    t = np.arange(N)*dt_step
    time = np.arange(N)*dt_step

    FFT_1 = ift.FFTOperator(domain=(time_space, h_space), space=0) * (1/t_vol)
    FFT_2 = ift.FFTOperator(domain=(h_space, h_space), space=1) * (1/h_vol)

    time_cast = time[:, None]
    k_freq_cast = k[None, :]
    xi_values_cast_as_rows = xi_time[:, None]

    print("\nCalculating stress...")

    print("\t Calculating zeta plus")
    zeta_plus = fieldify(np.exp(-np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))
    print("\t Calculating zeta minus")
    zeta_minus = fieldify(np.exp(np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))

    print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT_1(zeta_plus).val
    print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT_1(zeta_minus).val

    print("\t Calculating Phi matrix")
    Phi_val = tilde_zeta_plus * tilde_zeta_minus.conj()  # im putting the conjugate on the MINUS zeta since I also changed FFT convention by a sign wrt. Wikipedia...
    Phi_field = ift.Field(dt_((h_space, h_space)), val=Phi_val)

    print("\t Fourier-Transforming columns of Phi matrix")
    S = FFT_2(Phi_field)
    S_mat = S.val
    print("\t ... Done")

    dk = k[1] - k[0]        # safe in FFT ordering (first step is Δk)
    dt_dual = 1.0 / (N * dk)
    t_dual = np.arange(N) * dt_dual

    if not supress_print:
        diagnostic = np.abs(np.mean(S_mat.imag))
        if diagnostic < 1e-10:
            print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
        else:
            raise_warning(
                f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")

    return S_mat, t_dual, f

_smoothing_modes = Literal[
    "gaussian",
    "median",
    "uniform",
    "bilateral",
    "anisotropic"
]
def smooth_matrix(
    mat: np.ndarray,
    smoothing_lvl,
    mode: _smoothing_modes = "gaussian"
) -> np.ndarray:

    if mode == "gaussian":
        return gaussian_filter(mat, sigma=smoothing_lvl)

    elif mode == "median":
        size = int(max(1, smoothing_lvl))
        return median_filter(mat, size=size)

    elif mode == "uniform":
        size = int(max(1, smoothing_lvl))
        return uniform_filter(mat, size=size)

    elif mode == "bilateral":
        # smoothing_lvl used as intensity + spatial scale
        return cv2.bilateralFilter(
            mat.astype(np.float32),
            d=5,
            sigmaColor=smoothing_lvl * 20,
            sigmaSpace=smoothing_lvl * 5,
        )

    elif mode == "anisotropic":
        # total variation denoising
        return denoise_tv_chambolle(mat, weight=smoothing_lvl)

    else:
        raise ValueError(f"Unknown mode: {mode}")

def visualize_stress(stress_matrix, rows, cols, smooth=False, detect_outliers=False, tl="", hlines=None, vlines=None,
                     smoothing_level=5, cmap="plasma", **kwargs):

    stress_matrix = stress_matrix.real

    cols_are_increasing = np.all(np.diff(cols) > 0)  # strictly increasing
    rows_are_increasing = np.all(np.diff(rows) > 0)  # strictly increasing
    if not cols_are_increasing:
        raise ValueError("Columns must be increasing")
    if not rows_are_increasing:
        stress_matrix = np.fft.fftshift(stress_matrix, axes=0)  # shift DC frequency to middle
        rows = np.fft.fftshift(rows, axes=0)
        print("\t\tRows must be increasing, assuming a priori standard DFT order and moving DC to the middle")
        # Must be increasing because we want to plot from - frequency to 0 to + frequency on the y axis

    if smooth:
        stress_matrix = smooth_matrix(stress_matrix, smoothing_level)

    if detect_outliers:

        t_outlier, f_outlier = detect_outliers_in_stress(stress_matrix, fac=10, cols=cols, rows=rows)

        plt.plot(t_outlier, f_outlier, "b.", markersize=5)
        plt.show()

        plt.hist2d(t_outlier, f_outlier, bins=[50, 50], cmap='magma')
        plt.colorbar(label='Counts')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Outlier density in (t, f)')
        plt.show()

        bins = np.linspace(t_outlier.min(), t_outlier.max(), 50)
        bin_indices = np.digitize(t_outlier, bins)
        mean_f = [f_outlier[bin_indices == i].mean() for i in range(1, len(bins))]

        plt.plot(bins[:-1], mean_f, 'o-')
        plt.xlabel('Time')
        plt.ylabel('Mean frequency of outliers')
        plt.show()

    plt.figure(figsize=(10,6))
    plt.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap=cmap, interpolation='nearest')

    if hlines is not None:
        plt.hlines(hlines, 0, np.max(cols), color="r", ls="-")
    if vlines is not None:
        plt.vlines(vlines, 0, np.max(rows), color="r", ls="-")

    plt.colorbar(label='Stress')
    plt.tight_layout()

    usual_plot(xl='Time [s]', yl='Frequency [Hz]', title='Wigner function'+tl, **kwargs)


def detect_outliers_in_stress(stress_matrix, fac, cols, rows):
    mean_stress, std_stress = np.mean(stress_matrix), np.std(stress_matrix)
    thresh = mean_stress + fac * std_stress

    larger_than_threshhold_idcs = np.where(stress_matrix > thresh)
    t, f = cols[larger_than_threshhold_idcs[1]], rows[larger_than_threshhold_idcs[0]]

    return t, f


def fieldify(array, dom):
    return ift.Field(dt_(dom), array)

def dt_(dom):
    return ift.DomainTuple.make(dom)


class GaussianComb(jft.Model):
    def __init__(self, unique_k_lengths:jnp.array, list_of_peaks:jnp.array, list_of_amplitudes:jnp.array, rel_sigma_amp = .1, rel_sigma_widths=.1,
                 a_priori_width_of_peaks = 10, abs_width_sigma=1, abs_amp_sigma=1):
        """

        Generates a sum of Gaussian parametric peaks at fixed positions and with lognormal priors set on the
        widths and amplitudes.

        :param unique_k_lengths:        The unique frequencies, the operator is built in amplitude space.
        :param list_of_peaks:           Array of peak positions (frequencies)
        :param list_of_amplitudes:      Array of peak amplitudes (power units)
        :param rel_sigma_amp:           The relative standard deviation set on the lognormal amplitude prior
        :param rel_sigma_widths:        The relative standard deviation set on the lognormal frequency width prior
        :param a_priori_width_of_peaks: In Hz.
        """
        ## aaah : implement as two vectors: xi_g_amp and xi_g_width of length of the power spectrum domain

        self.f = unique_k_lengths
        self.N = len(list_of_peaks)
        self.frequency_widths = a_priori_width_of_peaks * jnp.ones(self.N)
        self.positions = list_of_peaks

        if not abs_width_sigma:
            self.sigma_widths = rel_sigma_widths * self.frequency_widths
        else:
            self.sigma_widths = abs_width_sigma

        if not abs_amp_sigma:
            self.sigma_amp = rel_sigma_amp * list_of_amplitudes
        else:
            self.sigma_amp = abs_amp_sigma

        self.xi_g_amp = jft.LogNormalPrior(mean=list_of_amplitudes, std=self.sigma_amp, name="xi_g_amp",
                                           dtype=jnp.float64, shape=(self.N,))
        # self.xi_g_amp = jft.NormalPrior(mean=0, std=list_of_amplitudes, name="xi_g_amp",
        #                                    dtype=jnp.float64, shape=(self.N,))
        self.xi_g_width = jft.LogNormalPrior(mean=self.frequency_widths, std=self.sigma_widths, name="xi_g_width",
                                             dtype=jnp.float64, shape=(self.N,))

        def single_gaussian(amp, width, pos):
            return amp**2 * jnp.exp(-0.5 * ((self.f - pos) / width) ** 2)

        self.sg = single_gaussian

        super().__init__(domain=self.xi_g_amp.domain | self.xi_g_width.domain)

    def __call__(self, xi):
        """
        Logic:

            gaussians = []
            for freq_center, amp, freq_width:

                    gaussian = amp * np.exp(-0.5 * ((f - freq_center) / freq_width) ** 2)
                    gaussians.append(gaussian)

            gaussian_comb = np.sum(gaussians, axis=0)

        :param xi:
        :return:
        """
        amplitude_vector = self.xi_g_amp(xi)
        width_vector = self.xi_g_width(xi)
        position_vector = self.positions

        gaussians = vmap(self.sg)(amplitude_vector, width_vector, position_vector)
        # comb = jnp.sum(gaussians, axis=0)
        # norm = jnp.trapezoid(comb, x=self.f)
        return jnp.sum(gaussians, axis=0)


class PowerSpectrumTemplate(jft.Model):
    def __init__(self, ps_template:jnp.array, scale=(1,.1)):
        """
        Creates an operator in power space that consists of a scalable power spectrum template.
        :param ps_template:     The fixed power spectrum template.
        """
        self.ps_template = ps_template
        self.scale = jft.LogNormalPrior(*scale, name="ps_template_scale")

        super().__init__(domain=self.scale.domain)

    def __call__(self, xi):
        scale_realization = self.scale(xi)
        return scale_realization * self.ps_template


def get_peaks_from_cache(only_positives = True, sigma_thresh=2, custom_norm=1, power_spectrum=None, custom_path=None):
    """
    Loads a saved xi_d file, detects peaks based on an ad-hoc threshhold and returns the position and amplitude of
    said peaks.
    :param only_positives:      bool,        Whether to report events at negative frequencies
    :param sigma_thresh:        float,       Threshold in standard deviations
    :param custom_norm:         float,       Normalization factor, e.g. max(posterior_pipe_1_ps_mean). By default,
                                             max(amplitude of peak) = 1.
    :param custom_path:                      Path to a saved xi_d file
    :param power_spectrum:                   If not none, will be used as weights
    :return:
    """
    path = "pipe2_xi_cache.txt" if custom_path is None else custom_path

    obj = np.loadtxt(path, dtype=np.complex128)

    xi, f = obj[:,0].real, obj[:,1].real
    if power_spectrum is None:
        power_spectrum = np.ones(len(f))
    if only_positives:
        to_del = np.where(xi<0)
        xi = np.delete(xi, to_del)
        f = np.delete(f, to_del)
        ps = np.delete(power_spectrum, to_del)

    adhoc_treshhold = sigma_thresh * np.mean(xi)

    where_peaks_in_xi = np.where(xi > adhoc_treshhold)
    peaks_k = f[where_peaks_in_xi]

    ps_weights = ps[where_peaks_in_xi]
    ps_weights_normed = ps_weights / max(ps_weights)

    amplitudes_k = xi[where_peaks_in_xi]
    normed_amplitudes_k = amplitudes_k * custom_norm / max(amplitudes_k) * ps_weights_normed

    return peaks_k, normed_amplitudes_k


def analyze_kl_callback(out_name, max_kl_iterations, lh, samples, vi_state):

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


class InvNoiseCovFromPs():
    def __init__(self, one_sided_noise_ps:jnp.array, data_grid, e_fac, n_dtps:int, custom_norm=1):
        r"""

        Don't delete yet, `downsampling` procedure and checks might be helpful when not using a welch average for
        the power spectrum.

        Takes one_sided_noise_ps as input and returns an operator which when called applies

            F^{-1} 1/full_noise_ps F.

        Here, I want to simply downsample and thus create periodic realizations of the noise. Therefore, the inference
        is going to be off at the edges.

        In this class, I assume that noise_ps is the mean posterior data power spectrum gained by previous inference
        runs and that the approximation p_d ~ p_n is to be employed. To circumvent biasing through periodic boundary
        conditions, the data were learned on an extended domain, i.e. the inferred data power spectrum has more points
        than there are data points, so this needs to be somehow downsampled. How?

            len(p_n_inferrred) > len(p_n_real)

        and n^* \hookleftarrow p_n_inferrred:

            len(n^*) > len(n_real).

        This class needs to implement an operator such that the operation

            op = (F^{-1} p_n F) * res

        is well-defined. Here, res is a non-periodic residual d-Rs of length N. But p_n is a MxM matrix with M>N.
        So, zero-pad the input to length M:

            res_prime[N:M] = 0

        This operation gives the same Fourier-Transform as just F(res) because it is just zeros
        (TODO: Maybe only up to a volume factor??)
        Then, after applying the full MxM p_n matrix and transforming back, we may simply cut as

            result = ... [:N].

        Problem:

        To be able to use the data power spectrum to create a noise covariance matrix, the power spectrum is down
        sampled to include only every second point.

        Actually, lets zero-pad the input, apply pd, iFFT and hope for the best and then do an analysis of gibbs
        ringing.

        The residual d minus R s can be made approximately periodic in the zero-padded region because s includes the
        variable xi_s, which can adapt to enforce boundary conditions. In a Gaussian likelihood method, the call method
        of this class would be applied to the residual d minus R s; if you draw a fixed sample without a xi_s variable,
        zero-padding does not enforce boundary conditions and the resulting FFT may appear a bit blurred.

        :param one_sided_noise_ps:    The noise power spectrum P_n(|k|).
        :param data_grid:   The real space data grid.
        :param e_fac:       The extension factor by which the data domain was extended to do the inference.
                            Will determine how much the power spectrum is downsampled.
        :param n_dtps:      The number of datapoints.
        :param custom_norm  A scalar multiplied onto the power spectrum to bring the data realizations to the order of
                            magnitude of the actual data (something about volume factors I evidently don't understand).
        """

        self.one_sided_noise_ps = one_sided_noise_ps[::e_fac] * custom_norm

        self.M_k_lengths = len(data_grid.harmonic_grid.relative_log_mode_lengths)
        self.M = len(one_sided_noise_ps)
        self.N = n_dtps
        self.k = data_grid.harmonic_grid.mode_lengths
        self.dk = self.k[1]-self.k[0]

        self.harmonic_data_grid_expander = data_grid.harmonic_grid.power_distributor # [0 1 2 ... 3 2 1], therefore,
        # if one_sided_noise_ps is ordered as [0, +1, +2, ..., +N/2], one_sided_noise_ps[power_distributor]
        # will be ordered as [0, +1, +2, ..., +N/2, +N/2-1, +2, +1].

        assert self.M == self.M_k_lengths  #  the noise power spectrum IS one-sided, i.e. supported by the correct
        # number of fourier modes. If the ps was gotten by an interpolation that didn't get it right, this will
        # throw an assertion error

        assert self.N == n_dtps  # downsampled correctly? edit 08.12: Forgot why I was doing this

        self.golden_fourier_norm = self.N * self.dk
        self.full_noise_ps = one_sided_noise_ps[self.harmonic_data_grid_expander]

        self.inv = self.full_noise_ps**(-1)
        self.sqrt = jnp.sqrt(self.full_noise_ps)

        self.uH = lambda p: fw_hartley(p, norm="ortho")
        self.iuH = lambda p: bw_hartley(p, norm="ortho")

        expected_var = np.sum(self.full_noise_ps * self.dk)
        print("Expected real-space variance:", expected_var)

        raise_warning("To do: Write assertions of the power spectrum and sample variances automatically")

        print("hä", np.sum(one_sided_noise_ps[1:])+one_sided_noise_ps[0]/2)

        print("\n\n\n and more diagnostics...\n")

        C = self.uH(np.diag(self.full_noise_ps * self.N * self.dk) @ self.iuH(np.eye(self.N)))

        print("np.mean(np.diag(C)[1:]) (exempting C[0][0] because I think its a special point):", np.mean(np.diag(C)[1:]))
        print("\n and C itself: ", C)



    def __call__(self, p):
        # Implements: N^{-1}(res) = F^{-1} p_n^{-1} F(res) where res = d-Rs for example
        # this assumes that the input is periodic. :TODO I believe that this will enforce the theoretical data Rs to be inferred as a periodic function

        fourier_input = self.uH(p)  # now, this is an i.i.d. variable in standard DFT order:
        # [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.]

        applying_inv_ps = 1/self.golden_fourier_norm * self.inv * fourier_input  # meaning, that self.inv must ALSO be in standard DFT order
        # [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.] which should be guaranteed through the use of
        # harmonic_data_grid_expander

        transforming_to_real_space = self.iuH(applying_inv_ps)
        res = transforming_to_real_space
        return res


    def N_sqrt(self, p):
        # Implements: xi_prime = N^{1/2} xi where xi is standard normal, such that xi_prime is from a Gaussian with covariance N.
        # These samples will be periodic.
        fourier_input = self.uH(p)
        applying_sqrt_ps = jnp.sqrt(self.golden_fourier_norm) * self.sqrt * fourier_input
        transforming_to_real_space = self.iuH(applying_sqrt_ps)
        res = transforming_to_real_space
        return res


    def get_samples(self, num):
        """
        0-centered Gaussian distributed samples with covariance N.
        :param num:    The numbers of samples to get.
        :return: list containing the samples
        """
        lt = []
        for _ in range(num):
            xi = np.random.standard_normal(self.N)
            sl = self.N_sqrt(xi)
            lt.append(sl)
        return lt


    def plot_samples(self, num):
        samples = self.get_samples(num)
        for sl in samples:
            plt.plot(self.time, sl)
        usual_plot()


class NoiseCovarianceFromPs():
    def __init__(self, one_sided_noise_ps:jnp.array, data_grid, callable_to_apply=None, silly_number=1):
        r"""
        Please see deprecated class `InvNoiseCovFromPs` as well.
        The call method of this class implements

            output = uH callable( h_vol * one_sided_noise_ps[expander]m) iuH ( input ),

        where uH and iuH are unitary Hartley and inverse Hartley transforms, respectively.

        :param one_sided_noise_ps:    The noise power spectrum P_n(|k|).
        :param data_grid:             The real space data grid to get the wavevectors.
        :param callable_to_apply:     A callable to apply the noise power spectrum and corresponding weights in Fourier
                                      space. For example, lambda x: x**-1 for an inverse power spectrum.
        """

        self.one_sided_noise_ps = one_sided_noise_ps
        self.apply_callable = callable_to_apply
        self.h_grid = data_grid.harmonic_grid
        self.k = self.h_grid.mode_lengths
        self.N = data_grid.shape[0]

        self.M_k_lengths = len(self.h_grid.relative_log_mode_lengths)
        self.M = len(one_sided_noise_ps)
        self.dk = self.k[1]-self.k[0]

        self.expand =  self.h_grid.power_distributor # [0 1 2 ... 3 2 1], therefore,
        # if one_sided_noise_ps is ordered as [0, +1, +2, ..., +N/2], one_sided_noise_ps[power_distributor]
        # will be ordered as [0, +1, +2, ..., +N/2, +N/2-1, +2, +1].

        assert self.M == self.M_k_lengths  #  the noise power spectrum IS one-sided, i.e. supported by the correct
        # number of fourier modes. If the ps was gotten by an interpolation that didn't get it right, this will
        # throw an assertion error

        self.h_vol = self.N * self.dk
        self.full_noise_ps = one_sided_noise_ps[self.expand]

        self.uH = lambda p: fw_hartley(p, norm="ortho")
        self.iuH = lambda p: bw_hartley(p, norm="ortho")

        self.silly_number = silly_number

        expected_var = np.sum(self.full_noise_ps * self.dk)
        print("Initiating noise covariance. σ^2 = ∫ ps(k) dk = ", expected_var,
              ". Callable to be applied: callable(2)=", self.apply_callable(2))

    def __call__(self, p):
        fourier_input = self.uH(p)  # An i.i.d. variable in standard DFT order [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.]
        kernel = self.apply_callable(self.full_noise_ps * self.h_vol)
        return self.iuH(kernel * fourier_input) * self.silly_number




def power_analyze_re(x_values, y_values):
    """
    Returns an estimate of the power spectrum by absolute squaring the fourier transform.
    :param x_values: The x values used to determine the spacing needed to calculate the fourier modes.
    :param y_values: A real space periodic array.
    :return:
    """
    N = len(x_values)
    dx = x_values[1] - x_values[0]
    ps = jnp.abs(jnp.fft.fft(y_values, n=N, norm="ortho"))**2
    k = jnp.fft.fftfreq(N, d=dx)
    return k, ps


def plot_histogram(key, mean: float, sigma: float, n_samples: int, mode="Lognormal"):
    """
    Plots a histogram visualizing the moment-matched lognormal transform.
    If `vlines` is provided, vertical lines will be drawn at the specified x-locations.
    Usage:

    plot_lognormal_histogram(mean=.06, sigma=0.03, n_samples=10000, vlines=[0.023, 0.05], save=True, show=True)

    :param mean:        The mean from which logmean is calculated with logsigma's help.
    :param sigma:       The sigma from which logsigma is calculated.
    :param n_samples:   How many samples to plot
    :return:
    """
    # fig = plt.figure(figsize=(10, 4))
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

    label = rf"{mode} with $(\mu, \sigma)=$" + f"$({mean}, {sigma})$" if not (mode=="Uniform") else rf"{mode} in " + r"$\mathrm{[0,1]}$"
    plt.hist(op_samples, bins=200, label=label,
             histtype='step', facecolor='white', color="black")

    plt.show()