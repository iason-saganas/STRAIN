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
               show=True, close=False):
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    ax = plt.gca()
    labels = ax.get_legend_handles_labels()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if labels != ([], []):
        plt.legend()
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
        "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/GW150914_strain.pickle",
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
            self.adjoint_zp = lambda arr, ext_fact: arr[:int(len(arr)/ext_fact+1)]  # TODO: correct cutting?
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
        self.inv_N_cov = None
        self.sqrt_inv_N_cov = None
        self.kl_kwargs = None
        self.nonlinearly_update_kwargs = None
        self.draw_linear_kwargs = None
        self.posterior_xi_samples = None
        self.parameter_choices = None
        self.model_prefix = None


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

        amplitude_op, parameter_choices, model_prefix = custom_signal_model.get_model_components()

        self.s_model = custom_signal_model
        self.amplitude_op = amplitude_op
        self.parameter_choices = parameter_choices
        self.model_prefix = model_prefix


    def add_cfm_signal_model(self, fluct:tuple, llslope:tuple, flex:tuple | None = None, asper:tuple | None=None,
                             offset_mean:float = 0, offset_std:tuple = (1e-16, 1e-16), model_prefix="s_",
                             add_power_spectrum_template=None, add_custom_power_op=(None,)):
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
                                   hack_custom_amplitude_operators=add_custom_power_op)

        parameter_choices = {
            f"{model_prefix}fluctuations": lambda xi: np.exp(fluct[0] + xi*fluct[1]),
            f"{model_prefix}loglogavgslope": lambda xi: llslope[0] + xi*llslope[1],
            f"{model_prefix}flexibility": lambda xi: np.exp(flex[0] + xi*flex[1]),
            f"{model_prefix}asperity": lambda xi: np.exp(asper[0] + xi*asper[1]),
            #"offset_mean": (offset_mean, "fix"),
            #"offset_std": (offset_std, "lognormal"),
        }


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
                super().__init__(init=signal_model.init)

            def __call__(self, xi):
                model_values_on_long_domain = self.sm(xi)
                return self.zp_adj(model_values_on_long_domain, self.ext_fac)

        s_prime = Response(signal_model=self.s_model, zp_adj=self.adjoint_zp, ext_fac=self.e_fac)

        return s_prime


    def add_noise_op(self, noise_var_level=1e-10,
                     inverse_noise_op:Optional[Callable[[jnp.array], jnp.array]]=None,
                     sqrt_inverse_noise_op:Optional[Callable[[jnp.array], jnp.array]]=None):

        if (inverse_noise_op is None) ^ (sqrt_inverse_noise_op is None):  # ^ = xOr operator!
            raise ValueError("One of inverse_noise_op or sqrt_inverse_noise_op was provided, but the other not,"
                             "likely want to provide both, eitherwise wrong metric in Gaussian likelihood.")

        if inverse_noise_op is None:
            # Both operators were not provided => Diagonal noise covariance
            self.inv_N_cov = lambda x: x/noise_var_level
            self.sqrt_inv_N_cov = lambda x: x/jnp.sqrt(noise_var_level)
        else:
            # Both operators were correctly provided.
            self.inv_N_cov = inverse_noise_op
            self.sqrt_inv_N_cov = sqrt_inverse_noise_op


    def add_minimizers(self, linear_loose=(0.02, 100), linear_strict=(0.02, 100), non_linear_loose=(0.5, 20),
                       non_linear_strict=(0.5, 20), kl_loose=(0.1, 35), kl_strict=(0.01, 50), use_strict=False):

        if use_strict:
            linear_energy, linear_iter = linear_strict
            nonlinear_energy, nonlinear_iter = non_linear_strict
            kl_energy, kl_iter = kl_strict
        else:
            linear_energy, linear_iter = linear_loose
            nonlinear_energy, nonlinear_iter = non_linear_loose
            kl_energy, kl_iter = kl_loose

        draw_linear_kwargs = dict(
            cg_name="linear_sampler",
            cg_kwargs=dict(absdelta=linear_energy, maxiter=linear_iter),
        )

        # Arguments for the minimizer in the nonlinear updating of the samples
        nonlinearly_update_kwargs = dict(
            minimize_kwargs=dict(
                name="non_linear_sampler",
                xtol=nonlinear_energy,
                cg_kwargs=dict(name=None),
                maxiter=nonlinear_iter,
            )
        )

        # Arguments for the minimizer of the KL-divergence cost potential
        kl_kwargs = dict(
            minimize_kwargs=dict(
                name="kl_minimizer", xtol=kl_energy, cg_kwargs=dict(name=None), maxiter=kl_iter
            )
        )

        self.draw_linear_kwargs = draw_linear_kwargs
        self.nonlinearly_update_kwargs = nonlinearly_update_kwargs
        self.kl_kwargs = kl_kwargs


    def build_lh(self):
        s_prime = self.signal_response()

        if self.inv_N_cov is None:
            level = 1e-10
            raise_warning(f"self.add_noise_op() was not called by the user, using Gaussian noise with default "
                          f"variance level {level}.")
            self.add_noise_op(noise_var_level=level)

        lh = jft.Gaussian(data=self.d, noise_cov_inv=self.inv_N_cov, noise_std_inv=self.sqrt_inv_N_cov).amend(s_prime)
        return lh

    def run_inference(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False, out_name="out",
                     resume=True, choose_low_kl_starting_pos=False, geoVi=True):
        lh = self.build_lh()

        if self.draw_linear_kwargs is None:
            self.add_minimizers(use_strict=use_strict_minimizers)

        if self.plotting_callback is not None:
            plotting_callback = lambda samples, vi_state: (
                self.plotting_callback(out_name, kl_iterations, lh, samples, vi_state))

        self.key, key_sampler, key_i = jax.random.split(self.key, 3)

        if choose_low_kl_starting_pos:
            initial_position = get_beneficial_position(key=key_i, lh=lh, samples_to_draw=2000)
        else:
            initial_position = jft.Vector(lh.init(key_i))

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

        posterior_parameters_samples = {
            key: [prior_distributions[key](xi[key]) for xi in post_xi_samples]
            for key in parameter_names
        }

        posterior_parameters_mean_std = {
            k: (np.mean(v), np.std(v)) for k, v in posterior_parameters_samples.items()
        }

        if print_posterior_parameters:
            for k, (mean, std) in posterior_parameters_mean_std.items():
                print(f"{k:20s}  mean = {mean:10.6f},  std = {std:10.6f}")

        return_list = [ps_mean_std, signal_mean_std, posterior_parameters_mean_std]
        if moment == "mean":
            return_ps, return_signal = [v[0] for v in return_list[:-1]]
            return_posterior_parameters = {k: v[0] for k, v in return_list[-1].items()}
        else:
            return_ps, return_signal, return_posterior_parameters = return_list

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
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
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
                           num=5, plot=True, plot_welch_average=False, rolling=False):
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
                self._plot_power_spectrum_and_sample()
            return

        samples = self.get_prior_samples(mode=mode, num=num)

        if not plot:
            return samples[0]  # peel away outer bracket

        xl = "Time"
        yl = "Strain"
        if mode == "signal":
            x = self.t_ss
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
            plt.plot(x, sl)
            if rolling:
                if plot_welch_average:
                    plot_welch_averaged_ps()
                    plt.loglog()
                usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}")

        if not rolling:
            if plot_welch_average:
                plot_welch_averaged_ps()
                plt.loglog()
            usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}", show=True, close=True)


    def _plot_power_spectrum_and_sample(self):
        pow_spec_samples = self.get_prior_samples(mode="power spectrum", num=1)
        xi = np.random.standard_normal(self.n_ss)
        expander = self.s_dom_harmonic.power_distributor
        signal_space_samples = [hartley(ps[expander] * xi, signal_grid=self.s_dom_real) for ps in pow_spec_samples]

        x_ps = self.k_signal
        x_s = self.t_ss
        xl = r"Unique $f$"
        yl = r"$\mathrm{Power}$"

        # remove 0-mode for plotting
        x_ps = x_ps[1:]
        pow_spec_samples = [sl[1:] for sl in pow_spec_samples]

        fig, axs = plt.subplots(nrows=2, ncols=1)
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


    def plot_posterior_signal(self, print_posterior_parameters=False, over_full_signal_space=False):
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
        plt.plot(self.t_ds, self.d-0.5, label="Data", color="orange")

        usual_plot()



    def plot_posterior_power_spectrum(self, print_posterior_parameters=False, plot_welch_average=True):
        ps_mean_std, _, _ = self.get_posterior_statistics(print_posterior_parameters)

        plt.errorbar(self.k_signal[1:], ps_mean_std[0][1:], yerr=ps_mean_std[1][1:],
                     label=r"Reconstructed power spectrum (with $1\sigma$ contour)", ecolor=light_blue, color=blue)

        print("Zeromode P_s(k=0) excluded in plot")  # because ~0 and then changes the y limits such that the majority
        # of the power spectrum lies in just the upper half of the coordinate system

        if plot_welch_average:
            plot_welch_averaged_ps()

        # plt.ylim(-3e-9, 1.4e-7)
        plt.loglog()
        usual_plot(xl="Frequency $f$", yl="Power")


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


    def plot_noise_sample_with_data(self, num, rolling=False):
        """
        Only if noise operator supports a call 'get_sample'.

        """
        noise_op = self.inv_N_cov
        try:
            noise_samples = noise_op.get_samples(num)
        except AttributeError:
            raise ValueError("self.inv_N_cov needs to have an implemented method 'get_samples'.")

        for sl in noise_samples:
            plt.plot(self.t_ds, sl)
            if rolling:
                plt.plot(self.t_ds, self.d, label="Actual data")
                usual_plot(close=True)

        if not rolling:
            plt.plot(self.t_ds, self.d, label="Actual data")
            usual_plot()


    def calculate_and_plot_penrose_xi(self, plot=True):
        penrose_xi = find_penrose_moore_solution(pipe=self, reload_from_cache=True, filename="my_penrose_xi.txt")
        if plot:

            mean_ps = (self.get_posterior_statistics(moment="mean", quantity="power spectrum"))[self.s_h_dom_expander]
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


import operator
from jax.numpy import fft
def hartley(p, signal_grid):

    # :TODO: looks inefficient

    tmp = fft.fftn(p, axes=None)
    # c = jft._config.get("hartley_convention")
    c = "non_canonical_hartley"
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub

    harmonic_dvol = 1.0 / signal_grid.total_volume
    return add_or_sub(tmp.real, tmp.imag) * harmonic_dvol


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


def get_welch_averaged_ps(interpolate_to=None):
    """

    :param interpolate_to:      If an integer, interpolates the power spectrum so its length matches the given integer.
    :return:
    """
    welch = "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle"
    _, k_lengths, power_spectrum = unpickle_me_this(welch, absolute_path=True)

    if interpolate_to is not None:
        power_spectrum = power_spectrum.val
        interpolator = interp1d(x=k_lengths, y=power_spectrum, kind="linear")
        new_k = np.linspace(min(k_lengths), max(k_lengths), interpolate_to)
        new_power_spectrum = interpolator(new_k)
        return jnp.array(new_k), jnp.array(new_power_spectrum)

    return jnp.array(k_lengths), jnp.array(power_spectrum.val)

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

        return hartley(amplitude_realization[self.power_distributor] * xi_s_realization, self.s_grid) * 1e2

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


    print("\nCalculating stress...")

    t_c = t[:, None]  # time cast
    k_c = k[None, :]  # shift frequencies cast
    xi_c = xi[:, None]  # xi values cast as rows

    print("\t Calculating zeta plus")
    zeta_plus = jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    print("\t Calculating zeta minus")
    zeta_minus = jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT(zeta_plus, ax=0)

    print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT(zeta_minus, ax=0)

    print("\t Calculating Phi matrix")
    Phi = tilde_zeta_plus * tilde_zeta_minus.conj()  # domain = (h_space, h_space)

    print("\t Inverse Fourier-Transforming columns of Phi matrix")
    S = iFFT(Phi, ax=1)
    S.block_until_ready()


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


def visualize_stress(stress_matrix, rows, cols, smooth=False, detect_outliers=True, tl="", hlines=None, vlines=None):

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
        stress_matrix = gaussian_filter(stress_matrix, sigma=5.0)   # sigma ~ 0.5..3 blur radius in pixels

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

    plt.figure(figsize=(8,6))
    plt.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap='viridis', interpolation='nearest')

    if hlines is not None:
        plt.hlines(hlines, 0, np.max(cols), color="r", ls="-")
    if vlines is not None:
        plt.vlines(vlines, 0, np.max(rows), color="r", ls="-")
    plt.colorbar(label='Stress')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [s]')
    plt.title('Time vs Frequency' + tl)
    plt.tight_layout()
    plt.show()


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
                 a_priori_width_of_peaks = 10):
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

        self.sigma_amp = rel_sigma_amp * list_of_amplitudes
        self.sigma_widths = rel_sigma_widths * self.frequency_widths

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


def get_peaks_from_cache(only_positives = True, sigma_thresh=2, custom_norm=1, custom_path=None):
    """
    Loads a saved xi_d file, detects peaks based on an ad-hoc threshhold and returns the position and amplitude of
    said peaks.
    :param only_positives:      bool,        Whether to report events at negative frequencies
    :param sigma_thresh:        float,       Threshold in standard deviations
    :param custom_norm:         float,       Normalization factor, e.g. max(posterior_pipe_1_ps_mean). By default,
                                             max(amplitude of peak) = 1.
    :param custom_path:                      Path to a saved xi_d file
    :return:
    """

    path = "pipe2_xi_cache.txt" if custom_path is None else custom_path

    obj = np.loadtxt(path, dtype=np.complex128)

    xi, f = obj[:,0].real, obj[:,1].real
    if only_positives:
        to_del = np.where(xi<0)
        xi = np.delete(xi, to_del)
        f = np.delete(f, to_del)

    adhoc_treshhold = sigma_thresh * np.mean(xi)

    where_peaks_in_xi = np.where(xi > adhoc_treshhold)
    peaks_k = f[where_peaks_in_xi]

    amplitudes_k = xi[where_peaks_in_xi]
    norm_amplitudes_k = amplitudes_k * custom_norm / max(amplitudes_k)

    return peaks_k, norm_amplitudes_k


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
    def __init__(self, noise_ps:jnp.array, data_grid, e_fac, n_dtps:int, custom_norm=1e-2):
        r"""

        Takes noise_ps as input and returns an operator which when called applies F^{-1} 1/noise_ps F.

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

        :param noise_ps:    The noise power spectrum.
        :param data_grid:   The real space data grid.
        :param e_fac:       The extension factor by which the data domain was extended to do the inference.
                            Will determine how much the power spectrum is downsampled.
        :param n_dtps:      The number of datapoints.
        :param custom_norm  A scalar multiplied onto the power spectrum to bring the data realizations to the order of
                            magnitude of the actual data (something about volume factors I evidently don't understand).
        """
        self.noise_ps = noise_ps[::e_fac] * custom_norm
        self.N = len(self.noise_ps)

        assert self.N == n_dtps  # downsampled correctly?

        self.inv_noise_ps = self.noise_ps**(-1)
        self.H = lambda p: jnp.fft.fft(p, n=self.N, norm="ortho")  # not scaling the forward transform
        self.iH = lambda p: jnp.fft.ifft(p, n=self.N, norm="ortho")  # not scaling the backward transform

        self.harmonic_dvol = 1.0 / data_grid.total_volume
        self.harmonic_dvol = 1.0  #  don't scale for now, testing something out


        self.time = np.arange(self.N) * data_grid.distances

        self.inv = self.inv_noise_ps
        self.sqrt = jnp.sqrt(self.noise_ps)


    def __call__(self, p):
        # Implements: N^{-1}(res) = F^{-1} p_n^{-1} F(res) where res = d-Rs for example
        # this assumes that the input is periodic. :TODO I believe that this will enforce the theoretical data Rs to be inferred as a periodic function
        fourier_input = self.H(p)
        applying_inv_ps = self.inv * fourier_input
        transforming_to_real_space = self.iH(applying_inv_ps)
        return transforming_to_real_space.real * self.harmonic_dvol


    def N_sqrt(self, p):
        # Implements: xi_prime = N^{1/2} xi where xi is standard normal, such that xi_prime is from a Gaussian with covariance N.
        # This samples will be periodic.
        fourier_input = self.H(p)
        applying_sqrt_ps = self.sqrt * fourier_input
        transforming_to_real_space = self.iH(applying_sqrt_ps)
        return transforming_to_real_space.real * self.harmonic_dvol  # :TODO guessing the volume factor here


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



