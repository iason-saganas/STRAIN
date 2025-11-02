from data.style_components.matplotlib_style import *
import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import nifty.nifty.re as jft
import jax
import warnings
from typing import Literal
from time import time


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
               show=True):
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
    else:
        plt.close()


def get_sample_data(norm=1e19, time_window=(15,17)):
    strain = unpickle_me_this("/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/"
                              "GW150914_strain.pickle", absolute_path=True)

    zero_time = 1126259446  # I got this zero time by looking at the caption of the figure produced by strain.plot().
    time = np.array(strain.times) - zero_time  # in seconds

    full_data = norm * strain.value
    full_time = time.copy()

    t_min, t_max = time_window
    indcs = np.where((t_min < time) & (time < t_max))
    data = full_data[indcs]
    time = time[indcs]

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
    def __init__(self, t, d, key, e_fac=2, r_fac=2):
        """
        Uses an RG for the data space.
        :TODO: Set Mask only as response iff r_fac \neq 1; Otherwise unneccessary computations

        Usage:

                pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key)
                pipe_1.add_cfm_signal_model(fluct=(5,2), llslope=(-4,1))
                pipe_1.add_noise_op(noise_var_level=1)

                latent_samples = pipe_1.run_inference(kl_iterations=5, use_strict_minimizers=False, out_name="re_pipe_1", resume=True)
                key = pipe_1.get_current_key()

                pipe_1.plot_posterior_power_spectrum()

        :param t:       The time at which the data were sampled.
        :param d:       The data.
        :param key:     The jax PRNG key.
        :param e_fac:   The factor by which to extend the length of the domain to ensure periodicity.
                        Default: 2.
        :param r_fac:   The factor by which to increase the resolution of the signal space. Default 2.
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
        self.n_ss = int(self.L_ss / self.dist_ss)

        self.t_ds = t
        self.t_ss = jnp.arange(self.n_ss)*self.dist_ss + t[0]

        self.adjoint_zp = lambda arr, ext_fact: arr[:int(len(arr)/ext_fact+1)]

        assert self.t_ds[0] == self.t_ss[0]  # same beginning?
        assert jnp.all(self.t_ds == self.adjoint_zp(self.t_ss, e_fac)[::r_fac])  # if you cut the extended
        # array up to the max of t_ds and then take make it coarser, do you get the same support points?


        self.d_dom_real = jft.correlated_field.make_grid(shape=(self.n_ds,), distances=(self.dist_ds,),
                                                     harmonic_type="Fourier")
        self.s_dom_real = jft.correlated_field.make_grid(shape=(self.n_ss,), distances=(self.dist_ss,),
                                                     harmonic_type="Fourier")

        self.d_dom_harmonic = self.d_dom_real.harmonic_grid
        self.s_dom_harmonic = self.s_dom_real.harmonic_grid

        self.k_data = self.d_dom_harmonic.mode_lengths
        self.k_signal = self.s_dom_harmonic.mode_lengths

        self.k_data_full = join_k_arrays(self.d_dom_harmonic)
        self.k_signal_full = join_k_arrays(self.s_dom_harmonic)

        self.amplitude_op = None
        self.s_model = None
        self.inv_N_cov = None
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
                             add_power_spectrum_template=None):
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
        :return:
        """

        cfm_maker = jft.CorrelatedFieldMaker(prefix=model_prefix)
        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=fluct,
                                   loglogavgslope=llslope, flexibility=flex, asperity=asper, harmonic_type="fourier",
                                   non_parametric_kind="power", hack_add_power_spectrum_template=add_power_spectrum_template)

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


    def add_noise_op(self, noise_op=None, noise_var_level=1e-10):
        if noise_op is None:
            self.inv_N_cov = lambda x: x/noise_var_level
        else:
            raise ValueError("Non-diagonal Gaussian noise operators not implemented yet.")


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

        lh = jft.Gaussian(data=self.d, noise_cov_inv=self.inv_N_cov).amend(s_prime)
        return lh

    def run_inference(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False, out_name="out",
                     resume=True):
        lh = self.build_lh()

        if self.draw_linear_kwargs is None:
            self.add_minimizers(use_strict=use_strict_minimizers)

        self.key, key_sampler, key_i = jax.random.split(self.key, 3)

        starting_time = time()

        samples, _ = jft.optimize_kl(
            likelihood=lh,
            position_or_samples=jft.Vector(lh.init(key_i)),
            key=key_sampler,
            n_total_iterations=kl_iterations,
            n_samples=n_samples,
            draw_linear_kwargs=self.draw_linear_kwargs,
            nonlinearly_update_kwargs=self.nonlinearly_update_kwargs,
            kl_kwargs=self.kl_kwargs,
            sample_mode="nonlinear_resample",
            resume=resume,
            odir=out_name,

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


    def get_posterior_statistics(self, print_posterior_parameters=False):
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

        return ps_mean_std, signal_mean_std, posterior_parameters_mean_std

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

    def plot_prior(self, mode="signal", num=500, plot=True):
        """

        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
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
        elif mode == "amplitude spectrum":
            x = self.k_signal
            xl = r"Unique $f$"
            yl = r"$\sqrt{\mathrm{Power}}$"
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


    def plot_prior_samples(self, mode:Literal["signal", "signal response", "power spectrum"]="signal",
                           num=5, plot=True, plot_welch_average=True):
        """

        :param plot_welch_average:
        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
        :return:
        """
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

            if plot_welch_average:
                _, k_lengths, power_spectrum = unpickle_me_this(
                    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                    absolute_path=True)
                k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
                spectrum_welch = power_spectrum.val[1:]

                plt.plot(k_lengths, spectrum_welch, label=r"Empirical estimate of $p(k)$", color="orange")

            # plt.ylim(-3e-9, 1.4e-7)
            plt.loglog()
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        for sl in samples:
            plt.plot(x, sl)

        usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}")


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
        plt.plot(self.t_ds, self.d, label="Data", color="orange")

        usual_plot()



    def plot_posterior_power_spectrum(self, print_posterior_parameters=False, plot_welch_average=True):
        ps_mean_std, _, _ = self.get_posterior_statistics(print_posterior_parameters)

        plt.errorbar(self.k_signal[1:], ps_mean_std[0][1:], yerr=ps_mean_std[1][1:],
                     label=r"Reconstructed power spectrum (with $1\sigma$ contour)", ecolor=light_blue, color=blue)

        print("Zeromode P_s(k=0) excluded in plot")  # because ~0 and then changes the y limits such that the majority
        # of the power spectrum lies in just the upper half of the coordinate system

        if plot_welch_average:
            _, k_lengths, power_spectrum = unpickle_me_this(
                "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                absolute_path=True)
            k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
            spectrum_welch = power_spectrum.val[1:]

            plt.plot(k_lengths, spectrum_welch, label="Empirical estimate", color="orange")

        # plt.ylim(-3e-9, 1.4e-7)
        plt.loglog()
        usual_plot(xl="Frequency $f$", yl="Power")


    def plot_posterior_harmonic_xi_s(self, multiply_with_posterior_amp_spec=False):
        posterior_latent_sl = self.posterior_xi_samples
        posterior_latent_mean_std = jft.mean_and_std(posterior_latent_sl)
        posterior_latent_mean, _ = posterior_latent_mean_std

        posterior_xi_s_mean = posterior_latent_mean[f"s_xi"]

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


def visualize_stress(stress_matrix, rows, cols, tl="", hlines=None, vlines=None):

    stress_matrix = stress_matrix.real

    cols_are_increasing = np.all(np.diff(cols) > 0)  # strictly increasing
    rows_are_increasing = np.all(np.diff(rows) > 0)  # strictly increasing
    if not cols_are_increasing:
        raise ValueError("Columns must be increasing")
    if not rows_are_increasing:
        stress_matrix = np.fft.fftshift(stress_matrix, axes=0)  # shift DC frequency to middle
        print("\t\tRows must be increasing, assuming a priori standard DFT order and moving DC to the middle")

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


def fieldify(array, dom):
    return ift.Field(dt_(dom), array)

def dt_(dom):
    return ift.DomainTuple.make(dom)