from data.style_components.matplotlib_style import *
import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import nifty.nifty.re as jft
import jax
import warnings


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
        self.n_ss = (self.L_ss / self.dist_ss).astype(int)

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

        self.amplitude_op = None
        self.s_model = None
        self.inv_N_cov = None
        self.kl_kwargs = None
        self.nonlinearly_update_kwargs = None
        self.draw_linear_kwargs = None
        self.posterior_xi_samples = None


    def add_custom_signal_model(self, custom_signal_op):
        raise ValueError("Not implemented yet. Be sure to implement an .init method/property to be fed into \n"
                         "jft.Model __init__, like init=cfm_maker.finalize().init")


    def add_cfm_signal_model(self, fluct:tuple, llslope:tuple, flex:tuple | None = None, asper:tuple | None=None,
                             offset_mean:float = 0, offset_std:tuple = (1e-16, 1e-16)):

        cfm_maker = jft.CorrelatedFieldMaker(prefix="s_")
        cfm_maker.set_amplitude_total_offset(offset_mean, offset_std)
        cfm_maker.add_fluctuations(shape=(self.n_ss,), distances=self.dist_ss, fluctuations=fluct,
                                   loglogavgslope=llslope, flexibility=flex, asperity=asper, harmonic_type="fourier",
                                   non_parametric_kind="power")

        s_model = cfm_maker.finalize()
        self.s_model = s_model
        self.amplitude_op = cfm_maker.amplitude


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

        # Arguements for the minimizer in the nonlinear updating of the samples
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

        self.posterior_xi_samples = samples
        print("\nSaved posterior latent samples as self.posterior_xi_samples")
        return samples


    def get_current_key(self):
        """
        Should always be called after calling .run_inference().
        """
        return self.key

    def report_statistics(self):
        if self.posterior_xi_samples is None:
            raise ValueError("Call 'run_inference()' before reporting statistics.")

        ps_amp_spec_samples = self.amplitude_op.force(self.posterior_xi_samples)


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
        elif mode == "amplitude spectrum":
            op = self.amplitude_op
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
            xl = r"Unique $\omega$"
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


    def plot_prior_samples(self, mode="signal", num=5, plot=True):
        """

        :param plot:    Whether to plot the prior or not. If not, only mean and std are returned.
        :param num:     Number of samples to compute mean and std from.
        :param mode:    Either 'signal' (default) or 'signal response' (lives in data space) or 'amplitude spectrum'.
        :return:
        """
        samples = self.get_prior_samples(mode=mode, num=num)

        if not plot:
            return samples

        xl = "Time"
        yl = "Strain"
        if mode == "signal":
            x = self.t_ss
        elif mode == "signal response":
            x = self.t_ds
        elif mode == "amplitude spectrum":
            x = self.k_signal
            xl = r"Unique $\omega$"
            yl = r"$\sqrt{\mathrm{Power}}$"
            plt.loglog()
        else:
            raise ValueError("Unknown mode '{}'".format(mode))

        for sl in samples:
            plt.plot(x, sl)

        usual_plot(xl=xl, yl=yl, title=f"Prior samples: {mode}")