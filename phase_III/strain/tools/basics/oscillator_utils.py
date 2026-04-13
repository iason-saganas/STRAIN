import nifty.re as jft
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from .plotting import blue, light_blue

def create_cfm(time_domain, prefix, offset_std, offset_mean, fluct, llslope, flex=None):
    """

    :param time_domain:   To speed up the solving of the differential equation, when using an adaptive step size
                                solver, the times may be chosen quite coarse in order to not force very small steps. Further,
                                since the adaptive step size method may require evaluation in between the support points, a
                                linear interpolator must be used at some point.
    :param prefix:              The model prefix.
    :param offset_std:
    :param fluct:
    :param llslope:
    :return:
    """
    N = len(time_domain)
    dt = time_domain[1] - time_domain[0]
    cfm_maker = jft.CorrelatedFieldMaker(prefix=prefix)
    cfm_maker.set_amplitude_total_offset(offset_mean=offset_mean, offset_std=offset_std)
    cfm_maker.add_fluctuations(shape=N, distances=dt, fluctuations=fluct, loglogavgslope=llslope,
                               flexibility=flex, non_parametric_kind="power", harmonic_type="fourier")
    correlated_field = cfm_maker.finalize()
    return correlated_field


def draw_realization(operator, sampling_key):
    sampling_key, subkey = jax.random.split(sampling_key)
    sample_domain = jft.random_like(sampling_key, primals=operator.domain)
    sample = operator(sample_domain)
    return sample, sampling_key


def get_oscillator_sample(oscillator_model, key, slice_samples=False,
                           custom_latent_pos=None):
    """
    Returns random samples for the coefficients of the SDE and the final waveform.
    :param oscillator_model:    A jft.model with attributes giving access to the coefficient operators of the oscillator.
    :param slice_samples:       If true, cuts samples so they are length N and therefore non-periodic.
    :param custom_latent_pos:   If not None, uses this latent position.
    :param key:
    :return:
    """
    t = oscillator_model.evolution_times
    gamma = oscillator_model.gamma
    omega = oscillator_model.omega
    force = oscillator_model.xi_force

    if custom_latent_pos is None:
        key, subkey = jax.random.split(key)
        sample_domain = jft.random_like(subkey, primals=oscillator_model.domain)
    else:
        sample_domain = custom_latent_pos

    if slice_samples:
        N = len(t)
    else:
        N = int(gamma.target.shape[0])
    waveform_sample = oscillator_model(sample_domain)[:N]
    omega_sample = omega(sample_domain)[:N]
    gamma_sample = gamma(sample_domain)[:N]
    force_sample = force(sample_domain)[:N]
    return key, waveform_sample, omega_sample, gamma_sample, force_sample


def draw_and_plot_field_realizations(times, diff_eq_solver_model, omega_op, gamma_op, xi_op, key, plot=True,
                                     custom_latent_position=None, tl=""):

    N = len(times)

    if custom_latent_position is None:
        key, subkey = jax.random.split(key)
        sample_domain = jft.random_like(subkey, primals=diff_eq_solver_model.domain)
    else:
        sample_domain = custom_latent_position

    print("draw_and_plot_field_realizations info: >> Slicing")

    waveform = diff_eq_solver_model(sample_domain)[:N]
    omega = omega_op(sample_domain)[:N]
    gamma = gamma_op(sample_domain)[:N]
    xi = xi_op(sample_domain)[:N]


    if not plot:
        return waveform, key

    if N > 10_000:
        print(f"draw_and_plot_field_realizations info: >> downsampling for plotting (N={N}  > 10.000)")
        times = times[::3]
        waveform = waveform[::3]
        omega = omega[::3]
        gamma = gamma[::3]
        xi = xi[::3]


    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].set_title(tl)
    axs[0].plot(times, omega, label=r"$\omega$ field sample")
    axs[1].plot(times, gamma, label=r"$\gamma$ field sample")
    axs[2].plot(times, xi, label=r"$\xi$ field sample")
    axs[3].plot(times, waveform , label=r"Resulting waveform")
    # axs[3].plot(times, waveform - np.cos(4000 * (times-times[0])), label=r"abs error to analytic solution")

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()
    return waveform, key


def plot_posterior(key, times, operator_list, label_list, latent_samples, plot_prior_samples=True,
                   save_fig=False):

    print("Imposing condition <jnp.max(jnp.abs(prior_sample)) < 2 * jnp.max(jnp.abs(mean_op))> on prior samples")

    N = len(operator_list)
    M = len(times)
    fig = plt.figure()

    for idx, operator in enumerate(operator_list):

        operator_samples = jnp.array([operator(xi) for xi in latent_samples])
        mean_op = jnp.mean(operator_samples, axis=0)
        op_std = jnp.std(operator_samples, axis=0)

        ax = fig.add_subplot(N, 1, idx + 1)

        if len(mean_op) < len(times):
            if save_fig:
                lb = None
            else:
                lb = "posterior " + label_list[idx]
            ax.plot(times[:len(mean_op)], mean_op, label=lb, color=blue, lw=3)
        else:
            if save_fig:
                lb = None
            else:
                lb = "posterior " + label_list[idx]
            ax.plot(times, mean_op[:M], label=lb, color=blue, lw=3)

            # shaded 1-sigma region
            plt.fill_between(times,
                             mean_op - op_std,
                             mean_op + op_std,
                             color=light_blue,
                             alpha=0.7)  # transparency

        if plot_prior_samples:
            for _ in range(3):
                key, subkey = jax.random.split(key)
                try:
                    rnd_domain = jft.random_like(key, operator.domain)
                except AttributeError:
                    continue
                prior_sample = operator(rnd_domain)

                if jnp.max(jnp.abs(prior_sample)) < 2 * jnp.max(jnp.abs(mean_op)):

                    if len(prior_sample) < len(times):
                        ax.plot(times[:len(prior_sample)], prior_sample, color="black", alpha=0.3)
                    else:
                        ax.plot(times, prior_sample[:M], color="black", alpha=0.3)

        ax.legend()
    fig.axes[0].set_title("Posterior fields (some prior samples in gray)")
    fig.axes[0].sharex(fig.axes[1])
    fig.axes[1].sharex(fig.axes[2])
    fig.axes[2].sharex(fig.axes[3])
    plt.tight_layout()
    if save_fig:
        plt.savefig("posterior.pdf")
    plt.show()

    return key