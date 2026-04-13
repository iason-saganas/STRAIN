from functools import reduce
from typing import Literal

from scipy.signal.windows import tukey

import nifty.re as jft
import diffrax
import jax.numpy as jnp
import jax

from phase_II.nifty_re_playground.strain_tools import raise_warning, Stress_jft, visualize_stress

from .inference import StochasticOscillatorPrior

from phase_III.useful.helpers import draw_and_plot_field_realizations, create_cfm

__all__ = ["HarmonicOscillator"]

def rhs_full(state, params):
    h, u = state
    ω, γ, ξ = params
    dh = u
    du = ξ - γ*u - ω**2*h
    return jnp.array([dh, du])

# Also define some important subcases for debugging and testing purposes
def rhs_simple_ho(state, params):
    h, u = state
    ω, _, _ = params
    dh = u
    du = - ω ** 2 * h
    return jnp.array([dh, du])


def rhs_driven_simple_ho(state, params):
    h, u = state
    ω, _, ξ = params
    dh = u
    du = ξ - ω**2*h
    return jnp.array([dh, du])


def rhs_damped_simple_ho(state, params):
    h, u = state
    ω, γ, _ = params
    dh = u
    du = - γ*u - ω**2*h
    return jnp.array([dh, du])


_differential_equations = dict(rhs_driven_simple_ho=rhs_driven_simple_ho,rhs_damped_simple_ho=rhs_damped_simple_ho,
                               rhs_full=rhs_full, rhs_simple_ho=rhs_simple_ho)


def rk4(times, omega, gamma, xi, y0, dfe):
    # dfe is a string, see `dfe` argument of `HarmonicOscllator`
    dt = times[1] - times[0]

    rhs = _differential_equations[dfe]

    def step(state, params):
        y = jnp.array(state)

        k1 = rhs(y, params)
        k2 = rhs(y + 0.5*dt*k1, params)
        k3 = rhs(y + 0.5*dt*k2, params)
        k4 = rhs(y + dt*k3, params)

        y_new = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        return (y_new[0], y_new[1]), y_new[0]

    params = jnp.stack([omega, gamma, xi], axis=1)
    (_, _), h_traj = jax.lax.scan(step, y0, params)

    return h_traj


def rk4_sample_t0(times, omega, gamma, xi, y0, t0, dfe):
    dt = times[1] - times[0]

    rhs = _differential_equations[dfe]

    # Ensure y0 is float64 so dtypes match across cond branches
    y0 = jnp.array(y0, dtype=jnp.float64)


    def step(carry, inputs):
        state, activated = carry
        params_t, t = inputs

        state, activated = jax.lax.cond(
            (t >= t0) & (~activated),
            lambda _: ((y0[0], y0[1]), True),
            lambda _: (state, activated),
            operand=None
        )

        y = jnp.array(state)

        k1 = rhs(y, params_t)
        k2 = rhs(y + 0.5 * dt * k1, params_t)
        k3 = rhs(y + 0.5 * dt * k2, params_t)
        k4 = rhs(y + dt * k3, params_t)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return ((y_new[0], y_new[1]), activated), y_new[0]

    params = jnp.stack([omega, gamma, xi], axis=1)
    # Match init_carry dtype to y0
    init_carry = ((jnp.float64(0.0), jnp.float64(0.0)), False)

    (_, _), h_traj = jax.lax.scan(step, init_carry, (params, times))

    return h_traj


def diffrax_solver(times, omega, gamma, xi, y0, dfe):
    # Diffrax needs not be feed the whole trajectory but instantenous coefficent values
    y0 = jnp.array(y0, dtype=jnp.float64)
    omega_i = diffrax.LinearInterpolation(ts=times, ys=omega)
    gamma_i = diffrax.LinearInterpolation(ts=times, ys=gamma)
    xi_i = diffrax.LinearInterpolation(ts=times, ys=xi)

    rhs = _differential_equations[dfe]

    def df_rhs(t_instant, y_instant, args):
        omega_interp, gamma_interp, xi_interp = args
        ω = omega_interp.evaluate(t_instant)
        γ = gamma_interp.evaluate(t_instant)
        ξ = xi_interp.evaluate(t_instant)
        return rhs(y_instant, (ω, γ, ξ))

    vector_field = diffrax.ODETerm(df_rhs)

    solver = diffrax.Tsit5()
    t0 = times[0]
    t1 = times[-1]
    dt0 = times[1]-times[0]
    saveat = diffrax.SaveAt(ts=times)

    return diffrax.diffeqsolve(
            vector_field,
            solver,
            t0,
            t1,
            dt0=dt0,
            args=(omega_i, gamma_i, xi_i),
            y0=y0,
            saveat=saveat,
            max_steps=50_000).ys[:, 0]


class HarmonicOscillator(jft.Model):
    def __init__(self, signal_domain_times, signal_prior:StochasticOscillatorPrior, normalize=False,
                 solver:Literal['rk4', 'rk4_init_condition_sampler', 'diffrax']='rk4',
                 dfe:Literal['rhs_full', 'rhs_simple_ho', 'rhs_damped_simple_ho', 'rhs_driven_simple_ho']='rhs_full',
                 tukey_window_alpha=.0, cfm_envelope=None,):
        """

        Represents a harmonic oscillator of frequency omega(t), damping gamma(t) and driven by xi_force(t).

        :param signal_domain_times:     The time points of the signal domain over which the oscillation solution
                                        is evaluated.
        :param signal_prior:            An instance of `StochasticOscillatorPrior` containing information on the
                                        priors and generative models.
        :param normalize:               If True, wavelet will be divided by its max such that the maximum amplitude
                                        is controlled by the inferred scaling factor.
        :param solver:                  The solver to use.
        :param dfe:                     The right hand side of the differential equation to use.
        :param tukey_window_alpha:      The tukey shape parameter used in the oscillator model on the final waveform.
                                        Default: no taper.
        :param cfm_envelope:            If a jft.Model (correlated field), will be multiplied onto final waveform as
                                        a non-parametric envelope. Use for development purposes.

        """

        self.prefix = "h_"
        self.evolution_times = signal_domain_times
        self.N_ss = len(self.evolution_times)

        if solver == 'rk4':
            self.solver = lambda **kwargs: rk4(**kwargs, y0=signal_prior.y0, dfe=dfe)
        elif solver == 'rk4_init_condition_sampler':
            self.solver = lambda **kwargs: rk4_sample_t0(**kwargs, y0=signal_prior.y0, dfe=dfe)
        elif solver == 'diffrax':
            raise_warning("manually fixing max steps to 50_000, revisit in future")
            self.solver = lambda **kwargs: diffrax_solver(**kwargs, y0=signal_prior.y0, dfe=dfe)
        else:
            raise ValueError(f"solver {solver} not recognized.")
        self.alpha = tukey_window_alpha

        self.omega = signal_prior.omega
        self.gamma = signal_prior.gamma
        self.xi_force = signal_prior.xi_force
        self.amplitude = signal_prior.amplitude
        self.win = jnp.array(tukey(len(self.evolution_times), alpha=self.alpha))

        self.dom = reduce(lambda d1, d2: d1 | d2, signal_prior.domains)
        self.targ = jax.ShapeDtypeStruct(shape=(self.N_ss,), dtype=jnp.float64)

        if cfm_envelope:
            log_env = create_cfm(signal_domain_times, prefix="envelope_", offset_std=(1e-16,1e-16), offset_mean=0,
                       fluct=cfm_envelope["fluct"], llslope=cfm_envelope["llslope"])
            self.env = lambda p: jnp.exp(log_env(p))
            self.env.domain = log_env.domain
            self.dom |= log_env.domain
        else:
            self.env = lambda p: 1

        if normalize:
            # self.normalize = lambda y, x: y / jnp.trapezoid(y=y, x=x)  # integral normalization
            eps = 1e-10
            self.normalize = lambda y, x: y / (jnp.max(y)+eps)  # amplitude normalization
        else:
            self.normalize = lambda y, x: y * 1

        super().__init__(domain=self.dom, target=self.targ)

    def __call__(self, p):
        amplitude = self.amplitude(p)
        envelope = self.env(p)

        omega = self.omega(p)
        gamma = self.gamma(p)
        xi = self.xi_force(p)

        waveform = self.solver(times=self.evolution_times, omega=omega, gamma=gamma, xi=xi)
        scaled_waveform = envelope * self.normalize(y=waveform * self.win, x=self.evolution_times) * amplitude

        return scaled_waveform


    def get_model_components(self):
        return lambda x: jnp.nan, {"none" : 1}, self.prefix

    def plot_samples(self, num, key, show_spectrogram=False, **kwargs):
        for idx in range(num):
            sample_waveform, key = draw_and_plot_field_realizations(times=self.evolution_times,
                                                                    diff_eq_solver_model=self,
                                                                    omega_op=self.omega, gamma_op=self.gamma,
                                                                    xi_op=self.xi_force,
                                                                    key=key, **kwargs)
            if show_spectrogram:
                N = len(sample_waveform)
                t = self.evolution_times
                if N > 10_000:
                    print("HarmonicOscillator.plot_samples info >> downsampling waveform for calculation of stress")
                    sample_waveform = sample_waveform[::2]
                    t = t[::2]
                S, t, f = Stress_jft(sample_waveform, time=t)
                visualize_stress(stress_matrix=S, rows=f, cols=t, smooth=True)
        print("Don't forget to get key from `HarmonicOscillator.plot_samples`")
        return key

