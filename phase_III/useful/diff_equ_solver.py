import jax.numpy as jnp
import jax
from scipy.signal.windows import tukey

jax.config.update("jax_enable_x64", True)
import nifty.nifty.re as jft
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5


def interpolation_based_stochastic_differential_equation(t, y, args):
    """
    Call (t,y args) where args = (omega_interpol, gamma_interpol, xi_interpol).

    :param t:                               The current/next time step to solve for.
    :param y:                               The current state vector (waveform and its derivative).
    :return:
    """
    h, u = y
    omega_interpol, gamma_interpol, xi_interpol = args
    omega_i, gamma_i, xi_i = (f(t) for f in (omega_interpol, gamma_interpol, xi_interpol))

    dh_dt = u
    du_dt = xi_i - gamma_i * u - omega_i**2 * h
    return jnp.array([dh_dt, du_dt])


def euler(times, omega, gamma, xi):
    dt = times[1] - times[0]

    def step(state, params):
        h, u = state
        ω, γ, ξ = params
        dh = u
        du = ξ - γ*u - ω**2*h
        h_new = h + dt*dh
        u_new = u + dt*du

        return (h_new, u_new), h_new

    params = jnp.stack([omega, gamma, xi], axis=1)
    (_, _), h_traj = jax.lax.scan(step, (0.0, 0.0), params)
    return h_traj


def rk4(times, omega, gamma, xi):
    dt = times[1] - times[0]

    def rhs(state, params):
        h, u = state
        ω, γ, ξ = params
        dh = u
        du = ξ - γ*u - ω**2*h
        return jnp.array([dh, du])

    def step(state, params):
        y = jnp.array(state)

        k1 = rhs(y, params)
        k2 = rhs(y + 0.5*dt*k1, params)
        k3 = rhs(y + 0.5*dt*k2, params)
        k4 = rhs(y + dt*k3, params)

        y_new = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        return (y_new[0], y_new[1]), y_new[0]

    params = jnp.stack([omega, gamma, xi], axis=1)
    (_, _), h_traj = jax.lax.scan(step, (0.0, 0.0), params)

    return h_traj


def diffrax_solver(evolution_times, omega, gamma, xi):
    """
    :param evolution_times: jnp.array,     the time support points of the data.
    :return: A solver object that needs an arg list (not unpacked) as input
    """
    vector_field = ODETerm(interpolation_based_stochastic_differential_equation)

    solver = Tsit5()
    t0 = evolution_times.min()
    t1 = evolution_times.max()
    dt0 = evolution_times[1]-evolution_times[0]
    saveat = SaveAt(ts=evolution_times)

    omega_interpol = lambda x: jnp.interp(x, evolution_times, omega)
    gamma_interpol = lambda x: jnp.interp(x, evolution_times, gamma)
    xi_interpol = lambda x: jnp.interp(x, evolution_times, xi)
    args = (omega_interpol, gamma_interpol, xi_interpol)

    return diffeqsolve(vector_field, solver, t0, t1, dt0=dt0, args=args, y0=jnp.array([.0,.0]), saveat=saveat,
                       max_steps=2*4096).ys[:, 0]



class AutoDiffEquationSolver(jft.Model):
    def __init__(self, prefix, reconstruction_times, cfm_sampling_times, omega_cfm, gamma_cfm, xi_cfm, scaling_constant,
                 solver=rk4, tukey_window=False, alpha_dont_change_default_for_legacy_reasons=0.3):

        self.prefix = prefix
        self.reconstruction_times = reconstruction_times
        self.cfm_sampling_times = cfm_sampling_times
        self.evolution_times = self.cfm_sampling_times
        self.solver = solver

        # self.N = len(reconstruction_times)
        # self.M = len(cfm_sampling_times)
        # if self.M >= self.N:
        #     print("Assuming cfm 0 time = reconstruction 0 time and cutting away last ", self.M-self.N, " points of "
        #           "all cfm realizations.")
        #     self.evolution_times = self.cfm_sampling_times[:self.N]
        # else:
        #     raise ValueError("? what did you do")
        self.omega_cfm = omega_cfm
        self.gamma_cfm = gamma_cfm
        self.xi_cfm = xi_cfm
        # self.diffeq_solver = diffrax_solver(self.cfm_sampling_times, self.reconstruction_times)
        self.dom = self.omega_cfm.domain | self.gamma_cfm.domain | self.xi_cfm.domain

        self.targ = jax.ShapeDtypeStruct(shape=(len(self.reconstruction_times),), dtype=jnp.float64)
        # if envelope_op:
        #     self.envelope_op = envelope_op
        #     self.dom |= self.envelope_op.domain
        # else:
        #     self.envelope_op = lambda *args: jnp.ones(self.N, dtype=jnp.float64)

        # if scaling_constant:
        self.scaling = jft.LogNormalPrior(*scaling_constant, name="scaling_")
        self.dom |= self.scaling.domain
        # else:
        #     self.scaling = lambda *args: 1

        if tukey_window:
            self.win = tukey(len(self.evolution_times), alpha=alpha_dont_change_default_for_legacy_reasons)
        else:
            self.win = 1

        super().__init__(domain=self.dom, target=self.targ)

    def __call__(self, p):
        omega = self.omega_cfm(p)
        gamma = self.gamma_cfm(p)
        xi = self.xi_cfm(p)
        # env = self.envelope_op(p)[:self.N]
        scaling = self.scaling(p)

        # waveform = euler(times=self.evolution_times[::self.step], omega=omega, gamma=gamma, xi=xi)
        waveform = self.solver(times=self.evolution_times, omega=omega, gamma=gamma, xi=xi)
        return waveform * scaling * self.win


    def get_model_components(self):
        return lambda x: jnp.nan, {"none" : 1}, self.prefix


class DomainCheckAndMask:
    def __init__(self, domain_time, target_time):
        """
        Takes the input and cuts away all points that don't lie exactly within target_time. If the resulting array
        is not of the length of target_time, an error is raised.
        :param domain_time:     jnp.array, for example a finely resolved correlated field time domain array
        :param target_time:     jnp.array, for example a coarse time domain corresponding to a strain series
        """
        self.domain = domain_time
        self.target = target_time

        start_points_aligned = (self.domain[0] == self.target[0])
        if not start_points_aligned:
            raise ValueError(f"First and last element of {self.domain} and {self.target} must be aligned.")

        self.N_dom = len(self.domain)
        self.N_target = len(self.target)

        self.ratio = self.N_dom // self.N_target
        self.masked_domain_array = self.domain[::self.ratio]

        successful_masking = jnp.array(self.masked_domain_array == self.target)
        if not jnp.all(successful_masking):
            raise ValueError(f"Masking procedure failed: self.masked_domain_array {self.masked_domain_array}\n must"
                             f"be equal to {self.target} but isn't")


    def __call__(self, p):
        return p[::self.ratio]