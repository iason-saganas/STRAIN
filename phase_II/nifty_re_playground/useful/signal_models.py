from nifty.nifty.re.prior import NormalPrior, LogNormalPrior, UniformPrior
from nifty.nifty.re.num import uniform_prior
import nifty.nifty.re as jft
from functools import partial
import operator
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def cfm_hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes=axes)
    c = "non_canonical_hartley"
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub
    return add_or_sub(tmp.real, tmp.imag)


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

from jax.tree_util import Partial
class ExponentialPrior(jft.WrappedCall):
    def __init__(self, *args, name):
        """

        :param *arg:        Positional arguments containing. The first one is the mean of the exponential distribution.
                            All other entries will be ignored.
        """
        mean, _ = args
        exponential_prior = Partial(self._exp_call, mean)
        super().__init__(call=exponential_prior, white_init=True, name=name)

    def _exp_call(self, mean, xi):
        create_uniform_variable = uniform_prior(a_min=0, a_max=1)
        return -mean * jnp.log(create_uniform_variable(xi))


class BrokenPowerLaw(jft.Model):
    def __init__(self, signal_grid:jft.correlated_field.RegularCartesianGrid,
                 pl_slope_left: tuple | float  = (0, None), pl_slope_right: tuple | float = (-1, None),
                 k_break: tuple | float = (10, 200), fluctuations: tuple | float = (4, 2),
                 peak_power:tuple | float = 1000., sigmoid_width: tuple | float = 100.,
                 envelope_fluctuations: tuple | float | None = (4, 2),
                 envelope_loglogavgslope: tuple | float | None = (-4 ,1), model_prefix="s_"):
        """
        If parameters are tuples, they should give:
            - mean, std if normal or lognormal
            - min, max if uniform
            - mean, None if exponential (second entry will be ignored)

        distribution.
        If constant, parameter stays fixed.

        Creates a signal realizations based on a broken power law model:

            ps = a_star * e^{alpha * l} * [ 1 - sigmoid(k, k_star, c)] + sigmoid(k, k_star, c) * b_star * e^{beta * l}
            ps /= integral(ps) * a

        where l are logarithmic fourier modes and the following abbreviations were used:

        pl_slope_left > alpha, pl_slope_right > beta, k_break > k_star, fluctuations > a
        peak_power > p_star and a_star = p_star *e^{-alpha * l} and similar for b_star.

        :param signal_grid:             The target grid.
        :param pl_slope_left:           The power law slope on the left side.
        :param pl_slope_right:          The power law slope on the right side.
        :param k_break:                 The fourier mode length at which peak power is reached.
        :param fluctuations:            Variance of the signal.
        :param peak_power:              The peak power at k_break.
        :param sigmoid_width:           The width of the sigmoid function on a linear scale.
        :param model_prefix:            The prefix of the model name.
        :e
        """
        print("\nInitializing signal model with broken power law.")
        self.signal_grid = signal_grid
        parameters = [pl_slope_left, pl_slope_right, k_break, fluctuations, peak_power, sigmoid_width,
                      envelope_fluctuations, envelope_loglogavgslope]
        parameter_names = ["pl_slope_left", "pl_slope_right", "k_break", "fluctuations", "peak_power", "sigmoid_width",
                           "envelope_fluctuations", "envelope_loglogavgslope"]
        parameter_names = [f"{model_prefix}{name}" for name in parameter_names]
        distributions = [ExponentialPrior, ExponentialPrior, UniformPrior, LogNormalPrior, NormalPrior, NormalPrior,
                         LogNormalPrior, NormalPrior]

        apply_cf_env = True
        if type(envelope_fluctuations) is not tuple or type(envelope_loglogavgslope) is not tuple:
            type_check = envelope_fluctuations is None and envelope_loglogavgslope is None
            apply_cf_env=False
            if not type_check:
                raise ValueError("To disable the envelope, both envelope_fluctuations and envelope_loglogavgslope\n"
                                 "must be None. Right now, they are: ", envelope_fluctuations, envelope_loglogavgslope)

        callables_list =  [self._make_callable(p, n, d) for p, n, d
                                                            in zip(parameters, parameter_names, distributions)]


        self.parameter_choices = {f"{n}": c for n, c in zip(parameter_names, callables_list) if c is not None}
        self.expander = self.signal_grid.harmonic_grid.power_distributor
        self.model_prefix = model_prefix
        self.k = self.signal_grid.harmonic_grid.mode_lengths
        self.dk = self.k[1]-self.k[0]
        self.N = self.signal_grid.shape[0]
        self.h_vol = self.N * self.dk
        self.shp = self.signal_grid.shape
        self.dist = self.signal_grid.distances

        self.ps = partial(self._create_power_spectrum, callables_list)
        self.amp = lambda xi: jnp.sqrt(self.ps(xi))

        single_ps_domains = [parameter.domain for name, parameter in zip(parameter_names, callables_list) if
                             hasattr(parameter, "domain")]
        self.ps_domain = {}
        for dict_domain in single_ps_domains:
            self.ps_domain |= dict_domain

        self.xi_s = jft.NormalPrior(mean=0, std=1, name=f"{model_prefix}xi",
                                    shape=jft.ShapeWithDtype(shape=self.shp, dtype=jnp.float64),)

        if apply_cf_env:
            self.env = self._create_cf_env(envelope_fluctuations, envelope_loglogavgslope)
            self.total_dom = self.ps_domain | self.env.domain | self.xi_s.domain
        else:
            self.env = None
            self.total_dom = self.ps_domain | self.xi_s.domain


        super().__init__(domain=self.total_dom, white_init=True)


    def _make_callable(self, parameter_information, parameter_name, distribution):
        """

        :param parameter_information:   A tuple of mean, std or similar, or a float, or None.
        :param parameter_name:          A string describing the parameter.
        :param distribution:            A jft.Model representing the prior distribution.
        :return:
        """
        if type(parameter_information) is not tuple and type(parameter_information) is not None:
            # Assuming type is some kind of float and should therefore be set as a constant function
            print(f"\tAssuming parameter {parameter_name} to have a constant value of ", parameter_information)
            return lambda *args: parameter_information
        elif parameter_information is not None:
            return distribution(*parameter_information, name=parameter_name)


    def _create_power_spectrum(self, variable_list, p):
        (call_pl_slope_left, call_pl_slope_right, call_k_break, call_fluctuations, call_peak_power, call_sigmoid_width,
         _, _) = variable_list

        # TODO: Make the transition linear in log(k_break), not in k_break (sigmoid_width * (k-k_break)) => ... ln(k)/ln(k_break)

        log = lambda x: jnp.log(x)  # natural log
        exp = lambda x: jnp.exp(x)

        # log = lambda x: jnp.log10(x)  # base-10 log
        # exp = lambda x: jnp.power(10, x)

        k = self.k[1:]  # exempt 0-mode and add power later on explicitly
        dk = k[1] - k[0]
        zm = 1e-10
        # zm = 1e-32

        # Get realizations
        k_break = call_k_break(p)
        peak_power = call_peak_power(p)
        pl_slope_left = call_pl_slope_left(p)
        pl_slope_right = call_pl_slope_right(p)
        sigmoid_width = call_sigmoid_width(p)
        fluctuations = call_fluctuations(p)

        l = log(k)
        l_break = log(k_break)

        a_break = peak_power * exp(-pl_slope_left * l_break)
        b_break = peak_power * exp(-pl_slope_right * l_break)

        sigmoid = 1/2 + 1/2*jnp.tanh(sigmoid_width*(k-k_break))
        anti_sigmoid = 1 - sigmoid
        ps = a_break * exp(pl_slope_left * l) * anti_sigmoid + b_break * exp(pl_slope_right * l) * sigmoid

        area = dk * jnp.sum(ps)

        # print("WARNING: >> BrokenPowerLaw model uses improper normalization in the ps right now. Please revert \n by uncommenting this line in the future.")
        # print("WARNING >> Please revert to prepending the zm AFTER integrating")
        # print("WARNING >> Please replace 0-mode with 1e-10 instead of 1e-32.")

        ps = ps / area * fluctuations**2

        # ps = ps / jnp.sum(ps) # to match nifty8 fw model


        ps = jnp.concatenate([jnp.array([zm]), ps])  # prepend the zero-mode
        return ps


    def _create_cf_env(self, fluct, llslope):

        cfm_maker = jft.CorrelatedFieldMaker(prefix=f"{self.model_prefix}envelope_")
        cfm_maker.set_amplitude_total_offset(0, (1e-16, 1e-16))
        cfm_maker.add_fluctuations(fluctuations=fluct, loglogavgslope=llslope,
                                   shape=self.shp, distances=self.dist, flexibility=None, asperity=None,
                                   harmonic_type="fourier", non_parametric_kind="power",
                                   hack_add_power_spectrum_template=None,
                                   hack_custom_amplitude_operators=None,
                                   hack_make_iwp_pos_definite=False)

        s_env = cfm_maker.finalize()
        log_s_env = lambda xi: jnp.exp(s_env(xi))
        log_s_env.domain = s_env.domain
        return s_env


    # def __call__(self, p):
    #     xi_s_real = self.xi_s(p)
    #     fourier_xi_s = fw_hartley(xi_s_real, norm="ortho")
    #
    #     amplitude_realization = self.amp(p)[self.expander]
    #     kernel = amplitude_realization * jnp.sqrt(self.h_vol) * fourier_xi_s
    #
    #     wavelet = bw_hartley(kernel, norm="ortho")
    #
    #     if self.env is not None:
    #         env_realization = self.env(p)
    #         wavelet = env_realization * wavelet
    #     return wavelet

    def __call__(self, p):
        # call to precisely match nifty8 implementation
        xi_s = self.xi_s(p)

        # fourier_xi_s = fw_hartley(xi_s_real, norm="ortho")  # also I think that fw_hartley(xi) is just another
        # xi the way I coded fw_hartley, so skip unnecessary computation in the future

        amplitude_realization = self.amp(p)[self.expander]
        kernel = amplitude_realization * xi_s

        # wavelet = bw_hartley(kernel, "ortho") * self.h_vol
        wavelet = cfm_hartley(kernel) * self.dk  # to match cfm convention

        if self.env is not None:
            env_realization = self.env(p)
            wavelet = env_realization * wavelet
        return wavelet


    def get_model_components(self):
        amplitude_op = self.amp
        parameter_choices = self.parameter_choices
        model_prefix = self.model_prefix
        return amplitude_op, parameter_choices, model_prefix

