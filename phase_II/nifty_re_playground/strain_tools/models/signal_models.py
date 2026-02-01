from nifty.nifty.re.prior import NormalPrior, LogNormalPrior, UniformPrior
from nifty.nifty.re.num import uniform_prior
from nifty.nifty.re.gauss_markov import IntegratedWienerProcess
import nifty.nifty.re as jft
from scipy.signal.windows import tukey
from functools import partial
import operator
import jax.numpy as jnp
import numpy as np
import jax
from jax.tree_util import Partial
jax.config.update("jax_enable_x64", True)

from ..basics.common_utils import fw_hartley, bw_hartley

__all__ = ["SignalModelCfmAsPowerSpectrum", "BrokenPowerLaw"]

# def tmp1(xi, amp, custom_norm_for_your_convenience=1):
# spits out a realization given a power spectrum.
# power spectrum should have length len(data), i.e. be already distributed
# return jnp.fft.ifft(custom_norm_for_your_convenience * amp * xi["s_hyper_xi"], norm="ortho").real

def _expand_rfft(f_unique, N):
    # work with unique k's and broadcast to full k's using this function.
    return np.concatenate([f_unique, f_unique[-2:0:-1].conj()]) if N % 2 == 0 else np.concatenate([f_unique, f_unique[-1:0:-1].conj()])


def _tmp2(k, p, length_of_data):
    # p = params, k fourier modes
    # assumes k[0] = 0 and np ordering of k

    slope = p[0]
    amplitude = p[1]

    tmp = np.abs(k.copy())  # negative modes are just the positive ones mirrored. If you remove abs you will get an error for slope =-1 for example, makes sense.
    tmp[0] = 1  # mask zeromode

    tmp = tmp**slope

    sorter = np.argsort(k)

    tmp = tmp / (np.trapezoid(tmp[sorter][1:], k[sorter][1:]))
    tmp = amplitude * tmp
    tmp[0] = 1e-30  # fix zeromode

    if not np.all(tmp >=0 ):

        raise ValueError("Power spectrum cannot be negative, p_s(k) = ", tmp)

    return _expand_rfft(tmp, length_of_data)


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

        dl_mean_hyper_prior_power_spectrum = _tmp2(self.k, [-10, 1e3], self.N_ss)
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


def cfm_hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes=axes)
    c = "non_canonical_hartley"
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub
    return add_or_sub(tmp.real, tmp.imag)


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
                 envelope_loglogavgslope: tuple | float | None = (-4 ,1),
                 flexibility: tuple | float | None = None,
                 model_prefix="s_"):
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
                      envelope_fluctuations, envelope_loglogavgslope, flexibility]
        parameter_names = ["pl_slope_left", "pl_slope_right", "k_break", "fluctuations", "peak_power", "sigmoid_width",
                           "envelope_fluctuations", "envelope_loglogavgslope", "flexibility"]
        parameter_names = [f"{model_prefix}{name}" for name in parameter_names]
        distributions = [LogNormalPrior, NormalPrior, UniformPrior, LogNormalPrior, NormalPrior, UniformPrior,
                         LogNormalPrior, NormalPrior, LogNormalPrior]


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
        self.log_vol = self.signal_grid.harmonic_grid.log_volume
        self.relative_log_mode_lengths = self.signal_grid.harmonic_grid.relative_log_mode_lengths
        self._deviations = None

        self.ps = partial(self._create_power_spectrum, callables_list)
        self.amp = lambda xi: jnp.sqrt(self.ps(xi))

        single_ps_domains = [parameter.domain for name, parameter in zip(parameter_names, callables_list) if
                             hasattr(parameter, "domain")]
        self.ps_domain = {}
        for dict_domain in single_ps_domains:
            self.ps_domain |= dict_domain

        self.xi_s = jft.NormalPrior(mean=0, std=1, name=f"{model_prefix}xi",
                                    shape=jft.ShapeWithDtype(shape=self.shp, dtype=jnp.float64),)
        self.set_deviations(flexibility)

        if apply_cf_env:
            self.env = self._create_cf_env(envelope_fluctuations, envelope_loglogavgslope)
            if self._deviations is not None:
                self.total_dom = self.ps_domain | self.env.domain | self.xi_s.domain | self._deviations.domain
            else:
                self.total_dom = self.ps_domain | self.env.domain | self.xi_s.domain
        else:
            self.env = None
            if self._deviations is not None:
                self.total_dom = self.ps_domain | self.xi_s.domain | self._deviations.domain
            else:
                self.total_dom = self.ps_domain | self.xi_s.domain

        self.win = tukey(self.N, alpha=0.3)

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
         _, _, _) = variable_list

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

        # sigmoid = 1/2 + 1/2*jnp.tanh(sigmoid_width*(k-k_break))
        rel_k = k/k_break
        sigmoid = 1/2 * (1 + jnp.tanh(sigmoid_width*jnp.log(rel_k)))
        anti_sigmoid = 1 - sigmoid

        # ps = a_break * exp(pl_slope_left * l) * anti_sigmoid + b_break * exp(pl_slope_right * l) * sigmoid
        ps = peak_power * (rel_k**pl_slope_left * anti_sigmoid + rel_k**pl_slope_right * sigmoid)

        if self._deviations is not None:
            log_iwp_spectrum = self._log_iwp(p)[1:]  # mask 0 value
            ps = ps * jnp.exp(log_iwp_spectrum)


        # print("WARNING: >> BrokenPowerLaw model uses improper normalization in the ps right now. Please revert \n by uncommenting this line in the future.")
        # print("WARNING >> Please revert to prepending the zm AFTER integrating")
        # print("WARNING >> Please replace 0-mode with 1e-10 instead of 1e-32.")
        # ps = ps / jnp.sum(ps) # to match nifty8 fw model

        area = dk * jnp.sum(ps)
        ps = ps / area * fluctuations**2



        ps = jnp.concatenate([jnp.array([zm]), ps])  # prepend the zero-mode
        return ps


    def _log_iwp(self, primals):

        twolog = self._deviations(primals)

        twolog = jnp.concatenate((jnp.zeros((1,)), twolog[:, 0]))
        detrended_log_iwp = self._remove_slope(self.relative_log_mode_lengths, twolog)

        return detrended_log_iwp


    def _remove_slope(self, rel_log_mode_dist, x):
        sc = rel_log_mode_dist / rel_log_mode_dist[-1]
        return x - x[-1] * sc


    def set_deviations(self, flex):
        if flex is not None:
            if not isinstance(flex, tuple):
                flex = LogNormalPrior(flex, 1e-16, name=f"{self.model_prefix}flexibility",)
            else:
                flex = LogNormalPrior(*flex, name=f"{self.model_prefix}flexibility",)

            self._deviations = IntegratedWienerProcess(
                jnp.zeros((2,)),
                flex,
                self.log_vol,
                name=self.model_prefix + "spectrum",
                asperity=1e-16,
                hack_make_pos_definite=False,
            )



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


    def __call__(self, p):
        xi_s_real = self.xi_s(p)
        fourier_xi_s = fw_hartley(xi_s_real, norm="ortho")

        amplitude_realization = self.amp(p)[self.expander]
        kernel = amplitude_realization * jnp.sqrt(self.h_vol) * fourier_xi_s

        wavelet = bw_hartley(kernel, norm="ortho")

        if self.env is not None:
            env_realization = self.env(p)
            wavelet = env_realization * wavelet
        return wavelet * self.win

    # def __call__(self, p):
    #     # call to precisely match nifty8 implementation
    #     xi_s = self.xi_s(p)
    #
    #     # fourier_xi_s = fw_hartley(xi_s_real, norm="ortho")  # also I think that fw_hartley(xi) is just another
    #     # xi the way I coded fw_hartley, so skip unnecessary computation in the future
    #
    #     amplitude_realization = self.amp(p)[self.expander]
    #     kernel = amplitude_realization * xi_s
    #
    #     # wavelet = bw_hartley(kernel, "ortho") * self.h_vol
    #     wavelet = cfm_hartley(kernel) * self.dk  # to match cfm convention
    #
    #     if self.env is not None:
    #         env_realization = self.env(p)
    #         wavelet = env_realization * wavelet
    #     return wavelet


    def get_model_components(self):
        amplitude_op = self.amp
        parameter_choices = self.parameter_choices
        model_prefix = self.model_prefix
        return amplitude_op, parameter_choices, model_prefix

