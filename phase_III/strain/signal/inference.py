from dataclasses import dataclass
from matplotlib import pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import welch as scipy_welch
from phase_II.nifty_re_playground.strain_tools import *
import numpy as np
import jax.numpy as jnp
import os
from typing import Literal, Any, Callable
import nifty.re as jft

from phase_III.strain.helpers import _save_plot_wh_bp_data_with_template, _save_plot_welch_average, _metadata_basics, _error_metadata
from phase_III.useful.helpers import *

__all__ = ["StrainSignalInference", "StochasticOscillatorPrior", "Mask", "_add_peak", "_multiply_op_2_to_op_1"]

class StochasticOscillatorPrior:
    def __init__(self, prior_dict, signal_time_domain, localize_force=(0., 0.5), couple_force_to_frequency=True):
        """
        Holds information on prior distribution on hyperparameters, assuming these are later fed into a stochastic
        harmonic oscillator solver and constructs correlated fields for the frequency, damping and force fields;
        sets a lognormal prior on the global amplitude.

        Usage:

        oscillator_prior_dct = {
            "frequency": {"offset_mean": 100, "offset_std": (50, 1e-16), "fluctuations": (1, 1), "loglogavgslope": (-2, 1e-16)},
            "damping": {"offset_mean": 25, "offset_std": (5, 1e-16), "fluctuations": (1e-16,1e-16), "loglogavgslope": (-2, 1e-16)},
            "force": {"offset_mean": 0, "offset_std": (1e-16, 1e-16), "fluctuations": (1e1, 1e1), "loglogavgslope": (0, 1)},
            "global_amplitude": (1e1, 1e1),
            "init_conditions": (.0,.0)
        }

        oscillator_prior = StochasticOscillatorPrior(oscillator_prior_dct)

        :param prior_dict: A dictionary containing keys for the frequency, damping and force field of the stochastic
                           harmonic oscillator solver, as well as the initial force value and global amplitude scaling.
                           Except for the offset_mean paramater and the initial_force parameter, all values should be
                           tuples rather than scalars, denoting prior mean and standard deviations of the parameters.
                           Example:

        :param signal_time_domain:  An array containing the sampling times over which the correlated fields are
                                    defined. Integration will be performed over this domain.


        """
        translated_prior_dct = prior_dict.copy()
        frequency_dct = translated_prior_dct["frequency"]
        translated_offset_mean = jnp.log(frequency_dct['offset_mean'] ** 2)
        translated_offset_std = jnp.abs(
            translated_offset_mean - 2 * jnp.log(frequency_dct['offset_mean'] + frequency_dct['offset_std'][0]))
        translated_prior_dct["frequency"] = {"offset_mean": translated_offset_mean,
                                             "offset_std": (translated_offset_std,
                                                            frequency_dct['offset_std'][1]),
                                             "fluctuations": frequency_dct['fluctuations'],
                                             "loglogavgslope": frequency_dct['loglogavgslope']}

        a = jnp.float64(jnp.round(frequency_dct['offset_mean'],2))
        b = jnp.float64(jnp.round(translated_prior_dct['frequency']['offset_mean'],2))
        c = jnp.float64(jnp.round(translated_prior_dct['frequency']['offset_std'][0],2))
        d = jnp.float64(jnp.round(translated_prior_dct['frequency']['offset_std'][1],2))

        print(f"\nYou are trying to set up the frequency prior for the oscillator with mean "
              f"{a}±{(frequency_dct['offset_std'])}. To ensure positivity, \nwe are exponentiating "
              f"internally and therefore changing the mean and standard deviation values to \n",
              f"{b}±({c,d}).")
        print(
            "If you are unsure this has the desired effect, check samples via `StochasticOscillatorPrior.plot_omega_samples`.")

        self.N = len(signal_time_domain)
        self.prior_dct = translated_prior_dct
        self.cfm_times = signal_time_domain
        self.localize_force = localize_force
        self.couple_force_to_frequency = couple_force_to_frequency

        # Fields to set
        self.omega, self.gamma, self.xi_force, self.amplitude, self.y0, self.domains = self._get_priors_from_dict()


    def plot_omega_samples(self, key):
        print("Ignoring the standard deviation on the standard deviation on offset_mean. You likely want to set this "
              "to 1e-16.")
        k = plot_histogram(key, mean=self.prior_dct["frequency"]["offset_mean"],
                       sigma=self.prior_dct["frequency"]["offset_std"][0], n_samples=500, mode="Lognormal",
                       apply_func=lambda s: jnp.sqrt(np.exp(s)), apply_func_descriptive_string=r"applying $\sqrt{e^s}$ on each sample $s$")
        print("Don't forget to get the key after calling this function.")
        return k

    def _get_priors_from_dict(self):
        if not 'init_condition' in self.prior_dct.keys():
            raise ValueError(r'Provide (h_0, \dot{h}_0) for integration.')

        y0 = self.prior_dct["init_condition"]

        names = ["frequency", "damping", "force", "global_amplitude"]
        missing = [n for n in names if n not in self.prior_dct]

        operator_container: dict[str, Callable | None] = {
            "frequency": None,
            "damping": None,
            "force": None,
            "global_amplitude": None,
        }
        valid_domains = []

        for op_name in names:
            if op_name == 'global_amplitude':
                # not a cfm so handle separately
                if op_name in missing:
                    op = lambda p: p * 1
                else:
                    op = jft.LogNormalPrior(*self.prior_dct["global_amplitude"], name="global_amplitude")
                    valid_domains.append(op.domain)
                operator_container[op_name] = op
            else:
                if op_name in missing:
                    op = lambda p: jnp.zeros(self.N)
                else:
                    prior = self.prior_dct[op_name]
                    op = create_cfm(
                        time_domain=self.cfm_times,
                        prefix=f"{op_name}_",
                        offset_std=prior["offset_std"],
                        offset_mean=prior["offset_mean"],
                        fluct=prior["fluctuations"],
                        llslope=prior["loglogavgslope"],
                        flex=prior.get("flex", None)  # if no flex provided set to None
                    )

                    if op_name == "frequency":
                        # make omega from sq_log_omega
                        log_sq_omega: Callable = op
                        op = lambda p: jnp.clip(jnp.sqrt(jnp.exp(log_sq_omega(p))), -jnp.inf, 1e4)
                        # noinspection PyUnresolvedReferences
                        op.domain = log_sq_omega.domain
                    valid_domains.append(op.domain)
                operator_container[op_name] = op


        omega, gamma, xi_force, amp = [operator_container[name] for name in names]

        if self.couple_force_to_frequency:
            xi_force = _multiply_op_2_to_op_1(xi_force, omega)
        if self.localize_force is not None:
            xi_force, new_dom = _add_peak(xi_force, times=self.cfm_times, prior=self.localize_force)
            valid_domains.append(new_dom)

        return omega, gamma, xi_force, amp, y0, valid_domains


def _add_peak(op, times, prior):
    t = times
    t0 = jft.NormalPrior(mean=prior[0], std=prior[1], name="t0")
    # sig = jft.NormalPrior(mean=.1, std=1e-16, name="sig")
    # t0 = 0
    sig = .1
    A = 1
    peak_model = lambda p: A*jnp.exp(-(t - t0(p))**2/(2*sig**2))

    op_tmp = lambda p: peak_model(p) * op(p)
    # op_tmp.domain = op.domain | t0.domain | sig.domain
    op_tmp.domain = op.domain | t0.domain
    return op_tmp, op_tmp.domain

def _multiply_op_2_to_op_1(op1, op2, e=2):
    # op_1_tmp = lambda p: op1(p)*op2(p)/jnp.max(op2(p))  # normed
    op_1_tmp = lambda p: op1(p)*op2(p)**e #  non-normed and squared
    # op_1_tmp = lambda p: op1(p)*op2(p) #  non-normed
    op_1_tmp.domain = op1.domain | op2.domain
    return op_1_tmp

class StrainSignalInference:
    def __init__(self,
                 key,
                 e_fac,
                 r_fac,
                 event_name="GW150914",
                 detector:Literal["H1", "L1"]="H1",
                 data_duration_of_hdf5_file:Literal["32sec", "4096sec"]="4096sec",
                 stationarity_time_scale=32,
                 custom_gps_center=None,
                 diagnostic_plots=True,
                 alpha_taper_on_data=.1,
                 out_name=''
                 ):
        """

        A thin wrapper around the `InferenceSchemeRe` class.

        :param key:                             The initial jax random key.
        :param e_fac:                           By how much to extend the domain to avoid boundary FFT artifacts
        :param r_fac:                           How much more resolved the signal time domain is w.r.t. the data domain.
        :param event_name:                      The unique GW event identifier that you wish to reconstruct.
        :param detector:                        The interferometer ID. Supported: L1, H1
        :param data_duration_of_hdf5_file:      The duration of the hdf5 file containing the strain data
        :param stationarity_time_scale:         The duration of the data to crop to, in order to calculate the Welch-
                                                average. In other words, assumed stationarity timescale.
                                                Time series will be centered around event. In seconds.
        :param custom_gps_center:               If None, default gps time found in readme file of strain data will be
                                                used as center. This default will only approximately be the merger time.
        :param diagnostic_plots:                If true, diagnostic plots will be created and stored.
        :param alpha_taper_on_data:             Data has to be periodic due to how the noise operator is built.
                                                This is the shape parameter of a Tukey window.
        """
        print("Assumed noise stationarity timescale for Welch-average: ", stationarity_time_scale, " seconds.")

        odir_main = f"STRAIN_{event_name}_{detector}_{out_name}"
        odir_diagnostic_plots = f"{odir_main}/diagnostic_plots/"
        odir_diagnostic_metadata = f"{odir_main}/diagnostic_metadata/"
        odir_errors = f"{odir_main}/errors/"
        if diagnostic_plots:
            os.makedirs(odir_diagnostic_plots, exist_ok=True)
        os.makedirs(odir_diagnostic_metadata, exist_ok=True)
        os.makedirs(odir_errors, exist_ok=True)

        tapering_function = lambda d: tukey(M=len(d), alpha=alpha_taper_on_data, sym=True)
        event_data = get_strain_from_disc(event_name=event_name, detector=detector,
                                      data_duration=data_duration_of_hdf5_file, center_on_event=True,
                                      desired_duration=stationarity_time_scale, add_whitened_data=True,
                                      tapering_function=tapering_function
                                      )

        T_mini_welch = max(event_data.event_time) - min(event_data.event_time)
        T_global_welch = max(event_data.time) - min(event_data.time)

        NR = get_waveform_template(event_name=event_name, detector=detector, gps_center=event_data.gps_center,
                                   silent=False, plot=False, force_online_fetch=False, model_approximant="IMRPhenomXPHM")


        # Attach basic fields

        self.key = key
        self.NR = NR
        self.event_data = event_data
        self.T_mini_welch = T_mini_welch
        self.T_global_welch = T_global_welch
        self.alpha_taper_on_data = alpha_taper_on_data
        self.stationarity_time_scale = stationarity_time_scale  # by construction T_global_welch ?
        self.f_welch, self.ps_welch = self._get_scipy_welch()

        # Create inference scheme object
        self.r_fac = r_fac
        self.e_fac = e_fac
        d = self.event_data.event_strain
        d = d - jnp.mean(d)  # tetrend
        d = tapering_function(d) * d
        t = self.event_data.event_time
        self.machinery = InferenceSchemeRe(t=t, d=d, e_fac=self.e_fac, r_fac=self.r_fac, key=key,
                                           plotting_callback=analyze_kl_callback)

        # Misc or to be set
        self.odir_main = odir_main
        self.oscillator = None

        # Metadata
        _metadata_basics(o=odir_diagnostic_metadata, e=event_name, s=event_data, det=detector, t_m=T_mini_welch,
                         t_g=T_global_welch,
                         t_d=data_duration_of_hdf5_file)
        _error_metadata(o=odir_errors)
        if diagnostic_plots:
            existing_files = os.listdir(odir_diagnostic_plots)

            plot_1_exists = np.sum(np.array([("whitened_bp_strain" in f) for f in existing_files]).astype(int)).astype(
                bool)
            plot_2_exists = np.sum(np.array([("welch_average" in f) for f in existing_files]).astype(int)).astype(bool)

            if not plot_1_exists:
                _save_plot_wh_bp_data_with_template(o=odir_diagnostic_plots, s=event_data, nr=NR, e=event_name)
            if not plot_2_exists:
                _save_plot_welch_average(o=odir_diagnostic_plots, f=self.f_welch, p=self.ps_welch)


    def add_signal_model(self, s_model, alpha=0.):
        self.oscillator = s_model
        self.machinery.add_custom_signal_model(custom_signal_model=self.oscillator, alpha=alpha)


    def add_noise_model(self, N_inv, N_sqrt_inv, N_sqrt):
        self.machinery.add_noise_op(inverse_noise_op=N_inv, sqrt_inverse_noise_op=N_sqrt_inv, sqrt_noise_op=N_sqrt)


    def run(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False,
                     resume=True, choose_low_kl_starting_pos=False, geoVi=True, chi2_threshold=jnp.inf, max_kl_iter=None,
                      **kwargs):
        # See documentation of `InferenceSchemeRe.run_inference`
        out_name = self.odir_main + '/nifty_out/'
        latent_post_samples, vi_info =  self.machinery.run_inference(kl_iterations=kl_iterations, n_samples=n_samples,
                                                                out_name=out_name,
                                                                use_strict_minimizers=use_strict_minimizers,
                                                                choose_low_kl_starting_pos=choose_low_kl_starting_pos,
                                                                max_kl_iter=max_kl_iter, **kwargs)
        return latent_post_samples, vi_info, self.machinery.get_current_key()


    def _get_scipy_welch(self):
        t = np.array(self.event_data.time)
        x = np.array(self.event_data.strain)  # full, untapered data

        # Stationarity check
        T = t.max() - t.min()
        if T > self.stationarity_time_scale:
            raise ValueError("Gotten data longer than stationarity time scale.")

        x = x - np.mean(x)  # detrend
        dt = t[1] - t[0]
        fs = 1.0 / dt
        k, ps = scipy_welch(
            x=x,
            fs=fs,
            window=("tukey", self.alpha_taper_on_data),
            nperseg=len(self.event_data.event_time),
            noverlap=None,  # => Default: 50% overlap
            detrend='constant',
            scaling="density",
            return_onesided=True,  # => Values of negative frequency range added to positive
        )
        ps /= 2  # => Therefore divide by two
        return k, ps

    def visualize_results(self, add_processed_data=False, plot_template=True, plot_oscillator_samples=True, **kwargs):
        t_min = self.event_data.event_time.min()
        t_max = self.event_data.event_time.max()

        if add_processed_data:
            welch_amp = np.sqrt(self.ps_welch[self.machinery.s_h_dom_expander])
            whitened_data = whiten(self.event_data.event_strain, amp=welch_amp)
            whitened_data_bp = bandpass(x=self.event_data.event_time, y=whitened_data)
            whitened_data_bp = whitened_data_bp/whitened_data_bp.max() * self.NR.strain.max()
            pass_to_whitened = (whitened_data_bp, 'Processed data')
        else:
            pass_to_whitened = None
        if plot_template:
            pass_to_maxL_template = (self.NR.time, self.NR.strain)
        else:
            pass_to_maxL_template = None
        self.machinery.plot_posterior_signal(print_posterior_parameters=True,
                                             maxL_template_xy=pass_to_maxL_template,
                                             whitened_data=pass_to_whitened,
                                             yl=r"$h(t)$ $\mathrm{[10^{-19}]}$",
                                             **kwargs)

        if plot_oscillator_samples:
            times = self.machinery.t_ss
            osc = self.oscillator
            oscillator_operators_list = [osc.omega, osc.gamma, osc.xi_force, osc]
            key = plot_posterior(self.machinery.get_current_key(), times=times, operator_list=oscillator_operators_list,
                                 latent_samples=self.machinery.posterior_xi_samples,
                                 label_list=["omega", "gamma res", "xi", "waveform"], save_fig=False)
            return key


class Mask:
    def __init__(self, signal_model, adjoint_zero_padder):
        """
        Masks a fine-resolution domain array to align exactly with a coarser target array.

        :param signal_model:            Signal model to use.
        :param adjoint_zero_padder:     The function that cuts and downsamples, e.g. from `InferenceSchemeRe.adjoint_zp`,
                                        which internally also performs domain and target checks
        """
        self.azp = adjoint_zero_padder
        self.signal_model = signal_model
        self.domain = self.signal_model.domain
        self.get_model_components = self.signal_model.get_model_components

    def __call__(self, p):
        fw_call = self.signal_model(p)
        return self.azp(fw_call)