from matplotlib import pyplot as plt
from scipy.signal.windows import tukey

from phase_II.nifty_re_playground.strain_tools import *
import numpy as np
import jax.numpy as jnp
import os
from typing import Literal
import nifty.nifty.re as jft

from phase_III.strain.helpers import _save_plot_wh_bp_data_with_template, _save_plot_welch_average, _metadata_basics, _error_metadata
from phase_III.useful.helpers import create_cfm

__all__ = ["StrainSignalInference", "StochasticOscillatorPrior", "Mask"]

class StochasticOscillatorPrior:
    def __init__(self, prior_dict, signal_time_domain, forceless=False):
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

        :param forceless:   Whether to include a driving force

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

        self.forceless = forceless
        self.N = len(signal_time_domain)
        self.prior_dct = translated_prior_dct
        self.cfm_times = signal_time_domain

        # Fields to set
        self.omega, self.gamma, self.xi_force = self._correlated_fields_from_dict_input()
        self.amplitude = jft.LogNormalPrior(*self.prior_dct["global_amplitude"], name="global_amplitude")
        self.y0 = self.prior_dct["init_conditions"]

    def plot_omega_samples(self, key):
        print("Ignoring the standard deviation on the standard deviation on offset_mean. You likely want to set this "
              "to 1e-16.")
        k = plot_histogram(key, mean=self.prior_dct["frequency"]["offset_mean"],
                       sigma=self.prior_dct["frequency"]["offset_std"][0], n_samples=500, mode="Lognormal",
                       apply_func=lambda s: jnp.sqrt(np.exp(s)), apply_func_descriptive_string=r"applying $\sqrt{e^s}$ on each sample $s$")
        print("Don't forget to get the key after calling this function.")
        return k

    def _correlated_fields_from_dict_input(self):

        if not self.forceless:
            names = ["frequency", "damping", "force"]
        else:
            names = ["frequency", "damping"]
        missing = [n for n in names if n not in self.prior_dct]
        if missing:
            raise ValueError(f"Missing keys in prior_dct: {missing}")

        cfm_dicts = {k: self.prior_dct[k] for k in names}

        results = []
        for name in names:
            prior = cfm_dicts[name]

            results.append(
                create_cfm(
                    time_domain=self.cfm_times,
                    prefix=f"{name}_",
                    offset_std=prior["offset_std"],
                    offset_mean=prior["offset_mean"],
                    fluct=prior["fluctuations"],
                    llslope=prior["loglogavgslope"],
                    flex=prior.get("flex", None)
                )
            )

        if not self.forceless:
            log_omega_sq, gamma, xi_force = results
        else:
            log_omega_sq, gamma = results
            xi_force = lambda p: jnp.zeros(self.N)

        def smooth_mask(times, t0, t1, width):
            return 0.5 * (
                    jnp.tanh((times - t0) / width)
                    - jnp.tanh((times - t1) / width)
            )

        def mask_operator(op):
            t0 = jft.NormalPrior(mean=-0.2, std=0.2, name="t0")
            t1 = jft.NormalPrior(mean=+0.2, std=0.2, name="t1")
            mask_width = jft.LogNormalPrior(mean=1e-2, std=1e-3, name="mask_width")
            # op_tmp = lambda p: op(p) * smooth_mask(self.cfm_times, t0=-.2, t1=.2, width=.001)
            op_tmp = lambda p: op(p) * smooth_mask(self.cfm_times, t0=t0(p), t1=t1(p), width=mask_width(p))
            op_tmp.domain = op.domain | t0.domain | t1.domain | mask_width.domain
            return op_tmp

        def add_peak(op):
            t = self.cfm_times
            t0 = jft.NormalPrior(mean=0, std=0.5, name="t0")
            # t0 = 0
            sig = .1
            A = 1
            peak_model = lambda p: A*jnp.exp(-(t - t0(p))**2/(2*sig**2))
            # peak_model = A/jnp.cosh((t-t0)/sig)

            # peak_model_concrete_values = A*jnp.exp(-(t - (-0.10998))**2/(2*sig**2))
            # plt.plot(self.cfm_times, peak_model_concrete_values)
            # plt.show()
            # stop

            op_tmp = lambda p: peak_model(p) * op(p)
            op_tmp.domain = op.domain | t0.domain
            return op_tmp


        def mask_inital(op, N_points):
            mask = jnp.concatenate((jnp.zeros(N_points), jnp.ones(len(self.cfm_times)-N_points)))
            op_tmp = lambda p: mask * op(p)
            op_tmp.domain = op.domain
            return op_tmp


        def op1_as_inverse_of_op2(op1, op2):
            op_1_tmp = lambda p: 1e2/(op2(p)+1)-1
            op_1_tmp.domain = op2.domain
            return op_1_tmp


        def multiply_op_2_to_op_1(op1, op2):
            # op_1_tmp = lambda p: op1(p)*op2(p)/jnp.max(op2(p))  # normed
            op_1_tmp = lambda p: op1(p)*op2(p)**2 #  non-normed and squared
            # op_1_tmp = lambda p: op1(p)*op2(p) #  non-normed
            op_1_tmp.domain = op1.domain | op2.domain
            return op_1_tmp

        import jax
        def give_op_1_smooth_amplitude_boost_with_op_2(op_1, op_2, boost=1e1, bound=1500):
            def op_1_tmp(p):
                mask = 1 + (boost - 1) * 0.5 * (1 + jnp.tanh(op_2(p) - bound))
                jax.debug.print("max boost through mask = {x}", x=jnp.max(mask))
                return op_1(p) * mask

            op_1_tmp.domain = op_1.domain | op_2.domain
            return op_1_tmp


        # omega = lambda p: jnp.sqrt(jnp.exp(log_omega_sq(p)))
        omega = lambda p: jnp.clip(jnp.sqrt(jnp.exp(log_omega_sq(p))), -jnp.inf, 1e4)
        omega.domain = log_omega_sq.domain

        # xi_force, = (mask_operator(op) for op in [xi_force])
        # xi_force = mask_inital(xi_force, int(len(self.cfm_times)*0.4))

        # omega, _, xi_force = (mask_operator(op) for op in (omega, gamma, xi_force))

        # omega = add_peak(omega)
        xi_force = multiply_op_2_to_op_1(xi_force, omega)
        xi_force = add_peak(xi_force)

        # Q = jft.LogNormalPrior(1e2, 1e2, name="quality_factor")
        # gamma = lambda p: omega(p)**(-1) / jnp.max(omega(p)**(-1)) * Q(p)
        # gamma.domain = log_omega_sq.domain | Q.domain

        # xi_force = give_op_1_smooth_amplitude_boost_with_op_2(xi_force, omega, boost=1e6)

        return omega, gamma, xi_force


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
                 taper_data=False
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
        :param taper_data:                      As the name suggests; you might want to set this to True for
                                                consistency's sake if the welch-averaged noise ps is used.
        """
        print("Assumed noise stationarity timescale for Welch-average: ", stationarity_time_scale, " seconds.")

        odir_main = f"STRAIN_{event_name}_{detector}"
        odir_diagnostic_plots = f"{odir_main}/diagnostic_plots/"
        odir_diagnostic_metadata = f"{odir_main}/diagnostic_metadata/"
        odir_errors = f"{odir_main}/errors/"
        if diagnostic_plots:
            os.makedirs(odir_diagnostic_plots, exist_ok=True)
        os.makedirs(odir_diagnostic_metadata, exist_ok=True)
        os.makedirs(odir_errors, exist_ok=True)

        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)
        strain = get_strain_from_disc(event_name=event_name, detector=detector,
                                      data_duration=data_duration_of_hdf5_file, center_on_event=True,
                                      desired_duration=stationarity_time_scale, add_whitened_data=True,
                                      tapering_function=tapering_function
                                      )

        f_welch = strain.aux.freqs
        ps_welch = strain.aux.ps_welch
        T_mini_welch = max(strain.event_time) - min(strain.event_time)
        T_global_welch = max(strain.time) - min(strain.time)

        NR = get_waveform_template(event_name=event_name, detector=detector, gps_center=strain.gps,
                                   silent=False, plot=False, force_online_fetch=False, model_approximant="IMRPhenomXPHM")


        _metadata_basics(o=odir_diagnostic_metadata, e=event_name, s=strain, det=detector, t_m=T_mini_welch, t_g=T_global_welch,
                         t_d=data_duration_of_hdf5_file)
        _error_metadata(o=odir_errors)
        if diagnostic_plots:
            existing_files = os.listdir(odir_diagnostic_plots)

            plot_1_exists = np.sum(np.array([("whitened_bp_strain" in f) for f in existing_files]).astype(int)).astype(bool)
            plot_2_exists = np.sum(np.array([("welch_average" in f) for f in existing_files]).astype(int)).astype(bool)

            if not plot_1_exists:
                _save_plot_wh_bp_data_with_template(o=odir_diagnostic_plots, s=strain, nr=NR, e=event_name)
            if not plot_2_exists:
                _save_plot_welch_average(o=odir_diagnostic_plots, f=f_welch, p=ps_welch)

        # Attach basic fields

        self.key = key
        self.NR = NR
        self.strain = strain
        self.ps_welch = ps_welch
        self.f_welch = f_welch
        self.T_mini_welch = T_mini_welch
        self.T_global_welch = T_global_welch

        # Create inference scheme object
        self.r_fac = r_fac
        self.e_fac = e_fac
        d = self.strain.event_strain
        if taper_data:
            # Override, delete later and probably match with Welch average
            tapering_function = lambda d: tukey(M=len(d), alpha=0.5, sym=True)
            d = tapering_function(d) * d
        self.pipe = InferenceSchemeRe(t=self.strain.event_time, d=d, e_fac=self.e_fac,
                                 r_fac=self.r_fac, key=key, plotting_callback=analyze_kl_callback)

        # Misc
        self.odir_main = odir_main

        # Fields to add later
        self.noise_ps = None


    def add_signal_model(self, s_model):
        self.pipe.add_custom_signal_model(custom_signal_model=s_model)


    def add_noise_model(self, N_inv, N_sqrt_inv, N_sqrt):
        self.pipe.add_noise_op(inverse_noise_op=N_inv, sqrt_inverse_noise_op=N_sqrt_inv, sqrt_noise_op=N_sqrt)


    def run(self, kl_iterations=10, n_samples=kl_sampling_rate, use_strict_minimizers=False,
                     resume=True, choose_low_kl_starting_pos=False, geoVi=True, chi2_threshold=jnp.inf, max_kl_iter=None,
                      **kwargs):
        # See documentation of `InferenceSchemeRe.run_inference`
        out_name = self.odir_main + '/nifty_out/'
        latent_post_samples, vi_info =  self.pipe.run_inference(kl_iterations=kl_iterations, n_samples=n_samples,
                                                                out_name=out_name,
                                                                use_strict_minimizers=use_strict_minimizers,
                                                                choose_low_kl_starting_pos=choose_low_kl_starting_pos,
                                                                max_kl_iter=max_kl_iter, **kwargs)
        return latent_post_samples, vi_info, self.pipe.get_current_key()

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