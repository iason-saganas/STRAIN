import nifty8

from minimization_control import *
import numpy as np
import matplotlib.pyplot as plt
import nifty8 as ift

def plot_histogram(mean, sigma, n_samples, mode="Lognormal"):
    if mode == "Normal":
        print("Normal distrubution")
        op = ift.NormalTransform(mean=mean, sigma=sigma, key="Normal for Histogram")
    elif mode == "Lognormal":
        op = ift.LognormalTransform(mean=mean, sigma=sigma, key='Lognormal for Histogram', N_copies=0)
    elif mode == "Uniform":
        raise ValueError("Not implemented")
        # print("Uniform distribution")
        # op = ift.StandardUniformTransform(key='Uniform for Histogram', N_copies=0,
        #                               upper_bound=sigma, shift=mean)
    else:
        raise ValueError("Unknown mode")

    op_samples = np.array([op(s).val for s in [ift.from_random(op.domain) for _ in range(n_samples)]])
    label = rf"{mode} with $(\mu, \sigma)=$" + f"$({mean}, {sigma})$" if not (mode=="Uniform") else rf"{mode} in " + r"$\mathrm{[0,1]}$"
    plt.hist(op_samples, bins=200, label=label, histtype='step', facecolor='white', color="black")

    plt.ylabel("Frequency", fontsize=30)
    plt.xlabel(r"Fluctuation parameter", fontsize=30)
    # Right-align the text in the legend
    # special_legend_II()
    # plt.xlim(-0.1, 1.1)
    # if save:
    #     filename = "data_storage/figures/histogram_of_lognormal_distribution"
    #     plt.tight_layout(pad=2)
    #     plt.savefig(filename + ".png", pad_inches=1)
    # if show:
    # plt.show()

class Mask(ift.LinearOperator):
    # Implements a mask response based on a sampling rate
    def __init__(self, domain, sampling_rate_hz, physical_start, physical_end):
        # sampling_rate is typically for ligo 4096Hz or 16384Hz
        # physical_start and end have to be correspondingly in seconds
        # Note that in my implementation, the points are probably not exactly equidistantly spaced (because of the `np.floor` function
        # Also domain.shape[0]-1 is so that the last element is definitely not returned as an index
        self._domain = domain

        physical_length_of_domain = physical_end - physical_start
        num_of_points_to_keep = int(sampling_rate_hz * physical_length_of_domain) + 1  # for the start point ???


        self._target = ift.DomainTuple.make(ift.UnstructuredDomain((num_of_points_to_keep, )))
        self._capability = self.TIMES | self.ADJOINT_TIMES

        self._where_to_keep = np.floor(np.linspace(0, domain.shape[0]-1, num_of_points_to_keep)).astype(int)
        print("Based on sampling rate, length and shape of domain, keeping ", num_of_points_to_keep, " points of continuous signal.")

    def apply(self, x, mode):
        self._check_input(x, mode)
        # x is a field
        if mode == self.ADJOINT_TIMES:
            # zero-pads
            values = np.zeros(self._domain.shape[0])
            values[self._where_to_keep] = x.val
            return ift.Field(self._domain, values)

        elif mode == self.TIMES:
            extract = np.array(x.val)
            idcs = np.array(self._where_to_keep)
            values = extract[idcs]
            return ift.Field(self._target, values)


class ExecuteRGSpaceKL:
    def __init__(self, discrete_time, d, cfm_model_name, n_pix_fac=2, x_fac=2, response="linear mask",
                 sampling_rate_hz=4096, kl_minimizations=10, fluct=(1,1), llslope=(-4,1),
                 gaussian_noise_level=1e-16, out_dir_name="out_index_my_playground", custom_noise_operator=None):
        """
        Parameters
        ----------
        discrete_time : np.array
            The x values at which discrete data occur.
        d : array-like
            Observed data to be analyzed.
        cfm_model_name : str
            What to call the CF ("key").
        n_pix_fac : int, optional
            Factor for determining the number of pixels in the reconstruction grid by multiplying this factor
            onto the number of datapoints. Default is 2.
        x_fac : int, optional
            Expansion factor for the spatial domain, s.t. extended domain is of length x_fac * n_pix_fac * len(d).
             Default is 2.
        response : str, optional
            Type of response model to apply (e.g., 'linear mask'). Default is 'linear mask'.
        sampling_rate_hz : float, optional
            Data sampling rate in Hertz. Default is 4096. Used to determine which points to mask in the signal domain.
        kl_minimizations : int, optional
            Number of KL divergence minimization steps. Default is 10.
        fluct : tuple of float, optional
            Parameters controlling the prior fluctuation amplitude. Default is (1, 1).
        llslope : tuple of float, optional
            Slope range for the log-log power spectrum prior. Default is (-4, 1).
        gaussian_noise_level : float, optional
            Assumed level of additive Gaussian noise. Default is 1e-16.
        out_dir_name : str, optional
            Where to store the output files. Default is "out_index_my_playground".
        """

        x0, xf = (np.min(discrete_time), np.max(discrete_time))

        n_dtps = len(d)
        n_pix = n_pix_fac * n_dtps

        print("Number of datapoints: ", n_dtps)
        print("Constructing RGspace of this number of points: ", n_pix, " (not extended)")

        length_of_domain = xf - x0

        self.out_dir_name = out_dir_name
        self.distances =  length_of_domain / n_pix
        self.domain = ift.RGSpace(shape=(n_pix,), distances=self.distances)
        self.domain_ext = ift.RGSpace(shape=(n_pix*x_fac,), distances=self.distances)
        self.data_space = ift.UnstructuredDomain(shape=(n_dtps, ))
        self.data_field = ift.Field(ift.DomainTuple.make((self.data_space,)),  val=d)

        self.X = ift.FieldZeroPadder(self.domain, new_shape=(n_pix*x_fac, ))

        if response != "linear mask":
            raise ValueError("Not implemented yet")

        mask_op = Mask(ift.DomainTuple.make((self.domain,)), sampling_rate_hz, x0, xf)
        self.R_physical = mask_op

        self.posterior_samples = None
        self.kl_minimizations = kl_minimizations
        self.fluct = fluct
        self.llslope = llslope
        self.cfm_model_name = cfm_model_name
        self.gaussian_noise_level = gaussian_noise_level
        self.prior_parameters = None
        self.n_dtps = n_dtps
        self.n_pix = n_pix

        mdl = self._create_model(fluct, llslope, cfm_model_name)

        self.model, self._cf = (mdl[0], mdl[1])
        self.domain_values = ift.Field(ift.DomainTuple.make(self.domain), np.linspace(x0, xf, n_pix))
        self.domain_values_ext = ift.Field(ift.DomainTuple.make(self.domain_ext), np.linspace(x0, (xf-x0)*x_fac+x0, n_pix*x_fac))
        self.discrete_domain_values = discrete_time

        self._R_full = self.R_physical @ self.X.adjoint @ self.model

        if custom_noise_operator is None:
            self._N = ift.ScalingOperator(self.data_field.domain, gaussian_noise_level, sampling_dtype=np.float64)
        else:
            self._N = custom_noise_operator


    def _create_model(self, fluctuations, llslope, cfm_model_name):
        s_offset = {
            "offset_mean": 0,
            # "offset_std": (1e-20, 1e-21),
            "offset_std": (1e-16, 1e-16),
        }

        s_fluctuations = {
            "target_subdomain": self.domain_ext,
            "fluctuations": (fluctuations[0], fluctuations[1]),
            "loglogavgslope": (llslope[0], llslope[1]),
            # "flexibility": (1., 0.5),
            "flexibility": (1e-16, 1e-16),
            # "asperity": (5., 0.5),
            "asperity": (1e-16, 1e-16)
        }

        s_model = ift.CorrelatedFieldMaker(cfm_model_name)
        s_model.set_amplitude_total_offset(**s_offset)
        s_model.add_fluctuations(**s_fluctuations)

        s_model_meta_class = s_model  # contains information on the power spectra etc
        s_model = s_model.finalize()  # this is solely the operator chauin
        self.prior_parameters = {**s_offset, **s_fluctuations}
        # self.model = s_model
        return s_model, s_model_meta_class

    def _create_gaussian_likelihood(self):
        N = self._N
        R = self._R_full
        likelihood_energy = ift.GaussianEnergy(self.data_field, N.inverse) @ R
        return likelihood_energy

    def _execute_kl(self, lh_energy, kl_minimizations = 10):

        posterior_samples = ift.optimize_kl(
            likelihood_energy=lh_energy,
            total_iterations=kl_minimizations,
            n_samples=kl_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory=self.out_dir_name,
            return_final_position=False,
            resume=True)

        print("Posterior samples saved. Analyze via class.plot_posterior()")
        self.posterior_samples = posterior_samples

    def run(self):
        print("Creating Gaussian likelihood...")
        lh_energy = self._create_gaussian_likelihood()
        print("Created Gaussian likelihood, running KL.")
        self._execute_kl(lh_energy, kl_minimizations=self.kl_minimizations)

    def plot_posterior(self, plot_signal_space=False, plot_with_variance=False):
        if self.posterior_samples is None:
            raise ValueError("Execute `run` first")

        s_m, s_v = self.posterior_samples.sample_stat(self.model)

        if plot_signal_space:
            plt.plot(self.discrete_domain_values, self.data_field.val, "r-", label="Data")  # DATA
            plt.plot(self.domain_values.val, self.X.adjoint(s_m).val, "b-", label="Posterior mean")
            if plot_with_variance:
                plt.errorbar(self.domain_values.val, self.X.adjoint(s_m).val, yerr=np.sqrt(self.X.adjoint(s_v).val))

        else:
            res_sq = (self.data_field.val - self.R_physical(self.X.adjoint(s_m)).val)**2
            chi_sq = np.sum(res_sq) / (self.gaussian_noise_level * self.n_dtps)
            print("chi_sq reduced in dataspace: ", chi_sq)
            plt.plot(self.discrete_domain_values, self.data_field.val, "r-", label="Data")  # DATA
            plt.plot(self.discrete_domain_values, self.R_physical(self.X.adjoint(s_m)).val, "b-", label="Data from posterior mean")

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Strain $[10^{-19}]$')
        plt.title("Reconstruction")
        plt.show()

    def plot_prior_samples(self, num=5, plot_in_data_space=False, apply_adjoint_zp = True):
        plt.ylabel("strain $[10^{-19}]$")
        if not plot_in_data_space:
            op = self.model
            samples = list(op(ift.from_random(op.domain)) for _ in range(num))
            if apply_adjoint_zp:
                plt.xlabel("time in seconds")
                y = [self.X.adjoint(sl) for sl in samples]
                [plt.plot(self.domain_values.val, sl.val) for sl in y]
                return y
            else:
                plt.xlabel("time in seconds (extended domain)")
                [plt.plot(self.domain_values_ext.val, sl.val) for sl in samples]
                return samples
        else:
            op = self.R_physical @ self.X.adjoint @ self.model
            samples = list(op(ift.from_random(op.domain)) for _ in range(num))
            for sample in samples:
                plt.plot(self.discrete_domain_values, sample.val)
            return samples

    def plot_data_realization(self):
        data_realization = self._R_full(ift.from_random(self.model.domain)) + self._N.draw_sample()
        plt.plot(self.discrete_domain_values, data_realization.val)

    def plot_prior_fluctuations_distribution(self):
        plot_histogram(mean=self.fluct[0], sigma=self.fluct[1], n_samples=1000, mode="Lognormal")

    def plot_pow_spec(self):
        a = self._cf.amplitude
        harmonic_samples = list(self.posterior_samples.local_iterator())
        # print("here: ",harmonic_samples[0].domain, harmonic_samples[0].val)
        spectrum_realizations = [a.force(sl) for sl in harmonic_samples]
        # print("here: ", spectrum_realizations[0].val)
        # print("here: ", spectrum_realizations[0].val[:3])
        # print("here: ", spectrum_realizations[0].val[-3:])

        arrs = [sp.val for sp in spectrum_realizations]
        mean_pow_spec = np.mean(arrs, axis=0)

        all_k_domains = [spec.domain[0].k_lengths for spec in spectrum_realizations]
        k_domain_lengths = all_k_domains[0]
        # print("Are all k-domains the same?", np.all(all_k_domains == all_k_domains[0]))

        for spec in spectrum_realizations:
            plt.plot(k_domain_lengths, spec.val, color="black", alpha=0.3)

        plt.plot(k_domain_lengths, mean_pow_spec, label="Posterior mean pow spec", lw=5)
        plt.loglog()

        plt.xlabel(r"Frequency $\omega$")
        plt.ylabel(r"$P_n(\omega)$")

        plt.legend()
        plt.ylim(1e-7,10)
        return k_domain_lengths, mean_pow_spec

    def get_mean_pow_spec(self):
        raise NotImplementedError


class NoiseOperatorFromPowerSpectrum(ift.EndomorphicOperator):
    def __init__(self, power_spectrum: ift.Field, harmonic_space: ift.RGSpace):
        """

        This class assumes the field to be analyzed is real and the data lives over a regular grid; applying it
        returns also a regular grid thus it's an endomorphic operator.
        The involved harmonic transforms as well as power spectra are normalized such that an instance of this class
        square rooted applied to Gaussian white noise gives a signal realization.

        :param power_spectrum ift.Field:

            The power spectrum to use, defined over a power space. The power space for a real field has less support
            points than the pixels of the field itself due to f(-k)=f(k)^{\ast}.

        :param harmonic_space ift.RGSpace:

            The dual space to the space the real field lives over. Used to get the HT operator, as well as to go
            from the lower-dimensional power space to the higher-dimensional harmonic space.

        """

        self.N_sqrt = None
        self.N = self.N_sqrt @ self.N_sqrt
