import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import nifty.nifty.re as jft
import numpy as np

from ..basics.common_utils import fw_hartley, bw_hartley, raise_warning
from ..basics.plotting import usual_plot


__all__ = ["NormedGaussianComb", "BaselineNormedGaussianComb", "ScaledPowerSpectrumTemplate",
           "InvNoiseCovFromPs", "NoiseCovarianceFromPs"]




class NormedGaussianComb(jft.Model):
    def __init__(self,
                 unique_k_lengths:jnp.array,
                 list_of_peaks:jnp.array,
                 list_of_amplitudes:jnp.array,

                 rel_sigma_amp = .1,
                 rel_sigma_widths=.1,
                 a_priori_width_of_peaks = 10,

                 abs_width_sigma=None,
                 abs_amp_sigma=None,

                 vary_positions=False,

                 norm=True
                 ):
        """

        Generates a sum of Gaussian parametric peaks at fixed positions and with lognormal priors set on the
        widths and amplitudes.

        The comb is normed by its area so the peak of the heights have no overall contributions to the global variance,
        but just inject power at specific frequencies. To separate functionality, this operator should not couple to
        the fluctuations parameter of the deviation field.

        :param unique_k_lengths:        The unique frequencies, the operator is built in amplitude space.
        :param list_of_peaks:           Array of peak positions (frequencies)
        :param list_of_amplitudes:      Array of peak amplitudes (power units)
        :param rel_sigma_amp:           The relative standard deviation set on the lognormal amplitude prior
        :param rel_sigma_widths:        The relative standard deviation set on the lognormal frequency width prior
        :param a_priori_width_of_peaks: In Hz.
        :param vary_positions:          Whether to set a normal prior on the peak positions with some internally set
                                        variance.
        :param abs_width_sigma:         The absolute standard deviation of the width. TAKES PRECEDENCE over
                                        rel_sigma_widths
        :param abs_amp_sigma:           The absolute standard deviation of the amplitude. TAKES PRECEDENCE over
                                        rel_sigma_widths
        :param norm:                    If false, skips the normalization (not recommended).
        """

        self.f = unique_k_lengths
        self.N = len(list_of_peaks)
        self.frequency_widths = a_priori_width_of_peaks * jnp.ones(self.N)
        self.positions = list_of_peaks

        if not abs_width_sigma:
            self.sigma_widths = rel_sigma_widths * self.frequency_widths
        else:
            self.sigma_widths = abs_width_sigma

        if not abs_amp_sigma:
            self.sigma_amp = rel_sigma_amp * list_of_amplitudes
        else:
            self.sigma_amp = abs_amp_sigma

        self.sigma_pos = self.positions / 1e3

        self.xi_g_amp = jft.LogNormalPrior(mean=list_of_amplitudes, std=self.sigma_amp, name="xi_g_amp",
                                           dtype=jnp.float64, shape=(self.N,))
        self.xi_g_width = jft.LogNormalPrior(mean=self.frequency_widths, std=self.sigma_widths, name="xi_g_width",
                                             dtype=jnp.float64, shape=(self.N,))
        self.xi_g_pos = jft.NormalPrior(mean=self.positions, std=self.sigma_pos, name="xi_g_pos",
                                             dtype=jnp.float64, shape=(self.N,))

        if vary_positions:
            self.xi_g_pos_vary_or_cst = lambda xi: self.xi_g_pos(xi)
            total_domain = self.xi_g_amp.domain | self.xi_g_width.domain | self.xi_g_pos.domain
        else:
            self.xi_g_pos_vary_or_cst = lambda xi: self.positions
            total_domain =  self.xi_g_amp.domain | self.xi_g_width.domain

        def single_gaussian(amp, width, pos):
            return amp * jnp.exp(-0.5 * ((self.f - pos) / width) ** 2)

        self.sg = single_gaussian
        self.normalize = lambda x, y: y / jnp.trapezoid(y=y, x=x) if norm else lambda x, y: y

        super().__init__(domain=total_domain)

    def __call__(self, xi):
        """
        Logic:

            gaussians = []
            for freq_center, amp, freq_width:

                    gaussian = amp * np.exp(-0.5 * ((f - freq_center) / freq_width) ** 2)
                    gaussians.append(gaussian)

            gaussian_comb = np.sum(gaussians, axis=0)

        :param xi:
        :return:
        """
        deprecate = True
        if deprecate:
            raise ValueError("If you use custom correlated field, the signature of the call inside non-parametric amplitude "
                         "will most likely not match this call")
        amplitude_vector = self.xi_g_amp(xi)
        width_vector = self.xi_g_width(xi)
        position_vector = self.xi_g_pos_vary_or_cst(xi)

        gaussians = vmap(self.sg)(amplitude_vector, width_vector, position_vector)
        # return jnp.sum(gaussians, axis=0)
        gc = jnp.sum(gaussians, axis=0)
        # norm = jnp.trapezoid(y=gc, x=self.f)
        return self.normalize(y=gc, x=self.f)


class BaselineNormedGaussianComb(jft.Model):
    def __init__(self,
                 list_of_amplitudes_above_baseline:jnp.array,
                 list_of_peaks:jnp.array,
                 unique_k_lengths:jnp.array,

                 rel_sigma_amp = .1,
                 rel_sigma_widths=.1,
                 a_priori_width_of_peaks = 10,

                 abs_width_sigma=None,
                 abs_amp_sigma=None,

                 norm=True
                 ):
        """

        Same as NormedGaussianComb but: the physical height of peak_i is multiplied with the current power spectrum
        value at the nearest frequency to k_i before all peaks are summed and normalized (if normalized).

        Therefore, list_of_amplitudes_above_baseline needs to contain amplitudes in orders of magnitude above the
        current baseline power spectrum, BEFORE normalization (complicates intuition a bit).

        For example, if a_100 is set to 1e2 and ps_100 is 1e-9, the physical peak height is 1e-7, which is two orders
        of magnitude above baseline, contrary to the 11 orders of magnitude difference you would get if you set
        a_100 is the peak height directly. This is to allow the baseline to increase itself, instead of high power at
        large k being explained only by very high peaks.

        Further, if a_100 < 1, the peak drowns under the baseline, which is more desirable than the latent parameter
        going to -np.inf.


        :param unique_k_lengths:        The unique frequencies, the operator is built in amplitude space.
        :param list_of_peaks:           Array of peak positions (frequencies)
        :param list_of_amplitudes_above_baseline:
                                        Array of relative peak amplitudes (power units). Physical peak amplitudes
                                        will be list_of_amplitudes_above_baseline * power_spectrum.
        :param rel_sigma_amp:           The relative standard deviation set on the lognormal amplitude prior
        :param rel_sigma_widths:        The relative standard deviation set on the lognormal frequency width prior
        :param a_priori_width_of_peaks: In Hz.
        :param abs_width_sigma:         The absolute standard deviation of the width. TAKES PRECEDENCE over
                                        rel_sigma_widths
        :param abs_amp_sigma:           The absolute standard deviation of the amplitude. TAKES PRECEDENCE over
                                        rel_sigma_widths
        :param norm:                    If false, skips the normalization (not recommended).
        """

        self.f = unique_k_lengths
        self.N = len(list_of_peaks)
        self.frequency_widths = a_priori_width_of_peaks * jnp.ones(self.N)
        self.positions = list_of_peaks

        if not abs_width_sigma:
            self.sigma_widths = rel_sigma_widths * self.frequency_widths
        else:
            self.sigma_widths = abs_width_sigma

        if not abs_amp_sigma:
            self.sigma_amp = rel_sigma_amp * list_of_amplitudes_above_baseline
        else:
            self.sigma_amp = abs_amp_sigma

        self.xi_g_amp = jft.LogNormalPrior(mean=list_of_amplitudes_above_baseline, std=self.sigma_amp, name="xi_g_amp",
                                           dtype=jnp.float64, shape=(self.N,))

        self.xi_g_width = jft.LogNormalPrior(mean=self.frequency_widths, std=self.sigma_widths, name="xi_g_width",
                                             dtype=jnp.float64, shape=(self.N,))

        def single_gaussian(amp, width, pos):
            return amp * jnp.exp(-0.5 * ((self.f - pos) / width) ** 2)

        self.sg = single_gaussian
        if norm:
            self.normalize = lambda x, y: y / jnp.trapezoid(y=y, x=x)
        else:
            self.normalize = lambda x, y: y

        super().__init__(domain=self.xi_g_amp.domain | self.xi_g_width.domain)

    def __call__(self, xi):
        """
        Logic:

            gaussians = []
            for freq_center, amp, freq_width:

                    gaussian = amp * np.exp(-0.5 * ((f - freq_center) / freq_width) ** 2)
                    gaussians.append(gaussian)

            gaussian_comb = np.sum(gaussians, axis=0)

        :param xi:
        :return:
        """
        amplitude_vector = self.xi_g_amp(xi)
        width_vector = self.xi_g_width(xi)
        position_vector = self.positions
        return [amplitude_vector, width_vector, position_vector]

    def finalize(self, peak_information, ps, ps_k_values):
        # Signature needs to match npa of custom correlated field!
        unweighted_amplitudes, widths, positions = peak_information
        extracted_weights = self.extract_weights(ps=ps, ps_k_values=ps_k_values)
        weighted_amplitudes = unweighted_amplitudes * extracted_weights

        # jax.debug.print("position, height before, ps, height after: {x}", x=[positions[0], unweighted_amplitudes[0],
        #                 extracted_weights[0], weighted_amplitudes[0]])

        weighted_gaussians = vmap(self.sg)(weighted_amplitudes, widths, positions)

        # jax.debug.print("max = amplitude = {x}", x=jnp.max(weighted_gaussians[0]))
        # jax.debug.print("expected height: {x}", x=weighted_amplitudes[0])

        gc = jnp.sum(weighted_gaussians, axis=0)
        return self.normalize(y=gc, x=self.f)

    def extract_weights(self, ps, ps_k_values):
        # Please unit test some time
        def weight_for_k(k):
            idx = jnp.argmin(jnp.abs(ps_k_values - k))
            return ps[idx]

        # Vectorize over all positions
        return vmap(weight_for_k)(self.positions)


class ScaledPowerSpectrumTemplate(jft.Model):
    def __init__(self, ps_template:jnp.array, scale=(1,.1)):
        """
        Creates an operator in power space that consists of a scalable power spectrum template.
        :param ps_template:      The fixed power spectrum template.
        :param scale:            Scaling factor: if e.g. 2, contributes twice the variance of the template to the
                                 overall process.
        """
        self.ps_template = ps_template
        self.scale = jft.LogNormalPrior(*scale, name="ps_template_scale")

        super().__init__(domain=self.scale.domain)

    def __call__(self, xi):
        scale_realization = self.scale(xi)
        return scale_realization * self.ps_template


class InvNoiseCovFromPs:
    def __init__(self, one_sided_noise_ps:jnp.array, data_grid, e_fac, n_dtps:int, custom_norm=1):
        r"""

        Don't delete yet, `downsampling` procedure and checks might be helpful when not using a welch average for
        the power spectrum.

        Takes one_sided_noise_ps as input and returns an operator which when called applies

            F^{-1} 1/full_noise_ps F.

        Here, I want to simply downsample and thus create periodic realizations of the noise. Therefore, the inference
        is going to be off at the edges.

        In this class, I assume that noise_ps is the mean posterior data power spectrum gained by previous inference
        runs and that the approximation p_d ~ p_n is to be employed. To circumvent biasing through periodic boundary
        conditions, the data were learned on an extended domain, i.e. the inferred data power spectrum has more points
        than there are data points, so this needs to be somehow downsampled. How?

            len(p_n_inferrred) > len(p_n_real)

        and n^* \hookleftarrow p_n_inferrred:

            len(n^*) > len(n_real).

        This class needs to implement an operator such that the operation

            op = (F^{-1} p_n F) * res

        is well-defined. Here, res is a non-periodic residual d-Rs of length N. But p_n is a MxM matrix with M>N.
        So, zero-pad the input to length M:

            res_prime[N:M] = 0

        This operation gives the same Fourier-Transform as just F(res) because it is just zeros
        (TODO: Maybe only up to a volume factor??)
        Then, after applying the full MxM p_n matrix and transforming back, we may simply cut as

            result = ... [:N].

        Problem:

        To be able to use the data power spectrum to create a noise covariance matrix, the power spectrum is down
        sampled to include only every second point.

        Actually, lets zero-pad the input, apply pd, iFFT and hope for the best and then do an analysis of gibbs
        ringing.

        The residual d minus R s can be made approximately periodic in the zero-padded region because s includes the
        variable xi_s, which can adapt to enforce boundary conditions. In a Gaussian likelihood method, the call method
        of this class would be applied to the residual d minus R s; if you draw a fixed sample without a xi_s variable,
        zero-padding does not enforce boundary conditions and the resulting FFT may appear a bit blurred.

        :param one_sided_noise_ps:    The noise power spectrum P_n(|k|).
        :param data_grid:   The real space data grid.
        :param e_fac:       The extension factor by which the data domain was extended to do the inference.
                            Will determine how much the power spectrum is downsampled.
        :param n_dtps:      The number of datapoints.
        :param custom_norm  A scalar multiplied onto the power spectrum to bring the data realizations to the order of
                            magnitude of the actual data (something about volume factors I evidently don't understand).
        """

        self.one_sided_noise_ps = one_sided_noise_ps[::e_fac] * custom_norm

        self.M_k_lengths = len(data_grid.harmonic_grid.relative_log_mode_lengths)
        self.M = len(one_sided_noise_ps)
        self.N = n_dtps
        self.k = data_grid.harmonic_grid.mode_lengths
        self.dk = self.k[1]-self.k[0]

        self.harmonic_data_grid_expander = data_grid.harmonic_grid.power_distributor # [0 1 2 ... 3 2 1], therefore,
        # if one_sided_noise_ps is ordered as [0, +1, +2, ..., +N/2], one_sided_noise_ps[power_distributor]
        # will be ordered as [0, +1, +2, ..., +N/2, +N/2-1, +2, +1].

        assert self.M == self.M_k_lengths  #  the noise power spectrum IS one-sided, i.e. supported by the correct
        # number of fourier modes. If the ps was gotten by an interpolation that didn't get it right, this will
        # throw an assertion error

        assert self.N == n_dtps  # downsampled correctly? edit 08.12: Forgot why I was doing this

        self.golden_fourier_norm = self.N * self.dk
        self.full_noise_ps = one_sided_noise_ps[self.harmonic_data_grid_expander]

        self.inv = self.full_noise_ps**(-1)
        self.sqrt = jnp.sqrt(self.full_noise_ps)

        self.uH = lambda p: fw_hartley(p, norm="ortho")
        self.iuH = lambda p: bw_hartley(p, norm="ortho")

        expected_var = np.sum(self.full_noise_ps * self.dk)
        print("Expected real-space variance:", expected_var)

        raise_warning("To do: Write assertions of the power spectrum and sample variances automatically")

        print("hä", np.sum(one_sided_noise_ps[1:])+one_sided_noise_ps[0]/2)

        print("\n\n\n and more diagnostics...\n")

        C = self.uH(np.diag(self.full_noise_ps * self.N * self.dk) @ self.iuH(np.eye(self.N)))

        print("np.mean(np.diag(C)[1:]) (exempting C[0][0] because I think its a special point):", np.mean(np.diag(C)[1:]))
        print("\n and C itself: ", C)



    def __call__(self, p):
        # Implements: N^{-1}(res) = F^{-1} p_n^{-1} F(res) where res = d-Rs for example
        # this assumes that the input is periodic. :TODO I believe that this will enforce the theoretical data Rs to be inferred as a periodic function

        fourier_input = self.uH(p)  # now, this is an i.i.d. variable in standard DFT order:
        # [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.]

        applying_inv_ps = 1/self.golden_fourier_norm * self.inv * fourier_input  # meaning, that self.inv must ALSO be in standard DFT order
        # [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.] which should be guaranteed through the use of
        # harmonic_data_grid_expander

        transforming_to_real_space = self.iuH(applying_inv_ps)
        res = transforming_to_real_space
        return res


    def N_sqrt(self, p):
        # Implements: xi_prime = N^{1/2} xi where xi is standard normal, such that xi_prime is from a Gaussian with covariance N.
        # These samples will be periodic.
        fourier_input = self.uH(p)
        applying_sqrt_ps = jnp.sqrt(self.golden_fourier_norm) * self.sqrt * fourier_input
        transforming_to_real_space = self.iuH(applying_sqrt_ps)
        res = transforming_to_real_space
        return res


    def get_samples(self, num):
        """
        0-centered Gaussian distributed samples with covariance N.
        :param num:    The numbers of samples to get.
        :return: list containing the samples
        """
        lt = []
        for _ in range(num):
            xi = np.random.standard_normal(self.N)
            sl = self.N_sqrt(xi)
            lt.append(sl)
        return lt


    def plot_samples(self, num):
        samples = self.get_samples(num)
        for sl in samples:
            plt.plot(self.time, sl)
        usual_plot()


class NoiseCovarianceFromPs:
    def __init__(self, one_sided_noise_ps:jnp.array, data_grid, callable_to_apply=None, apply_correction_factor=False,
                 correction_factor_dont_change_default_for_legacy_reasons=3.6142705042):
        r"""
        Please see deprecated class `InvNoiseCovFromPs` as well.
        The call method of this class implements

            output = uH callable( h_vol * one_sided_noise_ps[expander]m) iuH ( input ),

        where uH and iuH are unitary Hartley and inverse Hartley transforms, respectively.

        :param one_sided_noise_ps:    The noise power spectrum P_n(|k|).
        :param data_grid:             The real space data grid to get the wavevectors.
        :param callable_to_apply:     A callable to apply the noise power spectrum and corresponding weights in Fourier
                                      space. For example, lambda x: x**-1 for an inverse power spectrum.
        :param apply_correction_factor: If true, applies a correction factor to the power spectrum to get the right
                                        data variance for GW150914; somewhere along the lines I messed up some
                                        Fourier factors. Don't use in new code and remove in a few commits.
                                        The correction factor can be calculated as variance expected from full psd
                                        (with correction_factor ==1) divided by data variance.
        """
        self.one_sided_noise_ps = one_sided_noise_ps.copy()
        if apply_correction_factor:
            self.one_sided_noise_ps /= correction_factor_dont_change_default_for_legacy_reasons # needed to the correct data variance in old versions
        self.apply_callable = callable_to_apply
        self.h_grid = data_grid.harmonic_grid
        self.k = self.h_grid.mode_lengths
        self.N = data_grid.shape[0]

        self.M_k_lengths = len(self.h_grid.relative_log_mode_lengths)
        self.M = len(self.one_sided_noise_ps)
        self.dk = self.k[1]-self.k[0]
        self.dt = data_grid.distances[0]

        self.expand =  self.h_grid.power_distributor # [0 1 2 ... 3 2 1], therefore,
        # if one_sided_noise_ps is ordered as [0, +1, +2, ..., +N/2], one_sided_noise_ps[power_distributor]
        # will be ordered as [0, +1, +2, ..., +N/2, +N/2-1, +2, +1].

        assert self.M == self.M_k_lengths  #  the noise power spectrum IS one-sided, i.e. supported by the correct
        # number of fourier modes. If the ps was gotten by an interpolation that didn't get it right, this will
        # throw an assertion error

        self.h_vol = self.N * self.dk
        self.full_noise_ps = self.one_sided_noise_ps[self.expand]

        self.uH = lambda p: fw_hartley(p, norm="ortho")
        self.iuH = lambda p: bw_hartley(p, norm="ortho")

        expected_var = np.sum(self.full_noise_ps * self.dk)
        print("may I suggest the correction factor ", expected_var/ 5.381648179848571)
        print(
            f"\nInitialing noise covariance based on a power spectrum with total area σ^2 = ∫ ps(k) dk ~ "
            f"{expected_var:.15f}."
            f"\nApplying callable {_callable_repr(self.apply_callable)}"
        )

    def __call__(self, p):
        # fourier_input = self.uH(p)  # An i.i.d. variable in standard DFT order [0, +1, +2, ..., +N/2, -N/2+1, ..., -1.]
        # kernel = self.apply_callable(self.full_noise_ps * self.h_vol)
        # return self.iuH(kernel * fourier_input)

        # The call above seems to ever so slightly misrepresent the variance (by ~0.1)

        xi_tilde = jnp.fft.fft(p)  # full FFT
        df = 1.0 / (self.N * self.dt)  # frequency spacing
        # full_noise_ps already mirrors negative frequencies, multiply by N to correct ifft scaling
        kernel = self.apply_callable(self.full_noise_ps * df * self.N)
        return jnp.fft.ifft(kernel * xi_tilde).real


def _callable_repr(f):
    test_val = 2
    val = f(test_val)
    if val == 0.5:
        return "lambda x: x**(-1)"
    elif val == 2**0.5:
        return "lambda x: x**(1/2)"
    elif val == 2**(-0.5):
        return "lambda x: x**(-1/2)"
    else:
        return "<unknown callable>"
