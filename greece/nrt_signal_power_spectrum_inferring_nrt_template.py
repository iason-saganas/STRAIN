import numpy as np
from utils.generative_models import *

plot_data = False
plot_prior_samples_of_generative_model = False
plot_data_and_a_sample = False
plot_data_realization = False

def dt(domainoid):
    # domain tuple my dom
    return ift.DomainTuple.make(domainoid)


class Mask(ift.LinearOperator):
    # Implements a mask response
    def __init__(self, domain, target, keep_th: int):
        """

        :param domain:              The domain of the signal.
        :param target:              The data domain to which this operator maps.
        :param keep_th:             Assuming signal_field[0] == data_field[0], which elements of signal_field have to
                                    be masked to be equal to data_field? If every second, keep_th = 2. Others not
                                    implemented.
        :param physical_start:      min(time)
        :param physical_end:        max(time)
        """
        self._domain = domain
        self._target = target
        self.keep_th = keep_th
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        # x is a field

        if mode == self.ADJOINT_TIMES:
            # zero-pads
            values = np.zeros(self._domain.shape[0])
            where_to_insert = np.arange(0, self._domain.shape[0], self.keep_th)
            values[where_to_insert] = x.val
            return ift.Field(self._domain, values)

        elif mode == self.TIMES:
            # picks away every second, the first is kept
            values = np.array(x.val)[::self.keep_th]
            return ift.Field(self._target, values)



def generative_model(harmonic_space):
    """
    Returns a generative model based on a broken power law.

    k_masked = k.copy()
    k_masked[0] = 1

    gamma = ln(k_masked)

    power_law = exp (exponent * gamma)
    power_law[0] = zm

    :param harmonic_space:  The codomain to the signal domain.
    :return:
    """
    ### --- fixed parameters for now
    k_cut_off = 50
    normalization_constant = 1e2
    mask_zm = 1e-16  # if this is 0, no kidding, the conjugate gradient scheme in the inference fails.
    # message "can not find descent direction"
    # ---------


    # --- Prior choices
    prior_choices = {
        "alpha ": (1,2),
        "beta ": (-6,3),
        "cfm envelope fluctuations": (1, 0.5),
        "cfm envelope loglogavgslope": (-4, 1)
    }
    # ---------

    s_dom = harmonic_space.get_default_codomain()
    p_space = ift.PowerSpace(harmonic_space)

    k_masked = harmonic_space.get_unique_k_lengths()
    k_masked[0] = 1

    gamma_field = ift.Field(dt(p_space), val=np.log(k_masked))
    gamma_op = ift.DiagonalOperator(gamma_field)

    k_cut_off_index = np.argmin(np.abs(k_masked - k_cut_off))  # round to nearest such wavenumber

    alpha = ift.NormalTransform(*prior_choices["alpha "], key="alpha ")
    beta = ift.NormalTransform(*prior_choices["beta "], key="beta ")

    pspace_expander = ift.ContractionOperator(p_space, spaces=0).adjoint
    alpha, beta = pspace_expander @ alpha, pspace_expander @ beta

    power_law_left = np.exp(gamma_op @ alpha)
    power_law_right = np.exp(gamma_op @ beta)

    # Mask the zero modes

    mask_zm_val = np.ones(p_space.shape)
    mask_zm_val[0] = mask_zm
    mask_zm_op = ift.makeOp(ift.makeField(p_space, mask_zm_val))

    power_law_left, power_law_right = mask_zm_op @ power_law_left, mask_zm_op @ power_law_right

    # mask `power_law_left` at all positions larger than k_cut_off
    # mask `power_law_right` at all positions smaller than k_cut_off

    mask_pll_val = np.ones(p_space.shape)
    mask_pll_val[k_cut_off_index:] = 0
    mask_pll_op = ift.makeOp(ift.makeField(p_space, mask_pll_val))

    mask_plr_val = np.ones(p_space.shape)
    mask_plr_val[:k_cut_off_index] = 0
    mask_plr_op = ift.makeOp(ift.makeField(p_space, mask_plr_val))

    power_law_left, power_law_right = mask_pll_op @ power_law_left, mask_plr_op @ power_law_right

    # attach power law right unsmoothly at the position power law left stopped
    # Create selector field for index i = k_cut_off_index to match boundary conditions

    select_arr_l = np.zeros(p_space.shape)
    select_arr_r = np.zeros(p_space.shape)

    select_arr_l[k_cut_off_index-1] = 1
    select_arr_r[k_cut_off_index] = 1

    selector_1 = ift.makeOp(ift.makeField(p_space, select_arr_l))
    selector_2 = ift.makeOp(ift.makeField(p_space, select_arr_r))

    last_nonzero_left_el =  selector_1 @ power_law_left  # this will be a diagonal operator with one non-zero element
    # sitting somewhere => Sum it!
    first_nonzero_right_el = selector_2 @ power_law_right

    last_nonzero_left_el = pspace_expander.adjoint @ last_nonzero_left_el # one number
    # => Put it again on a field!
    first_nonzero_right_el = pspace_expander.adjoint @ first_nonzero_right_el

    last_nonzero_left_el = pspace_expander @ last_nonzero_left_el
    first_nonzero_right_el = pspace_expander @ first_nonzero_right_el

    power_law_right = power_law_right * first_nonzero_right_el.ptw("reciprocal") * last_nonzero_left_el

    ps = power_law_left + power_law_right
    integrator = ift.ContractionOperator(p_space, spaces=0)
    integral = integrator(ps) # scalar
    integral = pspace_expander @ integral # field

    ps = ps * integral.ptw("reciprocal")
    ps = normalization_constant * ps

    power_spectrum_sqrt = np.sqrt(ps)

    pd = ift.PowerDistributor(harmonic_space)
    power_spectrum_sqrt_full = pd @ power_spectrum_sqrt

    xi_s = ift.ducktape(harmonic_space, None,'xi_s')

    ht = ift.HartleyOperator(domain=s_dom)

    wavelet = ht.adjoint(power_spectrum_sqrt_full * xi_s)

    cf_env = ift.SimpleCorrelatedField(target=s_dom, offset_mean=None, offset_std=None,
                                         fluctuations=(1, 0.5), flexibility=None, asperity=None,
                                         loglogavgslope=(-4, 1), prefix="cfm envelope ", use_uniform_prior_on_fluctuations=False).ptw("exp")

    op = cf_env * wavelet

    op.prior_choices = prior_choices
    op.ps = ps

    return op

L = max(nrt_time_values) - min(nrt_time_values)
n_dtps = len(nrt_time_values)

pxl_vol_data_space = L/n_dtps
data_domain = dt(ift.RGSpace(shape=(n_dtps,), distances=pxl_vol_data_space))
data_domain_harmonic = data_domain[0].get_default_codomain()

n_pix = 2*n_dtps-1  # so actually every second support point in the signal domain is equal to that in that data domain!
pxl_vol_signal_space = L/n_pix
signal_domain = ift.RGSpace(shape=(n_pix,), distances=L/n_pix)

time_domain_values_signal_space = np.linspace(min(nrt_time_values), max(nrt_time_values), n_pix)
time_domain_values_data_space = nrt_time_values
# (time_domain_values_signal_space[::2] == time_domain_values_data_space) = True

h_space = signal_domain.get_default_codomain()
N = ift.ScalingOperator(data_domain, factor=1e-6, sampling_dtype=np.float64)
d = ift.Field(data_domain, val=nrt_strain_values)
R = Mask(dt(signal_domain), dt(data_domain), keep_th=2)
s = generative_model(h_space)

s_prime = R(s)

if plot_data:
    plt.plot(nrt_time_values, nrt_strain_values, label="Data")
    usual_plot()

if plot_prior_samples_of_generative_model:
    plot_power_spectrum = True
    if plot_power_spectrum:
        for _ in range(1):
            hs = signal_domain.get_default_codomain()
            ps = s.ps
            xi = ift.from_random(ps.domain)
            sl = ps(xi).val

            plt.plot(sl)
            plt.loglog()

    else:
        for _ in range(1):
            hs = signal_domain.get_default_codomain()
            xi = ift.from_random(s.domain)
            sl = s(xi).val

            plt.plot(time_domain_values_signal_space, sl)

    plt.vlines(100, 2.5e-19, 100, ls="--", color="black", label=r"$k_{\mathrm{cutoff}}$")
    usual_plot(xl=r"$|\vec{k}|$", yl="$P_s(k)$", )


if plot_data_and_a_sample:
    xi = ift.from_random(s.domain)
    sl = s(xi).val

    plt.plot(time_domain_values_signal_space, sl, label="Sample")
    plt.plot(nrt_time_values, nrt_strain_values, label="Data")
    usual_plot()

if plot_data_realization:
    xi = ift.from_random(s.domain)
    sl = s(xi)

    d_synth = R(sl) + N.draw_sample()

    plt.plot(nrt_time_values, d_synth.val, label="Data realization")
    plt.plot(time_domain_values_signal_space, sl.val, label="Underlying sample")
    plt.plot(nrt_time_values, nrt_strain_values, label="Real data")
    usual_plot()

energy = ift.GaussianEnergy(d, N.inverse) @ s_prime

posterior_samples = ift.optimize_kl(
            likelihood_energy=energy,
            total_iterations=20,
            # total_iterations=9,
            n_samples=kl_sampling_rate,
            kl_minimizer=descent_finder,
            sampling_iteration_controller=ic_sampling_lin,
            nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
            output_directory="outs/learning_nrt_template_2 ",
            return_final_position=False,
            resume=True)

s_mean, s_var = posterior_samples.sample_stat(s)
latent_sl = posterior_samples.sample_stat()[0]

print("\n", latent_sl.val, " and domain", latent_sl.domain,"\n")

# scalar_dom = ift.DomainTuple.scalar_domain()
# multi_dom = ift.MultiDomain.make({
#     "cfm envelope fluctuations": scalar_dom,
#     "cfm envelope loglogavgslope": scalar_dom,
#     "cfm envelope xi": h_space
# })
#
# # Step 2: Create a MultiField over the MultiDomain
# mf = ift.MultiField.from_dict(domain=multi_dom, dct={
#     "cfm envelope fluctuations": latent_sl["cfm envelope fluctuations"],
#     "cfm envelope loglogavgslope": latent_sl["cfm envelope loglogavgslope"],
#     "cfm envelope xi": latent_sl["cfm envelope xi"]
# })

# cf_env = ift.SimpleCorrelatedField(target=signal_domain, offset_mean=None, offset_std=None,
#                                  fluctuations=(1, 0.5), flexibility=None, asperity=None,
#                                  loglogavgslope=(-4, 1), prefix="cfm envelope ", use_uniform_prior_on_fluctuations=False).ptw("exp")
# cf_part = cf_env(mf)

# plt.plot(time_domain_values_signal_space, s_mean.val, label="Mean reconstruction")
# plt.plot(time_domain_values_signal_space, cf_part.val, label="Posterior cfm envelope")
plt.errorbar(time_domain_values_signal_space, s_mean.val, yerr=np.sqrt(s_var.val),label="Mean reconstruction with errorbars", color=blue, ecolor=light_blue)
plt.plot(nrt_time_values, nrt_strain_values, label="Actual nrt template", color="orange")
usual_plot()



# np.savetxt("data/xi_s_wavelet.txt", latent_sl.val["xi_s"])
# np.savetxt("data/xi_s_cfm_envelope.txt", latent_sl.val["cfm envelope xi"])
#
# pickle_me_this("data/best_latent_field", latent_sl)
# pickle_me_this("data/best_nrt_field", s_mean)

conclusions= """"
    We have build a generative model that can represent this numerical relativity template through an unsmooth 
    broken power law and simple correlated field envelope. 
    
    It does not capture all of the aspects of the NRT perfectly, some are outside the 1σ region. 
    I assume that there are optimizations you can do to fix this.
    For example the correlated field is allowed to be greater than one. For it to be a more predictable envelope,
    maybe we should enforce max(cf_envelope) = 1 somehow and then let the generative wavelet model part set the 
    amplitude of the signal, instead of the latter being adjusted through the correlated field. I think that's 
    what's happening at least.
    
    The optimal parameters found in this run were:   
    
        alpha = 15.61982054
        beta = -10.70962647
        cfm envelope fluctuations = 5.5207253465
        cfm envelope loglogavgslope = -2.48558889
        
    The two xi arrays are saved under `xi_s_wavelet` and `xi_s_cfm_envelope`.
    Also the whole posterior mean latent field is saved under `best_latent_field` and the best posterior field saved 
    as `best_nrt_field`
    
    The correct out folder is `learning_nrt_template_2` and the priors used were:
    
    cf_env = ift.SimpleCorrelatedField(target=s_dom, offset_mean=None, offset_std=None,
                                         fluctuations=(1, 0.5), flexibility=None, asperity=None,
                                         loglogavgslope=(-4, 1), prefix="cfm envelope ", use_uniform_prior_on_fluctuations=False).ptw("exp")  
                                         
    ### --- fixed parameters for now
    k_cut_off = 50
    normalization_constant = 1e2
    mask_zm = 1e-16  # if this is 0, no kidding, the conjugate gradient scheme in the inference fails.
    # message "can not find descent direction"
    # ---------


    # --- Prior choices
    prior_choices = {
        "alpha ": (1,2),
        "beta ": (-6,3),
        "cfm envelope fluctuations": (1, 0.5),
        "cfm envelope loglogavgslope": (-4, 1)
    }
    # ---------
    
"""

print(conclusions)