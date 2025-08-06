from quickview.greece.utils.config_jupyter_notebooks import *
import scipy as sc
from quickview.greece.data.style_components.matplotlib_style import *

nrt_strain_values = np.loadtxt("data/num_rel_template_strain_values.txt") * 1e19
nrt_time_values = np.loadtxt("data/num_rel_template_time_values.txt") - zero_time

def generative_model_4(extended_time_domain, extended_time_domain_values):
    """

    Generative model of numerical wavelet template with adjustable max height and position of peak

    :param extended_time_domain:
    :param extended_time_domain_values:
    :return:
    """

    # --- get numerical relativity template and cut it so it fits to the time domain --- #
    nrt_strain_values_copy = nrt_strain_values
    nrt_time_values_copy = nrt_time_values

    idcs_to_cut = np.where(nrt_time_values < np.min(signal_strip_time))
    nrt_strain_values_copy = np.delete(nrt_strain_values_copy, idcs_to_cut)
    nrt_time_values_copy = np.delete(nrt_time_values_copy, idcs_to_cut)

    len_difference = len(signal_strip_time) - len(nrt_time_values_copy)
    nrt_strain_values_copy = np.append(nrt_strain_values_copy, np.zeros(len_difference)).flatten()

    # now interpolate extended domain array
    interpolated_strain_values_large_domain = np.interp(extended_time_domain_values, signal_strip_time, nrt_strain_values_copy)

    interpolated_strain = sc.interpolate.interp1d(extended_time_domain_values, interpolated_strain_values_large_domain, kind="cubic", fill_value=0, bounds_error=False)

    def func(arg):
        return interpolated_strain(arg)

    def func_and_derv(arg):
        res = func(arg)
        return res, np.gradient(res, arg)

    ift.pointwise.ptw_dict["interpol_strain"] = func, func_and_derv

    # x_op = ift.DiagonalOperator(diagonal=ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values))
    x_field = ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values)
    x_adder = ift.Adder(x_field)
    shift = ift.NormalTransform(mean=1, sigma=0.5, key="shift ")
    # amp = ift.NormalTransform(mean=5*1e2, sigma=1e2, key="amplitude ")
    amp = ift.NormalTransform(mean=1, sigma=1e-16, key="amplitude ")

    expander = ift.ContractionOperator(extended_time_domain, spaces=0).adjoint

    amp = expander @ amp
    shift = expander @ shift

    x_shifted = x_adder @ shift

    unnormalized_strain = x_shifted.ptw("interpol_strain")
    op = unnormalized_strain * amp

    return op


def cosine_generative_model(extended_time_domain, extended_time_domain_values):
    # generates a*cos(bx)
    a = ift.NormalTransform(mean=1, sigma=1, key="amplitude ")
    b = ift.NormalTransform(mean=20, sigma=10, key="period ")

    expander = ift.ContractionOperator(extended_time_domain, spaces=0).adjoint

    a, b = expander @ a, expander @ b

    x_field = ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values)
    x = ift.DiagonalOperator(x_field)

    op = a * (x @ b).ptw("cos")

    xi = ift.NormalTransform(mean=1, sigma=0.1, key="envelope ", N_copies=len(extended_time_domain_values), custom_domain=extended_time_domain)
    # xi = ift.ducktape(extended_time_domain, None, 'xi')

    # for this pointwise operator multiplication, the domains of the operators actually get merged! And the targets need
    # to be the same!

    return op * xi


def cosine_with_exp_env_generative_model(extended_time_domain, extended_time_domain_values):
    # generates a*cos(b(x-c)) * gaussian_envelope
    # gaussian_envelope = e^{-1/2 (x-d)^2/e^2}

    prior_env_center = np.mean(extended_time_domain_values)

    prior_choices = {
        "amplitude ": (1,1),
        "period ": (20,10),
        "cosine shift ": (0,2),
        "envelope shift ": (0,2),
        "envelope dispersion ": (0.1,0.2),
    }

    a = ift.NormalTransform(*prior_choices["amplitude "], key="amplitude ")
    b = ift.NormalTransform(*prior_choices["period "], key="period ")
    c = ift.NormalTransform(*prior_choices["cosine shift "], key="cosine shift ")

    d = ift.NormalTransform(*prior_choices["envelope shift "], key="envelope shift ")
    e = ift.NormalTransform(*prior_choices["envelope dispersion "], key="envelope dispersion ")

    expander = ift.ContractionOperator(extended_time_domain, spaces=0).adjoint

    a, b, c, d, e = expander @ a, expander @ b, expander @ c, expander @ d, expander @ e

    x_field = ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values)
    x_field_prime = ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values-prior_env_center)

    x_adder = ift.Adder(x_field)
    x_adder_prime = ift.Adder(x_field_prime)

    shifted_x = x_adder @ c
    shifted_x_env = x_adder_prime @ d

    op = a * (shifted_x * b).ptw("cos")
    env = (-1/2* shifted_x_env**2 * e**(-2)).ptw("exp")

    res = op*env

    res.prior_choices = prior_choices

    return res


def wavelet_model_spline(extended_time_domain, extended_time_domain_values):

    prior_choices = {
        "shift ": (0, 0.5),
        "amplitude ": (1, 0.5)
    }

    from scipy.interpolate import UnivariateSpline
    spline = UnivariateSpline(nrt_time_values, nrt_strain_values, s=0)

    def truncated_spline(arg):
        base = spline(arg)
        base[np.where(arg < np.min(nrt_time_values))] = 0
        base[np.where(arg > np.max(nrt_time_values))] = 0
        return base

    def truncated_spline_and_derivative(arg):
        base = truncated_spline(arg)
        return base, np.gradient(base, arg)

    ift.pointwise.ptw_dict["truncated_spline_strain"] = truncated_spline, truncated_spline_and_derivative

    expander = ift.ContractionOperator(extended_time_domain, spaces=0).adjoint

    shift = ift.NormalTransform(*prior_choices["shift "], key="shift ")
    amp = ift.NormalTransform(*prior_choices["amplitude "], key="amplitude ")
    shift, amp = expander @ shift, expander @ amp

    x_field = ift.Field(ift.DomainTuple.make(extended_time_domain), val=extended_time_domain_values)
    x_adder = ift.Adder(x_field)

    shifted_x = x_adder @ shift

    op = amp*shifted_x.ptw("truncated_spline_strain")
    op.prior_choices = prior_choices

    return op


def usual_plot(xl=r"Time $t$ $\mathrm{[sec]}$", yl=r"Strain $h$ $\mathrm{[10^{-19}]}$", title=None, xlim=None, ylim=None):
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    ax = plt.gca()
    labels = ax.get_legend_handles_labels()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if labels != ([], []):
        plt.legend()
    plt.show()


def dt(domainoid):
    # domain tuple my dom
    return ift.DomainTuple.make(domainoid)


def generative_model_broken_power_law(harmonic_space, apply_envelope=True, add_extra_amplitude=False):
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
        "alpha ": (15,2),
        "beta ": (-6,3),
        "cfm envelope fluctuations": (4, 2),
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

    if apply_envelope:

        cf_env = ift.SimpleCorrelatedField(target=s_dom, offset_mean=None, offset_std=None,
                                             fluctuations=prior_choices["cfm envelope fluctuations"], flexibility=None, asperity=None,
                                             loglogavgslope=prior_choices["cfm envelope loglogavgslope"], prefix="cfm envelope ", use_uniform_prior_on_fluctuations=False).ptw("exp")
        op = cf_env * wavelet

    else:
        op = wavelet

    if add_extra_amplitude:
        amp = ift.NormalTransform(mean=500, sigma=500, key="amplitude ")
        expander = ift.ContractionOperator(s_dom, spaces=0).adjoint
        amp = expander @ amp
        op = amp * op


    op.prior_choices = prior_choices
    op.ps = ps

    return op

def generative_model_continuous_double_power_law(harmonic_space, apply_envelope=True,
                                                 exact_values_dict:dict=None):
    """

    EDIT
    -------
    What I learned from building this function: The conjugate gradient will blow up (either previous_gamma = INF or
    alpha ~ curvature < 0) if the power spectrum contains either 0's or almost zero values, something like smaller
    than 1e-30. I increased the numerical stability of my model by switching from clipping a logistic sigmoid function
    to using a more 'natural' tanh-based sigmoid function already built-in NIFTy and then checking for samples that
    the model build this way does not contain such small values.

    A potential problem could still be overflow issues when I do 10^(stuff) (although stuff ~ log10(...)).
    But during the inference I still get some overflow warnings connected to tmp = np.power(base, v) from the
    pointwise dictionary.

    Indeed, after running it for some iterations, it again cannot find the descent direction (infinite gamma error).
    I just tried out to fix c to 100, p0 to 1000 and center alpha and beta more strongly around values +14 and -10
    respectively (variance of 5); this does not throw a could not find descent direction error.
    -------

    A smooth version of the broken power law model, such that the dominant frequency can be learned by NIFTy.
    See desmos graph https://www.desmos.com/calculator/pv2qh01pyg.

    :param harmonic_space:          The codomain to the signal domain.
    :param apply_envelope:          Whether to apply the correlated field envelope or just leave the wavelet operator
                                    as is.
    :param exact_values_dict:       For debugging purposes. Takes a list of exact values to construct the operator from.
                                    The order of the list should be:

                                    [k0, p0, c, alpha, beta, cfm_envelope_fluctuations,
                                    cfm_envelope_loglogavgslope], e.g.
                                        exact_values_dict={
                                            "k0": 10,
                                            "p0": 80000,
                                            "c": 100,
                                            "alpha": +10,
                                            "beta": -10,
                                            "cfm_envelope_fluctuations": 5,
                                            "cfm_envelope_loglogavgslope": -6}

    :return:
    """
    ### --- fixed parameters for now
    # k_cut_off = 50
    # normalization_constant = 1e2
    # message "can not find descent direction"
    # ---------

    k = harmonic_space.get_unique_k_lengths()
    power_at_zm = 1e-16  # if this is 0, no kidding, the conjugate gradient scheme in the inference fails.
    # --- Prior choices
    prior_choices = {
        "k0 ": (0, np.max(k)),
        "p0 ": (1e3, 1e-16),
        "c ": (100, 1e-16),
        "alpha ": (+10,100),
        "beta ": (-10,100),
        "cfm_envelope_fluctuations ": (4, 2),
        "cfm_envelope_loglogavgslope ": (-4, 1)
    }
    # ---------

    s_dom = harmonic_space.get_default_codomain()
    p_space = ift.PowerSpace(harmonic_space)

    k[0] = 1  # I do this so np.log10(k) won't diverge at k[0]=0. Later I will set the power at k[0] to a fixed value
    # anyway.

    k_field = ift.Field(dt(p_space), val=k)
    gamma_field = ift.Field(dt(p_space), val=np.log10(k))
    gamma_op = ift.DiagonalOperator(gamma_field)

    k0 = ift.StandardUniformTransform(key="k0 ", upper_bound=prior_choices["k0 "][1]) # don't confuse with k[0], this is the k value where there is a peak in the power spectrum
    p0 = ift.NormalTransform(*prior_choices["p0 "], key="p0 ")
    c = ift.NormalTransform(*prior_choices["c "], key="c ")
    alpha = ift.NormalTransform(*prior_choices["alpha "], key="alpha ")
    beta = ift.NormalTransform(*prior_choices["beta "], key="beta ")

    if exact_values_dict is not None:
        [k0_val, p0_val, c_val, alpha_val, beta_val, _, _] = exact_values_dict.values()

        k0 = ift.NormalTransform(key="k0 ", mean=k0_val, sigma=1e-16)
        p0 = ift.NormalTransform(key="p0 ", mean=p0_val, sigma=1e-16)
        c = ift.NormalTransform(key="c ", mean=c_val, sigma=1e-16)
        alpha = ift.NormalTransform(key="alpha ", mean=alpha_val, sigma=1e-16)
        beta = ift.NormalTransform(key="beta ", mean=beta_val, sigma=1e-16)

    pspace_expander = ift.ContractionOperator(p_space, spaces=0).adjoint

    c = pspace_expander @ c
    k0 = pspace_expander @ k0
    p0 = pspace_expander @ p0
    alpha = pspace_expander @ alpha
    beta = pspace_expander @ beta


    k_field_adder = ift.Adder(a=k_field)
    add_one = ift.Adder(a=ift.Field(dt(p_space), val=np.ones(p_space.shape[0])))
    exponent = -c*(k_field_adder(-1*k0))

    C = ClipOperator(domain=exponent.target, clip_above=690, clip_below=-690)  # this value of `clip_above` gives power spectra that
    # deviate from the power spectra without this clipping operator by a maximum of ~10^-300.
    sigmoid = add_one(np.exp(C(exponent))).ptw("reciprocal")

    a0 = p0 * ( -alpha * np.log10(k0) ).ptw("exponentiate", 10)
    b0 = p0 * ( -beta * np.log10(k0) ).ptw("exponentiate", 10)


    # test = ift.from_random(exponent.domain)
    # test2 = exponent(test)
    # test3 = C(test2)
    # test4 = test3.ptw("exp")
    # print("test2.nifty_sigmoid", (-1*test2).ptw("sigmoid"))
    # print("test4 my sigmoid", add_one(test4).ptw("reciprocal"))

    my_sigmoid = sigmoid
    nifty_sigmoid = (-1*exponent).ptw("sigmoid")
    sigmoid_to_use = nifty_sigmoid

    ps = (a0*(gamma_op @ alpha).ptw("exponentiate", 10)-sigmoid_to_use*a0*(gamma_op @ alpha).ptw("exponentiate", 10)
          + sigmoid_to_use*b0*(gamma_op @ beta).ptw("exponentiate", 10))

    # ps = sigmoid_to_use*(gamma_op @ beta).ptw("exponentiate", 10)

    # for _ in range(10):
    #     test = ps(ift.from_random(ps.domain)).val
    #     print("0's in an exemplary pow spec? ", np.any(test < 1e-20))
    # stop

    integrator = ift.ContractionOperator(p_space, spaces=0)
    integral = integrator(ps) # scalar
    integral = pspace_expander @ integral # field

    ps = ps * integral.ptw("reciprocal")

    tmp = np.ones(p_space.shape)
    tmp[0] = power_at_zm
    set_zm_power = ift.makeOp(ift.makeField(p_space, tmp))

    # ps = set_zm_power(ps)

    amp_s = np.sqrt(ps)

    pd = ift.PowerDistributor(harmonic_space)
    amp_s_on_full_space = pd @ amp_s

    xi_s = ift.ducktape(harmonic_space, None,'xi_s')
    ht = ift.HartleyOperator(domain=s_dom)

    wavelet = ht.adjoint(amp_s_on_full_space * xi_s)

    if apply_envelope:
        fluctuations = prior_choices["cfm_envelope_fluctuations "]
        llslope = prior_choices["cfm_envelope_loglogavgslope "]
        if exact_values_dict is not None:
            [_, _, _, _, _, cfm_envelope_fluctuations_val, cfm_envelope_loglogavgslope_val] = exact_values_dict.values()
            fluctuations = (cfm_envelope_fluctuations_val, 1e-16)
            llslope = (cfm_envelope_loglogavgslope_val, 1e-16)
        cf_env = ift.SimpleCorrelatedField(target=s_dom, fluctuations=fluctuations, loglogavgslope=llslope,
                                           offset_mean=None, offset_std=None, flexibility=None, asperity=None,
                                            prefix="cfm_envelope_", use_uniform_prior_on_fluctuations=False).ptw("exp")
        op = cf_env * wavelet
    else:
        op = wavelet

    op.prior_choices = prior_choices
    op.ps = ps
    op.amp = amp_s_on_full_space


    return op


class SimpleMask(ift.LinearOperator):
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


class ClipOperator(ift.EndomorphicOperator):
    # When applied to a field, resets all field values that are larger than some threshold back to that threshold
    # itself in order to avoid overflow errors.
    # E.g. e^600 = some large number, e^700 = overflow error, so do e^C(700) = e^600 = large well-defined number
    # We use this to build a safe sigmoid operation. NIFTy cannot handle overflow errors (Numpy instead recasts the
    # overflow in 1/e^x for large x to 0).

    # For exp in 64bits, an overflow happens for an exponent of approximately 700.

    # C = ClipOperator(max=600, min=None, ...)
    #
    # C(x) = C([800, 100, 132,...]) = [700, 100, 132, ...]
    # C.adjoint(y) = C.adjoint([700, 100, 132, ...]) = [700, 100, 132, ...]

    # I.e. the adjoint operation is the identity, although one could think about making this operator unitary,
    # i.e. letting the adjoint be the inverse so that C.adjoint([700, 100, 132, ...]) = [800, 100, 132,...]
    # But really, it could also e.g. be that x_original = [1000, 100, 132,...]; so I guess it's better to stick with
    # the identity operator here.

    def __init__(self, domain, clip_above=None, clip_below=None):
        self._domain = domain
        self._target = domain
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self.clip_above = clip_above
        self.clip_below = clip_below

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            vals = x.val
            if self.clip_above is not None:
                vals = np.minimum(vals, self.clip_above)
            if self.clip_below is not None:
                vals = np.maximum(vals, self.clip_below)
            return ift.Field(dt(self._domain), vals)

        elif mode == self.ADJOINT_TIMES:
            return x

