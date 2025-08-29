import matplotlib.pyplot as plt
import numpy as np
import nifty8 as ift

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

import numpy as np

def _reshape_to_rows(arr, axis):
    """Move chosen axis to last and flatten leading dims -> shape (M, N)."""
    arr = np.asarray(arr)
    arr_moved = np.moveaxis(arr, axis, -1)
    prefix = arr_moved.shape[:-1]
    N = arr_moved.shape[-1]
    rows = arr_moved.reshape(-1, N)
    return rows, prefix, N

def _reshape_from_rows(rows, prefix):
    """Restore array shape from (M, N) with stored prefix shape."""
    N = rows.shape[1]
    out = rows.reshape(*prefix, N)
    return out

def hartley_to_fftshift(xi, axis=-1, flip_negatives=False):
    """
    Convert Hartley ordering -> zero-centered (fftshift) ordering along `axis`.
    Hartley ordering assumed: [0, pos..., (nyq if even), neg...]
    fftshift ordering: [neg..., 0, pos..., (nyq if even)]
    """
    rows, prefix, N = _reshape_to_rows(xi, axis)
    half = N // 2

    if N % 2 == 0:
        pos_len = half - 1
        zero = rows[:, 0:1]
        pos  = rows[:, 1:1 + pos_len]                # shape (M, pos_len)
        nyq  = rows[:, 1 + pos_len:1 + pos_len + 1]  # shape (M, 1)
        neg  = rows[:, 1 + pos_len + 1:]             # shape (M, pos_len)
        out_rows = np.concatenate([neg, zero, pos, nyq], axis=1)
        if flip_negatives and neg.size:
            out_rows[:, :neg.shape[1]] *= -1
    else:
        pos_len = half
        zero = rows[:, 0:1]
        pos  = rows[:, 1:1 + pos_len]
        neg  = rows[:, 1 + pos_len:]
        out_rows = np.concatenate([neg, zero, pos], axis=1)
        if flip_negatives and neg.size:
            out_rows[:, :neg.shape[1]] *= -1

    out = _reshape_from_rows(out_rows, prefix)
    return np.moveaxis(out, -1, axis)


def fftshift_to_hartley(xi_shifted, axis=-1, flip_negatives=False):
    """
    Inverse: zero-centered (fftshift) -> Hartley ordering [0, pos..., (nyq if even), neg...].
    Works along `axis`.
    """
    rows, prefix, N = _reshape_to_rows(xi_shifted, axis)
    half = N // 2

    if N % 2 == 0:
        pos_len = half - 1
        neg  = rows[:, :pos_len]                        # first block
        zero = rows[:, pos_len:pos_len + 1]
        pos  = rows[:, pos_len + 1: pos_len + 1 + pos_len]
        nyq  = rows[:, -1:]                             # last element
        if flip_negatives and neg.size:
            neg = -neg
        out_rows = np.concatenate([zero, pos, nyq, neg], axis=1)
    else:
        pos_len = half
        neg  = rows[:, :pos_len]
        zero = rows[:, pos_len:pos_len + 1]
        pos  = rows[:, pos_len + 1:]
        if flip_negatives and neg.size:
            neg = -neg
        out_rows = np.concatenate([zero, pos, neg], axis=1)

    out = _reshape_from_rows(out_rows, prefix)
    return np.moveaxis(out, -1, axis)


def dt_(dom):
    return ift.DomainTuple.make(dom)

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

    UPDATE: I found out what was happening. If the variance of alpha or beta was chosen to be very big, then it could
    be that e.g. beta was drawn as -100. As I noticed earlier, this would cause a problem for high key in 10^(gamma*beta),
    even though gamma scales with the log of k. But there is also the usage of beta in b0=k_0^-beta.
    So if now k0 was drawn as 1000, we do the operation  1000^100 which also gives an overflow error.

    Since the "worst" operation I do in this sense is k0^-beta, and if I fix beta to range strictly from 0 to -50
    which covers the most interesting ranges, k0 at most should be allowed to be of the order of ~1e6.

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

    k = harmonic_space.get_unique_k_lengths()
    power_at_zm = 1e-32  # The zm should, by assumption, be ~0. Setting it to small value to avoid nan's.
    # --- Prior choices
    prior_choices = {
        # "k0 ": (10, 200),  # Uniform
        # "k0 ": (1, 20),  # Uniform
        "k0 ": (20, 5),  # Gaussian
        "p0 ": (1e3, 1e-16),  # Gaussian
        "c ": (100, 1e-16),  # Gaussian
        "alpha ": (1,2),  # Gaussian
        # "alpha ": (2,2),  # Lognormal
        # "alpha ": (1, None),  # Exponential
        # "alpha ": (0,+20),  # Uniform
        "beta ": (-4,2),  # Gaussian
        # "beta ": (4,4),  # Lognormal
        # "beta ": (-1, None),  # Exponential
        # "beta ": (0,-20),  # Uniform
        "wavelet_fluct ": (4, 2), # Lognormal
        "cfm_envelope_fluctuations ": (4, 2),
        "cfm_envelope_loglogavgslope ": (-4, 1)
    }
    # ---------

    s_dom = harmonic_space.get_default_codomain()
    p_space = ift.PowerSpace(harmonic_space)

    k[0] = 1  # I do this so np.log10(k) won't diverge at k[0]=0. Later I will set the power at k[0] to a fixed value
    # anyway.

    k_field = ift.Field(dt_(p_space), val=k)
    gamma_field = ift.Field(dt_(p_space), val=np.log10(k))
    gamma_op = ift.DiagonalOperator(gamma_field)

    # k0 = ift.StandardUniformTransform(key="k0 ", shift=prior_choices["k0 "][0], upper_bound=prior_choices["k0 "][1]) # don't confuse with k[0], this is the k value where there is a peak in the power spectrum
    k0 = ift.NormalTransform(*prior_choices["k0 "], key="k0 ")
    p0 = ift.NormalTransform(*prior_choices["p0 "], key="p0 ")
    c = ift.NormalTransform(*prior_choices["c "], key="c ")
    alpha = ift.NormalTransform(*prior_choices["alpha "], key="alpha ")
    # alpha = ift.LognormalTransform(*prior_choices["alpha "], key="alpha ", N_copies=0)
    # alpha = ift.ExponentialTransform(prior_choices["alpha "][0], key="alpha ", N_copies=0)
    # alpha = ift.StandardUniformTransform(key="alpha ", upper_bound=prior_choices["alpha "][1])
    beta = ift.NormalTransform(*prior_choices["beta "], key="beta ")
    # beta = ift.ExponentialTransform(prior_choices["beta "][0], key="beta ", N_copies=0)
    # beta = ift.LognormalTransform(*prior_choices["beta "], key="beta ", N_copies=0)
    # beta = ift.StandardUniformTransform(key="beta ", upper_bound=prior_choices["beta "][1])
    wavelet_fluct = ift.LognormalTransform(*prior_choices["wavelet_fluct "], key="wavelet_fluct ", N_copies=0)

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
    wavelet_fluct = pspace_expander @ wavelet_fluct

    k_field_adder = ift.Adder(a=k_field)
    # add_one = ift.Adder(a=ift.Field(dt(p_space), val=np.ones(p_space.shape[0])))
    exponent = -c*(k_field_adder(-1*k0))

    a0 = p0 * ( -alpha * np.log10(k0) ).ptw("exponentiate", 10)
    b0 = p0 * ( -beta * np.log10(k0) ).ptw("exponentiate", 10)

    nifty_sigmoid = (-1*exponent).ptw("sigmoid")
    sigmoid_to_use = nifty_sigmoid

    ps = (a0*(gamma_op @ alpha).ptw("exponentiate", 10)-sigmoid_to_use*a0*(gamma_op @ alpha).ptw("exponentiate", 10)
          + sigmoid_to_use*b0*(gamma_op @ beta).ptw("exponentiate", 10))

    tmp = np.ones(p_space.shape)
    tmp[0] = power_at_zm
    set_zm_power = ift.makeOp(ift.makeField(p_space, tmp))

    ps = set_zm_power(ps)

    integrator = ift.ContractionOperator(p_space, spaces=0)
    integral = integrator(ps) # scalar
    integral = pspace_expander @ integral # field

    ps = ps * integral.ptw("reciprocal")

    ps = wavelet_fluct * ps

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


def _edges_from_centers(c):
    c = np.asarray(c, dtype=float)
    n = c.size
    if n == 1:
        h = 0.5 if c[0] == 0 else abs(c[0])*0.1
        return np.array([c[0]-h, c[0]+h])
    mid = (c[:-1] + c[1:]) / 2.0
    left = c[0] - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])


def visualize_stress_pcolormesh(stress_matrix, rows, cols,
                               cmap='viridis', shading='auto',
                               xlabel='Time [s]', ylabel='Frequency [Hz]',
                               sort_axes=False, vmin=None, vmax=None):
    """
    Plot stress_matrix with x axis=cols (horizontal) and y axis=rows (vertical).
    - rows: 1D array of y-centers (len == stress_matrix.shape[0])
    - cols: 1D array of x-centers (len == stress_matrix.shape[1])
    - if sort_axes=True the function will sort coords (useful if coords unordered)
    """
    Z = np.asarray(stress_matrix)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    if Z.shape != (rows.size, cols.size):
        raise ValueError(f"stress_matrix shape {Z.shape} does not match rows {rows.size}, cols {cols.size}")

    # optionally sort (keeps mapping to data)
    if sort_axes:
        r_idx = np.argsort(rows)
        c_idx = np.argsort(cols)
        rows = rows[r_idx]; cols = cols[c_idx]
        Z = Z[r_idx][:, c_idx]

    # build edges from centers (works for non-uniform spacing)
    y_edges = _edges_from_centers(rows)
    x_edges = _edges_from_centers(cols)

    fig, ax = plt.subplots(figsize=(8,6))
    mesh = ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Stress')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Time vs Frequency')
    plt.tight_layout()
    plt.show()
    return mesh
