import matplotlib.pyplot as plt
import nifty8 as ift
import pickle
import warnings
from matplotlib.patches import Rectangle
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt

def usual_plot(xl=r"Time $t$ $\mathrm{[sec]}$", yl=r"Strain $h$ $\mathrm{[10^{-19}]}$", title=None, xlim=None, ylim=None,
               show=True):
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    ax = plt.gca()
    labels = ax.get_legend_handles_labels()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if show:
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
        "k0 ": (4, 11),  # Uniform
        # "k0 ": (1, 20),  # Uniform
        # "k0 ": (20, 5),  # Gaussian
        # "p0 ": (1e3, 1e-16),  # Gaussian
        "p0 ": (1, 1),  # Lognormal
        "c ": (100, 1e-16),  # Gaussian
        "alpha ": (0,1),  # Gaussian
        # "alpha ": (2,2),  # Lognormal
        # "alpha ": (1, None),  # Exponential
        # "alpha ": (0,+20),  # Uniform
        "beta ": (-6,2),  # Gaussian
        # "beta ": (4,4),  # Lognormal
        # "beta ": (-1, None),  # Exponential
        # "beta ": (0,-20),  # Uniform
        "wavelet_fluct ": (12000, 1e-16), # Lognormal
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

    k0 = ift.StandardUniformTransform(key="k0 ", shift=prior_choices["k0 "][0], upper_bound=prior_choices["k0 "][1]) # don't confuse with k[0], this is the k value where there is a peak in the power spectrum
    # k0 = ift.NormalTransform(*prior_choices["k0 "], key="k0 ")
    # p0 = ift.NormalTransform(*prior_choices["p0 "], key="p0 ")
    p0 = ift.LognormalTransform(*prior_choices["p0 "], key="p0 ", N_copies=0)
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

    wavelet = ht.inverse(amp_s_on_full_space * xi_s)

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



def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def pickle_me_this(filename: str, data_to_pickle: object):
    path = filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()




def mad_outlier_positions(x, y, threshold=2.0, use_magnitude=False, one_sided=False):
    """
    Possibly delete later.
    Detect MAD outliers in y and return their corresponding x positions and z-scores.

    Parameters
    ----------
    x : 1D array
        Positions corresponding to y-values.
    y : 1D array (real or complex)
        Data values to detect outliers in.
    threshold : float
        Cutoff in scaled-MAD units.
    use_magnitude : bool
        If True and y is complex, use abs(y) for detection.
    one_sided : bool
        If True, detect only points above the median.

    Returns
    -------
    x_hits : array of x positions corresponding to outliers
    z_hits : array of MAD-scaled scores at those positions
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if np.iscomplexobj(y) and use_magnitude:
        y_proc = np.abs(y)
    else:
        y_proc = y.real

    med = np.median(y_proc)
    mad = np.median(np.abs(y_proc - med))
    if mad == 0:
        mad = np.mean(np.abs(y_proc - med)) + 1e-12

    z = 1.4826 * (y_proc - med) / mad

    if one_sided:
        mask = z > threshold
    else:
        mask = np.abs(z) > threshold

    x_hits = x[mask]
    z_hits = z[mask]

    return x_hits, z_hits


from scipy.ndimage import median_filter


def remove_baseline(signal, window=101, method='median'):
    """
    Possibly delete later.

    Remove baseline from a 1D signal.

    signal : array-like
        Input data (can have peaks)
    window : int
        Window length for local baseline estimation (odd number)
    method : str
        'median' or 'mean' for smoothing
    """
    signal = np.asarray(signal)
    if window % 2 == 0:
        window += 1  # enforce odd

    if method == 'median':
        baseline = median_filter(signal, size=window, mode='reflect')
    elif method == 'mean':
        # simple running mean
        kernel = np.ones(window) / window
        baseline = np.convolve(signal, kernel, mode='same')
    else:
        raise ValueError("method must be 'median' or 'mean'")

    residual = signal - baseline
    return residual, baseline



def fieldify(array, dom):
    return ift.Field(dt_(dom), array)


def raise_warning(msg):
    warnings.warn(msg, category=UserWarning, stacklevel=2)



def Stress(xi_field: ift.Field, supress_print=False):
    """

    DFT conventions:
        F[m] = sum_{n=0}^{N-1} f[n] * exp(2πi * m n / N)
        f[n] = (1/N) * sum_{m=0}^{N-1} F[m] * exp(-2πi * m n / N)

    :param xi_field:  ift.Field    The Field over real space to analyze. If harmonic, assumed to be in DFT standard order and is mapped to its real space counterpart.
    :param supress_print:          Whether to print imaginary part diagnostic.
    :return: (S_mat, t_dual, f)    S_mat is the calculated wigner matrix, t_dual is the dual time from fourier transforming
                                   the columns of an intermediate matrix in ascending, monotonic order and f are the
                                   frequencies of the setup in standard DFT order (i.e. 0 as the first element).

    """

    if xi_field.domain[0].harmonic:
        helper_fft = ift.FFTOperator(xi_field.domain, space=0)
        xi_time_field = helper_fft(xi_field)
    else:
        xi_time_field = xi_field


    time_space = xi_time_field.domain[0]
    h_space = time_space.get_default_codomain()

    N = time_space.size
    t_vol = time_space.scalar_dvol
    h_vol = h_space.scalar_dvol * N

    dt_step = time_space.distances[0]
    xi_time = xi_time_field.val.astype(np.complex128)
    N = len(xi_time)
    f = np.fft.fftfreq(N, d=dt_step)
    k = f.copy()
    t = np.arange(N)*dt_step
    time = np.arange(N)*dt_step

    FFT_1 = ift.FFTOperator(domain=(time_space, h_space), space=0) * (1/t_vol)
    FFT_2 = ift.FFTOperator(domain=(h_space, h_space), space=1) * (1/h_vol)

    time_cast = time[:, None]
    k_freq_cast = k[None, :]
    xi_values_cast_as_rows = xi_time[:, None]

    print("\nCalculating stress...")

    print("\t Calculating zeta plus")
    zeta_plus = fieldify(np.exp(-np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))
    print("\t Calculating zeta minus")
    zeta_minus = fieldify(np.exp(np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))

    print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT_1(zeta_plus).val
    print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT_1(zeta_minus).val

    print("\t Calculating Phi matrix")
    Phi_val = tilde_zeta_plus * tilde_zeta_minus.conj()  # im putting the conjugate on the MINUS zeta since I also changed FFT convention by a sign wrt. Wikipedia...
    Phi_field = ift.Field(dt_((h_space, h_space)), val=Phi_val)

    print("\t Fourier-Transforming columns of Phi matrix")
    S = FFT_2(Phi_field)
    S_mat = S.val
    print("\t ... Done")

    dk = k[1] - k[0]        # safe in FFT ordering (first step is Δk)
    dt_dual = 1.0 / (N * dk)
    t_dual = np.arange(N) * dt_dual

    if not supress_print:
        diagnostic = np.abs(np.mean(S_mat.imag))
        if diagnostic < 1e-10:
            print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
        else:
            raise_warning(
                f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")

    return S_mat, t_dual, f


def visualize_stress(stress_matrix, rows, cols, tl="", hlines=None):

    stress_matrix = stress_matrix.real

    cols_are_increasing = np.all(np.diff(cols) > 0)  # strictly increasing
    rows_are_increasing = np.all(np.diff(rows) > 0)  # strictly increasing
    if not cols_are_increasing:
        raise ValueError("Columns must be increasing")
    if not rows_are_increasing:
        stress_matrix = np.fft.fftshift(stress_matrix, axes=0)  # shift DC frequency to middle
        print("\t\tRows must be increasing, assuming a priori standard DFT order and moving DC to the middle")

    plt.figure(figsize=(8,6))
    plt.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap='viridis', interpolation='nearest')

    if hlines is not None:
        plt.hlines(hlines, 0, 1.75, color="r", ls="-")
    plt.colorbar(label='Stress')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [s]')
    plt.title('Time vs Frequency' + tl)
    plt.tight_layout()
    plt.show()


def generative_gaussian_comb(x_field: ift.Field, position_of_peaks:np.array, amplitude_of_peaks:np.array, half_width_of_peaks:np.array,
                             rel_sigma_lognormal=.1, rel_sigma_normal=0.5):
    """

    In the future, we probably should replace this with a probabilistic framework, an IFT-based peak finding algorithm;
    although being FAST would be nice. Ask in the group.

        gaussian = amp * np.exp(-0.5 * ((f - freq_center) / freq_width) ** 2)
        list_of_single_gaussians.append(gaussian)

        return sum(list_of_single_gaussians)

    :param rel_sigma_normal:                    How large is the standard deviation of normal transforms in terms of
                                                the mean.
    :param rel_sigma_lognormal:                 Same but for lognormals.
    :param x_field:                             The field values over which the comb is defined such that x_field.domain
                                                gives the domain in which everything takes place (e.g. power or harmonic
                                                domain).
    :param position_of_peaks:                   Human guess of position of peaks.
    :param amplitude_of_peaks:                  Human guess of amplitude of peaks.
    :param half_width_of_peaks:                      Human guess of the frequency HALF-width of peaks.

    :return: An operator that produces a Gaussian comb with parameters being each position, amplitude and width of the
             peaks.
    """
    list_of_single_gaussians = []
    running_idx = -1
    for f_cen, amplitude, f_half_width in zip(position_of_peaks, amplitude_of_peaks, half_width_of_peaks):
        running_idx += 1

        # amp = ift.NormalTransform(mean=amplitude, sigma=amplitude*rel_sigma_normal, key=f"gauss_peak_{running_idx}_amp", N_copies=0)
        # peak_center = ift.NormalTransform(mean=f_cen, sigma=f_cen*rel_sigma_normal, key=f"gauss_peak_{running_idx}_center", N_copies=0)

        amp = ift.LognormalTransform(mean=amplitude, sigma=amplitude*rel_sigma_lognormal, key=f"gauss_peak_{running_idx}_amp", N_copies=0)
        sigma = ift.LognormalTransform(mean=f_half_width, sigma=f_half_width*rel_sigma_lognormal, key=f"gauss_peak_{running_idx}_sigma", N_copies=0)
        peak_center = ift.LognormalTransform(mean=f_cen, sigma=f_cen*rel_sigma_lognormal, key=f"gauss_peak_{running_idx}_center", N_copies=0)



        expander = ift.ContractionOperator(x_field.domain, spaces=0).adjoint
        amp, peak_center, sigma = (expander @ amp, expander @ peak_center, expander @ sigma)

        x_adder = ift.Adder(a=x_field, neg=True)
        kernel = ( x_adder(peak_center) * sigma.ptw("reciprocal") ) ** 2
        gaussian = amp * (-1/2 * kernel).ptw("exp")

        list_of_single_gaussians.append(gaussian)

    op = None
    for idx, op_i in enumerate(list_of_single_gaussians):
        if idx == 0:
            op = op_i
        else:
            op = op + op_i

    return op


def welch_average(x, y, L, leave_out=None, debug=False, tapering_function=None,
                  output_on_full_harmonic_domain=False):
    """
    Subdivides data into little windows of length L. The data in the subwindows are zero-padded to ensure periodic
    boundary conditions, their fourier-transform absolutely squared and an average over all absolute squares is
    performed to get an estimate of the empirical power spectrum.

    All outputs are always in DFT standard order, i.e. 0 frequency first, then positive then negative frequencies.

    The windows have no overlap.

    EDIT: I am tapering the data, zero-padding a fixed dataset only increases the harmonic resolution, doesn't do
    anything against the boundary conditions.

    Note: Let the full dataset be of length L_1 and the small windows to subdivide it into L_2. The resolution of the
    power spectrum estimated by this function is 1/L_2 > 1/L_1, i.e. it is too coarse to be directly applied to
    the full dataset. Therefore, we can only whiten short records of length L_2 using this method.

    One could possibly interpolate in the future.

    :param x:
                                A np.array of time values over which the data are sampled.
    :param y:
                                The data. np.array or ift.Field
    :param L:
                                The real space length of little windows to subdivide the data under.
    :param debug:
                                If true plots diagnostics.
    :param tapering_function:
                                The callable to use to enforce periodic boundary conditions on the data. Default
                                is a Tukey window.
    :param leave_out:
                                A tuple (t_init, t_final) to exempt from the average. The dataset is then split into dataset 1
                                (t<t_init) and dataset 2 (t>t_final). The procedure is performed on dataset 1 and dataset 2
                                and their tranfsorms averaged again.
    :param output_on_full_harmonic_domain:
                                Whether the returned field is power distributed to the full harmonic domain.
    :return:
                                unique_k_lengths, power_spectrum if output_on_full_harmonic_domain == False
                                freqs, full_power_spectrum if output_on_full_harmonic_domain == True where
                                freqs contains negative frequencies as well.
    """
    if type(y) == ift.Field:
        y = y.val

    L_global = np.max(x)-np.min(x)
    if L_global < L:
        raise ValueError(f"Length of windows {L} larger than length of dataset {L_global}.")

    if leave_out is not None:
        x_init, x_final = leave_out

        cond_1 = np.where(x < x_init)
        cond_2 = np.where(x > x_final)
    else:
        cond_1 = np.where(x < np.inf)
        cond_2 = np.where(x > (-np.inf))

    y_strip_1 = y[cond_1]
    y_strip_2 = y[cond_2]

    x_in_strip_1 = x[cond_1]
    x_in_strip_2 = x[cond_2]

    if debug:
        welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2)

    lf_edges_ds1, r_edges_ds1 = get_lr_edges(x_in_strip_1, y_strip_1, L)
    lf_edges_ds2, r_edges_ds2 = get_lr_edges(x_in_strip_2, y_strip_2, L)

    num = len(lf_edges_ds1) + len(lf_edges_ds2) if leave_out is not None else len(lf_edges_ds1)
    print(f"\nConstructing {num} windows over which we average.\n")

    if debug:
        welch_average_debug_plot_II(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2,
                                    lf_edges_ds1, r_edges_ds1, lf_edges_ds2, r_edges_ds2)

    collection_of_small_datasets_strain = []
    collection_of_small_datasets_times = []

    for left_lim, right_lim in zip(lf_edges_ds1, r_edges_ds1):
        idcs = np.where((x > left_lim) & (x < right_lim))
        collection_of_small_datasets_strain.append(y[idcs])
        collection_of_small_datasets_times.append(x[idcs])

    if leave_out is not None:
        for left_lim, right_lim in zip(lf_edges_ds2, r_edges_ds2):
            idcs = np.where((x > left_lim) & (x < right_lim))
            collection_of_small_datasets_strain.append(y[idcs])
            collection_of_small_datasets_times.append(x[idcs])
    else:
        # dataset 1 and dataset 2, so just append one of them.
        pass

    if debug:
        welch_average_debug_plot_III(collection_of_small_datasets_strain, collection_of_small_datasets_times, x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2)

    if tapering_function is None:
        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)

    collection_of_small_datasets_strain_windowed = [d * tapering_function(d) for d in
                                                    collection_of_small_datasets_strain]

    if debug:
        welch_average_debug_plot_IV(collection_of_small_datasets_times, collection_of_small_datasets_strain,
                                    collection_of_small_datasets_strain_windowed)


    n_dtps = len(collection_of_small_datasets_times[0])
    time_domain = ift.RGSpace((n_dtps,), distances=L / n_dtps)
    F = ift.FFTOperator(_dt(time_domain))

    data_fields = [ift.Field(_dt(time_domain), val=small_data) for small_data in collection_of_small_datasets_strain_windowed]
    harmonic_data_fields = [F(df) for df in data_fields]

    empirical_power_spectra = [ift.power_analyze(h_df) for h_df in harmonic_data_fields]
    empirical_power_spectra_vals = [el.val for el in empirical_power_spectra]


    p_space = empirical_power_spectra[0].domain[0]
    unique_k_lengths = p_space.k_lengths
    power_spectrum_val = np.mean(empirical_power_spectra_vals, axis=0)
    power_spectrum = ift.Field(_dt(p_space), val=power_spectrum_val)

    if output_on_full_harmonic_domain:
        pd = ift.PowerDistributor(p_space._harmonic_partner)
        full_power_spectrum = pd(power_spectrum)
        freqs = np.fft.fftfreq(n=n_dtps, d=1/L)
        return freqs, full_power_spectrum
    else:
        return unique_k_lengths, power_spectrum



def _dt(dom):
    return ift.DomainTuple.make(dom, )


def get_lr_edges(x, y, L):
    lf_edges = np.arange(min(x), max(x), L)
    rght_edges = (lf_edges + L)[:-1]
    return lf_edges, rght_edges



def welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=True, alpha=1.):
    _ = plt.figure()
    plt.plot(x_in_strip_1, y_strip_1, label="Data in strip 1", alpha=alpha)
    plt.plot(x_in_strip_2, y_strip_2, label="Data in strip 2", alpha=alpha)
    usual_plot(title="Data (with a potential mask in a selected region)", show=show)


def welch_average_debug_plot_II(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2,
                                lf_edges_ds1, r_edges_ds1, lf_edges_ds2, r_edges_ds2):
    welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=False, alpha=.2)

    ax = plt.gca()

    # Plot Rectangles to visualize
    heights = [4, 4.5, 5] * len(lf_edges_ds1)
    for i, (left, right) in enumerate(zip(lf_edges_ds1, r_edges_ds1)):
        width = right - left
        if i % 2 != 0:
            c = (1, 0, 0)
            lw = 1
            height = 4
        else:
            c = (0, 0, 1)
            lw = 2
            height = 5
        rect = Rectangle((left, 0), width, heights[i], facecolor='none', linewidth=lw, edgecolor=c)
        ax.add_patch(rect)

    # Plot Rectangles to visualize
    for i, (left, right) in enumerate(zip(lf_edges_ds2, r_edges_ds2)):
        width = right - left
        if i % 2 != 0:
            c = (1, 0, 0)
            lw = 1
            height = 4
        else:
            c = (0, 0, 1)
            lw = 2
            height = 5
        rect = Rectangle((left, 0), width, heights[i], facecolor='none', linewidth=lw, edgecolor=c)
        ax.add_patch(rect)

    plt.text(-2, -4,
             f"Everything not in these windows, I will throw away.\n Number of windows: {len(lf_edges_ds1) + len(lf_edges_ds2)}"
             f"if something cut out, else divide by 2.",
             fontsize=12)
    plt.xlabel("Time [s]", fontsize=12)
    plt.show()


def welch_average_debug_plot_III(collection_of_small_datasets_strain, collection_of_small_datasets_times, x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2):

    welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=False, alpha=.2)

    for d, t in zip(collection_of_small_datasets_strain, collection_of_small_datasets_times):
        plt.plot(t, d)

    usual_plot(title="Data windows to be used in the average; full dataset has alpha ~ 0.2")


def welch_average_debug_plot_IV(collection_of_small_datasets_times, collection_of_small_datasets_strain, collection_of_small_datasets_strain_windowed):
    fig5 = plt.figure(figsize=(10, 6))
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Strain $[10^{-19}]$")

    plt.plot(collection_of_small_datasets_times[0], collection_of_small_datasets_strain[0], label="Example of original data in a window")
    plt.plot(collection_of_small_datasets_times[0], collection_of_small_datasets_strain_windowed[0], label="The same but tapered")
    usual_plot()



def whiten(y:np.array, amp:np.array, tapering_function = None):
    """

    :param y:                       The real-space data to whiten.
    :param amp:                     The custom amplitude spectrum to whiten with.
    :param tapering_function:       The tapering function to use; default is a Tukey window.
    :return:
    """
    if tapering_function is None:
        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)

    y = y*tapering_function(y)
    y_harmonic = np.fft.fft(y)
    whitened_y_harmonic = y_harmonic / amp
    return np.fft.ifft(whitened_y_harmonic).real

def bandpass(x, y, bp=(35, 350)):
    # copy of the function found in the file 'matched_filter_in_action.ipynb'
    dx = x[1] - x[0]
    fs = 1/dx
    low, high = bp
    res = butter(4, [low * 2. / fs, high * 2. / fs], btype='band', output='ba')
    bb, ab = res[0], res[1]
    normalization = np.sqrt((high - low) / (fs / 2))
    strain_bp = filtfilt(bb, ab, y) / normalization
    return strain_bp



