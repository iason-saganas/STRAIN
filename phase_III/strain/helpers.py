import matplotlib.pyplot as plt
import numpy as np
from phase_II.nifty_re_playground.strain_tools import *

import jax
import jax.numpy as jnp
import nifty.re as jft

__all__ = ["_save_plot_wh_bp_data_with_template", "_save_plot_welch_average", "_metadata_basics",
           "jft_model_vjp_jvp_stability", "_error_metadata"]

_metadata_title = lambda title: f"******** {title}" + "\n"
_metadata_endline = lambda: f"\n----------------" + "\n\n"

def _save_plot_wh_bp_data_with_template(o, s, e, nr):
    """
    Whitend and bandpassed strain together with the best-fit numerical relativity template projection

    :param o:   odir_diagnostic_plots
    :param s:   strain from `get_strain_from_disc`
    :param e:   event_name
    :param nr:  NR from `get_waveform_template`
    :return:
    """
    fn = o + "whitened_bp_strain_and_best_fit_template"
    _ = plt.figure(figsize=(8., 4.,))
    plt.xlim(-.1, .1)
    plt.plot(s.event_time, s.event_strain_white_bp, label=f"{e} strain, whitened and bandpassed")
    plt.plot(nr.time, nr.strain / max(nr.strain) * max(s.event_strain_white_bp), label="Scaled best-fit template")
    thesis_plot(mode="longer", yl="Strain", show=False, save_fig=True, save_path=fn, close=True)

def _save_plot_welch_average(o, f, p):
    """
    The calculated Welch-average for this event

    :param o:   odir_diagnostic_plots
    :param f:   f_welch from strain.aux.f
    :param p:   ps_welch from strain.aux.ps_welch
    :return:
    """
    fn = o + "welch_average"
    _ = plt.figure(figsize=(8., 4.,))

    pos_freqs = np.where(f > 0)
    f_pos = f[pos_freqs]
    ps_pos = p[pos_freqs]
    plt.plot(f_pos, ps_pos, color="black", label="")
    plt.loglog()
    thesis_plot(mode="longer", yl="Power", xl=r"Frequency $f$ $\mathrm{[Hz]}$", show=False, save_fig=True, save_path=fn,
                title="Welch-averaged noise power spectrum", close=True)

def _metadata_basics(o, e, s, det, t_m, t_g, t_d):
    """

    :param o:       odir_diagnostic_metadata
    :param e:       event_name
    :param s:       strain (object from get_strain_from_disc)
    :param det:     detector
    :param t_m:     T_mini_welch
    :param t_g:     T_global_welch
    :param t_d:     data_duration_of_hdf5_file

    :return:
    """
    fn = o + "metadata_basics"
    with open(fn, "w") as f:
        f.write(_metadata_title("Basics"))
        f.write("\nEvent name: " + e)
        f.write("\nDetector: " + det)
        f.write("\nData duration acc. to hdf5 file (sec): " + str(t_d))
        f.write("\nData duration used (sec) for calculating Welch: " + str(t_g))
        f.write("\nData duration over which reconstruction will occur: " + str(t_m))
        f.write(_metadata_endline())

        f.write(_metadata_title("Welch average kwargs (might differ if scipy welch is used)"))
        for key, value in s.aux.meta.items():
            f.write(f"\n{key}: {value}")
        f.write(_metadata_endline())

    f.close()

def _error_metadata(o):
    fn = o + "readme.txt"
    with open(fn, "w") as file:
        file.write("If all went well, this directory is empty; otherwise it contains the last forward model evaluation \n"
                   "before the geoVi sampling algorithm returned a nan energy error. Unpickle the pickle file stored \n"
                   "in this file and analyze it by applying the forward model to the latent positions.\n")
        file.write("\n")
        file.write("In order for this functionality to work, currently, the following lines need to be added to the \n"
                   "function `residual_vg` defined inside `nifty.re.evi.nonlinearly_update_residual`:")

        txt = f"""
        import datetime
        import jax
        import pickle
        current_date = datetime.datetime.now()
        fn = '{o}nan_state_'+str(current_date)+'.pkl'
        def save_nan_state(position, energy, filename=fn):
            if jnp.isnan(energy):
                pos_host = jax.tree_util.tree_map(jax.device_get, position)
                with open(filename, "wb") as f:
                    pickle.dump(pos_host, f)
                print(f"Saved NaN state")
        jax.debug.callback(save_nan_state, x, res)
        """
        file.write(txt)
        file.close()

def jft_model_vjp_jvp_stability(fwd:jft.Model, key, num=10):
    """

    Test absolute error between forward and backward propagation using the given fw-model; see
    `_stability_of_jax_vjp_jvp` doc.

    :param fwd:     A jft.Model whose __call__ method represents the forward model and which has a domain attribute
                    from which samples can be drawn.
    :param key      A random key with which random samples can be generated.
    :param num      How many vjp and jvp's to compute. The final error will be averaged
    :return:        The absolute error between <J^T v, w> and <v, J w>.
    """
    error_list = []
    for _ in range(num):
        key, key_i, key_j = jax.random.split(key, 3)
        primals_jft = jft.random_like(key=key_i, primals=fwd.domain)  # random position in parameter space
        tangents_jft = jft.random_like(key=key_j, primals=fwd.domain)  # random direciton in parameter space

        err = _stability_of_jax_vjp_jvp(fwd, primals_jft, tangents_jft)
        error_list.append(err)

    print("Mean error between <J^T v, w> and <v, J w>: ", jnp.mean(jnp.array(error_list)))
    print("Don't forget to get key from `jft_model_vjp_jvp_stability`.")
    return key

def _stability_of_jax_vjp_jvp(fwd, primals, tangents):
    """
    JAX implements two interesting functions:

        fwd(xi), J_T = jax.vjp(fwd, xi)
        fwd(xi), J(v) = jax.jvp(fwd, xi, v),

    where v is the tangent at xi and J is the Jacobian. more precisely, from the documentation:

        "J_T is a function from a cotangent vector with the same shape as fwd(xi) [so data space]
        to a tuple of cotangent vectors with the same number and shapes as xi [so parameter space],
        representing the vector-Jacobian product" (I think this function computes the transpose of
         the vjp, so the transpose of the Jacobian because it says 'a tuple' instead of a row vector, which
         is what the vjp should be; if in later applications you take jnp.vdot(vjp(primals), v), this is then
         completely equivalent to the vjp.)

    By definition of the transpose, the following relation should hold:

        <J^T v, w> = <v, J w>

    for any two vectors v,w.
    This function checks the stability of the vjp and jpv products by comparing the left-hand and right-hand sides
    of this equation. In particular,

        J^T v = jax.vjp(fwd, xi)[1] (v)

    and

        J w = jax.jvp(fwd, xi, w)[1].

    Simple example:
    ------------------
    import jax
    import jax.numpy as jnp

    def forward_model(x):
        # Simple 2D -> 2D example
        return jnp.array([x[0]**2, x[0] + x[1]])

    xi = jnp.array([1.0, 2.0])
    tangent = jnp.array([0.1, -0.2])

    err = _stability_of_jax_vjp_jvp(forward_model, xi, tangent)
    print("VJP-JVP identity error:", err)
    ------------------

    :param fwd:         A function that represents the forward model.
    :param primals:     The latent parameter onto which to apply the forward model.
    :param tangents:    Tangent vector in parameter space (w in JVP), used to check the identity.
    :return:            The absolute error between <J^T v, w> and <v, J w>.
    """
    # Choose a random cotangent vector in data space for testing
    rng = jax.random.PRNGKey(0)
    v = jax.random.normal(rng, shape=jax.tree_util.tree_leaves(fwd(primals))[0].shape) # essentially a random vector
    # in data space
    w = tangents

    # VJP
    y, vjp_fun = jax.vjp(fwd, primals)
    jT_v, = vjp_fun(v)  # J^T v

    # JVP
    y, j_w = jax.jvp(fwd, (primals,), (w,))  # J w

    try:
        w_vec = jnp.concatenate([jnp.ravel(p) for p in w.values()])  # get rid of domain keys
        jT_v_vec = jnp.concatenate([jnp.ravel(p) for p in jT_v.values()])  # get rid of domain keys
    except AttributeError:
        pass  #  pure arrays and not dicts

    # Compare the inner products
    lhs = jnp.vdot(jT_v_vec, w_vec)  # <J^T v, w>
    rhs = jnp.vdot(v, j_w)  # <v, J w>
    error = jnp.abs(lhs - rhs)

    return error