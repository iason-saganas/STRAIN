import os
from strain import *
from phase_II.nifty_re_playground.strain_tools import *

# ORIGINAL SDE SIGNAL RECONSTRUCTION

# from signal_reconstruction import *
#
# latent_samples = list(latent_samples)
#
# """
# Domain keys:
#
# {'gamma_cfm_fluctuations': Array(-0.16321983, dtype=float64), 'gamma_cfm_loglogavgslope': Array(0.08429145, dtype=float64), 'gamma_cfm_xi': Array([ 0.89547452, -0.63307033, -1.09459588, ...,  0.22184686,
#        -0.34380633, -0.13868742], dtype=float64), 'gamma_cfm_zeromode': Array(1.66552098, dtype=float64), 'omega_cfm_fluctuations': Array(0.91217251, dtype=float64), 'omega_cfm_loglogavgslope': Array(0.02047873, dtype=float64), 'omega_cfm_xi': Array([-0.59733346,  0.45187805,  1.09057151, ..., -1.05291997,
#         0.03313896, -2.41678309], dtype=float64), 'omega_cfm_zeromode': Array(-0.29671787, dtype=float64), 'scaling_': Array(0.27049293, dtype=float64), 'xi_cfm_fluctuations': Array(0.95022348, dtype=float64), 'xi_cfm_loglogavgslope': Array(1.42611434, dtype=float64), 'xi_cfm_xi': Array([ 0.30924953,  1.87874513, -1.01017265, ..., -1.07421293,
#        -0.76328741, -1.97841914], dtype=float64), 'xi_cfm_zeromode': Array(-1.63242172, dtype=float64)}
#
# """
#
# for sl in latent_samples:
#
#     xi = np.random.standard_normal(len(cfm_times))
#
#     domain_copy = sl._tree.copy()
#     domain_copy['xi_cfm_xi'] = xi
#     new_vec = jft.Vector(domain_copy)
#
#     sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet,
#                                                             omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
#                                                             key=key, custom_latent_position=sl, tl="posterior")
#
#     sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet,
#                                                             omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
#                                                             key=key, custom_latent_position=new_vec, tl="same but "
#                                                                                                         "relaxed xi cfm xi")

# UPDATED SDE SIGNAL RECONSTRUCTION

from signal_reconstruction_debug_debug import *

direc = "signal_reconstruction_sde_DEBUG_DEBUG/custom_callback/latent_posterior_means/"
latent_mean_files = os.listdir(direc)

latent_mean_files = sorted(
    latent_mean_files,
    key=lambda x: int(x.split('.')[0])
)

for file in latent_mean_files:
    iter_number = file.split(".")[0]  # for some reason gotten file list is not ordered by number
    latent_mean = unpickle_me_this(direc+file)

    sample_waveform, key = draw_and_plot_field_realizations(times=pipe_3.t_ss, diff_eq_solver_model=oscillator,
                                                                omega_op=oscillator.omega, gamma_op=oscillator.gamma,
                                                            xi_op=oscillator.xi_force,
                                                                key=key, custom_latent_position=latent_mean,
                                                            tl=f"Waveform from MEAN latent in iter {iter_number}")