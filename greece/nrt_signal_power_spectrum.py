import matplotlib.pyplot as plt

from utils.generative_models import *

def domain_tuple(domainoid):
    return ift.DomainTuple.make(domainoid)

n_dtps = len(nrt_time_values)
L = max(nrt_time_values) - min(nrt_time_values)
pxl_vol = L/n_dtps
data_domain = ift.RGSpace(shape=(n_dtps,), distances=pxl_vol)
data_domain_harmonic = data_domain.get_default_codomain()

HT = ift.HartleyOperator(data_domain, data_domain_harmonic)
power_spectrum_domain = ift.PowerSpace(data_domain_harmonic)
dof_distributor = ift.PowerDistributor(data_domain_harmonic)

def p_s(k):
    ### ---- parameters for the user to set
    std_to_set = 1
    variance_around_mean_field = 0
    power_law_exp_left = 2
    power_law_exp_right = -6
    k_cut_off = 50
    ### ----

    ### ---- Rename variables
    a = std_to_set**2
    zm = variance_around_mean_field
    ### ----

    k_cut_off_index = np.argmin(np.abs(k - k_cut_off))
    k_cut_approx = k[k_cut_off_index]

    def broken_power_law(k_modes, k_star, alpha, beta):
        left = []
        right = []
        for idx, k_mode in enumerate(k_modes):
            if k_mode < k_star:
                if k_mode == 0:
                    left.append(0)
                else:
                    left.append(k_mode ** alpha)
            else:
                right.append(k_mode ** beta)
        left, right = np.array(left), np.array(right)
        right = right / right[0] * left[-1]
        return np.concatenate((left, right))

    power_law = broken_power_law(k, k_cut_approx, power_law_exp_left, power_law_exp_right)

    vol = np.trapz(power_law[1:], k[1:])
    power_law[0] = zm

    res = a*power_law/vol

    return res

def p_s_sqrt(k):

    res = p_s(k)
    return np.sqrt(res)

k_array_reduced = data_domain_harmonic.get_unique_k_lengths()

plt.title("Power spectrum (non-square-rooted)")
plt.plot(k_array_reduced, p_s(k_array_reduced), "b.")
plt.loglog()
plt.show()

ps_sqrt = ift.Field(domain_tuple(power_spectrum_domain), p_s_sqrt(k_array_reduced))

ps_sqrt_full = dof_distributor(ps_sqrt)
ps_diag_sqrt = ift.DiagonalOperator(ps_sqrt_full)
k_field = data_domain_harmonic.get_k_length_array()

S_sqrt = HT.adjoint @ ps_diag_sqrt @ HT * (1/(pxl_vol**2 * n_dtps))

envelope = ift.SimpleCorrelatedField(target=data_domain, offset_mean=None, offset_std=None,
                                     fluctuations=(1,0.5), flexibility=None, asperity=None,
                                     loglogavgslope=(-4,1), prefix="cfm envelope ").ptw("exp")


for idx in range(1):

    xi_vals = np.random.normal(loc=0, scale=1, size=n_dtps)

    xi_w = ift.Field(S_sqrt.domain, xi_vals)
    xi_e = ift.from_random(envelope.domain)

    wavelet = S_sqrt(xi_w)
    env = envelope(xi_e)
    # s_val = env.val * wavelet.val
    s_val = wavelet.val  # just the wavelet without the envelope

    if idx == 0:
        plt.plot(nrt_time_values, s_val, label="wavelet sample")
        plt.plot(nrt_time_values, env.val/max(env.val)*max(wavelet.val), label="cfm envelope", ls="--")
    else:
        plt.plot(nrt_time_values, s_val)
        plt.plot(nrt_time_values, env.val / max(env.val) * max(wavelet.val), ls="--")
plt.legend()
plt.show()