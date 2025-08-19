#%%
# %matplotlib inline
#%%
import numpy as np
import matplotlib.pyplot as plt
from initial_ideas.utils.minimization_control import *
import pickle
#%%
# Construct ground truth model
cfm_params = {
    "fluctuations": (1,1),
    "loglogavgslope": (-3,1),
    "asperity": None,
    "flexibility": None,
    "offset_mean": 0,
    "offset_std": (1,2)
}

num_of_pixels = 2048
max_time = 1
time_domain = ift.DomainTuple.make(ift.RGSpace((num_of_pixels, ), distances=max_time/num_of_pixels), )
time_field = ift.Field(ift.DomainTuple.make(time_domain, ), val=np.linspace(0, max_time, num_of_pixels))

signal_model = ift.SimpleCorrelatedField(time_domain, **cfm_params)
#%%
# Extract random ground truth field
signal = signal_model(ift.from_random(signal_model.domain))
#%%
plt.plot(time_field.val, signal.val, "b-")
plt.xlabel("Time")
plt.ylabel("Signal value")
plt.title("Ground Truth")
plt.show()
#%% md
# Now, map this into data space by applying a masking operator and noise. For starters, model the noise through a scaling operator.
#%%
class Mask(ift.LinearOperator):
    def __init__(self, domain, num_of_points_to_keep):
        self._domain = domain
        self._target = ift.DomainTuple.make(ift.UnstructuredDomain((num_of_points_to_keep, )))
        self._capability = self.TIMES | self.ADJOINT_TIMES

        rng = np.random.default_rng()  # create a Generator instance
        self._where_to_keep = rng.integers(low=0, high=domain.shape, size=num_of_points_to_keep)

    def apply(self, x, mode):
        self._check_input(x, mode)
        # x is a field
        if mode == self.ADJOINT_TIMES:
            values = np.zeros(self._domain.shape[0])
            values[self._where_to_keep] = x.val
            return ift.Field(self._domain, values)

        elif mode == self.TIMES:
            extract = np.array(x.val)
            idcs = np.array(self._where_to_keep)
            values = extract[idcs]
            return ift.Field(self._target, values)

num_of_data = 1000
mask = Mask(time_domain, num_of_data)
data_domain = ift.DomainTuple.make(ift.UnstructuredDomain((num_of_data, ), ))

#%%
masked_signal_field = mask(signal)  # now of length 208
hermitianMaskedSignal = mask.adjoint(masked_signal_field)
masked_time_field = mask(time_field)
plt.plot(masked_time_field.val, masked_signal_field.val, "g.")
plt.title("Masked Data")
plt.show()
#%% md
# Now, bury this signal in noise of the same order of magnitude from a know operator
#%%
noise_operator = ift.ScalingOperator(data_domain, factor=5, sampling_dtype=np.float64)
estimated_noise_operator = ift.ScalingOperator(data_domain, factor=15, sampling_dtype=np.float64)
noisy_data = masked_signal_field + noise_operator.draw_sample()
plt.plot(masked_time_field.val, noisy_data.val, "r.")
plt.title("Added noise")
plt.show()
#%% md
# It should be possible to do the reconstruction via the Wiener Filter as well as the KL optimization. Because it is faster to code, I shall do the KL optimization for now.
#%%
likelihood_energy = ift.GaussianEnergy(data=noisy_data, inverse_covariance=estimated_noise_operator.inverse) @ mask @ signal_model

global_iterations = 32

posterior_samples, final_pos = ift.optimize_kl(
                                    likelihood_energy=likelihood_energy,
                                    total_iterations=global_iterations,
                                    n_samples=kl_sampling_rate,
                                    kl_minimizer=descent_finder,
                                    sampling_iteration_controller=ic_sampling_lin,
                                    nonlinear_sampling_minimizer=geoVI_sampling_minimizer,
                                    output_directory="out_synthetic",
                                    return_final_position=True,
                                    resume=False,
                                    )

def pickle_me_this(filename: str, data_to_pickle: object):
    path = "data_storage/" + filename + ".pickle"
    file = open(path, 'wb')
    pickle.dump(data_to_pickle, file)
    file.close()


def unpickle_me_this(filename: str, absolute_path=False):
    if absolute_path:
        file = open(filename, 'rb')
    else:
        file = open("data_storage/" + filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data

pickle_me_this("small_kl_run_lots_of_data_and_noise", posterior_samples)
# posterior_samples = unpickle_me_this("small_kl_run.pickle")
mean, var = posterior_samples.sample_stat(signal_model)
#%%
plt.errorbar(time_field.val, mean.val, yerr=np.sqrt(var.val), color="b", ls="-")
plt.plot(time_field.val, mean.val, color="orange", ls="-", lw=5)
plt.plot(masked_time_field.val, noisy_data.val, "r.")
plt.plot(time_field.val, signal.val, "g-")
plt.show()
#%%

#%% md
# 
#%%
