import nifty8 as ift
import numpy as np
import scipy
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

path = "outs/inference_with_continuous_double_power_law/"
name = "model_parameters_with_iidx.txt"

def inspect_sample(sample_list, idx):
    """
    :param sample_list:    The latest sample list
    :param idx:            The iteration index.
    :return:
    """
    real_space_values = get_real_space_values(sample_list)
    print_model_parameters(real_space_values, itr=idx)
    write_model_parameters(real_space_values, itr=idx, path=path)


def get_real_space_values(sample_list: list | ift.ResidualSampleList):
    r"""

        DISCUSSION
        ------------

        x is Gaussian if x = \mu + \sigma \xi. If you have the posterior mean \xi, <\xi>, then it is easy to see that
        <x> = \mu + \sigma <\xi>. But you already arrive at a non-linearity for the variance of x in terms of the
        variance of \xi. It is easy to show that in fact,

            Var(x) = <x^2> - <x>^2 = \sigma^2 * Var(\xi).

        For x lognormal, x = exp(\mu + \sigma \xi) there is already a non-linearity in <x> that can be handled if
        exp(\mu + \sigma \xi) is expanded around its known first and second moment, <\xi> and Var(\xi). This way, one
        gets

            <x> = exp(\mu + \sigma <\xi>) * (1 + 1/2 \sigma^2 Var(\xi))

        and

            Var(x) = exp(2\mu + 2\sigma <\xi>)(1+2\sigma^2 Var(\xi) - (1+1/2 \sigma^2 Var(\xi))^2)

        (see handwritten notes in `strain thoughts` file on iPad).

        Now, x = CDF_{iid}(\xi) is approximately uniformly distributed. One may show that

            <x>_x = <\xi>_\xi = \int d\xi P(\xi) CDF_{iid}(\xi)
                              ~  \int d\xi G(\xi-<\xi>, Var(\xi)) CDF_{iid}(\xi)
                              ~ CDF_{iid}(<\xi>/(1+Var(\xi)))

       see e.g. https://math.stackexchange.com/questions/449875/expected-value-of-normal-cdf.
       But since this is complicated and then building the variance from that is even more complicated,
       for this function we will just transform all samples by themselves via their transformation rules and then
       do the statistics (mean and UNBIASED variance) in real space.

        ------------

        This function calculates the found mean parameters \pm unbiased standard deviation in the current iteration,
        logs them and writes them to a file. If you change the distribution of the model parameters, please also do so here,
        since as of now it is hardcoded.

        :param sample_list:    The latest sample list
        :return:
        """

    distributions = {
        "k0 ": ["Uniform", (0, 2048.1250228923755)],
        "p0 ": ["Gaussian", (1e3, 1e-16)],
        "c ": ["Gaussian", (100, 1e-16)],
        "alpha ": ["Gaussian", (+10, 100)],
        "beta ": ["Gaussian", (-10, 100)],
        "cfm_envelope_fluctuations": ["Lognormal", (4, 2)],
        "cfm_envelope_loglogavgslope": ["Gaussian", (-4, 1)]
    }

    if type(sample_list) != list:
        latent_samples = list(sample_list.local_iterator())
    else:
        latent_samples = sample_list

    latent_samples_values = [el.val for el in latent_samples]  # dicts

    latent_space_values = {key: [] for key in distributions.keys()}
    for mf in latent_samples_values:
        for key in mf.keys():
            if not "xi" in key:  # ignore cfm and wavelet xi's
                latent_space_values[key].append(mf[key])

    real_space_values = {key: (np.float64, np.float64) for key in distributions.keys()}
    for key in latent_space_values:
        distribution_name, mean_sig = distributions[key]
        mean, sig = mean_sig
        if distribution_name == "Gaussian":
            x_values = [mean + sig * xi for xi in latent_space_values[key]]
        elif distribution_name == "Lognormal":
            x_values = [np.exp(mean + sig * xi) for xi in latent_space_values[key]]
        elif distribution_name == "Uniform":
            lower = mean
            upper = sig
            x_values = [lower + upper * scipy.stats.norm.cdf(xi) for xi in latent_space_values[key]]
        else:
            raise ValueError("Unknown distribution")

        real_space_values[key] = (np.mean(x_values), np.std(x_values, ddof=1))
    return real_space_values


def print_model_parameters(params, itr=None):
    """
    Prints in each iteration what model parameters were found.
    :param params:  The real space parameters.
    :param itr:     Default None, which global iteration.
    :return:
    """
    # Clean keys and prepare ordered list
    keys = list(params.keys())
    if 'cfm envelope fluctuations' in keys:
        keys.remove('cfm envelope fluctuations')

    # Print the current iteration's parameters
    print(f"\nThe model parameters in iteration {itr} were:")
    for key in keys:
        mean, std = params[key]
        print(f"  {key.strip()}: {mean:.2f} \u00b1 {std:.2f}")


def write_model_parameters(params, itr, path):
    """
    Creates a file with the model parameters found in each iteration.
    :param params:  The real space parameters.
    :param itr:     The global iteration index.
    :param path:    Where to write to.
    :return:
    """
    location = os.path.join(path, name)
    keys = list(params.keys())
    # If file doesn't exist, create and write header
    if not os.path.exists(location):
        with open(location, "w") as f:
            header = "iter " + " ".join(k.strip() for k in keys) + "\n"
            f.write(header)

    # Append the current values
    with open(location, "a") as f:
        values = " ".join(f"({params[k][0]:.2f} \u00b1 {params[k][1]:.2f})" for k in keys)
        f.write(f"{itr} {values}\n")


def read_parameter_file():
    """
    Reads the file that contains the real space mean and variance values of the model parameters.
    :return:
    """
    with open(path+name, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    headers = lines[0].split()[1:]  # skip 'iter'
    data = {h: [] for h in headers}
    iterations = []

    for line in lines[1:]:
        parts = line.split()
        iterations.append(int(parts[0]))
        i = 1
        for h in headers:
            val = " ".join(parts[i:i+3])  # grabs something like '(1285.15 ± 13.35)'
            match = re.match(r"\(\s*([-+0-9.eE]+)\s*±\s*([-+0-9.eE]+)\s*\)", val)
            if match:
                mu, sigma = map(float, match.groups())
                data[h].append((mu, sigma))
            else:
                data[h].append((None, None))
            i += 3

    return headers, data, iterations


def plot_parameter_evolution():
    """
    Creates a multi-subplot figure displaying the evolution of the real-space values of the model parameters
    against global iteration number.
    :return:
    """
    headers, data, iterations = read_parameter_file()
    n_params = len(headers)
    n_cols = 2
    n_rows = (n_params + 1) // 2  # ceil division

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    axs = axs.flatten()

    for i, h in enumerate(headers):
        means, stds = zip(*data[h])
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axs[i].errorbar(iterations, means, yerr=stds, fmt='o-', capsize=3)
        axs[i].set_ylabel(h)
        axs[i].grid(True)

    # Hide unused subplots if number of headers is odd
    for j in range(n_params, len(axs)):
        axs[j].axis("off")

    axs[-1].set_xlabel("Iteration")
    plt.tight_layout()
    plt.show()
