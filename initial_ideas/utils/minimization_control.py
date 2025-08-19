import nifty8 as ift

# Iteration control for `MGVI` and linear parts of the inference
ic_sampling_lin = ift.AbsDeltaEnergyController(name="Precise linear sampling", deltaE=0.02, iteration_limit=100)

# Iteration control for `geoVI`
ic_sampling_nl = ift.AbsDeltaEnergyController(name="Coarser, nonlinear sampling", deltaE=0.5, iteration_limit=20,
                                              convergence_level=2)
# For the non-linear sampling part of geoVI, the iteration controller has to be "promoted" to a minimizer:

# geoVI_sampling_minimizer = ift.NewtonCG(ic_sampling_nl)
geoVI_sampling_minimizer = ift.L_BFGS(ic_sampling_nl)

# KL Minimizer control, the same energy criterion as the geoVI iteration control, but more iteration steps
ic_newton = ift.AbsDeltaEnergyController(name='Newton Descent Finder', deltaE=0.1, convergence_level=2,
                                         iteration_limit=35)

from nifty8.minimization.line_search import LineSearch

line_searcher =LineSearch(preferred_initial_step_size=1., max_zoom_iterations=200)
descent_finder = ift.L_BFGS(ic_newton)

def kl_sampling_rate(index: int):
    """
    Callable for sampling of KL. KL minimization can be performed on its samples instead of computing it directly.
    First, get a ballpark, in later iteration increase sampling rate.
    :param index:
    :return:
    """
    if index < 20:
        return 7
    else:
        return 30