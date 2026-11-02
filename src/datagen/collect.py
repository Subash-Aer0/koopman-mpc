"""Closed-loop data collection with the LQR in the loop (Fig. 1).

At 10 Hz, record: tracking error e_x = x_c − x_r (position, velocity, rpy),
random perturbations u_d added to u_h, wind v_w, and the residual thrust
f_res = f_c − f_r (Model 2), where f_r is estimated from measured Ω_R.
"""


def random_perturbations(n_steps, scale, rng):
    raise NotImplementedError


def collect_episode(reference, wind, perturbations, params):
    """Simulate one episode and return a dict of logged arrays."""
    raise NotImplementedError


def collect_dataset(references, n_repeats, wind_factory, rng):
    raise NotImplementedError
