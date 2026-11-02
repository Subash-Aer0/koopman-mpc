"""Figures 2-4."""


def plot_prediction(t, true_states, pred_model1, pred_model2, path):
    """Fig. 2: position, velocity, orientation prediction."""
    raise NotImplementedError


def plot_tracking_error(t, errors_by_controller, path):
    """Fig. 3: absolute tracking error per axis, Model 1 vs Model 2."""
    raise NotImplementedError


def plot_mse(t, mse_by_controller, path):
    """Fig. 4: MSE over Monte Carlo runs."""
    raise NotImplementedError
