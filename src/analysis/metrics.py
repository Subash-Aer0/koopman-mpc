"""Error metrics: RMSE (Tables I, III), Euclidean norm (Table II), MSE (Fig. 4)."""


def rmse(err, axis=0):
    raise NotImplementedError


def euclidean_norm_error(err):
    raise NotImplementedError


def percent_reduction(baseline, value):
    raise NotImplementedError
