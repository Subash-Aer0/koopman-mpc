"""Low-level body-rate controller: (ω_des, f_c) -> commanded rotor speeds."""


class BodyRateController:
    def __init__(self, params, gains):
        raise NotImplementedError

    def __call__(self, x, u_h):
        raise NotImplementedError
