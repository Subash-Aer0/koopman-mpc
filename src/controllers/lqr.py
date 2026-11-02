"""High-level LQR tracking controller (Foehn & Scaramuzza [16]).

Outputs u_h = [ω_x, ω_y, ω_z, f_c] for the low-level controller; designed
without wind knowledge.
"""


class LQRTracker:
    def __init__(self, params, Q, R):
        raise NotImplementedError

    def __call__(self, x, x_ref):
        raise NotImplementedError
