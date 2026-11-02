"""Parameters reported in the paper (Tabassum & Bai, CCTA 2023).

Values not given in the paper (vehicle mass/inertia, drag coefficients,
LQR weights, input/state bounds) are left as TODO and must be chosen.
"""
import numpy as np

# Sampling (Sec. IV-A: data captured at 10 Hz)
DT = 0.1
TRAJ_DURATION = 14.0  # s, reference trajectories used for training

# Wind (Sec. IV-A, IV-C)
WIND_MEAN = np.array([2.7032, 1.7566, 0.0124])  # m/s, v̄w
WIND_COV = np.diag([0.5434, 0.4197, 0.0064])    # Σvw

# Koopman / EDMD (Sec. IV-B)
ERROR_STATE_DIM = 9   # position, velocity, roll/pitch/yaw tracking error
POLY_DEGREE = 2       # p
LIFTED_DIM = 55       # N = (n + p)! / (n! p!)
L1_LAMBDA = 20.0      # λ in Eq. (8)

# MPC (Sec. IV-C)
HORIZON = 5           # np
RISK_BETA = 0.99      # β, K-SMPC chance-constraint level

# Data collection (Sec. IV-A)
N_TRAIN_TRAJECTORIES = 2
N_REPEATS_PER_TRAJECTORY = 4
N_VALIDATION_SETS = 2
N_MONTE_CARLO = 5

# TODO: vehicle parameters (mass, inertia, arm length, rotor coefficients,
# drag matrix D) -- see dynamics/params.py
