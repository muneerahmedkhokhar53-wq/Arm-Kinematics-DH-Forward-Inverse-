"""Monte-Carlo reachable workspace + singularity analysis."""
import numpy as np
from .dh import Arm


def sample_workspace(arm: Arm, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.array([arm.random_q(rng) for _ in range(n)])
    P = np.array([arm.fkine(q)[:3, 3] for q in Q])
    M = np.array([arm.manipulability(q) for q in Q])
    return P, M


def workspace_stats(P):
    r = np.linalg.norm(P[:, :2], axis=1)
    return dict(max_reach=float(np.linalg.norm(P, axis=1).max()),
                z_min=float(P[:, 2].min()), z_max=float(P[:, 2].max()),
                r_max=float(r.max()))


def find_singularities(arm: Arm, n=5000, thresh=1e-3, seed=1):
    """Return sampled configurations with near-zero manipulability."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = arm.random_q(rng)
        if arm.manipulability(q) < thresh:
            out.append(q)
    return np.array(out)
