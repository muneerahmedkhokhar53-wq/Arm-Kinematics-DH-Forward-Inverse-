"""Cartesian trajectory generation + IK tracking."""
from dataclasses import dataclass
import numpy as np
from .dh import Arm
from .so3 import slerp, make_T, wrap
from .analytic import ik_spherical_wrist, nearest_solution
from .numeric import ik_dls


def quintic(s):
    """Minimum-jerk time scaling: zero vel/acc at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cartesian_line(T0, T1, n=100):
    S = quintic(np.linspace(0, 1, n))
    return [make_T(slerp(T0[:3, :3], T1[:3, :3], s), T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3])) for s in S]


def circle_path(center, radius, R, n=120, normal_axis=2):
    """Circle in the plane normal to `normal_axis`, fixed orientation R."""
    ax = [i for i in range(3) if i != normal_axis]
    out = []
    for t in np.linspace(0, 2 * np.pi, n):
        p = np.array(center, float)
        p[ax[0]] += radius * np.cos(t)
        p[ax[1]] += radius * np.sin(t)
        out.append(make_T(R, p))
    return out


@dataclass
class Tracking:
    Q: np.ndarray
    pos_err: np.ndarray
    manip: np.ndarray
    ok: np.ndarray


def track(arm: Arm, path, q0, method="dls", null_gain=0.0):
    """Follow a list of poses. method = 'dls' (warm-started) or 'analytic' (branch-continuous)."""
    Q, ok = [], []
    q = np.asarray(q0, float)
    for T in path:
        if method == "analytic":
            qn = nearest_solution(ik_spherical_wrist(arm, T), q)
            good = qn is not None
            q = qn if good else q
        else:
            r = ik_dls(arm, T, q, null_gain=null_gain, max_iter=100)
            q, good = r.q, r.success
        Q.append(q.copy()); ok.append(good)
    Q = np.array(Q)
    err = np.array([np.linalg.norm(arm.pose_error(arm.fkine(qi), T)[:3]) for qi, T in zip(Q, path)])
    man = np.array([arm.manipulability(qi) for qi in Q])
    return Tracking(Q, err, man, np.array(ok))
