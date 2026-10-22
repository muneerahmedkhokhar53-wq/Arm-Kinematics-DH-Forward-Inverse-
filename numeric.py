"""Numerical IK: damped-least-squares (Levenberg-Marquardt) with null-space control."""
from dataclasses import dataclass
import numpy as np
from .dh import Arm


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iters: int
    err_pos: float
    err_rot: float


def ik_dls(arm: Arm, T_goal, q0=None, max_iter=300, tol_pos=1e-6, tol_rot=1e-6,
           lam=0.05, step_max=0.5, null_gain=0.0, mask=None):
    """
    Damped least squares:  dq = J^T (J J^T + lam^2 I)^-1 e  (+ null-space centring).

    - lam^2 is scaled by ||e||^2 (Sugihara) so damping fades out near the solution
      -> quadratic-ish convergence away from singularities, stability at them.
    - null_gain > 0 pushes redundant joints toward mid-range via (I - J+ J) grad(H).
    - mask: 6-vector of 0/1 selecting which DOF of the error to track.
    """
    q = arm.qmid.copy() if q0 is None else np.asarray(q0, float).copy()
    lim = arm.qlim
    w = np.ones(6) if mask is None else np.asarray(mask, float)
    I6 = np.eye(6)
    ep = er = np.inf
    for it in range(1, max_iter + 1):
        e = arm.pose_error(arm.fkine(q), T_goal) * w
        ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        J = arm.jacobian(q) * w[:, None]
        damp = lam ** 2 * (e @ e) + 1e-9
        Jp = J.T @ np.linalg.inv(J @ J.T + damp * I6)
        dq = Jp @ e
        dq_null = 0.0
        if null_gain > 0 and arm.n > 6 - int((w == 0).sum()):
            half = (lim[:, 1] - lim[:, 0]) / 2
            grad = -(q - arm.qmid) / half ** 2          # gradient of joint-centring cost
            dq_null = (np.eye(arm.n) - Jp @ J) @ (null_gain * grad)
            dq = dq + dq_null
        # converged on the task AND (if requested) the self-motion has settled
        if ep < tol_pos and er < tol_rot and np.max(np.abs(dq_null)) < 1e-5:
            return IKResult(q, True, it - 1, ep, er)
        m = np.max(np.abs(dq))
        if m > step_max:
            dq *= step_max / m
        q = np.clip(q + dq, lim[:, 0], lim[:, 1])
    return IKResult(q, False, max_iter, ep, er)


def ik_multistart(arm: Arm, T_goal, tries=30, seed=0, q0=None, **kw):
    """Retry from random seeds - handles local minima / joint-limit traps."""
    rng = np.random.default_rng(seed)
    best = None
    kw.setdefault("max_iter", 80)
    for k in range(tries):
        start = q0 if (k == 0 and q0 is not None) else (arm.qmid if k == 0 else arm.random_q(rng))
        r = ik_dls(arm, T_goal, start, **kw)
        if r.success:
            return r
        if best is None or r.err_pos + r.err_rot < best.err_pos + best.err_rot:
            best = r
    return best
