"""Closed-form IK for a 6R arm with a spherical wrist (up to 8 solutions).

Geometry (standard DH, Spong-style elbow manipulator):
  A1: alpha=+90, d=d1     A2: a=a2     A3: a=a3
  A4: alpha=+90           A5: alpha=-90   A6: d=d6
The wrist centre is o4=o5 = p - d6 * z_goal, which decouples position (q1..q3)
from orientation (q4..q6; a ZYZ Euler extraction of R36).
"""
import numpy as np
from .dh import Arm, Link
from .so3 import Rz, wrap

DEG = np.pi / 180


def make_elbow_arm(d1=0.35, a2=0.45, a3=0.40, d6=0.10, name="Elbow6R", limits_deg=170):
    lim = (-limits_deg * DEG, limits_deg * DEG)
    L = [Link(0, np.pi / 2, d1, qlim=lim), Link(a2, 0, 0, qlim=lim), Link(a3, 0, 0, qlim=lim),
         Link(0, np.pi / 2, 0, qlim=lim), Link(0, -np.pi / 2, 0, qlim=lim), Link(0, 0, d6, qlim=lim)]
    return Arm(L, "std", name)


def ik_spherical_wrist(arm: Arm, T_goal, check_limits=True, tol=1e-6):
    """Return a list of joint vectors (each verified through FK). Empty if unreachable."""
    d1, a2, a3, d6 = arm.links[0].d, arm.links[1].a, arm.links[2].a, arm.links[5].d
    T = T_goal @ np.linalg.inv(arm.tool) @ np.linalg.inv(arm.base)
    R, p = T[:3, :3], T[:3, 3]
    pc = p - d6 * R[:, 2]
    xc, yc, zc = pc

    sols = []
    base_q1 = np.arctan2(yc, xc) if np.hypot(xc, yc) > 1e-9 else 0.0
    for q1 in (base_q1, base_q1 + np.pi):                      # shoulder front / back
        r = xc * np.cos(q1) + yc * np.sin(q1)                   # signed reach in the arm plane
        s = zc - d1
        c3 = (r * r + s * s - a2 * a2 - a3 * a3) / (2 * a2 * a3)
        if abs(c3) > 1 + 1e-9:
            continue                                            # out of reach
        c3 = np.clip(c3, -1, 1)
        for q3 in (np.arccos(c3), -np.arccos(c3)):              # elbow up / down
            q2 = np.arctan2(s, r) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))
            R03 = arm.fkine([q1, q2, q3], upto=3)[:3, :3]
            R36 = R03.T @ R
            # R36 = Rz(q4) Ry(-q5) Rz(q6)  (ZYZ Euler, beta = -q5)
            sb = np.hypot(R36[0, 2], R36[1, 2])
            if sb < 1e-7:                                       # wrist singularity: q4+q6 only
                q4 = 0.0
                q5 = 0.0 if R36[2, 2] > 0 else np.pi
                q6 = np.arctan2(R36[1, 0], R36[0, 0]) if R36[2, 2] > 0 else np.arctan2(-R36[1, 0], -R36[0, 0])
                cands = [(q4, q5, q6)]
            else:
                b = np.arctan2(sb, R36[2, 2])
                a1 = np.arctan2(R36[1, 2], R36[0, 2]); c1 = np.arctan2(R36[2, 1], -R36[2, 0])
                a2_ = np.arctan2(-R36[1, 2], -R36[0, 2]); c2 = np.arctan2(-R36[2, 1], R36[2, 0])
                cands = [(a1, -b, c1), (a2_, b, c2)]            # wrist no-flip / flip
            for q4, q5, q6 in cands:
                q = wrap([q1, q2, q3, q4, q5, q6])
                if np.linalg.norm(arm.pose_error(arm.fkine(q), T_goal)) > tol:
                    continue
                if check_limits and not arm.within_limits(q):
                    continue
                if not any(np.allclose(q, s_, atol=1e-6) for s_ in sols):
                    sols.append(q)
    return sols


def nearest_solution(sols, q_ref):
    """Branch selection: solution with the smallest joint-space distance to q_ref."""
    if not sols:
        return None
    return min(sols, key=lambda q: np.linalg.norm(wrap(q - q_ref)))
