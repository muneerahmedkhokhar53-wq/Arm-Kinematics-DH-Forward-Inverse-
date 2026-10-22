"""Denavit-Hartenberg kinematics implemented from scratch (NumPy only).

Supports both conventions:
  * "std" - classic DH:      A = Rz(theta) Tz(d) Tx(a) Rx(alpha)
  * "mod" - Craig / modified: A = Rx(alpha) Tx(a) Rz(theta) Tz(d)
Only revolute joints are modelled (covers the vast majority of arms).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .so3 import log_so3


@dataclass(frozen=True)
class Link:
    a: float = 0.0
    alpha: float = 0.0
    d: float = 0.0
    offset: float = 0.0
    qlim: tuple = (-np.pi, np.pi)


class Arm:
    def __init__(self, links, convention="std", name="arm", tool=None, base=None):
        assert convention in ("std", "mod")
        self.links = list(links)
        self.convention = convention
        self.name = name
        self.tool = np.eye(4) if tool is None else np.asarray(tool, float)
        self.base = np.eye(4) if base is None else np.asarray(base, float)

    # ------------------------------------------------------------------ basics
    @property
    def n(self):
        return len(self.links)

    @property
    def qlim(self):
        return np.array([l.qlim for l in self.links])

    @property
    def qmid(self):
        return self.qlim.mean(axis=1)

    def random_q(self, rng=None):
        rng = np.random.default_rng(rng)
        lim = self.qlim
        return rng.uniform(lim[:, 0], lim[:, 1])

    def within_limits(self, q, tol=1e-9):
        lim = self.qlim
        return bool(np.all(q >= lim[:, 0] - tol) and np.all(q <= lim[:, 1] + tol))

    # ------------------------------------------------------- forward kinematics
    def link_transform(self, link: Link, q: float) -> np.ndarray:
        th = q + link.offset
        ct, st = np.cos(th), np.sin(th)
        ca, sa = np.cos(link.alpha), np.sin(link.alpha)
        a, d = link.a, link.d
        if self.convention == "std":
            return np.array([[ct, -st * ca, st * sa, a * ct],
                             [st, ct * ca, -ct * sa, a * st],
                             [0, sa, ca, d],
                             [0, 0, 0, 1.0]])
        return np.array([[ct, -st, 0, a],
                         [st * ca, ct * ca, -sa, -sa * d],
                         [st * sa, ct * sa, ca, ca * d],
                         [0, 0, 0, 1.0]])

    def frames(self, q):
        """Cumulative frames [T_base, T_1, ..., T_n] (tool NOT included)."""
        T = self.base.copy()
        out = [T]
        for link, qi in zip(self.links, q):
            T = T @ self.link_transform(link, qi)
            out.append(T)
        return out

    def fkine(self, q, upto=None):
        """End-effector pose (base . A1..An . tool). `upto=k` -> first k links, no tool."""
        q = np.asarray(q, float)
        if upto is not None:
            return self.frames(q[:upto])[upto] if upto else self.base.copy()
        return self.frames(q)[-1] @ self.tool

    def joint_positions(self, q):
        """Origins of all frames (and the tool tip) for plotting."""
        pts = [T[:3, 3] for T in self.frames(q)]
        pts.append((self.frames(q)[-1] @ self.tool)[:3, 3])
        return np.array(pts)

    # ---------------------------------------------------------------- Jacobian
    def jacobian(self, q):
        """Geometric Jacobian in the base frame (6 x n): [v; w]."""
        q = np.asarray(q, float)
        F = self.frames(q)
        pe = (F[-1] @ self.tool)[:3, 3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            # std: joint i spins about z_{i-1};  mod: about z_i
            Tj = F[i] if self.convention == "std" else F[i + 1]
            z, o = Tj[:3, 2], Tj[:3, 3]
            J[:3, i] = np.cross(z, pe - o)
            J[3:, i] = z
        return J

    def manipulability(self, q):
        """Yoshikawa measure sqrt(det(J J^T))."""
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def condition_number(self, q):
        s = np.linalg.svd(self.jacobian(q), compute_uv=False)
        return float(s[0] / max(s[-1], 1e-15))

    # ----------------------------------------------------------------- helpers
    def pose_error(self, T_cur, T_goal):
        """6-vector [dp; dw] taking T_cur to T_goal (base frame)."""
        dp = T_goal[:3, 3] - T_cur[:3, 3]
        dw = log_so3(T_goal[:3, :3] @ T_cur[:3, :3].T)
        return np.r_[dp, dw]

    @classmethod
    def from_rtb(cls, robot):
        """Convert a roboticstoolbox DHRobot into an Arm (all-revolute only)."""
        links = []
        for L in robot.links:
            assert L.isrevolute, "only revolute joints supported"
            qlim = tuple(L.qlim) if L.qlim is not None else (-np.pi, np.pi)
            links.append(Link(L.a, L.alpha, L.d, L.offset, qlim))
        conv = "mod" if robot.mdh else "std"
        return cls(links, conv, robot.name,
                   tool=np.asarray(robot.tool.A), base=np.asarray(robot.base.A))

    def to_rtb(self):
        import roboticstoolbox as rtb
        cls = rtb.RevoluteMDH if self.convention == "mod" else rtb.RevoluteDH
        L = [cls(a=l.a, alpha=l.alpha, d=l.d, offset=l.offset, qlim=np.array(l.qlim))
             for l in self.links]
        return rtb.DHRobot(L, name=self.name)
