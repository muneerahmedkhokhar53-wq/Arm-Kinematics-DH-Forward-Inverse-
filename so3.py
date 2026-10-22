"""Small SO(3) toolkit: exp/log maps, slerp, rotation constructors."""
import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    """Axis-angle vector of R, robust near 0 and pi."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - th < 1e-6:  # near pi: recover axis from R + I
        B = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        v = B[:, i] / np.sqrt(max(B[i, i], 1e-15))
        return th * v / np.linalg.norm(v)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return th * w


def slerp(R0, R1, s):
    return R0 @ exp_so3(s * log_so3(R0.T @ R1))


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if p is not None: T[:3, 3] = p
    return T


def wrap(q):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi
