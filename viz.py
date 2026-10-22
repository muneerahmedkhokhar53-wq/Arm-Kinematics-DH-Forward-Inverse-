"""Matplotlib visualisation (works headless)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _frame_axes(ax, T, s=0.06):
    o = T[:3, 3]
    for i, c in enumerate("rgb"):
        ax.plot(*zip(o, o + s * T[:3, i]), color=c, lw=1.5)


def _setup(ax, lim=1.0):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 1.5 * lim)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 0.75))


def draw_arm(ax, arm, q, color="tab:blue", frames=True, alpha=1.0):
    P = arm.joint_positions(q)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=color, lw=3, ms=5, alpha=alpha)
    if frames:
        for T in arm.frames(q)[1:]:
            _frame_axes(ax, T, 0.05)
        _frame_axes(ax, arm.fkine(q), 0.09)


def plot_ik_branches(arm, sols, T_goal, path):
    fig = plt.figure(figsize=(13, 6))
    n = len(sols)
    cols = min(4, n); rows = int(np.ceil(n / cols))
    for i, q in enumerate(sols):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw_arm(ax, arm, q, color=plt.cm.tab10(i), frames=False)
        ax.scatter(*T_goal[:3, 3], c="k", marker="*", s=120)
        _setup(ax, 0.9)
        ax.set_title(f"solution {i+1}\nq={np.round(np.degrees(q)).astype(int)}", fontsize=7)
    fig.suptitle(f"Analytic IK: {n} distinct solutions for one end-effector pose")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_workspace(P, M, path):
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(P[::6, 0], P[::6, 1], P[::6, 2], c=P[::6, 2], s=1, cmap="viridis")
    ax.set_title("Reachable workspace"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax2 = fig.add_subplot(132)
    r = np.hypot(P[:, 0], P[:, 1])
    sc = ax2.scatter(r, P[:, 2], c=M, s=1, cmap="plasma")
    fig.colorbar(sc, ax=ax2, label="manipulability"); ax2.set_xlabel("radial reach r [m]")
    ax2.set_ylabel("z [m]"); ax2.set_title("Side view coloured by dexterity"); ax2.set_aspect("equal")
    ax3 = fig.add_subplot(133); ax3.hist(M, bins=50, color="tab:purple")
    ax3.set_xlabel("Yoshikawa manipulability"); ax3.set_title("Dexterity distribution")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_tracking(tr_dict, path, t=None):
    """tr_dict: {label: Tracking}"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for lab, tr in tr_dict.items():
        x = np.arange(len(tr.Q)) if t is None else t
        axs[0].semilogy(x, tr.pos_err + 1e-12, label=lab)
        axs[1].plot(x, tr.manip, label=lab)
    axs[0].set_title("Position tracking error [m]"); axs[1].set_title("Manipulability along path")
    first = next(iter(tr_dict.values()))
    for j in range(first.Q.shape[1]):
        axs[2].plot(np.degrees(first.Q[:, j]), label=f"q{j+1}")
    axs[2].set_title(f"Joint angles [deg] ({next(iter(tr_dict))})"); axs[2].legend(fontsize=6, ncol=2)
    axs[0].legend(); axs[1].legend()
    for a in axs: a.set_xlabel("waypoint"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def animate(arm, Q, path_xyz, out, fps=20, lim=0.9, title=""):
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    path_xyz = np.asarray(path_xyz)

    def upd(k):
        ax.cla(); _setup(ax, lim); ax.set_title(title)
        ax.plot(*path_xyz.T, "k--", lw=1, alpha=.5)
        draw_arm(ax, arm, Q[k]); ax.plot(*zip(*[arm.fkine(q)[:3, 3] for q in Q[:k + 1]]), "r", lw=2)
        ax.view_init(25, 35 + 0.3 * k)
    FuncAnimation(fig, upd, frames=range(0, len(Q), 2)).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
