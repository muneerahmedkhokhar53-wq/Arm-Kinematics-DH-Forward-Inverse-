import numpy as np
from robot_arm_kinematics import (make_elbow_arm, ik_spherical_wrist, nearest_solution,
                                  trajectory, workspace)
from robot_arm_kinematics import viz            # viz forces the "Agg" backend...
import matplotlib
matplotlib.use("TkAgg", force=True)            # ...so switch to a window backend afterwards
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

arm = make_elbow_arm()
q_true = np.radians([30, -40, 60, 20, 50, -30])
T_goal = arm.fkine(q_true)
sols = ik_spherical_wrist(arm, T_goal)
best = nearest_solution(sols, q_true)

path = trajectory.circle_path([0.5, 0.0, 0.5], 0.1, T_goal[:3, :3], n=80)
tr = trajectory.track(arm, path, q0=best, method="analytic")
path_xyz = np.array([T[:3, 3] for T in path])
tip = np.array([arm.fkine(q)[:3, 3] for q in tr.Q])

# ---- Window 1: live animated arm (drag with the mouse to rotate) ----
fig1 = plt.figure("Arm animation", figsize=(7, 7))
ax1 = fig1.add_subplot(111, projection="3d")

def update(k):
    ax1.cla(); viz._setup(ax1, 0.9)
    ax1.set_title("Arm following a circle (drag to rotate)")
    ax1.plot(*path_xyz.T, "k--", lw=1, alpha=0.5)
    viz.draw_arm(ax1, arm, tr.Q[k])
    ax1.plot(*tip[:k + 1].T, "r", lw=2)

anim = FuncAnimation(fig1, update, frames=len(tr.Q), interval=50, repeat=True)

# ---- Window 2: all 8 IK solutions ----
fig2 = plt.figure("IK solutions", figsize=(13, 6))
cols = min(4, len(sols)); rows = int(np.ceil(len(sols) / cols))
for i, q in enumerate(sols):
    ax = fig2.add_subplot(rows, cols, i + 1, projection="3d")
    viz.draw_arm(ax, arm, q, color=plt.cm.tab10(i), frames=False)
    ax.scatter(*T_goal[:3, 3], c="k", marker="*", s=120)
    viz._setup(ax, 0.9)
    ax.set_title(f"solution {i+1}\nq={np.round(np.degrees(q)).astype(int)}", fontsize=7)
fig2.tight_layout()

# ---- Window 3: workspace ----
P, M = workspace.sample_workspace(arm, n=5000)
fig3 = plt.figure("Workspace", figsize=(10, 5))
a = fig3.add_subplot(121, projection="3d")
a.scatter(P[::3, 0], P[::3, 1], P[::3, 2], c=P[::3, 2], s=1, cmap="viridis")
a.set_title("Reachable workspace")
b = fig3.add_subplot(122)
sc = b.scatter(np.hypot(P[:, 0], P[:, 1]), P[:, 2], c=M, s=1, cmap="plasma")
fig3.colorbar(sc, ax=b, label="manipulability")
b.set_xlabel("radial reach [m]"); b.set_ylabel("z [m]"); b.set_aspect("equal")

# ---- Window 4: tracking ----
fig4, axs = plt.subplots(1, 3, figsize=(14, 4))
fig4.canvas.manager.set_window_title("Tracking")
axs[0].semilogy(tr.pos_err + 1e-12); axs[0].set_title("Position error [m]")
axs[1].plot(tr.manip); axs[1].set_title("Manipulability")
for j in range(6):
    axs[2].plot(np.degrees(tr.Q[:, j]), label=f"q{j+1}")
axs[2].set_title("Joint angles [deg]"); axs[2].legend(fontsize=6, ncol=2)
for x in axs: x.set_xlabel("waypoint"); x.grid(alpha=.3)
fig4.tight_layout()

plt.show()
