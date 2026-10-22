import numpy as np
from robot_arm_kinematics import make_elbow_arm, ik_spherical_wrist, nearest_solution, ik_multistart, so3, trajectory, workspace
from robot_arm_kinematics import viz

arm = make_elbow_arm()

# 1) Forward kinematics -> pose
q_true = np.radians([30, -40, 60, 20, 50, -30])
T_goal = arm.fkine(q_true)
print("End-effector pose:\n", np.round(T_goal, 3))

# 2) Analytic IK (all solutions)
sols = ik_spherical_wrist(arm, T_goal)
print(f"\nAnalytic IK found {len(sols)} solutions")
for q in sols:
    print(np.round(np.degrees(q), 1))
best = nearest_solution(sols, q_true)
print("Nearest to original q:", np.round(np.degrees(best), 1))

# 3) Numerical IK
r = ik_multistart(arm, T_goal)
print("\nNumeric IK success:", r.success, "pos err:", r.err_pos)

# 4) Trajectory tracking on a circle
path = trajectory.circle_path(center=[0.5, 0.0, 0.5], radius=0.1, R=T_goal[:3, :3], n=60)
tr = trajectory.track(arm, path, q0=best, method="analytic")
print("\nTracking OK for all waypoints:", tr.ok.all(), "max err:", tr.pos_err.max())

# 5) Plots (saved as PNG/GIF files)
viz.plot_ik_branches(arm, sols, T_goal, "ik_solutions.png")
P, M = workspace.sample_workspace(arm, n=5000)
viz.plot_workspace(P, M, "workspace.png")
viz.plot_tracking({"analytic": tr}, "tracking.png")
viz.animate(arm, tr.Q, [T[:3, 3] for T in path], "arm.gif")
print("\nSaved: ik_solutions.png, workspace.png, tracking.png, arm.gif")
