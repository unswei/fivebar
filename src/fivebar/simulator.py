import argparse

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from matplotlib.patches import Circle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from .config import DEFAULT_LINKAGE, DEFAULT_SIMULATOR, load_config_file
from .singularity_plotter import base_points, circle_intersection


D = DEFAULT_LINKAGE.d
BASE_Y = DEFAULT_LINKAGE.base_y
L1 = DEFAULT_LINKAGE.l1
L2 = DEFAULT_LINKAGE.l2
L3 = DEFAULT_LINKAGE.l3
L4 = DEFAULT_LINKAGE.l4
L5 = DEFAULT_LINKAGE.l5
TIP_MASS_KG = DEFAULT_SIMULATOR.tip_mass_kg
DEFAULT_GRAVITY_X = DEFAULT_SIMULATOR.gravity_x
DEFAULT_GRAVITY_Y = DEFAULT_SIMULATOR.gravity_y
TIP_LOAD_FORCE = TIP_MASS_KG * np.array([DEFAULT_GRAVITY_X, DEFAULT_GRAVITY_Y])
DEFAULT_LEFT_ANGLE_DEG = DEFAULT_SIMULATOR.left_angle_deg
DEFAULT_RIGHT_ANGLE_DEG = DEFAULT_SIMULATOR.right_angle_deg
DEFAULT_DISTANCE = DEFAULT_SIMULATOR.distance
DEFAULT_TARGET_ANGLE_DEG = DEFAULT_SIMULATOR.target_angle_deg
DEFAULT_BRANCH = DEFAULT_SIMULATOR.closure_branch
DEFAULT_IK_LEFT_BRANCH = DEFAULT_LINKAGE.left_branch
DEFAULT_IK_RIGHT_BRANCH = DEFAULT_LINKAGE.right_branch
DISTANCE_DETENTS = (0.5, 1.0)
TARGET_ANGLE_DETENTS = (90.0,)
GRAVITY_DETENTS = (-9.81, 0.0, 9.81)
DISTANCE_DETENT_TOLERANCE = 0.008
TARGET_ANGLE_DETENT_TOLERANCE = 0.5
GRAVITY_DETENT_TOLERANCE = 0.08
NEAR_SINGULAR_CONDITION = 15.0
NEAR_SINGULAR_PARALLEL_SIN = 0.15
NEAR_SINGULAR_SERIAL_MARGIN = 0.03
SWITCH_PLAN_STEP_DEG = 3.0
SWITCH_PLAN_MARGIN_DEG = 12.0


def apply_runtime_config(linkage_config, simulator_config):
    """Apply a loaded config to the module-level constants used by the simulator."""
    global D, BASE_Y, L1, L2, L3, L4, L5
    global TIP_MASS_KG, DEFAULT_GRAVITY_X, DEFAULT_GRAVITY_Y, TIP_LOAD_FORCE
    global DEFAULT_LEFT_ANGLE_DEG, DEFAULT_RIGHT_ANGLE_DEG
    global DEFAULT_DISTANCE, DEFAULT_TARGET_ANGLE_DEG, DEFAULT_BRANCH
    global DEFAULT_IK_LEFT_BRANCH, DEFAULT_IK_RIGHT_BRANCH

    D = linkage_config.d
    BASE_Y = linkage_config.base_y
    L1 = linkage_config.l1
    L2 = linkage_config.l2
    L3 = linkage_config.l3
    L4 = linkage_config.l4
    L5 = linkage_config.l5

    TIP_MASS_KG = simulator_config.tip_mass_kg
    DEFAULT_GRAVITY_X = simulator_config.gravity_x
    DEFAULT_GRAVITY_Y = simulator_config.gravity_y
    TIP_LOAD_FORCE = TIP_MASS_KG * np.array([DEFAULT_GRAVITY_X, DEFAULT_GRAVITY_Y])

    DEFAULT_LEFT_ANGLE_DEG = simulator_config.left_angle_deg
    DEFAULT_RIGHT_ANGLE_DEG = simulator_config.right_angle_deg
    DEFAULT_DISTANCE = simulator_config.distance
    DEFAULT_TARGET_ANGLE_DEG = simulator_config.target_angle_deg
    DEFAULT_BRANCH = simulator_config.closure_branch
    DEFAULT_IK_LEFT_BRANCH = linkage_config.left_branch
    DEFAULT_IK_RIGHT_BRANCH = linkage_config.right_branch


def normalise_angle_deg(angle):
    """Normalise an angle to the Matplotlib slider range."""
    return (angle + 180.0) % 360.0 - 180.0


def angle_change_deg(a, b):
    """Smallest absolute angular change between two degree values."""
    return abs(normalise_angle_deg(a - b))


def interpolate_angle_deg(start, end, fraction):
    """Interpolate along the shortest signed angular path."""
    return normalise_angle_deg(start + normalise_angle_deg(end - start) * fraction)


def elbow_from_inward_angle(base, length, angle_rad, side):
    """Position of a proximal elbow measured from the inward horizontal."""
    if side == "left":
        absolute_angle = angle_rad
    elif side == "right":
        absolute_angle = np.pi - angle_rad
    else:
        raise ValueError(f"unknown side: {side}")

    return base + length * np.array([np.cos(absolute_angle), np.sin(absolute_angle)])


def inward_angle_from_elbow(base, elbow, side):
    """Return the signed inward angle that places an elbow at the given point."""
    absolute_angle = np.rad2deg(np.arctan2(elbow[1] - base[1], elbow[0] - base[0]))
    if side == "left":
        return normalise_angle_deg(absolute_angle)
    if side == "right":
        return normalise_angle_deg(180.0 - absolute_angle)
    raise ValueError(f"unknown side: {side}")


def point_from_polar(distance, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([distance * np.cos(angle_rad), distance * np.sin(angle_rad)])


def solve_endpoint(C, D_joint, branch):
    """Solve the distal-link closure point from the two elbow positions."""
    return circle_intersection(C, L2, D_joint, L4, branch=branch)


def tool_endpoint(C, wrist):
    """Endpoint of L5, rigidly extending the left distal link through the wrist."""
    if wrist is None:
        return None
    direction = wrist - C
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return None
    return wrist + L5 * direction / length


def wrist_from_tool_endpoint(C, endpoint):
    """Recover the five-bar wrist point from the L5 endpoint and left elbow."""
    return C + (L2 / (L2 + L5)) * (endpoint - C)


def pose_from_angles(A0, B0, left_deg, right_deg, branch):
    left_angle = np.deg2rad(left_deg)
    right_angle = np.deg2rad(right_deg)
    C = elbow_from_inward_angle(A0, L1, left_angle, side="left")
    D_joint = elbow_from_inward_angle(B0, L3, right_angle, side="right")
    P = solve_endpoint(C, D_joint, branch)
    return C, D_joint, P


def closure_branch_for_point(C, D_joint, P):
    """Return the distal closure branch whose endpoint is closest to P."""
    options = []
    for branch in (1, -1):
        endpoint = solve_endpoint(C, D_joint, branch)
        if endpoint is not None:
            options.append((np.linalg.norm(endpoint - P), branch))
    if not options:
        return None
    return min(options)[1]


def singularity_score(metrics):
    """Dimensionless clearance score; larger means further from singularity."""
    condition, parallel_sin, serial_min = metrics
    condition_score = 0.0 if not np.isfinite(condition) else 1.0 / condition
    serial_score = serial_min / max(L1, L2, L3, L4)
    return min(condition_score, parallel_sin, serial_score)


def is_near_singularity(metrics):
    condition, parallel_sin, serial_min = metrics
    return (
        not np.isfinite(condition)
        or condition > NEAR_SINGULAR_CONDITION
        or parallel_sin < NEAR_SINGULAR_PARALLEL_SIN
        or serial_min < NEAR_SINGULAR_SERIAL_MARGIN
    )


def branch_pair_label(left_branch, right_branch):
    return f"({left_branch:+d}, {right_branch:+d})"


def inverse_kinematics_solutions(
    A0,
    B0,
    target_endpoint,
    current_left_deg,
    current_right_deg,
    branch_pair=None,
    closure_branch_constraint=None,
):
    """Return all valid IK solutions for a target L5 endpoint."""
    solutions = []
    left_branches = (branch_pair[0],) if branch_pair is not None else (1, -1)
    right_branches = (branch_pair[1],) if branch_pair is not None else (1, -1)
    for left_branch in left_branches:
        C = circle_intersection(A0, L1, target_endpoint, L2 + L5, branch=left_branch)
        if C is None:
            continue

        target_wrist = wrist_from_tool_endpoint(C, target_endpoint)
        for right_branch in right_branches:
            D_joint = circle_intersection(B0, L3, target_wrist, L4, branch=right_branch)
            if D_joint is None:
                continue

            closure_branch = closure_branch_for_point(C, D_joint, target_wrist)
            if closure_branch is None:
                continue
            if (
                closure_branch_constraint is not None
                and closure_branch != closure_branch_constraint
            ):
                continue

            wrist = solve_endpoint(C, D_joint, closure_branch)
            metrics = singularity_metrics(A0, B0, C, D_joint, wrist)
            tool = tool_endpoint(C, wrist)
            if tool is None:
                continue
            left_angle_deg = inward_angle_from_elbow(A0, C, side="left")
            right_angle_deg = inward_angle_from_elbow(B0, D_joint, side="right")
            change = angle_change_deg(left_angle_deg, current_left_deg) + angle_change_deg(
                right_angle_deg, current_right_deg
            )
            absolute_angle_size = abs(left_angle_deg) + abs(right_angle_deg)
            solutions.append(
                {
                    "change": change,
                    "absolute_angle_size": absolute_angle_size,
                    "left_angle_deg": left_angle_deg,
                    "right_angle_deg": right_angle_deg,
                    "left_branch": left_branch,
                    "right_branch": right_branch,
                    "closure_branch": closure_branch,
                    "C": C,
                    "D_joint": D_joint,
                    "P": wrist,
                    "E": tool,
                    "metrics": metrics,
                    "score": singularity_score(metrics),
                }
            )

    return solutions


def choose_inverse_kinematics_solution(
    A0,
    B0,
    P,
    current_left_deg,
    current_right_deg,
    branch_pair=None,
    closure_branch_constraint=None,
    prevent_branch_switching=False,
):
    """Choose a continuous IK solution, unless singularity avoidance should switch branches."""
    solutions = inverse_kinematics_solutions(
        A0,
        B0,
        P,
        current_left_deg,
        current_right_deg,
        branch_pair=branch_pair,
        closure_branch_constraint=closure_branch_constraint,
    )
    if not solutions:
        return None

    continuous = min(
        solutions,
        key=lambda solution: (solution["change"], solution["absolute_angle_size"]),
    )
    if (
        branch_pair is not None
        or prevent_branch_switching
        or not is_near_singularity(continuous["metrics"])
        or len(solutions) == 1
    ):
        continuous["selection_reason"] = "fixed branch pair" if branch_pair is not None else "continuous"
        continuous["switched_for_singularity"] = False
        return continuous

    safest = max(
        solutions,
        key=lambda solution: (
            solution["score"],
            -solution["change"],
            -solution["absolute_angle_size"],
        ),
    )
    safest["switched_for_singularity"] = safest is not continuous
    safest["selection_reason"] = (
        "singularity avoidance" if safest["switched_for_singularity"] else "continuous"
    )
    return safest


def inverse_kinematics_from_point(A0, B0, P, current_left_deg, current_right_deg):
    """Compatibility wrapper returning the chosen IK solution."""
    return choose_inverse_kinematics_solution(A0, B0, P, current_left_deg, current_right_deg)


def infer_ik_branch_pair(A0, B0, endpoint, current_left_deg, current_right_deg):
    """Infer the IK branch pair that best matches the current actuator angles."""
    if endpoint is None:
        return None
    solutions = inverse_kinematics_solutions(
        A0,
        B0,
        endpoint,
        current_left_deg,
        current_right_deg,
    )
    if not solutions:
        return None
    closest = min(
        solutions,
        key=lambda solution: (solution["change"], solution["absolute_angle_size"]),
    )
    return closest["left_branch"], closest["right_branch"]


def closure_gap(C, D_joint):
    """Return metres beyond the distal-link closure range; zero means closable."""
    elbow_distance = np.linalg.norm(D_joint - C)
    if elbow_distance > L2 + L4:
        return elbow_distance - (L2 + L4)
    if elbow_distance < abs(L2 - L4):
        return abs(L2 - L4) - elbow_distance
    return 0.0


def angle_axis_between(start, end, step_deg=SWITCH_PLAN_STEP_DEG, margin_deg=SWITCH_PLAN_MARGIN_DEG):
    lo = max(-180.0, min(start, end) - margin_deg)
    hi = min(180.0, max(start, end) + margin_deg)
    values = list(np.arange(lo, hi + step_deg * 0.5, step_deg))
    values.extend([start, end])
    return np.array(sorted({round(value, 6) for value in values}), dtype=float)


def plan_min_endpoint_switch_path(
    A0,
    B0,
    start_left_deg,
    start_right_deg,
    start_branch,
    end_left_deg,
    end_right_deg,
    end_branch,
):
    """Plan a branch-change path that minimises endpoint travel on a local joint grid."""
    left_axis = angle_axis_between(start_left_deg, end_left_deg)
    right_axis = angle_axis_between(start_right_deg, end_right_deg)
    branch_axis = [1, -1]

    start_i = int(np.argmin(np.abs(left_axis - start_left_deg)))
    start_j = int(np.argmin(np.abs(right_axis - start_right_deg)))
    end_i = int(np.argmin(np.abs(left_axis - end_left_deg)))
    end_j = int(np.argmin(np.abs(right_axis - end_right_deg)))
    start_k = branch_axis.index(start_branch)
    end_k = branch_axis.index(end_branch)

    endpoints = {}
    for i, left_deg in enumerate(left_axis):
        for j, right_deg in enumerate(right_axis):
            for k, branch in enumerate(branch_axis):
                C, _, wrist = pose_from_angles(A0, B0, left_deg, right_deg, branch)
                endpoint = tool_endpoint(C, wrist)
                if endpoint is not None:
                    endpoints[(i, j, k)] = endpoint

    start = (start_i, start_j, start_k)
    goal = (end_i, end_j, end_k)
    if start not in endpoints or goal not in endpoints:
        return None

    queue = [(0.0, 0.0, start)]
    costs = {start: (0.0, 0.0)}
    previous = {}
    neighbour_steps = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while queue:
        max_step_cost, length_cost, node = heappop(queue)
        if node == goal:
            break
        if (max_step_cost, length_cost) != costs[node]:
            continue

        i, j, k = node
        neighbours = [(i, j, 1 - k)]
        neighbours.extend((i + di, j + dj, k) for di, dj in neighbour_steps)

        for neighbour in neighbours:
            ni, nj, nk = neighbour
            if not (0 <= ni < len(left_axis) and 0 <= nj < len(right_axis)):
                continue
            if neighbour not in endpoints:
                continue

            endpoint_move = np.linalg.norm(endpoints[neighbour] - endpoints[node])
            joint_move = abs(left_axis[ni] - left_axis[i]) + abs(right_axis[nj] - right_axis[j])
            switch_penalty = 0.02 if nk != k else 0.0
            next_cost = (
                max(max_step_cost, endpoint_move + switch_penalty),
                length_cost + endpoint_move + 0.0005 * joint_move + switch_penalty,
            )

            if next_cost < costs.get(neighbour, (np.inf, np.inf)):
                costs[neighbour] = next_cost
                previous[neighbour] = node
                heappush(queue, (next_cost[0], next_cost[1], neighbour))

    if goal not in costs:
        return None

    nodes = []
    node = goal
    while node != start:
        nodes.append(node)
        node = previous[node]
    nodes.append(start)
    nodes.reverse()

    points = [endpoints[node] for node in nodes]
    segment_lengths = [
        np.linalg.norm(points[index] - points[index - 1]) for index in range(1, len(points))
    ]
    return {
        "points": points,
        "length": sum(segment_lengths),
        "max_step": max(segment_lengths, default=0.0),
        "nodes": nodes,
    }


def singularity_metrics(A0, B0, C, D_joint, P):
    """Return compact singularity/conditioning metrics for the current pose."""
    if P is None:
        return None

    u = P - C
    v = P - D_joint
    parallel_sin = abs(u[0] * v[1] - u[1] * v[0]) / (
        np.linalg.norm(u) * np.linalg.norm(v)
    )
    serial_left = np.dot(u, np.array([-(C - A0)[1], (C - A0)[0]]))
    serial_right = np.dot(v, np.array([-(D_joint - B0)[1], (D_joint - B0)[0]]))

    A_mat = np.vstack([u, v])
    B_mat = np.diag([serial_left, serial_right])
    if abs(np.linalg.det(A_mat)) < 1e-10 or abs(serial_left * serial_right) < 1e-10:
        condition = np.inf
    else:
        condition = np.linalg.cond(np.linalg.solve(A_mat, B_mat))

    return condition, parallel_sin, min(abs(serial_left), abs(serial_right))


def tip_jacobian(A0, B0, left_deg, right_deg, branch, step_rad=1e-5):
    """Numerically estimate dE/dq for the L5 tip in inward-angle radians."""
    C, _, P = pose_from_angles(A0, B0, left_deg, right_deg, branch)
    centre = tool_endpoint(C, P)
    if centre is None:
        return None

    step_deg = np.rad2deg(step_rad)
    jacobian = np.empty((2, 2), dtype=float)
    for column, (left_delta, right_delta) in enumerate(((step_deg, 0.0), (0.0, step_deg))):
        C_plus, _, P_plus = pose_from_angles(
            A0, B0, left_deg + left_delta, right_deg + right_delta, branch
        )
        plus = tool_endpoint(C_plus, P_plus)
        C_minus, _, P_minus = pose_from_angles(
            A0, B0, left_deg - left_delta, right_deg - right_delta, branch
        )
        minus = tool_endpoint(C_minus, P_minus)

        if plus is not None and minus is not None:
            jacobian[:, column] = (plus - minus) / (2.0 * step_rad)
        elif plus is not None:
            jacobian[:, column] = (plus - centre) / step_rad
        elif minus is not None:
            jacobian[:, column] = (centre - minus) / step_rad
        else:
            return None

    return jacobian


def holding_torques_for_tip_load(A0, B0, left_deg, right_deg, branch, force=None):
    """Return actuator torques needed to hold a static force at the L5 tip."""
    jacobian = tip_jacobian(A0, B0, left_deg, right_deg, branch)
    if jacobian is None:
        return None
    if force is None:
        force = TIP_LOAD_FORCE
    load_torque = jacobian.T @ force
    return -load_torque


def draw_target_semicircle(ax, radius=1.0):
    theta = np.linspace(0.0, np.pi, 300)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k--", linewidth=1)
    ax.plot([-radius, radius], [0.0, 0.0], "k--", linewidth=1)


def set_slider_without_callback(slider, value):
    eventson = slider.eventson
    slider.eventson = False
    slider.set_val(value)
    slider.eventson = eventson


def set_radio_without_callback(radio, branch):
    index = 0 if branch == 1 else 1
    eventson = radio.eventson
    radio.eventson = False
    radio.set_active(index)
    radio.eventson = eventson


def snap_slider_to_detent(slider, detents, tolerance):
    closest = min(detents, key=lambda detent: abs(slider.val - detent))
    if abs(slider.val - closest) <= tolerance and slider.val != closest:
        set_slider_without_callback(slider, closest)
    return slider.val


def draw_slider_detents(slider, detents):
    for detent in detents:
        if slider.valmin <= detent <= slider.valmax:
            slider.ax.axvline(detent, ymin=0.2, ymax=0.8, color="0.2", alpha=0.45, linewidth=1)


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive five-bar linkage simulator.")
    parser.add_argument(
        "--config",
        help="Path to a TOML config file with [linkage] and optional [simulator] settings.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        linkage_config, simulator_config, _ = load_config_file(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    apply_runtime_config(linkage_config, simulator_config)

    A0, B0 = base_points(D, base_y=BASE_Y)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(left=0.27, bottom=0.43, right=0.76)

    ax.set_title("Interactive five-bar linkage")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.0, 1.25)
    draw_target_semicircle(ax)

    base_line, = ax.plot([A0[0], B0[0]], [A0[1], B0[1]], "k-", linewidth=2)
    left_link, = ax.plot([], [], "o-", linewidth=3, markersize=6, label="left chain")
    right_link, = ax.plot([], [], "o-", linewidth=3, markersize=6, label="right chain")
    endpoint_marker, = ax.plot([], [], "ro", markersize=7, label="end point")
    wrist_marker, = ax.plot([], [], "ko", markersize=4, alpha=0.65, label="five-bar wrist")
    target_marker, = ax.plot([], [], "x", color="tab:green", markersize=8, label="polar target")
    trace_line, = ax.plot([], [], color="tab:red", alpha=0.35, linewidth=1, label="trajectory")
    switch_trace_line, = ax.plot(
        [],
        [],
        color="tab:purple",
        alpha=0.85,
        linewidth=1.5,
        linestyle="--",
        label="switch trajectory",
    )
    left_reach_circle = Circle(
        (0.0, 0.0),
        L2,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        color="tab:blue",
        alpha=0.0,
    )
    right_reach_circle = Circle(
        (0.0, 0.0),
        L4,
        fill=False,
        linestyle=":",
        linewidth=1.2,
        color="tab:orange",
        alpha=0.0,
    )
    ax.add_patch(left_reach_circle)
    ax.add_patch(right_reach_circle)
    status_ax = fig.add_axes([0.02, 0.43, 0.22, 0.46])
    status_ax.set_axis_off()
    status_text = status_ax.text(
        0.0,
        1.0,
        "",
        transform=status_ax.transAxes,
        va="top",
        ha="left",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    left_slider_ax = fig.add_axes([0.28, 0.32, 0.48, 0.03])
    right_slider_ax = fig.add_axes([0.28, 0.27, 0.48, 0.03])
    distance_slider_ax = fig.add_axes([0.28, 0.19, 0.48, 0.03])
    target_angle_slider_ax = fig.add_axes([0.28, 0.14, 0.48, 0.03])
    gravity_x_slider_ax = fig.add_axes([0.28, 0.09, 0.48, 0.03])
    gravity_y_slider_ax = fig.add_axes([0.28, 0.04, 0.48, 0.03])
    clear_ax = fig.add_axes([0.28, 0.005, 0.14, 0.03])
    branch_ax = fig.add_axes([0.80, 0.38, 0.16, 0.12])
    ik_branch_ax = fig.add_axes([0.80, 0.18, 0.16, 0.16])
    prevent_switch_ax = fig.add_axes([0.80, 0.11, 0.16, 0.05])

    left_slider = Slider(
        left_slider_ax,
        "left inward angle",
        -180.0,
        180.0,
        valinit=DEFAULT_LEFT_ANGLE_DEG,
        valstep=0.1,
    )
    right_slider = Slider(
        right_slider_ax,
        "right inward angle",
        -180.0,
        180.0,
        valinit=DEFAULT_RIGHT_ANGLE_DEG,
        valstep=0.1,
    )
    distance_slider = Slider(
        distance_slider_ax,
        "target distance",
        0.0,
        1.8,
        valinit=DEFAULT_DISTANCE,
        valstep=0.001,
    )
    target_angle_slider = Slider(
        target_angle_slider_ax,
        "target angle",
        0.0,
        180.0,
        valinit=DEFAULT_TARGET_ANGLE_DEG,
        valstep=0.1,
    )
    gravity_x_slider = Slider(
        gravity_x_slider_ax,
        "gravity x [m/s^2]",
        -20.0,
        20.0,
        valinit=DEFAULT_GRAVITY_X,
        valstep=0.01,
    )
    gravity_y_slider = Slider(
        gravity_y_slider_ax,
        "gravity y [m/s^2]",
        -20.0,
        20.0,
        valinit=DEFAULT_GRAVITY_Y,
        valstep=0.01,
    )
    branch_radio = RadioButtons(
        branch_ax,
        ("branch +1", "branch -1"),
        active=0 if DEFAULT_BRANCH == 1 else 1,
    )
    ik_branch_labels = (
        "auto",
        "(+1, +1)",
        "(+1, -1)",
        "(-1, +1)",
        "(-1, -1)",
    )
    ik_branch_pairs = {
        "(+1, +1)": (1, 1),
        "(+1, -1)": (1, -1),
        "(-1, +1)": (-1, 1),
        "(-1, -1)": (-1, -1),
    }
    ik_branch_radio = RadioButtons(ik_branch_ax, ik_branch_labels, active=0)
    ik_branch_ax.set_title("IK branch pair")
    prevent_switch_check = CheckButtons(
        prevent_switch_ax,
        ("prevent switch",),
        (False,),
    )
    clear_button = Button(clear_ax, "clear trace")
    draw_slider_detents(distance_slider, DISTANCE_DETENTS)
    draw_slider_detents(target_angle_slider, TARGET_ANGLE_DETENTS)
    draw_slider_detents(gravity_x_slider, GRAVITY_DETENTS)
    draw_slider_detents(gravity_y_slider, GRAVITY_DETENTS)

    trace_x = []
    trace_y = []
    switch_trace_x = []
    switch_trace_y = []
    is_updating = False
    current_ik_pair = [DEFAULT_IK_LEFT_BRANCH, DEFAULT_IK_RIGHT_BRANCH]

    def current_branch():
        return 1 if branch_radio.value_selected == "branch +1" else -1

    def fixed_ik_branch_pair():
        return ik_branch_pairs.get(ik_branch_radio.value_selected)

    def prevent_branch_switching():
        return prevent_switch_check.get_status()[0]

    def active_ik_branch_pair():
        selected_pair = fixed_ik_branch_pair()
        if selected_pair is not None:
            return selected_pair
        if prevent_branch_switching():
            return tuple(current_ik_pair)
        return None

    def current_tip_force():
        return TIP_MASS_KG * np.array([gravity_x_slider.val, gravity_y_slider.val])

    def append_switch_trajectory(
        start_left_deg,
        start_right_deg,
        end_left_deg,
        end_right_deg,
        start_branch,
        preferred_branch,
    ):
        """Draw a locally planned branch-change path with low endpoint travel."""
        plan = plan_min_endpoint_switch_path(
            A0,
            B0,
            start_left_deg,
            start_right_deg,
            start_branch,
            end_left_deg,
            end_right_deg,
            preferred_branch,
        )
        if plan is None:
            return None

        switch_trace_x.append(np.nan)
        switch_trace_y.append(np.nan)
        for point in plan["points"]:
            switch_trace_x.append(point[0])
            switch_trace_y.append(point[1])

        if len(switch_trace_x) > 2400:
            del switch_trace_x[: len(switch_trace_x) - 2400]
            del switch_trace_y[: len(switch_trace_y) - 2400]
        switch_trace_line.set_data(switch_trace_x, switch_trace_y)
        return plan

    def draw_pose(C, D_joint, P, target=None, target_status=None, record_trace=True):
        E = tool_endpoint(C, P)
        left_reach_circle.center = C
        right_reach_circle.center = D_joint
        if target is None:
            target_marker.set_data([], [])
        else:
            target_marker.set_data([target[0]], [target[1]])

        if P is None:
            gap = closure_gap(C, D_joint)
            left_link.set_data([A0[0], C[0]], [A0[1], C[1]])
            right_link.set_data([B0[0], D_joint[0]], [B0[1], D_joint[1]])
            endpoint_marker.set_data([], [])
            wrist_marker.set_data([], [])
            left_reach_circle.set_alpha(0.75)
            right_reach_circle.set_alpha(0.75)
            status_text.set_text(
                "no closed five-bar pose\nclosure gap = {:.4f} m\nelbow distance = {:.4f} m\nmax distal span = {:.4f} m".format(
                    gap, np.linalg.norm(D_joint - C), L2 + L4
                )
            )
            fig.canvas.draw_idle()
            return

        left_reach_circle.set_alpha(0.0)
        right_reach_circle.set_alpha(0.0)

        left_link.set_data([A0[0], C[0], P[0], E[0]], [A0[1], C[1], P[1], E[1]])
        right_link.set_data([B0[0], D_joint[0], P[0]], [B0[1], D_joint[1], P[1]])
        endpoint_marker.set_data([E[0]], [E[1]])
        wrist_marker.set_data([P[0]], [P[1]])

        if record_trace:
            trace_x.append(E[0])
            trace_y.append(E[1])
            if len(trace_x) > 600:
                del trace_x[: len(trace_x) - 600]
                del trace_y[: len(trace_y) - 600]
            trace_line.set_data(trace_x, trace_y)

        metrics = singularity_metrics(A0, B0, C, D_joint, P)
        condition, parallel_sin, serial_min = metrics
        polar_distance = np.linalg.norm(E)
        polar_angle = np.rad2deg(np.arctan2(E[1], E[0]))
        if polar_angle < 0.0:
            polar_angle += 360.0

        holding_torque = holding_torques_for_tip_load(
            A0,
            B0,
            left_slider.val,
            right_slider.val,
            current_branch(),
            force=current_tip_force(),
        )
        if holding_torque is None:
            torque_status = "\nhold torque (+in) = unavailable"
        else:
            tip_force = current_tip_force()
            torque_status = (
                "\ntip force = ({:.2f}, {:.2f}) N\nhold torque (+in)\n  left = {:.2f} Nm\n  right = {:.2f} Nm".format(
                    tip_force[0],
                    tip_force[1],
                    holding_torque[0],
                    holding_torque[1],
                )
            )

        extra_status = "" if target_status is None else f"\n{target_status}"
        ik_status = "\nIK branch pair = {} ({})".format(
            branch_pair_label(current_ik_pair[0], current_ik_pair[1]),
            "switching prevented" if prevent_branch_switching() else "switching allowed",
        )
        status_text.set_text(
            "E = ({:.3f}, {:.3f})\nwrist = ({:.3f}, {:.3f})\nr = {:.3f}, angle = {:.1f} deg\ncondition = {:.2f}\nparallel sin = {:.3f}\nserial margin = {:.3f}{}{}{}".format(
                E[0],
                E[1],
                P[0],
                P[1],
                polar_distance,
                polar_angle,
                condition,
                parallel_sin,
                serial_min,
                ik_status,
                torque_status,
                extra_status,
            )
        )
        fig.canvas.draw_idle()

    def update_from_base(_=None):
        nonlocal is_updating
        if is_updating:
            return
        is_updating = True

        left_angle = np.deg2rad(left_slider.val)
        right_angle = np.deg2rad(right_slider.val)

        C = elbow_from_inward_angle(A0, L1, left_angle, side="left")
        D_joint = elbow_from_inward_angle(B0, L3, right_angle, side="right")
        P = solve_endpoint(C, D_joint, current_branch())
        E = tool_endpoint(C, P)

        if E is not None:
            branch_pair = infer_ik_branch_pair(A0, B0, E, left_slider.val, right_slider.val)
            if branch_pair is not None:
                current_ik_pair[:] = branch_pair
            polar_distance = np.linalg.norm(E)
            polar_angle = np.rad2deg(np.arctan2(E[1], E[0]))
            if polar_angle < 0.0:
                polar_angle += 360.0
            set_slider_without_callback(distance_slider, min(distance_slider.valmax, polar_distance))
            set_slider_without_callback(
                target_angle_slider,
                min(target_angle_slider.valmax, max(target_angle_slider.valmin, polar_angle)),
            )

        draw_pose(C, D_joint, P)
        is_updating = False

    def update_from_load(_=None):
        snap_slider_to_detent(gravity_x_slider, GRAVITY_DETENTS, GRAVITY_DETENT_TOLERANCE)
        snap_slider_to_detent(gravity_y_slider, GRAVITY_DETENTS, GRAVITY_DETENT_TOLERANCE)

        left_angle = np.deg2rad(left_slider.val)
        right_angle = np.deg2rad(right_slider.val)

        C = elbow_from_inward_angle(A0, L1, left_angle, side="left")
        D_joint = elbow_from_inward_angle(B0, L3, right_angle, side="right")
        P = solve_endpoint(C, D_joint, current_branch())
        draw_pose(C, D_joint, P, record_trace=False)

    def update_from_polar(_=None):
        nonlocal is_updating
        if is_updating:
            return
        is_updating = True

        snap_slider_to_detent(distance_slider, DISTANCE_DETENTS, DISTANCE_DETENT_TOLERANCE)
        snap_slider_to_detent(
            target_angle_slider,
            TARGET_ANGLE_DETENTS,
            TARGET_ANGLE_DETENT_TOLERANCE,
        )
        target = point_from_polar(distance_slider.val, target_angle_slider.val)
        previous_left_deg = left_slider.val
        previous_right_deg = right_slider.val
        solution = choose_inverse_kinematics_solution(
            A0,
            B0,
            target,
            current_left_deg=previous_left_deg,
            current_right_deg=previous_right_deg,
            branch_pair=active_ik_branch_pair(),
            closure_branch_constraint=current_branch() if prevent_branch_switching() else None,
            prevent_branch_switching=prevent_branch_switching(),
        )
        if solution is None:
            target_marker.set_data([target[0]], [target[1]])
            status_text.set_text(
                "polar target unreachable\nr = {:.3f}, angle = {:.1f} deg".format(
                    distance_slider.val, target_angle_slider.val
                )
            )
            fig.canvas.draw_idle()
            is_updating = False
            return

        left_angle_deg = solution["left_angle_deg"]
        right_angle_deg = solution["right_angle_deg"]
        closure_branch = solution["closure_branch"]
        current_ik_pair[:] = [solution["left_branch"], solution["right_branch"]]
        C = solution["C"]
        D_joint = solution["D_joint"]
        switch_plan = None

        if solution["switched_for_singularity"]:
            switch_plan = append_switch_trajectory(
                previous_left_deg,
                previous_right_deg,
                left_angle_deg,
                right_angle_deg,
                current_branch(),
                closure_branch,
            )

        set_slider_without_callback(left_slider, left_angle_deg)
        set_slider_without_callback(right_slider, right_angle_deg)
        set_radio_without_callback(branch_radio, closure_branch)

        P = solve_endpoint(C, D_joint, closure_branch)
        target_status = "polar IK target active"
        if solution["selection_reason"] == "singularity avoidance":
            target_status = "polar IK target active; singularity-avoidance branch"
            if switch_plan is not None:
                target_status += (
                    "\nswitch path length = {:.3f} m, max step = {:.3f} m".format(
                        switch_plan["length"], switch_plan["max_step"]
                    )
                )
        elif solution["selection_reason"] == "fixed branch pair":
            target_status = "polar IK target active; fixed IK branch"
        elif prevent_branch_switching():
            target_status = "polar IK target active; branch switching prevented"
        draw_pose(C, D_joint, P, target=target, target_status=target_status)
        is_updating = False

    def clear_trace(_=None):
        trace_x.clear()
        trace_y.clear()
        switch_trace_x.clear()
        switch_trace_y.clear()
        trace_line.set_data([], [])
        switch_trace_line.set_data([], [])
        fig.canvas.draw_idle()

    left_slider.on_changed(update_from_base)
    right_slider.on_changed(update_from_base)
    branch_radio.on_clicked(update_from_base)
    ik_branch_radio.on_clicked(update_from_polar)
    prevent_switch_check.on_clicked(update_from_polar)
    distance_slider.on_changed(update_from_polar)
    target_angle_slider.on_changed(update_from_polar)
    gravity_x_slider.on_changed(update_from_load)
    gravity_y_slider.on_changed(update_from_load)
    clear_button.on_clicked(clear_trace)

    update_from_base()
    plt.show()


if __name__ == "__main__":
    main()
