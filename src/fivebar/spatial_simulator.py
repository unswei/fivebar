import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from . import simulator as planar
from .config import load_config_file
from .singularity_plotter import base_points


DEFAULT_ROLL_DEG = 0.0


def local_to_world(point, roll_deg):
    """Rotate a local planar point about the world x-axis."""
    point = np.asarray(point, dtype=float)
    roll = np.deg2rad(roll_deg)
    return np.array(
        [
            point[0],
            point[1] * np.cos(roll),
            point[1] * np.sin(roll),
        ],
        dtype=float,
    )


def local_polyline_to_world(points, roll_deg):
    return np.array([local_to_world(point, roll_deg) for point in points], dtype=float)


def set_line3d(line, points):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        line.set_data([], [])
        line.set_3d_properties([])
        return
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def draw_hemisphere(ax, radius=1.0):
    """Draw a light wireframe for the z >= 0 target half-sphere."""
    alpha = np.linspace(0.0, np.pi, 18)
    beta = np.linspace(0.0, np.pi, 18)
    alpha_grid, beta_grid = np.meshgrid(alpha, beta)
    x = radius * np.cos(alpha_grid)
    planar_y = radius * np.sin(alpha_grid)
    y = planar_y * np.cos(beta_grid)
    z = planar_y * np.sin(beta_grid)
    ax.plot_wireframe(x, y, z, color="0.72", linewidth=0.45, alpha=0.45)


def branch_pair_label(left_branch, right_branch):
    return f"({left_branch:+d}, {right_branch:+d})"


def format_value(value, digits=3):
    if np.isfinite(value):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "inf"


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive 3D five-bar linkage simulator.")
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
    planar.apply_runtime_config(linkage_config, simulator_config)

    A0, B0 = base_points(planar.D, base_y=planar.BASE_Y)

    fig = plt.figure(figsize=(11.5, 8.0))
    fig.subplots_adjust(left=0.28, bottom=0.42, right=0.78)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Interactive 3D five-bar linkage")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.55, 1.2)
    ax.set_box_aspect((1, 1, 0.75))
    ax.view_init(elev=24, azim=-58)
    draw_hemisphere(ax, radius=1.0)

    left_link, = ax.plot([], [], [], "o-", linewidth=3, markersize=5, label="left chain")
    right_link, = ax.plot([], [], [], "o-", linewidth=3, markersize=5, label="right chain")
    base_line, = ax.plot([], [], [], "k-", linewidth=2, label="base")
    endpoint_marker, = ax.plot([], [], [], "ro", markersize=7, label="end point")
    wrist_marker, = ax.plot([], [], [], "ko", markersize=4, alpha=0.7, label="five-bar wrist")
    target_marker, = ax.plot([], [], [], "x", color="tab:green", markersize=8, label="local polar target")
    trace_line, = ax.plot([], [], [], color="tab:red", alpha=0.35, linewidth=1, label="trajectory")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    status_ax = fig.add_axes([0.02, 0.42, 0.22, 0.48])
    status_ax.set_axis_off()
    status_text = status_ax.text(
        0.0,
        1.0,
        "",
        transform=status_ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    roll_slider_ax = fig.add_axes([0.30, 0.34, 0.46, 0.03])
    left_slider_ax = fig.add_axes([0.30, 0.29, 0.46, 0.03])
    right_slider_ax = fig.add_axes([0.30, 0.24, 0.46, 0.03])
    distance_slider_ax = fig.add_axes([0.30, 0.16, 0.46, 0.03])
    target_angle_slider_ax = fig.add_axes([0.30, 0.11, 0.46, 0.03])
    clear_ax = fig.add_axes([0.30, 0.045, 0.14, 0.035])
    branch_ax = fig.add_axes([0.80, 0.40, 0.16, 0.12])
    ik_branch_ax = fig.add_axes([0.80, 0.19, 0.16, 0.16])
    prevent_switch_ax = fig.add_axes([0.80, 0.12, 0.16, 0.05])

    roll_slider = Slider(
        roll_slider_ax,
        "base x roll",
        0.0,
        180.0,
        valinit=DEFAULT_ROLL_DEG,
        valstep=0.1,
    )
    left_slider = Slider(
        left_slider_ax,
        "left inward angle",
        -180.0,
        180.0,
        valinit=planar.DEFAULT_LEFT_ANGLE_DEG,
        valstep=0.1,
    )
    right_slider = Slider(
        right_slider_ax,
        "right inward angle",
        -180.0,
        180.0,
        valinit=planar.DEFAULT_RIGHT_ANGLE_DEG,
        valstep=0.1,
    )
    distance_slider = Slider(
        distance_slider_ax,
        "target distance",
        0.0,
        1.8,
        valinit=planar.DEFAULT_DISTANCE,
        valstep=0.001,
    )
    target_angle_slider = Slider(
        target_angle_slider_ax,
        "target angle",
        0.0,
        180.0,
        valinit=planar.DEFAULT_TARGET_ANGLE_DEG,
        valstep=0.1,
    )
    branch_radio = RadioButtons(
        branch_ax,
        ("branch +1", "branch -1"),
        active=0 if planar.DEFAULT_BRANCH == 1 else 1,
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
    prevent_switch_check = CheckButtons(prevent_switch_ax, ("prevent switch",), (False,))
    clear_button = Button(clear_ax, "clear trace")
    planar.draw_slider_detents(distance_slider, planar.DISTANCE_DETENTS)
    planar.draw_slider_detents(target_angle_slider, planar.TARGET_ANGLE_DETENTS)
    roll_slider.ax.axvline(90.0, ymin=0.2, ymax=0.8, color="0.2", alpha=0.45, linewidth=1)

    trace_points = []
    current_ik_pair = [planar.DEFAULT_IK_LEFT_BRANCH, planar.DEFAULT_IK_RIGHT_BRANCH]
    is_updating = False

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

    def draw_pose(C, D_joint, wrist, target=None, target_status=None, record_trace=True):
        roll_deg = roll_slider.val
        E = planar.tool_endpoint(C, wrist)
        target_world = local_to_world(target, roll_deg) if target is not None else None

        set_line3d(base_line, local_polyline_to_world([A0, B0], roll_deg))
        if target_world is None:
            set_line3d(target_marker, [])
        else:
            set_line3d(target_marker, [target_world])

        if wrist is None or E is None:
            set_line3d(left_link, local_polyline_to_world([A0, C], roll_deg))
            set_line3d(right_link, local_polyline_to_world([B0, D_joint], roll_deg))
            set_line3d(endpoint_marker, [])
            set_line3d(wrist_marker, [])
            status_text.set_text("no closed five-bar pose")
            fig.canvas.draw_idle()
            return

        left_points = local_polyline_to_world([A0, C, wrist, E], roll_deg)
        right_points = local_polyline_to_world([B0, D_joint, wrist], roll_deg)
        E_world = local_to_world(E, roll_deg)
        wrist_world = local_to_world(wrist, roll_deg)
        set_line3d(left_link, left_points)
        set_line3d(right_link, right_points)
        set_line3d(endpoint_marker, [E_world])
        set_line3d(wrist_marker, [wrist_world])

        if record_trace:
            trace_points.append(E_world)
            if len(trace_points) > 800:
                del trace_points[: len(trace_points) - 800]
            set_line3d(trace_line, trace_points)

        metrics = planar.singularity_metrics(A0, B0, C, D_joint, wrist)
        condition, parallel_sin, serial_min = metrics
        polar_distance = np.linalg.norm(E)
        polar_angle = np.rad2deg(np.arctan2(E[1], E[0]))
        if polar_angle < 0.0:
            polar_angle += 360.0

        extra_status = "" if target_status is None else f"\n{target_status}"
        status_text.set_text(
            "E world = ({:.3f}, {:.3f}, {:.3f})\n"
            "E local = ({:.3f}, {:.3f})\n"
            "roll x = {:.1f} deg\n"
            "r = {:.3f}, angle = {:.1f} deg\n"
            "condition = {:.2f}\n"
            "parallel sin = {:.3f}\n"
            "serial margin = {:.3f}\n"
            "IK branch pair = {} ({}){}".format(
                E_world[0],
                E_world[1],
                E_world[2],
                E[0],
                E[1],
                roll_deg,
                polar_distance,
                polar_angle,
                condition,
                parallel_sin,
                serial_min,
                branch_pair_label(current_ik_pair[0], current_ik_pair[1]),
                "switching prevented" if prevent_branch_switching() else "switching allowed",
                extra_status,
            )
        )
        fig.canvas.draw_idle()

    def update_from_base(_=None):
        nonlocal is_updating
        if is_updating:
            return
        is_updating = True
        C, D_joint, wrist = planar.pose_from_angles(
            A0,
            B0,
            left_slider.val,
            right_slider.val,
            current_branch(),
        )
        E = planar.tool_endpoint(C, wrist)
        if E is not None:
            branch_pair = planar.infer_ik_branch_pair(
                A0,
                B0,
                E,
                left_slider.val,
                right_slider.val,
            )
            if branch_pair is not None:
                current_ik_pair[:] = branch_pair
            polar_distance = np.linalg.norm(E)
            polar_angle = np.rad2deg(np.arctan2(E[1], E[0]))
            if polar_angle < 0.0:
                polar_angle += 360.0
            planar.set_slider_without_callback(
                distance_slider,
                min(distance_slider.valmax, polar_distance),
            )
            planar.set_slider_without_callback(
                target_angle_slider,
                min(target_angle_slider.valmax, max(target_angle_slider.valmin, polar_angle)),
            )
        draw_pose(C, D_joint, wrist)
        is_updating = False

    def update_from_roll(_=None):
        C, D_joint, wrist = planar.pose_from_angles(
            A0,
            B0,
            left_slider.val,
            right_slider.val,
            current_branch(),
        )
        target = planar.point_from_polar(distance_slider.val, target_angle_slider.val)
        draw_pose(C, D_joint, wrist, target=target, record_trace=False)

    def update_from_target(_=None):
        nonlocal is_updating
        if is_updating:
            return
        is_updating = True
        planar.snap_slider_to_detent(
            distance_slider,
            planar.DISTANCE_DETENTS,
            planar.DISTANCE_DETENT_TOLERANCE,
        )
        planar.snap_slider_to_detent(
            target_angle_slider,
            planar.TARGET_ANGLE_DETENTS,
            planar.TARGET_ANGLE_DETENT_TOLERANCE,
        )
        target = planar.point_from_polar(distance_slider.val, target_angle_slider.val)
        solution = planar.choose_inverse_kinematics_solution(
            A0,
            B0,
            target,
            current_left_deg=left_slider.val,
            current_right_deg=right_slider.val,
            branch_pair=active_ik_branch_pair(),
            closure_branch_constraint=current_branch() if prevent_branch_switching() else None,
            prevent_branch_switching=prevent_branch_switching(),
        )
        if solution is None:
            status_text.set_text(
                "local polar target unreachable\n"
                "r = {:.3f}, angle = {:.1f} deg".format(
                    distance_slider.val,
                    target_angle_slider.val,
                )
            )
            is_updating = False
            fig.canvas.draw_idle()
            return

        current_ik_pair[:] = [solution["left_branch"], solution["right_branch"]]
        planar.set_slider_without_callback(left_slider, solution["left_angle_deg"])
        planar.set_slider_without_callback(right_slider, solution["right_angle_deg"])
        planar.set_radio_without_callback(branch_radio, solution["closure_branch"])

        target_status = "local polar IK target active"
        if solution["selection_reason"] == "fixed branch pair":
            target_status = "local polar IK target active; fixed IK branch"
        elif prevent_branch_switching():
            target_status = "local polar IK target active; branch switching prevented"

        draw_pose(
            solution["C"],
            solution["D_joint"],
            solution["P"],
            target=target,
            target_status=target_status,
        )
        is_updating = False

    def clear_trace(_=None):
        trace_points.clear()
        set_line3d(trace_line, [])
        fig.canvas.draw_idle()

    roll_slider.on_changed(update_from_roll)
    left_slider.on_changed(update_from_base)
    right_slider.on_changed(update_from_base)
    branch_radio.on_clicked(update_from_base)
    ik_branch_radio.on_clicked(update_from_target)
    prevent_switch_check.on_clicked(update_from_target)
    distance_slider.on_changed(update_from_target)
    target_angle_slider.on_changed(update_from_target)
    clear_button.on_clicked(clear_trace)

    update_from_base()
    plt.show()


if __name__ == "__main__":
    main()
