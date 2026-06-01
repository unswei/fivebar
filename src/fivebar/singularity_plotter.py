import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

from .config import load_config_file


def base_points(d, base_y=0.0):
    """Return the two fixed base pivots."""
    return np.array([-d / 2.0, base_y]), np.array([d / 2.0, base_y])


def cross2(a, b):
    """2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def rot90(v):
    """Rotate a 2D vector by +90 degrees."""
    return np.array([-v[1], v[0]], dtype=float)


def circle_intersection(c0, r0, c1, r1, branch=1, eps=1e-12):
    """
    Return one intersection point of two circles.

    c0, r0: centre and radius of first circle.
    c1, r1: centre and radius of second circle.
    branch: +1 or -1 selects one of the two possible intersections.
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    delta = c1 - c0
    R = np.linalg.norm(delta)

    if R < eps:
        return None

    a = (r0**2 - r1**2 + R**2) / (2.0 * R)
    h2 = r0**2 - a**2

    if h2 < -1e-9:
        return None

    h = np.sqrt(max(0.0, h2))
    e = delta / R
    p = c0 + a * e
    p_perp = rot90(e)

    return p + branch * h * p_perp


def tool_endpoint(C, P, l2, l5):
    """Endpoint of L5, rigidly extending the left distal link through the wrist."""
    if l5 <= 0:
        return P

    direction = P - C
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return None
    return P + l5 * direction / length


def wrist_from_tool_endpoint(C, E, l2, l5):
    """Recover the five-bar wrist point from the L5 endpoint and left elbow."""
    if l5 <= 0:
        return E
    return C + (l2 / (l2 + l5)) * (E - C)


def fivebar_metrics(
    P,
    d,
    l1,
    l2,
    l3,
    l4,
    left_branch=1,
    right_branch=-1,
    base_y=0.0,
):
    """
    Compute reachability, elbow positions, and singularity metrics for one point P.

    Geometry:
        A -- l1 -- C -- l2 -- P
        B -- l3 -- D -- l4 -- P

    A and B are the fixed base pivots, separated by distance d.
    C and D are passive elbow joints.
    P is the wrist-centre / end point of the five-bar stage.
    """
    A0, B0 = base_points(d, base_y=base_y)
    P = np.asarray(P, dtype=float)

    rA = np.linalg.norm(P - A0)
    rB = np.linalg.norm(P - B0)

    # Serial singularities occur at the folded and stretched chain boundaries.
    if not (abs(l1 - l2) <= rA <= l1 + l2):
        return None
    if not (abs(l3 - l4) <= rB <= l3 + l4):
        return None

    C = circle_intersection(A0, l1, P, l2, branch=left_branch)
    D = circle_intersection(B0, l3, P, l4, branch=right_branch)

    if C is None or D is None:
        return None

    u = P - C
    v = P - D

    # Constraint Jacobian form:
    #     A_mat * Pdot = B_mat * qdot
    # Parallel/direct singularity occurs when det(A_mat) = 0.
    # Serial/inverse singularity occurs when det(B_mat) = 0.
    A_mat = np.vstack([u, v])

    b1 = np.dot(u, rot90(C - A0))
    b2 = np.dot(v, rot90(D - B0))
    B_mat = np.diag([b1, b2])

    det_A = np.linalg.det(A_mat)
    det_B = b1 * b2

    parallel_sin = abs(cross2(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))

    if abs(det_A) < 1e-10 or abs(det_B) < 1e-10:
        cond = np.inf
    else:
        J = np.linalg.solve(A_mat, B_mat)
        cond = np.linalg.cond(J)

    return {
        "C": C,
        "D": D,
        "cond": cond,
        "parallel_sin": parallel_sin,
        "det_A": det_A,
        "det_B": det_B,
        "serial_left": b1,
        "serial_right": b2,
        "reachable": True,
    }


def fivebar_tool_metrics(
    E,
    d,
    l1,
    l2,
    l3,
    l4,
    l5,
    left_branch=1,
    right_branch=-1,
    base_y=0.0,
):
    """Compute five-bar singularity metrics for an L5 tool endpoint E."""
    if l5 <= 0:
        result = fivebar_metrics(
            E,
            d,
            l1,
            l2,
            l3,
            l4,
            left_branch=left_branch,
            right_branch=right_branch,
            base_y=base_y,
        )
        if result is not None:
            result["E"] = np.asarray(E, dtype=float)
            result["P"] = np.asarray(E, dtype=float)
        return result

    A0, B0 = base_points(d, base_y=base_y)
    E = np.asarray(E, dtype=float)

    C = circle_intersection(A0, l1, E, l2 + l5, branch=left_branch)
    if C is None:
        return None

    P = wrist_from_tool_endpoint(C, E, l2, l5)
    D = circle_intersection(B0, l3, P, l4, branch=right_branch)
    if D is None:
        return None

    u = P - C
    v = P - D

    A_mat = np.vstack([u, v])
    b1 = np.dot(u, rot90(C - A0))
    b2 = np.dot(v, rot90(D - B0))
    B_mat = np.diag([b1, b2])

    det_A = np.linalg.det(A_mat)
    det_B = b1 * b2
    parallel_sin = abs(cross2(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v))

    if abs(det_A) < 1e-10 or abs(det_B) < 1e-10:
        cond = np.inf
    else:
        J = np.linalg.solve(A_mat, B_mat)
        cond = np.linalg.cond(J)

    return {
        "C": C,
        "D": D,
        "P": P,
        "E": E,
        "cond": cond,
        "parallel_sin": parallel_sin,
        "det_A": det_A,
        "det_B": det_B,
        "serial_left": b1,
        "serial_right": b2,
        "reachable": True,
    }


def sample_workspace(
    d=0.45,
    l1=0.65,
    l2=0.65,
    l3=0.65,
    l4=0.65,
    l5=0.0,
    left_branch=1,
    right_branch=-1,
    base_y=0.0,
    nx=350,
    ny=300,
):
    """Sample the five-bar wrist/tool workspace on a grid."""
    max_reach = max(l1 + l2 + l5, l3 + l4 + l5)
    xmin = -d / 2.0 - max_reach * 0.95
    xmax = d / 2.0 + max_reach * 0.95
    ymin = base_y - max_reach * 0.95
    ymax = base_y + max_reach * 1.15

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    reachable = np.zeros_like(X, dtype=bool)
    log_cond = np.full_like(X, np.nan, dtype=float)
    condition_number = np.full_like(X, np.nan, dtype=float)
    parallel_badness = np.full_like(X, np.nan, dtype=float)

    for i in range(ny):
        for j in range(nx):
            result = fivebar_tool_metrics(
                np.array([X[i, j], Y[i, j]]),
                d,
                l1,
                l2,
                l3,
                l4,
                l5,
                left_branch=left_branch,
                right_branch=right_branch,
                base_y=base_y,
            )

            if result is None:
                continue

            reachable[i, j] = True

            cond = result["cond"]
            condition_number[i, j] = cond
            if np.isfinite(cond):
                log_cond[i, j] = np.log10(cond)
            else:
                log_cond[i, j] = 4.0

            # parallel_sin is zero at a parallel singularity.
            # Badness increases as we approach it.
            parallel_badness[i, j] = -np.log10(result["parallel_sin"] + 1e-6)

    return X, Y, reachable, log_cond, condition_number, parallel_badness


def summarise_upper_semicircle(X, Y, reachable, condition_number, radius=1.0):
    """Return reachability and max condition for the upper half-disc."""
    in_semicircle = (Y >= 0.0) & (X * X + Y * Y <= radius * radius)
    sample_count = int(np.count_nonzero(in_semicircle))
    reachable_count = int(np.count_nonzero(in_semicircle & reachable))
    fraction = reachable_count / sample_count if sample_count else np.nan

    reachable_conditions = condition_number[in_semicircle & reachable]
    if reachable_conditions.size:
        max_condition = float(np.nanmax(reachable_conditions))
    else:
        max_condition = np.nan

    return sample_count, reachable_count, fraction, max_condition


def draw_circle(ax, centre, radius, **kwargs):
    if radius <= 0:
        return

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        centre[0] + radius * np.cos(theta),
        centre[1] + radius * np.sin(theta),
        **kwargs,
    )


def draw_right_serial_tool_curve(
    ax,
    A0,
    B0,
    l1,
    l2,
    l3,
    l4,
    l5,
    left_branch,
    radius,
    **kwargs,
):
    """Draw right-chain serial wrist circle mapped into L5 tool-tip space."""
    if radius <= 0:
        return

    theta = np.linspace(0, 2 * np.pi, 800)
    points = []
    for angle in theta:
        wrist = B0 + radius * np.array([np.cos(angle), np.sin(angle)])
        C = circle_intersection(A0, l1, wrist, l2, branch=left_branch)
        if C is None:
            if points and points[-1] is not None:
                points.append(None)
            continue

        endpoint = tool_endpoint(C, wrist, l2, l5)
        if endpoint is None:
            if points and points[-1] is not None:
                points.append(None)
            continue
        points.append(endpoint)

    segment = []
    for point in points + [None]:
        if point is None:
            if len(segment) > 1:
                segment = np.array(segment)
                ax.plot(segment[:, 0], segment[:, 1], **kwargs)
            segment = []
            continue
        segment.append(point)


def plot_fivebar_singularities(
    d=0.45,
    l1=0.65,
    l2=0.65,
    l3=0.65,
    l4=0.65,
    l5=0.0,
    left_branch=1,
    right_branch=-1,
    base_y=0.0,
    target_radius=None,
    target_shape="semicircle",
):
    """Create plots for one fixed assembly condition, selectable by branch pair."""
    A0, B0 = base_points(d, base_y=base_y)
    branch_options = {
        "(+1, +1)": (1, 1),
        "(+1, -1)": (1, -1),
        "(-1, +1)": (-1, 1),
        "(-1, -1)": (-1, -1),
    }
    labels = list(branch_options)
    active_label = f"({left_branch:+d}, {right_branch:+d})"
    active_index = labels.index(active_label) if active_label in labels else 0

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.subplots_adjust(left=0.06, right=0.82, bottom=0.11, top=0.84, wspace=0.25)
    radio_ax = fig.add_axes([0.85, 0.52, 0.12, 0.24])
    radio = RadioButtons(radio_ax, labels, active=active_index)
    radio_ax.set_title("fixed branch")
    summary_ax = fig.add_axes([0.85, 0.23, 0.13, 0.20])
    summary_ax.set_axis_off()
    summary_text = summary_ax.text(0.0, 1.0, "", va="top", ha="left", fontsize=9)
    colourbar = [None]

    def draw_target_overlays():
        if target_radius is None:
            return

        if target_shape == "circle":
            theta = np.linspace(0, 2 * np.pi, 600)
        else:
            theta = np.linspace(0, np.pi, 400)
        target_x = target_radius * np.cos(theta)
        target_y = target_radius * np.sin(theta)
        for target_ax in axes:
            target_ax.plot(target_x, target_y, color="black", linewidth=1.2)
            if target_shape != "circle":
                target_ax.plot(
                    [-target_radius, target_radius],
                    [0.0, 0.0],
                    color="black",
                    linewidth=1.2,
                )

    def redraw(branch_label):
        current_left_branch, current_right_branch = branch_options[branch_label]
        if colourbar[0] is not None:
            colourbar[0].remove()
            colourbar[0] = None

        for axis in axes:
            axis.clear()

        X, Y, reachable, log_cond, condition_number, parallel_badness = sample_workspace(
            d=d,
            l1=l1,
            l2=l2,
            l3=l3,
            l4=l4,
            l5=l5,
            left_branch=current_left_branch,
            right_branch=current_right_branch,
            base_y=base_y,
        )
        summary_radius = 1.0 if target_radius is None else target_radius
        (
            semicircle_samples,
            semicircle_reachable,
            semicircle_fraction,
            semicircle_max_condition,
        ) = summarise_upper_semicircle(
            X,
            Y,
            reachable,
            condition_number,
            radius=summary_radius,
        )

        ax = axes[0]
        ax.contourf(X, Y, reachable.astype(float), levels=[0.5, 1.5], alpha=0.3)
        ax.scatter([A0[0], B0[0]], [A0[1], B0[1]], marker="o", label="base pivots")

        left_effective = l2 + l5
        draw_circle(
            ax,
            A0,
            l1 + left_effective,
            color="tab:green",
            linestyle="--",
            linewidth=1,
            label="left serial singularity at tool tip",
        )
        draw_circle(
            ax,
            A0,
            abs(l1 - left_effective),
            color="tab:green",
            linestyle=":",
            linewidth=1,
        )
        draw_right_serial_tool_curve(
            ax,
            A0,
            B0,
            l1,
            l2,
            l3,
            l4,
            l5,
            current_left_branch,
            l3 + l4,
            color="tab:blue",
            linestyle="--",
            linewidth=1,
            label="right serial singularity at tool tip",
        )
        draw_right_serial_tool_curve(
            ax,
            A0,
            B0,
            l1,
            l2,
            l3,
            l4,
            l5,
            current_left_branch,
            abs(l3 - l4),
            color="tab:blue",
            linestyle=":",
            linewidth=1,
        )

        ax.set_title("Reachable tool workspace and serial singularity curves")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend(loc="lower right")

        ax = axes[1]
        capped = np.clip(log_cond, 0, 4)
        im = ax.pcolormesh(X, Y, capped, shading="auto")
        colourbar[0] = fig.colorbar(im, ax=ax)
        colourbar[0].set_label("log10(condition number), capped at 4")

        finite_parallel = np.isfinite(parallel_badness)
        if finite_parallel.any():
            ax.contour(X, Y, parallel_badness, levels=[1.5, 2.0, 2.5], linewidths=1)

        ax.scatter([A0[0], B0[0]], [A0[1], B0[1]], marker="o", label="base pivots")
        ax.set_title("Kinematic conditioning and near-parallel singularities")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        draw_target_overlays()

        fig.suptitle(
            f"Fixed assembly condition {branch_label}; "
            f"d={d}, l1={l1}, l2={l2}, l3={l3}, l4={l4}, l5={l5}, base_y={base_y}"
        )
        if np.isfinite(semicircle_max_condition):
            max_condition_text = f"{semicircle_max_condition:.2f}"
        else:
            max_condition_text = "n/a"
        summary_text.set_text(
            f"{summary_radius:g} m upper semicircle\n"
            f"reachable: {semicircle_reachable} / {semicircle_samples} "
            f"({100.0 * semicircle_fraction:.1f}%)\n"
            f"max condition: {max_condition_text}"
        )
        fig.canvas.draw_idle()

    radio.on_clicked(redraw)
    redraw(labels[active_index])

    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(description="Plot five-bar linkage singularities.")
    parser.add_argument(
        "--config",
        help="Path to a TOML config file with [linkage] and optional [singularity_plot] settings.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        linkage_config, _, plot_config = load_config_file(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    plot_fivebar_singularities(
        d=linkage_config.d,
        l1=linkage_config.l1,
        l2=linkage_config.l2,
        l3=linkage_config.l3,
        l4=linkage_config.l4,
        l5=linkage_config.l5,
        left_branch=linkage_config.left_branch,
        right_branch=linkage_config.right_branch,
        base_y=linkage_config.base_y,
        target_radius=plot_config.target_radius,
        target_shape=plot_config.target_shape,
    )


if __name__ == "__main__":
    main()
