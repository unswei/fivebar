import argparse

import numpy as np
import matplotlib.pyplot as plt

from .config import load_config_file
from .singularity_plotter import fivebar_tool_metrics


def branch_pair(value):
    """Parse a branch pair of the form +1,-1."""
    parts = value.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("branch pair must look like +1,-1")
    try:
        pair = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("branch values must be +1 or -1") from error
    if pair[0] not in (-1, 1) or pair[1] not in (-1, 1):
        raise argparse.ArgumentTypeError("branch values must be +1 or -1")
    return pair


def spatial_to_planar(point):
    """Map a 3D point to the equivalent planar five-bar target.

    The 3D actuator rotates the planar mechanism about the world x-axis. For a
    point (x, y, z), the equivalent planar target is (x, sqrt(y^2 + z^2)).
    """
    x, y, z = point
    return np.array([x, np.hypot(y, z)], dtype=float)


def is_reachable(point, linkage, left_branch, right_branch):
    """Return the planar five-bar metrics for a 3D target point, or None."""
    planar_target = spatial_to_planar(point)
    return fivebar_tool_metrics(
        planar_target,
        linkage.d,
        linkage.l1,
        linkage.l2,
        linkage.l3,
        linkage.l4,
        linkage.l5,
        left_branch=left_branch,
        right_branch=right_branch,
        base_y=linkage.base_y,
    )


def sample_half_ball(linkage, radius, grid, left_branch, right_branch):
    """Sample reachability inside the z >= 0 half-ball."""
    xs = np.linspace(-radius, radius, grid)
    ys = np.linspace(-radius, radius, grid)
    zs = np.linspace(0.0, radius, max(2, grid // 2 + 1))

    sample_count = 0
    reachable_count = 0
    max_condition = 0.0

    for x in xs:
        for y in ys:
            for z in zs:
                if x * x + y * y + z * z > radius * radius:
                    continue
                sample_count += 1
                metrics = is_reachable((x, y, z), linkage, left_branch, right_branch)
                if metrics is None:
                    continue
                reachable_count += 1
                max_condition = max(max_condition, metrics["cond"])

    fraction = reachable_count / sample_count if sample_count else np.nan
    return {
        "sample_count": sample_count,
        "reachable_count": reachable_count,
        "fraction": fraction,
        "max_condition": max_condition if reachable_count else np.nan,
    }


def sample_hemisphere_surface(linkage, radius, polar_samples, roll_samples, left_branch, right_branch):
    """Sample the z >= 0 hemisphere surface swept by the x-axis actuator."""
    points = []
    reachable = []
    condition = []

    # alpha parameterises the original 2D upper semicircle:
    # x = r cos(alpha), planar_y = r sin(alpha), alpha in [0, pi].
    # roll then rotates planar_y around x by beta in [0, pi].
    for alpha in np.linspace(0.0, np.pi, polar_samples):
        x = radius * np.cos(alpha)
        planar_y = radius * np.sin(alpha)
        for beta in np.linspace(0.0, np.pi, roll_samples):
            point = np.array(
                [
                    x,
                    planar_y * np.cos(beta),
                    planar_y * np.sin(beta),
                ],
                dtype=float,
            )
            metrics = is_reachable(point, linkage, left_branch, right_branch)
            points.append(point)
            reachable.append(metrics is not None)
            condition.append(metrics["cond"] if metrics is not None else np.nan)

    return np.array(points), np.array(reachable, dtype=bool), np.array(condition, dtype=float)


def format_fraction(summary):
    return (
        f"{summary['reachable_count']} / {summary['sample_count']} "
        f"({100.0 * summary['fraction']:.1f}%)"
    )


def plot_spatial_reachability(
    linkage,
    radius=1.0,
    left_branch=1,
    right_branch=-1,
    volume_grid=51,
    polar_samples=61,
    roll_samples=61,
):
    """Plot reachability for a five-bar stage rotated around the x-axis."""
    volume_summary = sample_half_ball(
        linkage,
        radius,
        volume_grid,
        left_branch,
        right_branch,
    )
    surface_points, surface_reachable, surface_condition = sample_hemisphere_surface(
        linkage,
        radius,
        polar_samples,
        roll_samples,
        left_branch,
        right_branch,
    )

    surface_summary = {
        "sample_count": int(surface_reachable.size),
        "reachable_count": int(np.count_nonzero(surface_reachable)),
        "fraction": float(np.mean(surface_reachable)) if surface_reachable.size else np.nan,
        "max_condition": (
            float(np.nanmax(surface_condition[surface_reachable]))
            if np.any(surface_reachable)
            else np.nan
        ),
    }
    print(
        "Hemisphere surface reachable: "
        f"{format_fraction(surface_summary)}; "
        f"max condition: {surface_summary['max_condition']:.2f}"
    )
    print(
        "Half-ball volume reachable: "
        f"{format_fraction(volume_summary)}; "
        f"max condition: {volume_summary['max_condition']:.2f}"
    )

    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    summary_ax = fig.add_subplot(1, 2, 2)
    summary_ax.set_axis_off()

    reachable_points = surface_points[surface_reachable]
    unreachable_points = surface_points[~surface_reachable]
    if unreachable_points.size:
        ax.scatter(
            unreachable_points[:, 0],
            unreachable_points[:, 1],
            unreachable_points[:, 2],
            s=5,
            c="0.82",
            alpha=0.25,
            label="unreachable",
        )
    if reachable_points.size:
        log_condition = np.log10(np.clip(surface_condition[surface_reachable], 1.0, 1e4))
        scatter = ax.scatter(
            reachable_points[:, 0],
            reachable_points[:, 1],
            reachable_points[:, 2],
            s=7,
            c=log_condition,
            cmap="viridis_r",
            vmin=0.0,
            vmax=4.0,
            alpha=0.85,
            label="reachable",
        )
        colourbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.05)
        colourbar.set_label("log10(condition number), capped at 4")

    ax.set_title(f"{radius:g} m hemisphere surface reachability")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_box_aspect((1, 1, 0.55))
    ax.legend(loc="upper left")

    max_surface_condition = surface_summary["max_condition"]
    max_volume_condition = volume_summary["max_condition"]
    surface_condition_text = f"{max_surface_condition:.2f}" if np.isfinite(max_surface_condition) else "n/a"
    volume_condition_text = f"{max_volume_condition:.2f}" if np.isfinite(max_volume_condition) else "n/a"
    summary_ax.text(
        0.0,
        1.0,
        "\n".join(
            [
                "3D five-bar reachability",
                "",
                "Model:",
                "planar five-bar swept around the base x-axis",
                "base x-axis rotation range: 0 to 180 deg",
                "",
                f"branch pair: ({left_branch:+d}, {right_branch:+d})",
                f"radius: {radius:g} m",
                "",
                "Hemisphere surface:",
                f"reachable: {format_fraction(surface_summary)}",
                f"max condition: {surface_condition_text}",
                "",
                "Half-ball volume:",
                f"reachable: {format_fraction(volume_summary)}",
                f"max condition: {volume_condition_text}",
                "",
                "Notes:",
                "z >= 0 is sampled.",
                "Each 3D point maps to planar target",
                "(x, sqrt(y^2 + z^2)).",
                "Condition is the planar five-bar condition",
                "reported by the existing singularity model.",
            ]
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(
        "Five-bar spatial reachability with one base x-axis actuator",
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot 3D reachability for a five-bar stage swept around the base x-axis."
    )
    parser.add_argument(
        "--config",
        help="Path to a TOML config file with [linkage] settings.",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Half-sphere radius in metres.")
    parser.add_argument(
        "--branch",
        type=branch_pair,
        help="Fixed branch pair, for example +1,-1. Defaults to the config branch pair.",
    )
    parser.add_argument(
        "--volume-grid",
        type=int,
        default=51,
        help="Cartesian grid resolution for sampled half-ball volume.",
    )
    parser.add_argument(
        "--polar-samples",
        type=int,
        default=61,
        help="Samples along the original planar semicircle for the hemisphere surface.",
    )
    parser.add_argument(
        "--roll-samples",
        type=int,
        default=61,
        help="Samples across the 0 to 180 degree base x-axis roll.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        linkage, _, _ = load_config_file(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    left_branch, right_branch = args.branch or (linkage.left_branch, linkage.right_branch)
    plot_spatial_reachability(
        linkage,
        radius=args.radius,
        left_branch=left_branch,
        right_branch=right_branch,
        volume_grid=args.volume_grid,
        polar_samples=args.polar_samples,
        roll_samples=args.roll_samples,
    )


if __name__ == "__main__":
    main()
