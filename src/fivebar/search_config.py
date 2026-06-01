import argparse
from dataclasses import dataclass

import numpy as np

from .singularity_plotter import base_points


@dataclass(frozen=True)
class Candidate:
    max_link: float
    d: float
    base_y: float
    l1: float
    l2: float
    l3: float
    l4: float
    left_branch: int
    right_branch: int
    min_parallel_sin: float
    min_serial: float
    max_condition: float


def semicircle_points(radius=1.0, radial_samples=41, angular_samples=121):
    """Sample the filled upper semicircle, with extra points on the boundary."""
    rs = np.linspace(0.0, radius, radial_samples)
    theta = np.linspace(0.0, np.pi, angular_samples)

    points = [[r * np.cos(t), r * np.sin(t)] for r in rs for t in theta]
    points.extend([[-radius + 2.0 * radius * i / 300.0, 0.0] for i in range(301)])
    points.extend([[radius * np.cos(t), radius * np.sin(t)] for t in theta])
    return np.array(points, dtype=float)


def rot90(v):
    """Rotate 2D vectors by +90 degrees."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def circle_intersections(c0, r0, points, r1, branch, eps=1e-12):
    """Vectorised circle intersection for one fixed circle and many target points."""
    delta = points - c0
    R = np.linalg.norm(delta, axis=1)
    safe_R = np.where(R > eps, R, 1.0)
    a = (r0**2 - r1**2 + R**2) / (2.0 * safe_R)
    h2 = r0**2 - a**2
    ok = (R > eps) & (h2 >= -1e-9)

    e = delta / safe_R[:, None]
    p = c0 + a[:, None] * e
    return p + branch * np.sqrt(np.maximum(0.0, h2))[:, None] * rot90(e), ok


def evaluate_candidate(points, candidate, min_parallel_sin, min_serial, compute_condition=False):
    A0, B0 = base_points(candidate.d, base_y=candidate.base_y)
    C, ok_left = circle_intersections(
        A0, candidate.l1, points, candidate.l2, candidate.left_branch
    )
    D, ok_right = circle_intersections(
        B0, candidate.l3, points, candidate.l4, candidate.right_branch
    )
    if not np.all(ok_left & ok_right):
        return None

    u = points - C
    v = points - D
    u_norm = np.linalg.norm(u, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    parallel = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / (u_norm * v_norm)

    b1 = np.sum(u * rot90(C - A0), axis=1)
    b2 = np.sum(v * rot90(D - B0), axis=1)
    serial = np.minimum(np.abs(b1), np.abs(b2))

    min_seen_parallel = parallel.min()
    min_seen_serial = serial.min()
    if min_seen_parallel < min_parallel_sin or min_seen_serial < min_serial:
        return None

    max_seen_condition = np.nan
    if compute_condition:
        A_mats = np.stack([u, v], axis=1)
        B_mats = np.zeros((len(points), 2, 2), dtype=float)
        B_mats[:, 0, 0] = b1
        B_mats[:, 1, 1] = b2

        try:
            jacobians = np.linalg.solve(A_mats, B_mats)
        except np.linalg.LinAlgError:
            return None

        singular_values = np.linalg.svd(jacobians, compute_uv=False)
        conditions = singular_values[:, 0] / singular_values[:, -1]
        if not np.all(np.isfinite(conditions)):
            return None

        max_seen_condition = conditions.max()

    return Candidate(
        max_link=candidate.max_link,
        d=candidate.d,
        base_y=candidate.base_y,
        l1=candidate.l1,
        l2=candidate.l2,
        l3=candidate.l3,
        l4=candidate.l4,
        left_branch=candidate.left_branch,
        right_branch=candidate.right_branch,
        min_parallel_sin=min_seen_parallel,
        min_serial=min_seen_serial,
        max_condition=max_seen_condition,
    )


def minimum_equal_link(points, d, base_y):
    """Smallest equal link length that reaches every sampled point."""
    A0, B0 = base_points(d, base_y=base_y)
    max_radius = max(
        np.linalg.norm(points - A0, axis=1).max(),
        np.linalg.norm(points - B0, axis=1).max(),
    )
    return max_radius / 2.0


def search(args):
    points = semicircle_points(
        radius=args.radius,
        radial_samples=args.radial_samples,
        angular_samples=args.angular_samples,
    )

    candidates = []
    branch_pairs = [(1, -1), (-1, 1), (1, 1), (-1, -1)]
    d_values = np.linspace(args.min_d, args.max_d, args.d_steps)
    base_y_values = -np.linspace(args.min_base_clearance, args.max_base_depth, args.base_y_steps)
    link_factors = np.linspace(args.min_link_factor, args.max_link_factor, args.link_factor_steps)

    for d in d_values:
        for base_y in base_y_values:
            link_min = minimum_equal_link(points, d, base_y)
            for factor in link_factors:
                link = link_min * factor
                if link > args.max_link:
                    continue

                for left_branch, right_branch in branch_pairs:
                    rough = Candidate(
                        max_link=link,
                        d=d,
                        base_y=base_y,
                        l1=link,
                        l2=link,
                        l3=link,
                        l4=link,
                        left_branch=left_branch,
                        right_branch=right_branch,
                        min_parallel_sin=0.0,
                        min_serial=0.0,
                        max_condition=0.0,
                    )
                    candidate = evaluate_candidate(
                        points,
                        rough,
                        min_parallel_sin=args.min_parallel_sin,
                        min_serial=args.min_serial,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

    candidates = sorted(
        candidates,
        key=lambda item: (item.max_link, -item.min_parallel_sin, -item.min_serial),
    )
    return candidates, points


def build_parser():
    parser = argparse.ArgumentParser(
        description="Search symmetric five-bar configurations for a singularity-free upper semicircle."
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--min-d", type=float, default=0.1)
    parser.add_argument("--max-d", type=float, default=1.0)
    parser.add_argument("--d-steps", type=int, default=25)
    parser.add_argument("--min-base-clearance", type=float, default=0.05)
    parser.add_argument("--max-base-depth", type=float, default=1.0)
    parser.add_argument("--base-y-steps", type=int, default=25)
    parser.add_argument("--max-link", type=float, default=1.0)
    parser.add_argument("--min-link-factor", type=float, default=1.01)
    parser.add_argument("--max-link-factor", type=float, default=1.35)
    parser.add_argument("--link-factor-steps", type=int, default=10)
    parser.add_argument("--min-parallel-sin", type=float, default=0.1)
    parser.add_argument("--min-serial", type=float, default=0.02)
    parser.add_argument("--radial-samples", type=int, default=31)
    parser.add_argument("--angular-samples", type=int, default=91)
    parser.add_argument("--top", type=int, default=12)
    return parser


def main():
    args = build_parser().parse_args()
    candidates, points = search(args)

    if not candidates:
        print("No candidates found. Try relaxing the margins or increasing --max-link.")
        return

    for candidate in candidates[: args.top]:
        candidate = evaluate_candidate(
            points,
            candidate,
            min_parallel_sin=args.min_parallel_sin,
            min_serial=args.min_serial,
            compute_condition=True,
        )
        if candidate is None:
            continue
        print(
            "max_link={:.4f} d={:.4f} base_y={:.4f} "
            "l1=l2=l3=l4={:.4f} branches=({:+d},{:+d}) "
            "min_parallel_sin={:.4f} min_serial={:.4f} max_condition={:.2f}".format(
                candidate.max_link,
                candidate.d,
                candidate.base_y,
                candidate.l1,
                candidate.left_branch,
                candidate.right_branch,
                candidate.min_parallel_sin,
                candidate.min_serial,
                candidate.max_condition,
            )
        )


if __name__ == "__main__":
    main()
