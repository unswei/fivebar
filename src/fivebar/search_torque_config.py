import argparse
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_LINKAGE
from .search_config import circle_intersections, semicircle_points
from .singularity_plotter import base_points


CURRENT_CONFIG = {
    "d": DEFAULT_LINKAGE.d,
    "base_y": DEFAULT_LINKAGE.base_y,
    "l1": DEFAULT_LINKAGE.l1,
    "l2": DEFAULT_LINKAGE.l2,
    "l3": DEFAULT_LINKAGE.l3,
    "l4": DEFAULT_LINKAGE.l4,
    "l5": DEFAULT_LINKAGE.l5,
    "left_branch": DEFAULT_LINKAGE.left_branch,
    "right_branch": DEFAULT_LINKAGE.right_branch,
}


@dataclass(frozen=True)
class Geometry:
    d: float
    base_y: float
    l1: float
    l2: float
    l3: float
    l4: float
    l5: float
    left_branch: int
    right_branch: int

    @property
    def max_link(self):
        return max(self.l1, self.l2, self.l3, self.l4, self.l5)

    @property
    def total_link(self):
        return self.l1 + self.l2 + self.l3 + self.l4 + self.l5


@dataclass(frozen=True)
class EvaluatedCandidate:
    geometry: Geometry
    max_abs_torque: float
    max_abs_left_torque: float
    max_abs_right_torque: float
    min_parallel_sin: float
    min_serial: float
    max_condition: float
    worst_point: tuple[float, float]
    worst_joint: str


def rot90_rows(v):
    """Rotate an array of 2D row vectors by +90 degrees."""
    return np.stack([-v[:, 1], v[:, 0]], axis=1)


def evaluate_geometry(points, geometry, force, min_parallel_sin, min_serial, max_condition):
    """Evaluate reachability, singularity margins, and static tip-load torque."""
    if min(geometry.l1, geometry.l2, geometry.l3, geometry.l4) <= 0.0 or geometry.l5 < 0.0:
        return None

    A0, B0 = base_points(geometry.d, base_y=geometry.base_y)

    # For the left chain, L5 makes the tool point lie on a longer ray from C.
    left_tool_span = geometry.l2 + geometry.l5
    C, ok_left = circle_intersections(
        A0,
        geometry.l1,
        points,
        left_tool_span,
        geometry.left_branch,
    )
    if not np.all(ok_left):
        return None

    P = C + (geometry.l2 / left_tool_span) * (points - C)
    D, ok_right = circle_intersections(
        B0,
        geometry.l3,
        P,
        geometry.l4,
        geometry.right_branch,
    )
    if not np.all(ok_right):
        return None

    u = P - C
    v = P - D
    u_norm = np.linalg.norm(u, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    if np.any(u_norm < 1e-12) or np.any(v_norm < 1e-12):
        return None

    parallel_sin = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / (u_norm * v_norm)
    min_seen_parallel = parallel_sin.min()
    if min_seen_parallel < min_parallel_sin:
        return None

    c_dot = rot90_rows(C - A0)
    # The right slider is an inward angle, so its derivative is -rot90(D - B0).
    d_dot = -rot90_rows(D - B0)
    b_left = np.sum(u * c_dot, axis=1)
    b_right = np.sum(v * d_dot, axis=1)
    serial = np.minimum(np.abs(b_left), np.abs(b_right))
    min_seen_serial = serial.min()
    if min_seen_serial < min_serial:
        return None

    a_mats = np.stack([u, v], axis=1)
    b_mats = np.zeros((len(points), 2, 2), dtype=float)
    b_mats[:, 0, 0] = b_left
    b_mats[:, 1, 1] = b_right

    try:
        wrist_jacobians = np.linalg.solve(a_mats, b_mats)
    except np.linalg.LinAlgError:
        return None

    singular_values = np.linalg.svd(wrist_jacobians, compute_uv=False)
    conditions = singular_values[:, 0] / singular_values[:, -1]
    if not np.all(np.isfinite(conditions)):
        return None

    max_seen_condition = conditions.max()
    if max_seen_condition > max_condition:
        return None

    alpha = left_tool_span / geometry.l2
    tip_jacobians = alpha * wrist_jacobians
    tip_jacobians[:, :, 0] += (1.0 - alpha) * c_dot

    load_torques = np.einsum("nik,i->nk", tip_jacobians, force)
    holding_torques = -load_torques
    abs_torques = np.abs(holding_torques)
    worst_flat_index = int(np.argmax(abs_torques))
    worst_point_index, worst_joint_index = np.unravel_index(worst_flat_index, abs_torques.shape)

    return EvaluatedCandidate(
        geometry=geometry,
        max_abs_torque=float(abs_torques[worst_point_index, worst_joint_index]),
        max_abs_left_torque=float(abs_torques[:, 0].max()),
        max_abs_right_torque=float(abs_torques[:, 1].max()),
        min_parallel_sin=float(min_seen_parallel),
        min_serial=float(min_seen_serial),
        max_condition=float(max_seen_condition),
        worst_point=(float(points[worst_point_index, 0]), float(points[worst_point_index, 1])),
        worst_joint="left" if worst_joint_index == 0 else "right",
    )


def random_geometry(rng, args, branch_pair):
    d = rng.uniform(args.min_d, args.max_d)
    base_y = -rng.uniform(args.min_base_clearance, args.max_base_depth)
    return Geometry(
        d=d,
        base_y=base_y,
        l1=rng.uniform(args.min_link, args.max_link),
        l2=rng.uniform(args.min_link, args.max_link),
        l3=rng.uniform(args.min_link, args.max_link),
        l4=rng.uniform(args.min_link, args.max_link),
        l5=rng.uniform(args.min_l5, args.max_l5),
        left_branch=branch_pair[0],
        right_branch=branch_pair[1],
    )


def clip_geometry(values, args, branch_pair):
    return Geometry(
        d=float(np.clip(values[0], args.min_d, args.max_d)),
        base_y=float(-np.clip(-values[1], args.min_base_clearance, args.max_base_depth)),
        l1=float(np.clip(values[2], args.min_link, args.max_link)),
        l2=float(np.clip(values[3], args.min_link, args.max_link)),
        l3=float(np.clip(values[4], args.min_link, args.max_link)),
        l4=float(np.clip(values[5], args.min_link, args.max_link)),
        l5=float(np.clip(values[6], args.min_l5, args.max_l5)),
        left_branch=branch_pair[0],
        right_branch=branch_pair[1],
    )


def perturb_geometry(rng, geometry, args, scale):
    values = np.array(
        [
            geometry.d,
            geometry.base_y,
            geometry.l1,
            geometry.l2,
            geometry.l3,
            geometry.l4,
            geometry.l5,
        ],
        dtype=float,
    )
    spans = np.array(
        [
            args.max_d - args.min_d,
            args.max_base_depth - args.min_base_clearance,
            args.max_link - args.min_link,
            args.max_link - args.min_link,
            args.max_link - args.min_link,
            args.max_link - args.min_link,
            args.max_l5 - args.min_l5,
        ],
        dtype=float,
    )
    trial_values = values + rng.normal(0.0, scale, size=len(values)) * spans
    return clip_geometry(trial_values, args, (geometry.left_branch, geometry.right_branch))


def candidate_key(candidate):
    return (
        candidate.max_abs_torque,
        candidate.geometry.max_link,
        candidate.geometry.total_link,
        -candidate.min_parallel_sin,
        -candidate.min_serial,
    )


def add_if_valid(candidates, points, geometry, force, args):
    evaluated = evaluate_geometry(
        points,
        geometry,
        force,
        min_parallel_sin=args.min_parallel_sin,
        min_serial=args.min_serial,
        max_condition=args.max_condition,
    )
    if evaluated is not None:
        candidates.append(evaluated)


def search(args):
    rng = np.random.default_rng(args.seed)
    points = semicircle_points(
        radius=args.radius,
        radial_samples=args.radial_samples,
        angular_samples=args.angular_samples,
    )
    force = args.mass * np.array([args.gravity_x, args.gravity_y], dtype=float)
    branch_pairs = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    candidates = []

    if args.include_current:
        add_if_valid(
            candidates,
            points,
            Geometry(**CURRENT_CONFIG),
            force,
            args,
        )

    for index in range(args.samples):
        branch_pair = branch_pairs[index % len(branch_pairs)] if args.all_branches else branch_pairs[0]
        geometry = random_geometry(rng, args, branch_pair)
        add_if_valid(candidates, points, geometry, force, args)

    candidates.sort(key=candidate_key)
    candidates = candidates[: args.keep]

    for round_index in range(args.refine_rounds):
        if not candidates:
            break
        scale = args.refine_scale * (args.refine_decay**round_index)
        parents = candidates[: args.refine_parents]
        refined = list(candidates)
        for parent in parents:
            for _ in range(args.refine_children):
                geometry = perturb_geometry(rng, parent.geometry, args, scale)
                add_if_valid(refined, points, geometry, force, args)
        refined.sort(key=candidate_key)
        candidates = refined[: args.keep]

    candidates.sort(key=candidate_key)
    return candidates, points, force


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Search L1..L5 configurations that cover a sampled upper semicircle "
            "without singularities while minimising worst single-joint holding torque."
        )
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--gravity-x", type=float, default=9.81)
    parser.add_argument("--gravity-y", type=float, default=0.0)
    parser.add_argument("--min-d", type=float, default=0.15)
    parser.add_argument("--max-d", type=float, default=0.25)
    parser.add_argument("--min-base-clearance", type=float, default=0.05)
    parser.add_argument("--max-base-depth", type=float, default=0.9)
    parser.add_argument("--min-link", type=float, default=0.25)
    parser.add_argument("--max-link", type=float, default=1.2)
    parser.add_argument("--min-l5", type=float, default=0.0)
    parser.add_argument("--max-l5", type=float, default=0.8)
    parser.add_argument("--min-parallel-sin", type=float, default=0.08)
    parser.add_argument("--min-serial", type=float, default=0.015)
    parser.add_argument("--max-condition", type=float, default=50.0)
    parser.add_argument("--radial-samples", type=int, default=25)
    parser.add_argument("--angular-samples", type=int, default=73)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--refine-rounds", type=int, default=4)
    parser.add_argument("--refine-parents", type=int, default=12)
    parser.add_argument("--refine-children", type=int, default=120)
    parser.add_argument("--refine-scale", type=float, default=0.06)
    parser.add_argument("--refine-decay", type=float, default=0.55)
    parser.add_argument("--all-branches", dest="all_branches", action="store_true")
    parser.add_argument("--single-branch", dest="all_branches", action="store_false")
    parser.add_argument("--no-include-current", dest="include_current", action="store_false")
    parser.set_defaults(all_branches=True, include_current=True)
    return parser


def format_candidate(index, candidate):
    g = candidate.geometry
    return (
        "#{index} max_abs_torque={torque:.3f} Nm "
        "left_max={left:.3f} Nm right_max={right:.3f} Nm "
        "d={d:.4f} base_y={base_y:.4f} "
        "l1={l1:.4f} l2={l2:.4f} l3={l3:.4f} l4={l4:.4f} l5={l5:.4f} "
        "branches=({lb:+d},{rb:+d}) "
        "min_parallel_sin={parallel:.4f} min_serial={serial:.4f} "
        "max_condition={condition:.2f} "
        "worst_joint={joint} worst_point=({x:.3f},{y:.3f})"
    ).format(
        index=index,
        torque=candidate.max_abs_torque,
        left=candidate.max_abs_left_torque,
        right=candidate.max_abs_right_torque,
        d=g.d,
        base_y=g.base_y,
        l1=g.l1,
        l2=g.l2,
        l3=g.l3,
        l4=g.l4,
        l5=g.l5,
        lb=g.left_branch,
        rb=g.right_branch,
        parallel=candidate.min_parallel_sin,
        serial=candidate.min_serial,
        condition=candidate.max_condition,
        joint=candidate.worst_joint,
        x=candidate.worst_point[0],
        y=candidate.worst_point[1],
    )


def main():
    args = build_parser().parse_args()
    candidates, points, force = search(args)
    print(
        "sampled_points={} force=({:.3f},{:.3f}) N d_range=({:.3f},{:.3f})".format(
            len(points),
            force[0],
            force[1],
            args.min_d,
            args.max_d,
        )
    )
    if not candidates:
        print("No candidates found. Try increasing --samples or relaxing the singularity margins.")
        return

    for index, candidate in enumerate(candidates[: args.top], start=1):
        print(format_candidate(index, candidate))


if __name__ == "__main__":
    main()
