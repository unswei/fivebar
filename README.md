# Fivebar

Five-bar linkage simulation and singularity exploration tools.

This repository contains Python/Matplotlib tools and a static JavaScript browser simulator for exploring a planar five-bar mechanism with a rigid L5 tool extension.

## Current Tools

- `fivebar-simulator`: interactive linkage simulator with base joint control, polar target control, closure and IK branch selection, optional branch-switch prevention, singularity readout, trajectory trace, and static tip-load torque estimates.
- `fivebar-spatial-simulator`: interactive 3D simulator for the five-bar stage swept around the base x-axis.
- `fivebar-plot-singularities`: workspace and singularity plots for a fixed assembly branch.
- `fivebar-plot-spatial-reachability`: first 3D reachability plot for a planar five-bar swept 180 degrees around the base x-axis.
- `fivebar-search-config`: coarse search tool for candidate linkage dimensions.
- `fivebar-search-torque-config`: random/refinement search for L1..L5 dimensions that cover the radius-1 upper semicircle while minimising worst single-joint holding torque.
- `web/`: static browser simulator and singularity plot for GitHub Pages.

## Repository Layout

```text
.
├── configs/default.toml      Example runtime config
├── src/fivebar/              Python package and Matplotlib tools
│   └── config.py             Shared linkage/default configuration
├── web/                      Static JavaScript simulator and singularity plot
├── pyproject.toml            Python project metadata and command entry points
├── uv.lock                   Locked Python dependencies
├── README.md
└── LICENSE
```

Edit `src/fivebar/config.py` to change the built-in defaults. To run with a separate TOML file, use `--config`.

## Setup

Install dependencies with uv:

```bash
uv sync
```

Run the interactive simulator:

```bash
uv run fivebar-simulator
uv run fivebar-simulator --config configs/default.toml
```

Run the interactive 3D simulator:

```bash
uv run fivebar-spatial-simulator
uv run fivebar-spatial-simulator --config configs/default.toml
```

Run the singularity plotter:

```bash
uv run fivebar-plot-singularities
uv run fivebar-plot-singularities --config configs/default.toml
```

Run the first 3D reachability plot:

```bash
uv run fivebar-plot-spatial-reachability
uv run fivebar-plot-spatial-reachability --branch +1,-1 --radius 1
```

Run the search tool:

```bash
uv run fivebar-search-config
```

Run the torque-optimising search:

```bash
uv run fivebar-search-torque-config
```

This is a sampled optimisation, not a formal proof. Increase `--samples`, `--radial-samples`, and `--angular-samples` for a more stringent search.

## Browser Simulator

The static JavaScript tools live in `web/` and have no build step. They can run from GitHub Pages directly.

Live version:

```text
https://unswei.github.io/fivebar/
```

The simulator is at `/web/`, and the fixed-branch singularity plot is at `/web/singularity.html`.

For local preview:

```bash
python3 -m http.server 4173
```

Then open:

```text
http://127.0.0.1:4173/web/
```

If GitHub Pages is configured to serve the repository root, the root `index.html` redirects to the web simulator.

## Notes

The simulator torque readout is quasi-static. It estimates the actuator holding torques for a point load at the L5 tip via the tip Jacobian. It does not include link masses, friction, inertia, gearbox effects, or actuator limits.

The browser implementation is intentionally static and client-side so it can run from GitHub Pages without a build step or server.
