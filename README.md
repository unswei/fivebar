# Fivebar

Five-bar linkage simulation and singularity exploration tools.

This repository currently contains Python/Matplotlib tools for exploring a planar five-bar mechanism with a rigid L5 tool extension. The layout also reserves a `web/` folder for a future JavaScript/GitHub Pages version.

## Current Tools

- `fivebar-simulator`: interactive linkage simulator with base joint control, polar target control, branch selection, singularity readout, trajectory trace, and static tip-load torque estimates.
- `fivebar-plot-singularities`: workspace and singularity plots for a fixed assembly branch.
- `fivebar-search-config`: coarse search tool for candidate linkage dimensions.
- `fivebar-search-torque-config`: random/refinement search for L1..L5 dimensions that cover the radius-1 upper semicircle while minimising worst single-joint holding torque.

## Repository Layout

```text
.
├── configs/default.toml      Example runtime config
├── src/fivebar/              Python package and Matplotlib tools
│   └── config.py             Shared linkage/default configuration
├── web/                      Placeholder for a future static JavaScript app
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

Run the singularity plotter:

```bash
uv run fivebar-plot-singularities
uv run fivebar-plot-singularities --config configs/default.toml
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

The static JavaScript simulator lives in `web/` and has no build step. It can run from GitHub Pages directly.

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

The future browser implementation should keep the shared geometry concepts close to the Python version, but it can be built as a static client-side app suitable for GitHub Pages.
