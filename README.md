# Fivebar

Interactive five-bar linkage simulator, singularity visualisation, and configuration search tools for planar parallel robotics and mechanism design.

Live tools: <https://unswei.github.io/fivebar/>

Code: <https://github.com/unswei/fivebar>

Lab: <https://unswei.github.io/>

## What This Is

This repository explores a planar five-bar linkage with a rigid `L5` tool extension. The mechanism has two actuated base joints, two passive elbow joints, and a closed-chain wrist point. The `L5` link extends the left distal link rigidly beyond the wrist, so the tool tip is not the same point as the five-bar loop closure.

The project is aimed at research and early mechanism design. It helps answer practical questions such as:

- which tool-tip positions are reachable for a fixed assembly branch
- where serial or parallel singularities occur
- how well conditioned the linkage is across a target workspace
- what idealised holding torque the two base actuators see under a point load
- how a 2D five-bar stage behaves when swept into 3D by an added base x-axis actuator

Search keywords: five-bar linkage, fivebar linkage, planar parallel robot, parallel mechanism, linkage kinematics, singularity analysis, condition number, static torque, mechanism design, robotics simulator.

## Web Tools

The static JavaScript tools run directly from GitHub Pages and require no server:

- [Landing page](https://unswei.github.io/fivebar/)
- [Interactive simulator](https://unswei.github.io/fivebar/web/)
- [Fixed-branch singularity plot](https://unswei.github.io/fivebar/web/singularity.html)

The browser simulator shows:

- base joint angle control
- polar target control by distance and angle
- closure branch and IK branch-pair selection
- optional branch-switch prevention
- tool-tip, wrist, and trajectory traces
- condition number, serial margin, and parallel-singularity margin
- ideal quasi-static holding torque for a configurable tip load

The singularity page samples a fixed assembly branch across the workspace. It reports full plot reachability, reachability over the 1 m upper semicircle, and max condition number for that semicircle.

## Python Tools

Install dependencies with uv:

```bash
uv sync
```

Run the interactive 2D simulator:

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

Run the 3D reachability plot:

```bash
uv run fivebar-plot-spatial-reachability
uv run fivebar-plot-spatial-reachability --branch +1,-1 --radius 1
```

Run the search tools:

```bash
uv run fivebar-search-config
uv run fivebar-search-torque-config
```

The search tools are sampled optimisations, not formal proofs. Increase `--samples`, `--radial-samples`, and `--angular-samples` for more stringent checks.

## Singularity Terms

**Condition number** is the ratio between the largest and smallest singular values of the linkage Jacobian. In this project it measures how unevenly actuator motion maps to tool/wrist motion. A condition number near `1` is well conditioned. Larger values mean one direction is becoming weak or hard to control. Very large values indicate proximity to a singularity.

**Serial singularity** is an inverse-kinematic singularity of one side chain. It occurs when an actuated proximal link and its distal link are folded or stretched so that the actuator loses useful leverage over the wrist/tool point.

**Parallel singularity** is a direct-kinematic singularity of the closed five-bar loop. It occurs when the two distal constraints become locally dependent. Near this condition the mechanism can lose stiffness or gain an uncontrolled motion direction.

The visualisations keep a fixed assembly branch when requested. This matters because showing the best branch point-by-point can hide discontinuities that a real mechanism cannot pass through without changing assembly mode.

## Current Best Practical Config

The current default configuration is the best practical short-link configuration found so far for the 1 m upper semicircle target, including the `L5` extension.

```toml
[linkage]
d = 0.1917
base_y = -0.3745
l1 = 0.5829
l2 = 0.4632
l3 = 0.3308
l4 = 0.7360
l5 = 0.3790
left_branch = 1
right_branch = -1
```

For the default `(+1, -1)` branch, the current web singularity sampler reports the 1 m upper semicircle as fully reachable at the sampled resolution, with max condition around `6.61`. The 3D swept model, using a base x-axis actuator over `0..180 deg`, also samples the 1 m `z >= 0` hemisphere surface as fully reachable for the same branch in the current coarse check.

## Repository Layout

```text
.
├── configs/default.toml      Example runtime config
├── src/fivebar/              Python package and Matplotlib tools
├── web/                      Static JavaScript simulator and singularity plot
├── index.html                Static landing page for GitHub Pages
├── pyproject.toml            Python project metadata and command entry points
├── uv.lock                   Locked Python dependencies
├── README.md
└── LICENSE
```

Edit `src/fivebar/config.py` to change the built-in defaults. To run with a separate TOML file, use `--config`.

## Limitations

- The models are ideal kinematic and quasi-static tools.
- Link masses, actuator masses, friction, bearing losses, backlash, compliance, cable stretch, and gearbox effects are not included.
- Torque estimates use a point load at the `L5` tip via the tip Jacobian.
- Dynamic effects such as acceleration, vibration, impacts, and controller limits are not modelled.
- Singularity and reachability claims are based on sampled grids unless otherwise stated.
- The 3D tools currently model the planar linkage swept around a base x-axis; they do not yet include a full 3-DOF wrist or full 6-DOF task-space Jacobian.

## Citation And Contact

This is an exploratory research tool rather than an archival publication. If it supports your work, cite the GitHub repository URL and include the commit hash used for your results.

Suggested citation:

```text
Obst, O. Fivebar: five-bar linkage simulation and singularity exploration tools.
GitHub repository, https://github.com/unswei/fivebar
```

Contact: [embodied intelligence and collective robotics](https://unswei.github.io/) at UNSW.

## Licence

MIT. See [LICENSE](LICENSE).
