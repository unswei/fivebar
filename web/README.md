# Web App Placeholder

This folder contains the static JavaScript version of the five-bar simulator and singularity plot. The simulator exposes the same closure branch, IK branch-pair, and branch-switch prevention controls as the Python simulator.

The intended deployment target is GitHub Pages, so the web app should remain client-side only: HTML, CSS, JavaScript, and static assets. If a bundler is introduced later, keep generated output such as `dist/` out of source control unless deployment requires it.
