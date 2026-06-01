const config = {
  linkage: {
    d: 0.173,
    baseY: -0.486,
    l1: 0.556,
    l2: 0.544,
    l3: 0.544,
    l4: 0.557,
    l5: 0.406,
    leftBranch: -1,
    rightBranch: 1,
  },
  simulator: {
    leftAngleDeg: 15.534233824250094,
    rightAngleDeg: 43.922888743202634,
    distance: 0.5,
    targetAngleDeg: 90,
    branch: -1,
    massKg: 1,
    gravityX: -9.81,
    gravityY: 0,
    preventBranchSwitching: false,
  },
};

const constants = {
  nearCondition: 15,
  distanceDetents: [0.5, 1.0],
  targetAngleDetents: [90],
  gravityDetents: [-9.81, 0, 9.81],
};

const state = {
  leftAngleDeg: config.simulator.leftAngleDeg,
  rightAngleDeg: config.simulator.rightAngleDeg,
  targetDistance: config.simulator.distance,
  targetAngleDeg: config.simulator.targetAngleDeg,
  branch: config.simulator.branch,
  ikLeftBranch: config.linkage.leftBranch,
  ikRightBranch: config.linkage.rightBranch,
  preventBranchSwitching: config.simulator.preventBranchSwitching,
  gravityX: config.simulator.gravityX,
  gravityY: config.simulator.gravityY,
  trace: [],
  mode: "angles",
};

const svg = document.getElementById("plot");
const layer = document.getElementById("plotLayer");
const readout = document.getElementById("readout");
const controls = {
  leftAngle: document.getElementById("leftAngle"),
  rightAngle: document.getElementById("rightAngle"),
  targetDistance: document.getElementById("targetDistance"),
  targetAngle: document.getElementById("targetAngle"),
  gravityX: document.getElementById("gravityX"),
  gravityY: document.getElementById("gravityY"),
  ikBranchPair: document.getElementById("ikBranchPair"),
  preventBranchSwitching: document.getElementById("preventBranchSwitching"),
};
const outputs = {
  leftAngle: document.getElementById("leftAngleValue"),
  rightAngle: document.getElementById("rightAngleValue"),
  targetDistance: document.getElementById("distanceValue"),
  targetAngle: document.getElementById("targetAngleValue"),
  gravityX: document.getElementById("gravityXValue"),
  gravityY: document.getElementById("gravityYValue"),
};

const plot = {
  width: 900,
  height: 650,
  xmin: -1.65,
  xmax: 1.65,
  ymin: -1.0,
  ymax: 1.25,
};

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1]];
}

function mul(a, scalar) {
  return [a[0] * scalar, a[1] * scalar];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

function cross(a, b) {
  return a[0] * b[1] - a[1] * b[0];
}

function norm(a) {
  return Math.hypot(a[0], a[1]);
}

function rot90(a) {
  return [-a[1], a[0]];
}

function normaliseAngleDeg(angle) {
  return ((((angle + 180) % 360) + 360) % 360) - 180;
}

function angleChangeDeg(a, b) {
  return Math.abs(normaliseAngleDeg(a - b));
}

function basePoints() {
  const { d, baseY } = config.linkage;
  return [
    [-d / 2, baseY],
    [d / 2, baseY],
  ];
}

function circleIntersection(c0, r0, c1, r1, branch = 1) {
  const delta = sub(c1, c0);
  const radius = norm(delta);
  if (radius < 1e-12) return null;

  const a = (r0 * r0 - r1 * r1 + radius * radius) / (2 * radius);
  const h2 = r0 * r0 - a * a;
  if (h2 < -1e-9) return null;

  const h = Math.sqrt(Math.max(0, h2));
  const e = mul(delta, 1 / radius);
  const p = add(c0, mul(e, a));
  return add(p, mul(rot90(e), branch * h));
}

function elbowFromInwardAngle(base, length, angleDeg, side) {
  const angleRad = (angleDeg * Math.PI) / 180;
  const absoluteAngle = side === "left" ? angleRad : Math.PI - angleRad;
  return add(base, [length * Math.cos(absoluteAngle), length * Math.sin(absoluteAngle)]);
}

function inwardAngleFromElbow(base, elbow, side) {
  const absolute = (Math.atan2(elbow[1] - base[1], elbow[0] - base[0]) * 180) / Math.PI;
  return side === "left" ? normaliseAngleDeg(absolute) : normaliseAngleDeg(180 - absolute);
}

function pointFromPolar(distance, angleDeg) {
  const angleRad = (angleDeg * Math.PI) / 180;
  return [distance * Math.cos(angleRad), distance * Math.sin(angleRad)];
}

function solveEndpoint(c, dJoint, branch) {
  const { l2, l4 } = config.linkage;
  return circleIntersection(c, l2, dJoint, l4, branch);
}

function toolEndpoint(c, wrist) {
  if (!wrist) return null;
  const { l5 } = config.linkage;
  const direction = sub(wrist, c);
  const length = norm(direction);
  if (length < 1e-12) return null;
  return add(wrist, mul(direction, l5 / length));
}

function wristFromToolEndpoint(c, endpoint) {
  const { l2, l5 } = config.linkage;
  return add(c, mul(sub(endpoint, c), l2 / (l2 + l5)));
}

function poseFromAngles(leftDeg, rightDeg, branch) {
  const [a0, b0] = basePoints();
  const { l1, l3 } = config.linkage;
  const c = elbowFromInwardAngle(a0, l1, leftDeg, "left");
  const dJoint = elbowFromInwardAngle(b0, l3, rightDeg, "right");
  const wrist = solveEndpoint(c, dJoint, branch);
  return { a0, b0, c, dJoint, wrist, endpoint: toolEndpoint(c, wrist) };
}

function closureBranchForPoint(c, dJoint, wrist) {
  const options = [1, -1]
    .map((branch) => {
      const endpoint = solveEndpoint(c, dJoint, branch);
      return endpoint ? { branch, distance: norm(sub(endpoint, wrist)) } : null;
    })
    .filter(Boolean);
  if (!options.length) return null;
  options.sort((a, b) => a.distance - b.distance);
  return options[0].branch;
}

function singularityMetrics(a0, b0, c, dJoint, wrist) {
  if (!wrist) return null;
  const u = sub(wrist, c);
  const v = sub(wrist, dJoint);
  const parallelSin = Math.abs(cross(u, v)) / (norm(u) * norm(v));
  const serialLeft = dot(u, rot90(sub(c, a0)));
  const serialRight = dot(v, rot90(sub(dJoint, b0)));
  const detA = cross(u, v);
  const detB = serialLeft * serialRight;
  let condition = Infinity;

  if (Math.abs(detA) > 1e-10 && Math.abs(detB) > 1e-10) {
    const invA = [
      [v[1] / detA, -u[1] / detA],
      [-v[0] / detA, u[0] / detA],
    ];
    const j = [
      [invA[0][0] * serialLeft, invA[0][1] * serialRight],
      [invA[1][0] * serialLeft, invA[1][1] * serialRight],
    ];
    condition = condition2x2(j);
  }

  return {
    condition,
    parallelSin,
    serialMin: Math.min(Math.abs(serialLeft), Math.abs(serialRight)),
  };
}

function condition2x2(m) {
  const a = m[0][0];
  const b = m[0][1];
  const c = m[1][0];
  const d = m[1][1];
  const s1 = a * a + b * b + c * c + d * d;
  const det = a * d - b * c;
  const inner = Math.max(0, s1 * s1 - 4 * det * det);
  const lambdaMax = (s1 + Math.sqrt(inner)) / 2;
  const lambdaMin = (s1 - Math.sqrt(inner)) / 2;
  if (lambdaMin <= 1e-18) return Infinity;
  return Math.sqrt(lambdaMax / lambdaMin);
}

function branchPairLabel(leftBranch, rightBranch) {
  const left = leftBranch > 0 ? "+1" : "-1";
  const right = rightBranch > 0 ? "+1" : "-1";
  return `(${left}, ${right})`;
}

function parseBranchPair(value) {
  if (value === "auto") return null;
  const [leftBranch, rightBranch] = value.split(",").map(Number);
  return { leftBranch, rightBranch };
}

function activeBranchPairConstraint() {
  const selected = parseBranchPair(controls.ikBranchPair.value);
  if (selected) return selected;
  if (!state.preventBranchSwitching) return null;
  return { leftBranch: state.ikLeftBranch, rightBranch: state.ikRightBranch };
}

function inverseKinematicsSolutions(targetEndpoint, branchPair = null, closureBranchConstraint = null) {
  const [a0, b0] = basePoints();
  const { l1, l2, l3, l4, l5 } = config.linkage;
  const solutions = [];
  const leftBranches = branchPair ? [branchPair.leftBranch] : [1, -1];
  const rightBranches = branchPair ? [branchPair.rightBranch] : [1, -1];

  for (const leftBranch of leftBranches) {
    const c = circleIntersection(a0, l1, targetEndpoint, l2 + l5, leftBranch);
    if (!c) continue;
    const targetWrist = wristFromToolEndpoint(c, targetEndpoint);

    for (const rightBranch of rightBranches) {
      const dJoint = circleIntersection(b0, l3, targetWrist, l4, rightBranch);
      if (!dJoint) continue;
      const closureBranch = closureBranchForPoint(c, dJoint, targetWrist);
      if (closureBranch === null) continue;
      if (closureBranchConstraint !== null && closureBranch !== closureBranchConstraint) continue;
      const wrist = solveEndpoint(c, dJoint, closureBranch);
      const endpoint = toolEndpoint(c, wrist);
      if (!endpoint) continue;
      const leftAngleDeg = inwardAngleFromElbow(a0, c, "left");
      const rightAngleDeg = inwardAngleFromElbow(b0, dJoint, "right");
      const metrics = singularityMetrics(a0, b0, c, dJoint, wrist);
      const change =
        angleChangeDeg(leftAngleDeg, state.leftAngleDeg) +
        angleChangeDeg(rightAngleDeg, state.rightAngleDeg);
      const singularScore = Math.min(
        Number.isFinite(metrics.condition) ? 1 / metrics.condition : 0,
        metrics.parallelSin,
        metrics.serialMin / Math.max(l1, l2, l3, l4),
      );

      solutions.push({
        c,
        dJoint,
        wrist,
        endpoint,
        leftAngleDeg,
        rightAngleDeg,
        leftBranch,
        rightBranch,
        closureBranch,
        metrics,
        change,
        singularScore,
      });
    }
  }
  return solutions;
}

function chooseIkSolution(targetEndpoint) {
  const branchPair = activeBranchPairConstraint();
  const closureBranch = state.preventBranchSwitching ? state.branch : null;
  const solutions = inverseKinematicsSolutions(targetEndpoint, branchPair, closureBranch);
  if (!solutions.length) return null;
  solutions.sort((a, b) => a.change - b.change || b.singularScore - a.singularScore);
  const continuous = solutions[0];
  continuous.selectionReason = branchPair ? "fixed branch pair" : "continuous";
  continuous.switchedForSingularity = false;
  if (branchPair || continuous.metrics.condition < constants.nearCondition) return continuous;
  solutions.sort((a, b) => b.singularScore - a.singularScore || a.change - b.change);
  const safest = solutions[0];
  safest.selectionReason = safest === continuous ? "continuous" : "singularity avoidance";
  safest.switchedForSingularity = safest !== continuous;
  return safest;
}

function inferIkBranchPair(endpoint, leftDeg, rightDeg) {
  const solutions = inverseKinematicsSolutions(endpoint);
  if (!solutions.length) return null;
  solutions.sort(
    (a, b) =>
      angleChangeDeg(a.leftAngleDeg, leftDeg) +
      angleChangeDeg(a.rightAngleDeg, rightDeg) -
      (angleChangeDeg(b.leftAngleDeg, leftDeg) + angleChangeDeg(b.rightAngleDeg, rightDeg)),
  );
  return {
    leftBranch: solutions[0].leftBranch,
    rightBranch: solutions[0].rightBranch,
  };
}

function tipJacobian(leftDeg, rightDeg, branch) {
  const centre = poseFromAngles(leftDeg, rightDeg, branch).endpoint;
  if (!centre) return null;
  const stepRad = 1e-5;
  const stepDeg = (stepRad * 180) / Math.PI;
  const columns = [];

  for (const [leftDelta, rightDelta] of [
    [stepDeg, 0],
    [0, stepDeg],
  ]) {
    const plus = poseFromAngles(leftDeg + leftDelta, rightDeg + rightDelta, branch).endpoint;
    const minus = poseFromAngles(leftDeg - leftDelta, rightDeg - rightDelta, branch).endpoint;
    if (plus && minus) {
      columns.push(mul(sub(plus, minus), 1 / (2 * stepRad)));
    } else if (plus) {
      columns.push(mul(sub(plus, centre), 1 / stepRad));
    } else if (minus) {
      columns.push(mul(sub(centre, minus), 1 / stepRad));
    } else {
      return null;
    }
  }
  return columns;
}

function holdingTorques() {
  const jacobian = tipJacobian(state.leftAngleDeg, state.rightAngleDeg, state.branch);
  if (!jacobian) return null;
  const force = [config.simulator.massKg * state.gravityX, config.simulator.massKg * state.gravityY];
  return [-dot(jacobian[0], force), -dot(jacobian[1], force)];
}

function worldToSvg(point) {
  const x = ((point[0] - plot.xmin) / (plot.xmax - plot.xmin)) * plot.width;
  const y = plot.height - ((point[1] - plot.ymin) / (plot.ymax - plot.ymin)) * plot.height;
  return [x, y];
}

function pathFromWorld(points) {
  return points
    .map((point, index) => {
      const [x, y] = worldToSvg(point);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function svgElement(name, attrs = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value);
  }
  return element;
}

function drawPolyline(points, className) {
  const path = svgElement("path", { d: pathFromWorld(points), class: className });
  layer.append(path);
}

function drawCircle(point, radius, attrs) {
  const [cx, cy] = worldToSvg(point);
  layer.append(svgElement("circle", { cx, cy, r: radius, ...attrs }));
}

function drawTargetCross(point) {
  const [x, y] = worldToSvg(point);
  layer.append(svgElement("path", { d: `M ${x - 8} ${y - 8} L ${x + 8} ${y + 8}`, class: "target" }));
  layer.append(svgElement("path", { d: `M ${x + 8} ${y - 8} L ${x - 8} ${y + 8}`, class: "target" }));
}

function drawGuides() {
  const xAxis = [worldToSvg([plot.xmin, 0]), worldToSvg([plot.xmax, 0])];
  const yAxis = [worldToSvg([0, plot.ymin]), worldToSvg([0, plot.ymax])];
  layer.append(svgElement("path", { d: `M ${xAxis[0][0]} ${xAxis[0][1]} L ${xAxis[1][0]} ${xAxis[1][1]}`, class: "axis-line" }));
  layer.append(svgElement("path", { d: `M ${yAxis[0][0]} ${yAxis[0][1]} L ${yAxis[1][0]} ${yAxis[1][1]}`, class: "axis-line" }));

  const arc = [];
  for (let i = 0; i <= 160; i += 1) {
    const theta = (Math.PI * i) / 160;
    arc.push([Math.cos(theta), Math.sin(theta)]);
  }
  drawPolyline(arc, "guide-line");
  drawPolyline([[-1, 0], [1, 0]], "guide-line");
}

function render() {
  syncControls();
  const pose = poseFromAngles(state.leftAngleDeg, state.rightAngleDeg, state.branch);
  const target = pointFromPolar(state.targetDistance, state.targetAngleDeg);
  layer.replaceChildren();
  drawGuides();

  if (state.trace.length > 1) {
    drawPolyline(state.trace, "trace");
  }

  drawPolyline([pose.a0, pose.b0], "base");
  if (!pose.wrist || !pose.endpoint) {
    drawPolyline([pose.a0, pose.c], "left-link");
    drawPolyline([pose.b0, pose.dJoint], "right-link");
    drawTargetCross(target);
    updateReadout(null, target);
    return;
  }

  drawPolyline([pose.a0, pose.c, pose.wrist, pose.endpoint], "left-link");
  drawPolyline([pose.b0, pose.dJoint, pose.wrist], "right-link");
  drawCircle(pose.a0, 5, { fill: "#20242a" });
  drawCircle(pose.b0, 5, { fill: "#20242a" });
  drawCircle(pose.c, 7, { fill: "var(--blue)", class: "node" });
  drawCircle(pose.dJoint, 7, { fill: "var(--orange)", class: "node" });
  drawCircle(pose.wrist, 5, { fill: "#2f343b", opacity: "0.78" });
  drawCircle(pose.endpoint, 9, { fill: "var(--red)", class: "node" });
  drawTargetCross(target);

  if (state.mode !== "load") {
    state.trace.push(pose.endpoint);
    if (state.trace.length > 600) state.trace.shift();
  }

  updateReadout(pose, target);
}

function metric(label, value, className = "") {
  return `<div class="metric ${className}"><dt>${label}</dt><dd>${value}</dd></div>`;
}

function updateReadout(pose, target) {
  if (!pose || !pose.endpoint) {
    readout.innerHTML = metric("Status", "No closed five-bar pose", "unreachable");
    return;
  }
  const metrics = singularityMetrics(pose.a0, pose.b0, pose.c, pose.dJoint, pose.wrist);
  const torque = holdingTorques();
  const endpoint = pose.endpoint;
  const polarDistance = norm(endpoint);
  let polarAngle = (Math.atan2(endpoint[1], endpoint[0]) * 180) / Math.PI;
  if (polarAngle < 0) polarAngle += 360;

  readout.innerHTML = [
    metric("Endpoint E", `${fmt(endpoint[0])}, ${fmt(endpoint[1])} m`),
    metric("Wrist", `${fmt(pose.wrist[0])}, ${fmt(pose.wrist[1])} m`),
    metric("Polar", `r ${fmt(polarDistance)}, angle ${fmt(polarAngle, 1)}°`),
    metric("Condition", Number.isFinite(metrics.condition) ? fmt(metrics.condition, 2) : "∞"),
    metric("Parallel sin", fmt(metrics.parallelSin, 3)),
    metric("Serial margin", `${fmt(metrics.serialMin, 3)} m²`),
    metric(
      "IK branch pair",
      `${branchPairLabel(state.ikLeftBranch, state.ikRightBranch)}<br />${
        state.preventBranchSwitching ? "switching prevented" : "switching allowed"
      }`,
    ),
    metric("Target", `${fmt(target[0])}, ${fmt(target[1])} m`),
    metric("Tip force", `${fmt(config.simulator.massKg * state.gravityX, 2)}, ${fmt(config.simulator.massKg * state.gravityY, 2)} N`),
    metric(
      "Hold torque (+in)",
      torque ? `L ${fmt(torque[0], 2)} Nm<br />R ${fmt(torque[1], 2)} Nm` : "unavailable",
    ),
  ].join("");
}

function fmt(value, digits = 3) {
  if (Object.is(value, -0)) value = 0;
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

function snap(value, detents, tolerance) {
  let best = value;
  let distance = Infinity;
  for (const detent of detents) {
    const diff = Math.abs(value - detent);
    if (diff < distance) {
      distance = diff;
      best = detent;
    }
  }
  return distance <= tolerance ? best : value;
}

function syncControls() {
  controls.leftAngle.value = state.leftAngleDeg;
  controls.rightAngle.value = state.rightAngleDeg;
  controls.targetDistance.value = state.targetDistance;
  controls.targetAngle.value = state.targetAngleDeg;
  controls.gravityX.value = state.gravityX;
  controls.gravityY.value = state.gravityY;
  controls.preventBranchSwitching.checked = state.preventBranchSwitching;

  outputs.leftAngle.textContent = fmt(state.leftAngleDeg, 1);
  outputs.rightAngle.textContent = fmt(state.rightAngleDeg, 1);
  outputs.targetDistance.textContent = fmt(state.targetDistance, 3);
  outputs.targetAngle.textContent = fmt(state.targetAngleDeg, 1);
  outputs.gravityX.textContent = fmt(state.gravityX, 2);
  outputs.gravityY.textContent = fmt(state.gravityY, 2);

  document.querySelector(`input[name="branch"][value="${state.branch}"]`).checked = true;
}

function updateFromAngles() {
  state.mode = "angles";
  state.leftAngleDeg = Number(controls.leftAngle.value);
  state.rightAngleDeg = Number(controls.rightAngle.value);
  const pose = poseFromAngles(state.leftAngleDeg, state.rightAngleDeg, state.branch);
  if (pose.endpoint) {
    const branchPair = inferIkBranchPair(pose.endpoint, state.leftAngleDeg, state.rightAngleDeg);
    if (branchPair) {
      state.ikLeftBranch = branchPair.leftBranch;
      state.ikRightBranch = branchPair.rightBranch;
    }
    state.targetDistance = norm(pose.endpoint);
    let angle = (Math.atan2(pose.endpoint[1], pose.endpoint[0]) * 180) / Math.PI;
    if (angle < 0) angle += 360;
    state.targetAngleDeg = Math.max(0, Math.min(180, angle));
  }
  render();
}

function updateFromTarget() {
  state.mode = "target";
  state.targetDistance = snap(Number(controls.targetDistance.value), constants.distanceDetents, 0.008);
  state.targetAngleDeg = snap(Number(controls.targetAngle.value), constants.targetAngleDetents, 0.5);
  const solution = chooseIkSolution(pointFromPolar(state.targetDistance, state.targetAngleDeg));
  if (solution) {
    state.leftAngleDeg = solution.leftAngleDeg;
    state.rightAngleDeg = solution.rightAngleDeg;
    state.branch = solution.closureBranch;
    state.ikLeftBranch = solution.leftBranch;
    state.ikRightBranch = solution.rightBranch;
  }
  render();
}

function updateFromLoad() {
  state.mode = "load";
  state.gravityX = snap(Number(controls.gravityX.value), constants.gravityDetents, 0.08);
  state.gravityY = snap(Number(controls.gravityY.value), constants.gravityDetents, 0.08);
  render();
}

function setup() {
  const { linkage } = config;
  document.getElementById("configSummary").textContent =
    `d ${linkage.d} m; L = ${linkage.l1}, ${linkage.l2}, ${linkage.l3}, ${linkage.l4}, ${linkage.l5} m`;

  controls.leftAngle.value = state.leftAngleDeg;
  controls.rightAngle.value = state.rightAngleDeg;
  controls.targetDistance.value = state.targetDistance;
  controls.targetAngle.value = state.targetAngleDeg;
  controls.gravityX.value = state.gravityX;
  controls.gravityY.value = state.gravityY;
  controls.ikBranchPair.value = "auto";
  controls.preventBranchSwitching.checked = state.preventBranchSwitching;

  controls.leftAngle.addEventListener("input", updateFromAngles);
  controls.rightAngle.addEventListener("input", updateFromAngles);
  controls.targetDistance.addEventListener("input", updateFromTarget);
  controls.targetAngle.addEventListener("input", updateFromTarget);
  controls.gravityX.addEventListener("input", updateFromLoad);
  controls.gravityY.addEventListener("input", updateFromLoad);
  controls.ikBranchPair.addEventListener("change", updateFromTarget);
  controls.preventBranchSwitching.addEventListener("change", () => {
    state.preventBranchSwitching = controls.preventBranchSwitching.checked;
    updateFromTarget();
  });
  document.querySelectorAll('input[name="branch"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.mode = "angles";
      state.branch = Number(radio.value);
      const pose = poseFromAngles(state.leftAngleDeg, state.rightAngleDeg, state.branch);
      if (pose.endpoint) {
        const branchPair = inferIkBranchPair(pose.endpoint, state.leftAngleDeg, state.rightAngleDeg);
        if (branchPair) {
          state.ikLeftBranch = branchPair.leftBranch;
          state.ikRightBranch = branchPair.rightBranch;
        }
      }
      render();
    });
  });
  document.getElementById("clearTrace").addEventListener("click", () => {
    state.trace = [];
    state.mode = "load";
    render();
  });
  render();
}

setup();
