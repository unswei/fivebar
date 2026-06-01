const config = {
  d: 0.1917,
  baseY: -0.3745,
  l1: 0.5829,
  l2: 0.4632,
  l3: 0.3308,
  l4: 0.736,
  l5: 0.379,
  leftBranch: 1,
  rightBranch: -1,
  targetRadius: 1.0,
};

const view = {
  xmin: -1.35,
  xmax: 1.35,
  ymin: -1.1,
  ymax: 1.25,
  nx: 150,
  ny: 132,
};

const workspaceCanvas = document.getElementById("workspaceCanvas");
const conditionCanvas = document.getElementById("conditionCanvas");
const branchSelect = document.getElementById("branchPair");
const summary = document.getElementById("summary");

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

function basePoints() {
  return [
    [-config.d / 2, config.baseY],
    [config.d / 2, config.baseY],
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
  return add(add(c0, mul(e, a)), mul(rot90(e), branch * h));
}

function wristFromToolEndpoint(c, endpoint) {
  return add(c, mul(sub(endpoint, c), config.l2 / (config.l2 + config.l5)));
}

function toolEndpoint(c, wrist) {
  const direction = sub(wrist, c);
  const length = norm(direction);
  if (length < 1e-12) return null;
  return add(wrist, mul(direction, config.l5 / length));
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

function fivebarToolMetrics(endpoint, leftBranch, rightBranch) {
  const [a0, b0] = basePoints();
  const c = circleIntersection(a0, config.l1, endpoint, config.l2 + config.l5, leftBranch);
  if (!c) return null;

  const wrist = wristFromToolEndpoint(c, endpoint);
  const dJoint = circleIntersection(b0, config.l3, wrist, config.l4, rightBranch);
  if (!dJoint) return null;

  const u = sub(wrist, c);
  const v = sub(wrist, dJoint);
  const detA = cross(u, v);
  const b1 = dot(u, rot90(sub(c, a0)));
  const b2 = dot(v, rot90(sub(dJoint, b0)));
  const detB = b1 * b2;
  const parallelSin = Math.abs(detA) / (norm(u) * norm(v));
  let condition = Infinity;

  if (Math.abs(detA) > 1e-10 && Math.abs(detB) > 1e-10) {
    const invA = [
      [v[1] / detA, -u[1] / detA],
      [-v[0] / detA, u[0] / detA],
    ];
    condition = condition2x2([
      [invA[0][0] * b1, invA[0][1] * b2],
      [invA[1][0] * b1, invA[1][1] * b2],
    ]);
  }

  return {
    c,
    dJoint,
    wrist,
    condition,
    parallelSin,
    serialMin: Math.min(Math.abs(b1), Math.abs(b2)),
  };
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * ratio));
  canvas.height = Math.max(1, Math.round(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function worldToCanvas(point, width, height) {
  return [
    ((point[0] - view.xmin) / (view.xmax - view.xmin)) * width,
    height - ((point[1] - view.ymin) / (view.ymax - view.ymin)) * height,
  ];
}

function drawAxes(ctx, width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  const x0 = worldToCanvas([view.xmin, 0], width, height);
  const x1 = worldToCanvas([view.xmax, 0], width, height);
  const y0 = worldToCanvas([0, view.ymin], width, height);
  const y1 = worldToCanvas([0, view.ymax], width, height);
  ctx.beginPath();
  ctx.moveTo(x0[0], x0[1]);
  ctx.lineTo(x1[0], x1[1]);
  ctx.moveTo(y0[0], y0[1]);
  ctx.lineTo(y1[0], y1[1]);
  ctx.stroke();
  ctx.restore();
}

function drawGrid(ctx, width, height) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e8edf3";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = Math.ceil(view.xmin * 5) / 5; x <= view.xmax; x += 0.2) {
    const p = worldToCanvas([x, 0], width, height);
    ctx.moveTo(p[0], 0);
    ctx.lineTo(p[0], height);
  }
  for (let y = Math.ceil(view.ymin * 5) / 5; y <= view.ymax; y += 0.2) {
    const p = worldToCanvas([0, y], width, height);
    ctx.moveTo(0, p[1]);
    ctx.lineTo(width, p[1]);
  }
  ctx.stroke();
}

function drawWorldPolyline(ctx, points, width, height, style) {
  let open = false;
  ctx.save();
  Object.assign(ctx, style);
  if (style.dash) ctx.setLineDash(style.dash);
  ctx.beginPath();
  for (const point of points) {
    if (!point) {
      open = false;
      continue;
    }
    const [x, y] = worldToCanvas(point, width, height);
    if (!open) {
      ctx.moveTo(x, y);
      open = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function circlePoints(centre, radius, steps = 360) {
  return Array.from({ length: steps + 1 }, (_, index) => {
    const theta = (2 * Math.PI * index) / steps;
    return [centre[0] + radius * Math.cos(theta), centre[1] + radius * Math.sin(theta)];
  });
}

function drawTargetCircle(ctx, width, height) {
  drawWorldPolyline(ctx, circlePoints([0, 0], config.targetRadius), width, height, {
    strokeStyle: "#111111",
    lineWidth: 1.4,
    dash: [6, 5],
  });
}

function drawSerialCurves(ctx, width, height, leftBranch) {
  const [a0, b0] = basePoints();
  const leftEffective = config.l2 + config.l5;
  for (const radius of [config.l1 + leftEffective, Math.abs(config.l1 - leftEffective)]) {
    drawWorldPolyline(ctx, circlePoints(a0, radius), width, height, {
      strokeStyle: "#2ca02c",
      lineWidth: 1.3,
      dash: radius > 0.01 ? [7, 5] : [],
    });
  }

  for (const radius of [config.l3 + config.l4, Math.abs(config.l3 - config.l4)]) {
    const points = [];
    for (let index = 0; index <= 420; index += 1) {
      const theta = (2 * Math.PI * index) / 420;
      const wrist = add(b0, [radius * Math.cos(theta), radius * Math.sin(theta)]);
      const c = circleIntersection(a0, config.l1, wrist, config.l2, leftBranch);
      const endpoint = c ? toolEndpoint(c, wrist) : null;
      points.push(endpoint);
    }
    drawWorldPolyline(ctx, width ? points : [], width, height, {
      strokeStyle: "#1f77b4",
      lineWidth: 1.3,
      dash: radius > 0.01 ? [7, 5] : [],
    });
  }
}

function colourForCondition(condition) {
  const capped = Math.max(0, Math.min(4, Math.log10(condition)));
  const t = capped / 4;
  const hue = 210 - 190 * t;
  const light = 92 - 42 * t;
  return `hsl(${hue}, 72%, ${light}%)`;
}

function insideUpperSemicircle(point, radius) {
  return point[1] >= 0 && point[0] * point[0] + point[1] * point[1] <= radius * radius;
}

function render() {
  const [leftBranch, rightBranch] = branchSelect.value.split(",").map(Number);
  const workspace = prepareCanvas(workspaceCanvas);
  const condition = prepareCanvas(conditionCanvas);
  drawGrid(workspace.ctx, workspace.width, workspace.height);
  drawGrid(condition.ctx, condition.width, condition.height);

  const cellW = workspace.width / view.nx;
  const cellH = workspace.height / view.ny;
  let reachable = 0;
  let minParallel = Infinity;
  let minSerial = Infinity;
  let maxCondition = 0;
  let semicircleSamples = 0;
  let semicircleReachable = 0;
  let semicircleMaxCondition = 0;

  for (let iy = 0; iy < view.ny; iy += 1) {
    for (let ix = 0; ix < view.nx; ix += 1) {
      const x = view.xmin + ((ix + 0.5) / view.nx) * (view.xmax - view.xmin);
      const y = view.ymin + ((iy + 0.5) / view.ny) * (view.ymax - view.ymin);
      const inSemicircle = insideUpperSemicircle([x, y], config.targetRadius);
      const metrics = fivebarToolMetrics([x, y], leftBranch, rightBranch);
      const px = ix * cellW;
      const py = condition.height - (iy + 1) * cellH;
      if (inSemicircle) semicircleSamples += 1;

      if (!metrics) {
        workspace.ctx.fillStyle = "rgba(242, 244, 247, 0.78)";
        condition.ctx.fillStyle = "rgba(242, 244, 247, 0.78)";
      } else {
        reachable += 1;
        minParallel = Math.min(minParallel, metrics.parallelSin);
        minSerial = Math.min(minSerial, metrics.serialMin);
        maxCondition = Math.max(maxCondition, metrics.condition);
        if (inSemicircle) {
          semicircleReachable += 1;
          semicircleMaxCondition = Math.max(semicircleMaxCondition, metrics.condition);
        }
        workspace.ctx.fillStyle = "rgba(31, 119, 180, 0.18)";
        condition.ctx.fillStyle = colourForCondition(metrics.condition);
      }
      workspace.ctx.fillRect(px, py, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
      condition.ctx.fillRect(px, py, Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
  }

  for (const target of [workspace, condition]) {
    drawAxes(target.ctx, target.width, target.height);
    drawTargetCircle(target.ctx, target.width, target.height);
    const [a0, b0] = basePoints();
    for (const point of [a0, b0]) {
      const [x, y] = worldToCanvas(point, target.width, target.height);
      target.ctx.fillStyle = "#111111";
      target.ctx.beginPath();
      target.ctx.arc(x, y, 4, 0, 2 * Math.PI);
      target.ctx.fill();
    }
  }
  drawSerialCurves(workspace.ctx, workspace.width, workspace.height, leftBranch);

  const total = view.nx * view.ny;
  summary.innerHTML = [
    metric("Reachable samples", `${reachable} / ${total} (${format((100 * reachable) / total, 1)}%)`),
    metric(
      `${format(config.targetRadius)} m upper semicircle reachable`,
      `${semicircleReachable} / ${semicircleSamples} (${format((100 * semicircleReachable) / semicircleSamples, 1)}%)`,
    ),
    metric(
      `${format(config.targetRadius)} m upper semicircle max condition`,
      semicircleReachable ? format(semicircleMaxCondition, 2) : "n/a",
    ),
    metric("Min parallel sin", reachable ? format(minParallel, 4) : "n/a"),
    metric("Min serial margin", reachable ? `${format(minSerial, 4)} m²` : "n/a"),
    metric("Max condition", reachable ? format(maxCondition, 2) : "n/a"),
    metric("Branch pair", `(${leftBranch > 0 ? "+" : ""}${leftBranch}, ${rightBranch > 0 ? "+" : ""}${rightBranch})`),
  ].join("");
}

function metric(label, value) {
  return `<div class="metric"><dt>${label}</dt><dd>${value}</dd></div>`;
}

function format(value, digits = 3) {
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

function setup() {
  document.getElementById("configSummary").textContent =
    `d ${config.d} m; L = ${config.l1}, ${config.l2}, ${config.l3}, ${config.l4}, ${config.l5} m`;
  branchSelect.value = `${config.leftBranch},${config.rightBranch}`;
  branchSelect.addEventListener("change", render);
  window.addEventListener("resize", render);
  render();
}

setup();
