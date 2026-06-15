(function () {
  const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";
  const LABELS = {
    0: "reaches xy=1 without crossing",
    1: "reaches xy=1 after crossing",
    2: "to saddle (0, 0)",
    3: "blow-up",
    4: "unresolved within budget",
  };
  const NONCONV_COLORS = {
    2: "#4c78a8",
    3: "#7b3294",
    4: "#bab0ac",
  };

  const plotConfig = { displayModeBar: false, responsive: true };
  let plotlyPromise = null;

  function loadPlotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (plotlyPromise) return plotlyPromise;

    plotlyPromise = new Promise((resolve, reject) => {
      const found = document.querySelector(`script[src="${PLOTLY_URL}"]`);
      if (found) {
        found.addEventListener("load", () => resolve(window.Plotly), { once: true });
        found.addEventListener("error", reject, { once: true });
        return;
      }

      const script = document.createElement("script");
      script.src = PLOTLY_URL;
      script.onload = () => resolve(window.Plotly);
      script.onerror = reject;
      document.head.appendChild(script);
    });

    return plotlyPromise;
  }

  function linspace(a, b, n) {
    if (n === 1) return [a];
    const h = (b - a) / (n - 1);
    return Array.from({ length: n }, (_, i) => a + i * h);
  }

  function median(values) {
    if (!values.length) return null;
    const s = values.slice().sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
  }

  function snap(value, grid) {
    const h = (grid.max - grid.min) / (grid.n - 1);
    const i = Math.max(0, Math.min(grid.n - 1, Math.round((value - grid.min) / h)));
    return grid.min + i * h;
  }

  function state(x, y) {
    const residual = x * y - 1;
    return {
      residual,
      imbalance: x * x - y * y,
    };
  }

  function sharpness(imbalance, residual) {
    return Math.sqrt(imbalance * imbalance + 4 * (residual + 1) * (residual + 1));
  }

  function finalSharpness(imbalance) {
    return Math.sqrt(imbalance * imbalance + 4);
  }

  function gdRun(x0, y0, eta, steps) {
    let x = x0;
    let y = y0;
    let current = state(x, y);
    let previousResidual = current.residual;
    let signFlips = 0;
    let stopStep = 0;
    let label = null;
    let residualHits = Math.abs(current.residual) < 1e-7 ? 1 : 0;
    let originHits = Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6 && Math.abs(current.residual + 1) < 1e-6 ? 1 : 0;

    const xs = [x];
    const ys = [y];
    const residuals = [current.residual];
    const s = [sharpness(current.imbalance, current.residual)];
    const etaS = [eta * s[0]];
    const t = [0];

    if (originHits >= 5) label = 2;
    if (residualHits >= 5) label = 0;

    for (let k = 1; k <= steps && label === null; k += 1) {
      const r = x * y - 1;
      const nextX = x - eta * r * y;
      const nextY = y - eta * r * x;
      x = nextX;
      y = nextY;
      current = state(x, y);

      xs.push(x);
      ys.push(y);
      residuals.push(current.residual);
      s.push(sharpness(current.imbalance, current.residual));
      etaS.push(eta * s[s.length - 1]);
      t.push(k);

      if (previousResidual * current.residual < 0) signFlips += 1;
      previousResidual = current.residual;

      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e8 || Math.abs(y) > 1e8) {
        label = 3;
        stopStep = k;
        break;
      }

      residualHits = Math.abs(current.residual) < 1e-7 ? residualHits + 1 : 0;
      originHits = Math.abs(x) < 1e-6 && Math.abs(y) < 1e-6 && Math.abs(current.residual + 1) < 1e-6 ? originHits + 1 : 0;

      if (originHits >= 5) {
        label = 2;
        stopStep = k;
      } else if (residualHits >= 5) {
        label = signFlips ? 1 : 0;
        stopStep = k;
      }
    }

    if (label === null) {
      label = 4;
      stopStep = steps;
    }

    const initialResidual = Math.max(Math.abs(residuals[0]), 1e-12);
    const tailStart = residuals.findIndex((r) => Math.abs(r) <= 0.01 * initialResidual);
    const tail = tailStart < 0 ? [] : etaS.slice(tailStart);
    const tailGap = tail.map((v) => Math.abs(v - 2));

    return {
      label,
      stopStep,
      signFlips,
      xs,
      ys,
      t,
      s,
      etaS,
      tailStart: tailStart < 0 ? null : tailStart,
      tailMedianEtaS: median(tail),
      nearEdge: median(tailGap) !== null && median(tailGap) <= 0.1,
      finalSharpness: finalSharpness(state(x, y).imbalance),
    };
  }

  function decodeGrid(values, n, transform) {
    const rows = [];
    for (let i = 0; i < n; i += 1) {
      rows.push(Array.from(values.slice(i * n, (i + 1) * n), transform));
    }
    return rows;
  }

  function hyperbola(a, b) {
    const pos = b > 1 / 3 ? linspace(Math.max(a, 1 / 3), b, 240) : [];
    const neg = a < -1 / 3 ? linspace(a, Math.min(b, -1 / 3), 240) : [];
    return [
      line(pos, pos.map((x) => 1 / x), "#111111", 2.2),
      line(neg, neg.map((x) => 1 / x), "#111111", 2.2),
    ];
  }

  function imbalanceCurves(c, a, b, style) {
    const traces = [];
    const grid = linspace(a, b, 360);
    const add = (x, y) => {
      if (x.length > 1) traces.push(line(x, y, style.color, style.width, style.dash, style.opacity));
    };

    if (Math.abs(c) < 1e-12) {
      add(grid, grid);
      add(grid, grid.map((x) => -x));
      return traces;
    }

    if (c > 0) {
      const xp = [];
      const yp = [];
      const xn = [];
      const yn = [];
      grid.forEach((y) => {
        const x = Math.sqrt(y * y + c);
        if (x >= a && x <= b) {
          xp.push(x);
          yp.push(y);
        }
        if (-x >= a && -x <= b) {
          xn.push(-x);
          yn.push(y);
        }
      });
      add(xp, yp);
      add(xn, yn);
      return traces;
    }

    const shift = -c;
    const xt = [];
    const yt = [];
    const xb = [];
    const yb = [];
    grid.forEach((x) => {
      const y = Math.sqrt(x * x + shift);
      if (y >= a && y <= b) {
        xt.push(x);
        yt.push(y);
      }
      if (-y >= a && -y <= b) {
        xb.push(x);
        yb.push(-y);
      }
    });
    add(xt, yt);
    add(xb, yb);
    return traces;
  }

  function line(x, y, color, width, dash = "solid", opacity = 1) {
    return {
      type: "scatter",
      mode: "lines",
      x,
      y,
      line: { color, width, dash },
      opacity,
      hoverinfo: "skip",
      showlegend: false,
    };
  }

  function frame(run, min, max) {
    const cap = 2 * (max - min);
    const points = run.xs
      .map((x, i) => [x, run.ys[i]])
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
      .filter(([x, y]) => run.label !== 3 || Math.max(Math.abs(x), Math.abs(y)) <= cap);

    if (!points.length) {
      return { min, max, x: [min, max], y: [min, max] };
    }

    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    const half = 0.58 * Math.max(x1 - x0, y1 - y0, 0.25 * (max - min));
    const cx = 0.5 * (x0 + x1);
    const cy = 0.5 * (y0 + y1);
    const xr = [cx - half, cx + half];
    const yr = [cy - half, cy + half];

    return {
      min: Math.min(xr[0], yr[0]),
      max: Math.max(xr[1], yr[1]),
      x: xr,
      y: yr,
    };
  }

  function fmt(x) {
    return Number.isFinite(x) ? x.toFixed(3) : "n/a";
  }

  function outcome(run) {
    if (run.label === 0) return `reaches xy=1 after ${run.stopStep} steps, no crossing`;
    if (run.label === 1) return `reaches xy=1 after ${run.stopStep} steps, with crossing`;
    if (run.label === 3) return `blow-up detected by step ${run.stopStep}`;
    if (run.label === 4) return `not settled within ${run.stopStep} steps`;
    return `${LABELS[run.label]} after ${run.stopStep} steps`;
  }

  async function init(root) {
    root.innerHTML = '<div class="phase-explorer-loading">loading interactive explorer...</div>';

    const [Plotly, response] = await Promise.all([
      loadPlotly(),
      fetch(root.dataset.src),
    ]);
    if (!response.ok) throw new Error(`could not load ${root.dataset.src}`);

    const data = await response.json();
    const etas = data.etas.map(Number);
    const grid = {
      min: Number(data.grid_min),
      max: Number(data.grid_max),
      n: Number(data.grid_size),
    };
    const values = linspace(grid.min, grid.max, grid.n);
    const total = grid.n * grid.n;
    const colorMax = Math.max(
      2,
      ...Object.values(data.sharpness_milli).flat().filter((v) => v >= 0).map((v) => 2 + v / 1000)
    );
    const cache = {};
    const key = (i) => etas[i].toFixed(2);
    const labels = (i) => cache[`labels-${key(i)}`] ||= data.labels[key(i)];
    const sharpnessGrid = (i) => cache[`sharp-${key(i)}`] ||= decodeGrid(
      data.sharpness_milli[key(i)],
      grid.n,
      (v) => v < 0 ? null : 2 + v / 1000
    );
    const nonconverged = (i, label) => {
      const cacheKey = `points-${key(i)}-${label}`;
      if (cache[cacheKey]) return cache[cacheKey];
      const xs = [];
      const ys = [];
      const lab = labels(i);
      for (let k = 0; k < lab.length; k += 1) {
        if (Number(lab[k]) === label) {
          xs.push(values[k % grid.n]);
          ys.push(values[Math.floor(k / grid.n)]);
        }
      }
      cache[cacheKey] = { xs, ys };
      return cache[cacheKey];
    };

    root.innerHTML = `
      <div class="phase-explorer-controls">
        <div class="phase-explorer-range-row">
          <label>learning rate</label>
          <input class="phase-explorer-slider" type="range" min="0" max="${etas.length - 1}" step="1">
          <div class="phase-explorer-readout"></div>
        </div>
        <div class="phase-explorer-ticks">${etas.map((eta) => `<span>${eta.toFixed(2)}</span>`).join("")}</div>
      </div>
      <div class="phase-explorer-panels">
        <div class="phase-explorer-plot"></div>
        <div class="phase-explorer-plot"></div>
      </div>
      <div class="phase-explorer-legend">
        <span class="phase-explorer-legend-item"><span class="phase-explorer-swatch" style="background:linear-gradient(90deg, #440154, #21908d, #fde725)"></span><span>color: final sharpness when GD converges</span></span>
        ${[2, 3, 4].map((label) => `<span class="phase-explorer-legend-item"><span class="phase-explorer-swatch" style="background:${NONCONV_COLORS[label]}"></span><span>${LABELS[label]}</span></span>`).join("")}
      </div>
      <p class="phase-explorer-note">Left: starts colored by final sharpness. White curves have the same value of $x^2-y^2$. Right: GD path and sharpness over time.</p>
      <p class="phase-explorer-status"></p>
    `;

    const textColor = getComputedStyle(document.body).color;
    const plots = root.querySelectorAll(".phase-explorer-plot");
    const slider = root.querySelector(".phase-explorer-slider");
    const readout = root.querySelector(".phase-explorer-readout");
    const status = root.querySelector(".phase-explorer-status");
    const tickEls = root.querySelectorAll(".phase-explorer-ticks span");
    let etaIndex = Math.max(0, etas.findIndex((eta) => Math.abs(eta - Number(root.dataset.initialEta || 0.75)) < 1e-9));
    let selected = {
      x: snap(Number(root.dataset.initialX || 1.2), grid),
      y: snap(Number(root.dataset.initialY || 0.8), grid),
    };

    const baseLayout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: textColor },
    };
    const axisStyle = {
      gridcolor: "rgba(128,128,128,0.20)",
      zerolinecolor: "rgba(128,128,128,0.45)",
    };

    function render() {
      const eta = etas[etaIndex];
      const run = gdRun(selected.x, selected.y, eta, data.steps);
      const counts = data.counts[key(etaIndex)];
      const startImbalance = selected.x * selected.x - selected.y * selected.y;
      const unsettled = (100 * (counts["4"] || 0) / total).toFixed(2);

      readout.textContent = `eta = ${eta.toFixed(2)} | grid = ${grid.n}x${grid.n} | steps = ${data.steps}`;
      tickEls.forEach((el, i) => el.classList.toggle("current", i === etaIndex));
      status.textContent = [
        `start = (${fmt(selected.x)}, ${fmt(selected.y)})`,
        `x^2-y^2 = ${fmt(startImbalance)}`,
        `outcome = ${outcome(run)}`,
        `final sharpness = ${run.label <= 1 ? fmt(run.finalSharpness) : "n/a"}`,
        run.tailStart === null ? "late statistic unavailable" : `late part starts at step ${run.tailStart} | median late eta*sharpness = ${fmt(run.tailMedianEtaS)} | near edge = ${run.nearEdge ? "yes" : "no"}`,
        `unsettled on this eta grid = ${unsettled}%`,
      ].join(" | ");

      Plotly.react(plots[0], [
        {
          type: "heatmap",
          x: values,
          y: values,
          z: sharpnessGrid(etaIndex),
          zmin: 2,
          zmax: colorMax,
          colorscale: "Viridis",
          colorbar: { title: { text: "final sharpness" }, thickness: 12 },
          hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>sharpness=%{z:.3f}<extra></extra>",
          zsmooth: false,
        },
        ...data.imbalance_levels.flatMap((c) => imbalanceCurves(c, grid.min, grid.max, { color: "rgba(255,255,255,0.65)", width: 1.1 })),
        ...[2, 3, 4].map((label) => {
          const pts = nonconverged(etaIndex, label);
          return {
            type: "scattergl",
            mode: "markers",
            x: pts.xs,
            y: pts.ys,
            marker: { color: NONCONV_COLORS[label], size: 4, symbol: "square", opacity: label === 4 ? 0.75 : 0.95 },
            hovertemplate: `x=%{x:.2f}<br>y=%{y:.2f}<br>${LABELS[label]}<extra></extra>`,
            showlegend: false,
          };
        }),
        {
          type: "scatter",
          mode: "markers",
          x: [selected.x],
          y: [selected.y],
          marker: { color: "#ffffff", size: 11, line: { color: "#111111", width: 2 } },
          hoverinfo: "skip",
          showlegend: false,
        },
      ], {
        ...baseLayout,
        margin: { l: 48, r: 12, t: 38, b: 44 },
        title: { text: "final sharpness from each start", x: 0.5, xanchor: "center", font: { size: 16 } },
        xaxis: { title: "x0", range: [grid.min, grid.max], scaleanchor: "y", scaleratio: 1, ...axisStyle },
        yaxis: { title: "y0", range: [grid.min, grid.max], ...axisStyle },
        uirevision: "post5-map",
      }, plotConfig);

      const fr = frame(run, grid.min, grid.max);
      const pathTraces = [
        ...hyperbola(fr.min, fr.max),
        ...imbalanceCurves(startImbalance, fr.min, fr.max, { color: "#8ecae6", width: 1.8, dash: "dot" }),
        {
          type: "scatter",
          mode: "lines+markers",
          x: run.xs,
          y: run.ys,
          line: { color: "#f94144", width: 2.6 },
          marker: { color: "#f94144", size: 4 },
          hoverinfo: "skip",
          showlegend: false,
        },
        {
          type: "scatter",
          mode: "markers",
          x: [run.xs[0], run.xs[run.xs.length - 1]],
          y: [run.ys[0], run.ys[run.ys.length - 1]],
          marker: { color: ["#ffffff", "#f94144"], size: [10, 8], line: { color: "#111111", width: 2 } },
          hoverinfo: "skip",
          showlegend: false,
        },
      ];
      const sharpnessTraces = [
        line(run.t, run.s, "#4c78a8", 2.4),
        line(run.t, run.t.map(() => 2), "#111111", 1.6, "dot"),
        line(run.t, run.t.map(() => 2 / eta), "#f4a261", 1.6, "dash"),
      ].map((trace) => ({ ...trace, xaxis: "x2", yaxis: "y2" }));

      Plotly.react(plots[1], [...pathTraces, ...sharpnessTraces], {
        ...baseLayout,
        margin: { l: 56, r: 24, t: 52, b: 46 },
        showlegend: false,
        xaxis: { domain: [0, 1], range: fr.x, title: "x", scaleanchor: "y", scaleratio: 1, ...axisStyle },
        yaxis: { domain: [0.46, 1], range: fr.y, title: "y", ...axisStyle },
        xaxis2: { domain: [0, 1], anchor: "y2", title: "step", ...axisStyle },
        yaxis2: { domain: [0, 0.28], anchor: "x2", title: "sharpness", range: [1.95, 1.02 * Math.max(2.05, 2 / eta, ...run.s)], ...axisStyle },
        annotations: [{ x: 0.5, y: 1.07, xref: "paper", yref: "paper", text: "path and sharpness", showarrow: false, font: { size: 16 } }],
        shapes: run.tailStart === null ? [] : [{
          type: "line",
          xref: "x2",
          yref: "paper",
          x0: run.tailStart,
          x1: run.tailStart,
          y0: 0,
          y1: 0.28,
          line: { color: "rgba(120,120,120,0.7)", width: 1.4, dash: "dot" },
        }],
        uirevision: "post5-path",
      }, plotConfig);
    }

    slider.value = String(etaIndex);
    slider.addEventListener("input", () => {
      etaIndex = Number(slider.value);
      render();
    });

    await Promise.all([
      Plotly.newPlot(plots[0], [], {}, plotConfig),
      Plotly.newPlot(plots[1], [], {}, plotConfig),
    ]);
    render();

    plots[0].on("plotly_click", (event) => {
      const point = event?.points?.[0];
      if (!point) return;
      selected = {
        x: snap(point.x, grid),
        y: snap(point.y, grid),
      };
      render();
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".sharpness-explorer").forEach((root) => {
      init(root).catch((error) => {
        root.innerHTML = `<div class="phase-explorer-error">interactive explorer unavailable: ${error.message}</div>`;
      });
    });
  });
})();
