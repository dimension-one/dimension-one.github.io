(function () {
  const COLORS = {
    0: "#2ca02c",
    1: "#ff7f0e",
    2: "#1f77b4",
    3: "#7b3294",
    4: "#d0d0d0",
  };

  const CRITICAL_COLOR = "#d62728";
  const TRAJECTORY_COLOR = "#00c2ff";

  const LABEL_TEXT = {
    0: "to xy=1, no crossing",
    1: "to xy=1, with crossing",
    2: "to saddle (0, 0)",
    3: "blow-up",
    4: "unresolved within budget",
  };

  const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";

  function ensurePlotly() {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }
    if (!window.__post4PhaseExplorerPlotlyPromise) {
      window.__post4PhaseExplorerPlotlyPromise = new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[src="${PLOTLY_URL}"]`);
        if (existing) {
          existing.addEventListener("load", () => resolve(window.Plotly), { once: true });
          existing.addEventListener("error", reject, { once: true });
          return;
        }
        const script = document.createElement("script");
        script.src = PLOTLY_URL;
        script.onload = () => resolve(window.Plotly);
        script.onerror = reject;
        document.head.appendChild(script);
      });
    }
    return window.__post4PhaseExplorerPlotlyPromise;
  }

  function linspace(min, max, n) {
    if (n === 1) {
      return [min];
    }
    const step = (max - min) / (n - 1);
    return Array.from({ length: n }, (_, i) => min + i * step);
  }

  function roundToGrid(value, min, max, size) {
    const step = (max - min) / (size - 1);
    const index = Math.max(0, Math.min(size - 1, Math.round((value - min) / step)));
    return {
      index,
      value: min + index * step,
    };
  }

  function loss(x, y) {
    return 0.5 * (1 - x * y) ** 2;
  }

  function gdStep(x, y, eta) {
    const n = x * y - 1;
    return {
      x: x - eta * n * y,
      y: y - eta * n * x,
    };
  }

  function decodeGrid(encoded, size) {
    const rows = new Array(size);
    let cursor = 0;
    for (let i = 0; i < size; i += 1) {
      const row = new Array(size);
      for (let j = 0; j < size; j += 1) {
        row[j] = Number(encoded.charAt(cursor));
        cursor += 1;
      }
      rows[i] = row;
    }
    return rows;
  }

  function discreteColorscale() {
    return [
      [0.0, COLORS[0]],
      [0.1249, COLORS[0]],
      [0.125, COLORS[1]],
      [0.3749, COLORS[1]],
      [0.375, COLORS[2]],
      [0.6249, COLORS[2]],
      [0.625, COLORS[3]],
      [0.8749, COLORS[3]],
      [0.875, COLORS[4]],
      [1.0, COLORS[4]],
    ];
  }

  function analyzeTrajectory(x0, y0, eta, options) {
    const {
      steps = 320,
      tolN = 1e-7,
      tolOrigin = 1e-6,
      settleWindow = 5,
      blowupRadius = 1e8,
      plotLimit = 3.2,
    } = options || {};

    let x = x0;
    let y = y0;
    let n = x * y - 1;
    let previousN = n;
    let signFlips = 0;
    let nHits = Math.abs(n) < tolN ? 1 : 0;
    let originHits = Math.abs(x) < tolOrigin && Math.abs(y) < tolOrigin && Math.abs(n + 1) < tolOrigin ? 1 : 0;
    let label = null;
    let stopStep = 0;
    let clipped = false;

    const xs = [x];
    const ys = [y];
    const zs = [Math.log1p(loss(x, y))];

    function pushForPlot(px, py) {
      if (clipped) {
        return;
      }
      if (!Number.isFinite(px) || !Number.isFinite(py) || Math.abs(px) > plotLimit || Math.abs(py) > plotLimit) {
        clipped = true;
        return;
      }
      xs.push(px);
      ys.push(py);
      zs.push(Math.log1p(loss(px, py)));
    }

    if (originHits >= settleWindow) {
      label = 2;
    } else if (nHits >= settleWindow) {
      label = 0;
    }

    for (let step = 1; step <= steps && label === null; step += 1) {
      const next = gdStep(x, y, eta);
      x = next.x;
      y = next.y;
      pushForPlot(x, y);

      n = x * y - 1;
      if (previousN * n < 0) {
        signFlips += 1;
      }
      previousN = n;

      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > blowupRadius || Math.abs(y) > blowupRadius) {
        label = 3;
        stopStep = step;
        break;
      }

      nHits = Math.abs(n) < tolN ? nHits + 1 : 0;
      originHits = Math.abs(x) < tolOrigin && Math.abs(y) < tolOrigin && Math.abs(n + 1) < tolOrigin ? originHits + 1 : 0;

      if (originHits >= settleWindow) {
        label = 2;
        stopStep = step;
      } else if (nHits >= settleWindow) {
        label = signFlips === 0 ? 0 : 1;
        stopStep = step;
      }
    }

    if (label === null) {
      label = 4;
      stopStep = steps;
    }

    return {
      label,
      stopStep,
      signFlips,
      clipped,
      xs,
      ys,
      zs,
      finalX: x,
      finalY: y,
    };
  }

  function buildSurface(min, max, size) {
    const grid = linspace(min, max, size);
    const z = grid.map((y) => grid.map((x) => Math.log1p(loss(x, y))));
    return { grid, z };
  }

  function buildHyperbola() {
    const pos = linspace(1 / 3, 3, 220);
    const neg = linspace(-3, -1 / 3, 220);
    return {
      posX: pos,
      posY: pos.map((x) => 1 / x),
      negX: neg,
      negY: neg.map((x) => 1 / x),
    };
  }

  function formatNumber(value) {
    return value >= 0 ? value.toFixed(2) : value.toFixed(2);
  }

  function formatOutcome(run, budget) {
    if (run.label === 0 || run.label === 1 || run.label === 2) {
      return `${LABEL_TEXT[run.label]} after ${run.stopStep} step${run.stopStep === 1 ? "" : "s"}`;
    }
    if (run.label === 3) {
      return `blow-up detected by step ${run.stopStep}`;
    }
    return `not settled within ${budget} steps`;
  }

  function countFraction(counts, label, total) {
    return (100 * (counts[String(label)] || 0)) / total;
  }

  async function initExplorer(root) {
    if (root.dataset.initialized === "true") {
      return;
    }
    root.dataset.initialized = "true";
    root.innerHTML = '<div class="phase-explorer-loading">loading interactive explorer…</div>';

    try {
      const src = root.dataset.src;
      const initialEta = Number(root.dataset.initialEta || 0.55);
      const initialX = Number(root.dataset.initialX || 1.2);
      const initialY = Number(root.dataset.initialY || 0.8);

      const [Plotly, dataResponse] = await Promise.all([
        ensurePlotly(),
        fetch(src),
      ]);

      if (!dataResponse.ok) {
        throw new Error(`could not load ${src}`);
      }

      const data = await dataResponse.json();
      const textColor = getComputedStyle(document.body).color;
      const etas = data.etas.map(Number);
      const gridMin = Number(data.grid_min);
      const gridMax = Number(data.grid_max);
      const gridSize = Number(data.grid_size);
      const gridValues = linspace(gridMin, gridMax, gridSize);
      const totalPoints = gridSize * gridSize;
      const hyperbola = buildHyperbola();
      const surface = buildSurface(gridMin, gridMax, 45);
      const gridCache = new Map();

      root.innerHTML = `
        <div class="phase-explorer-controls">
          <div class="phase-explorer-range-row">
            <label>learning rate</label>
            <input class="phase-explorer-slider" type="range" min="0" max="${etas.length - 1}" step="1">
            <div class="phase-explorer-readout"></div>
          </div>
          <div class="phase-explorer-ticks"></div>
        </div>
        <div class="phase-explorer-panels">
          <div class="phase-explorer-plot"></div>
          <div class="phase-explorer-plot"></div>
        </div>
        <div class="phase-explorer-legend"></div>
        <p class="phase-explorer-note">The right panel shows the sampled ${gridSize}x${gridSize} grid with nearest-neighbor coloring. The left panel shows the trajectory on the surface of log(1 + L).</p>
        <p class="phase-explorer-status"></p>
      `;

      const slider = root.querySelector(".phase-explorer-slider");
      const readout = root.querySelector(".phase-explorer-readout");
      const ticks = root.querySelector(".phase-explorer-ticks");
      const leftPlot = root.querySelectorAll(".phase-explorer-plot")[0];
      const rightPlot = root.querySelectorAll(".phase-explorer-plot")[1];
      const legend = root.querySelector(".phase-explorer-legend");
      const status = root.querySelector(".phase-explorer-status");

      legend.innerHTML = [
        [0, LABEL_TEXT[0]],
        [1, LABEL_TEXT[1]],
        [2, LABEL_TEXT[2]],
        [3, LABEL_TEXT[3]],
        [4, LABEL_TEXT[4]],
      ].map(([label, text]) => `
        <span class="phase-explorer-legend-item">
          <span class="phase-explorer-swatch" style="background:${COLORS[label]}"></span>
          <span>${text}</span>
        </span>
      `).join("");

      ticks.innerHTML = etas.map((eta) => `<span>${eta.toFixed(2)}</span>`).join("");

      function etaKey(index) {
        return etas[index].toFixed(2);
      }

      function getGridFor(index) {
        const key = etaKey(index);
        if (!gridCache.has(key)) {
          gridCache.set(key, decodeGrid(data.labels[key], gridSize));
        }
        return gridCache.get(key);
      }

      let selected = {
        x: roundToGrid(initialX, gridMin, gridMax, gridSize).value,
        y: roundToGrid(initialY, gridMin, gridMax, gridSize).value,
      };

      let currentEtaIndex = etas.findIndex((eta) => Math.abs(eta - initialEta) < 1e-9);
      if (currentEtaIndex < 0) {
        currentEtaIndex = 0;
      }

      function updateTicks(index) {
        ticks.querySelectorAll("span").forEach((span, i) => {
          span.classList.toggle("current", i === index);
        });
      }

      function surfaceTrace() {
        return {
          type: "surface",
          x: surface.grid,
          y: surface.grid,
          z: surface.z,
          colorscale: "Viridis",
          showscale: false,
          opacity: 0.95,
          hoverinfo: "skip",
        };
      }

      function minimizerTrace3d(xs, ys) {
        return {
          type: "scatter3d",
          mode: "lines",
          x: xs,
          y: ys,
          z: xs.map(() => 0.035),
          line: { color: CRITICAL_COLOR, width: 7 },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function saddleTrace3d() {
        const z = Math.log1p(loss(0, 0)) + 0.12;
        const radius = 0.16;
        return {
          type: "scatter3d",
          mode: "lines",
          x: [-radius, radius, null, -radius, radius],
          y: [-radius, radius, null, radius, -radius],
          z: [z, z, null, z, z],
          line: { color: CRITICAL_COLOR, width: 8 },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function trajectoryTrace3d(run) {
        return {
          type: "scatter3d",
          mode: "lines+markers",
          x: run.xs,
          y: run.ys,
          z: run.zs,
          line: { color: TRAJECTORY_COLOR, width: 6 },
          marker: {
            size: 3,
            color: TRAJECTORY_COLOR,
            line: { color: "#111111", width: 1 },
          },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function startTrace3d() {
        return {
          type: "scatter3d",
          mode: "markers",
          x: [selected.x],
          y: [selected.y],
          z: [Math.log1p(loss(selected.x, selected.y))],
          marker: {
            size: 6,
            color: "#ffffff",
            line: { color: "#111111", width: 2 },
          },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function heatmapTrace(index) {
        return {
          type: "heatmap",
          x: gridValues,
          y: gridValues,
          z: getGridFor(index),
          zmin: 0,
          zmax: 4,
          colorscale: discreteColorscale(),
          showscale: false,
          xgap: 0,
          ygap: 0,
          zsmooth: false,
          hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        };
      }

      function minimizerTrace2d(xs, ys) {
        return {
          type: "scatter",
          mode: "lines",
          x: xs,
          y: ys,
          line: { color: CRITICAL_COLOR, width: 3.2 },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function saddleTrace2d() {
        return {
          type: "scatter",
          mode: "markers",
          x: [0],
          y: [0],
          marker: {
            color: CRITICAL_COLOR,
            symbol: "x",
            size: 12,
            line: { color: CRITICAL_COLOR, width: 3 },
          },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function trajectoryTrace2d(run) {
        return {
          type: "scatter",
          mode: "lines+markers",
          x: run.xs,
          y: run.ys,
          line: { color: TRAJECTORY_COLOR, width: 3 },
          marker: {
            color: TRAJECTORY_COLOR,
            size: 5,
            line: { color: "#111111", width: 1 },
          },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      function startTrace2d() {
        return {
          type: "scatter",
          mode: "markers",
          x: [selected.x],
          y: [selected.y],
          marker: {
            color: "#ffffff",
            size: 10,
            line: { color: "#111111", width: 2 },
          },
          hoverinfo: "skip",
          showlegend: false,
        };
      }

      const commonLayout = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: textColor },
      };

      const leftLayout = {
        ...commonLayout,
        margin: { l: 0, r: 0, t: 38, b: 0 },
        title: { text: "trajectory on log(1 + L(x, y))", x: 0.5, xanchor: "center", font: { size: 16 } },
        scene: {
          xaxis: { title: "x", range: [gridMin, gridMax], gridcolor: "rgba(128,128,128,0.25)", zerolinecolor: "rgba(128,128,128,0.45)" },
          yaxis: { title: "y", range: [gridMin, gridMax], gridcolor: "rgba(128,128,128,0.25)", zerolinecolor: "rgba(128,128,128,0.45)" },
          zaxis: { title: "log(1 + L)", range: [0, 4.05], gridcolor: "rgba(128,128,128,0.25)" },
          aspectmode: "cube",
          camera: { eye: { x: -0.85, y: -0.85, z: 1.70 } },
        },
        uirevision: "post4-surface",
      };

      const rightLayout = {
        ...commonLayout,
        margin: { l: 48, r: 12, t: 38, b: 44 },
        title: { text: "phase portrait on the sampled grid", x: 0.5, xanchor: "center", font: { size: 16 } },
        xaxis: {
          title: "x",
          range: [gridMin, gridMax],
          scaleanchor: "y",
          scaleratio: 1,
          gridcolor: "rgba(128,128,128,0.20)",
          zerolinecolor: "rgba(128,128,128,0.45)",
        },
        yaxis: {
          title: "y",
          range: [gridMin, gridMax],
          gridcolor: "rgba(128,128,128,0.20)",
          zerolinecolor: "rgba(128,128,128,0.45)",
        },
        uirevision: "post4-phase",
      };

      function render() {
        const eta = etas[currentEtaIndex];
        const counts = data.counts[etaKey(currentEtaIndex)];
        const run = analyzeTrajectory(selected.x, selected.y, eta, { steps: data.steps });
        const unresolved = countFraction(counts, 4, totalPoints).toFixed(2);

        readout.textContent = `eta = ${eta.toFixed(2)} | grid = ${gridSize}x${gridSize} | steps = ${data.steps}`;
        updateTicks(currentEtaIndex);

        status.textContent = `start = (${formatNumber(selected.x)}, ${formatNumber(selected.y)}) | outcome = ${formatOutcome(run, data.steps)} | unresolved on this eta grid = ${unresolved}%${run.clipped ? " | plotted path clipped after leaving the window [-3,3]^2" : ""}`;

        Plotly.react(
          leftPlot,
          [
            surfaceTrace(),
            minimizerTrace3d(hyperbola.posX, hyperbola.posY),
            minimizerTrace3d(hyperbola.negX, hyperbola.negY),
            trajectoryTrace3d(run),
            saddleTrace3d(),
            startTrace3d(),
          ],
          leftLayout,
          { displayModeBar: false, responsive: true }
        );

        Plotly.react(
          rightPlot,
          [
            heatmapTrace(currentEtaIndex),
            minimizerTrace2d(hyperbola.posX, hyperbola.posY),
            minimizerTrace2d(hyperbola.negX, hyperbola.negY),
            trajectoryTrace2d(run),
            saddleTrace2d(),
            startTrace2d(),
          ],
          rightLayout,
          { displayModeBar: false, responsive: true }
        );
      }

      slider.value = String(currentEtaIndex);
      slider.addEventListener("input", () => {
        currentEtaIndex = Number(slider.value);
        render();
      });

      await Promise.all([
        Plotly.newPlot(leftPlot, [], leftLayout, { displayModeBar: false, responsive: true }),
        Plotly.newPlot(rightPlot, [], rightLayout, { displayModeBar: false, responsive: true }),
      ]);

      render();

      if (typeof rightPlot.on === "function") {
        rightPlot.on("plotly_click", (event) => {
          const point = event?.points?.[0];
          if (!point) {
            return;
          }
          selected = {
            x: roundToGrid(point.x, gridMin, gridMax, gridSize).value,
            y: roundToGrid(point.y, gridMin, gridMax, gridSize).value,
          };
          render();
        });
      }
    } catch (error) {
      root.innerHTML = `<div class="phase-explorer-error">interactive explorer unavailable: ${error.message}</div>`;
    }
  }

  function setupDeferredInit(root) {
    const details = root.closest("details");
    if (!details || details.open) {
      if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) {
              continue;
            }
            observer.disconnect();
            initExplorer(root);
            break;
          }
        }, { rootMargin: "200px 0px" });
        observer.observe(root);
      } else {
        initExplorer(root);
      }
      return;
    }

    const handler = () => {
      if (!details.open) {
        return;
      }
      details.removeEventListener("toggle", handler);
      initExplorer(root);
    };

    details.addEventListener("toggle", handler);
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".phase-explorer").forEach(setupDeferredInit);
  });
})();
