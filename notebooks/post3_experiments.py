"""Reproducible fixed-noise and across-noise experiments for post 3.

Run with: python notebooks/post3_experiments.py
The notebook embeds this source so its Colab copy is self-contained.
"""
import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np


def _compensated_prefix(initial, increments):
    """Neumaier summation: avoid the drift of a long ordinary cumulative sum."""
    out = np.empty(len(increments) + 1, dtype=np.float64)
    total, correction = float(initial), 0.0
    out[0] = total
    for i, value in enumerate(increments, start=1):
        value = float(value)
        updated = total + value
        correction += ((total - updated) + value if abs(total) >= abs(value)
                       else (value - updated) + total)
        total = updated
        out[i] = total + correction
    return out


def geometry_stats(w, b, v, gammas=(0.1, 0.2, 0.5, 1.0)):
    """Integrate Delta(x)=sum_j v_j relu(w_j*x+b_j) on [-1,1].

    Leb_wrong and band_mass_* are LENGTHS (divide by 2 for uniform mass).
    The classifier predicts 1 iff Delta > 0, matching two-logit argmax's
    choice of class 0 on ties. nb_flips counts changes between consecutive
    positive-length decision intervals, including a zero plateau (class 0),
    but excluding an isolated zero touch. nb_regions counts activation
    intervals, as in the original notebook, even if a knot cancels in Delta.
    Lavg_flip preserves the original length-weighted slope over activation
    intervals with a root STRICTLY inside them; a root at a knot contributes
    to nb_flips but not Lavg_flip.

    Sorting and a compensated float64 sweep cost O(width log width).
    Relative roundoff tolerances identify numerically zero affine pieces
    and snap roots at knots. There is no absolute coefficient cutoff, so
    rescaling a nonzero network by a tiny positive number preserves its
    decisions. These are analytic interval integrals, subject to float64
    arithmetic, rather than estimates from a sampled test grid.
    """
    w, b, v = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (w, b, v)]
    if not (w.shape == b.shape == v.shape):
        raise ValueError('w, b and v must have the same length')
    if not all(np.all(np.isfinite(a)) for a in (w, b, v)):
        raise ValueError('network parameters must be finite')
    gammas = tuple(float(g) for g in gammas)
    if any(not np.isfinite(g) or g < 0 for g in gammas):
        raise ValueError('gammas must be finite and nonnegative')

    nz = w != 0
    roots = np.full(w.shape, np.inf)
    roots[nz] = -b[nz] / w[nz]
    inside = nz & (roots > -1) & (roots < 1)
    knots, inverse = np.unique(roots[inside], return_inverse=True)
    bounds = np.r_[-1.0, knots, 1.0]
    left, right = bounds[:-1], bounds[1:]
    length = right - left
    mid0 = left[0] + 0.5 * length[0]
    active = w * mid0 + b > 0
    vw, vb = v * w, v * b
    m0 = math.fsum(vw[active])
    c0 = math.fsum(vb[active])
    jump_m = np.bincount(inverse, weights=v[inside] * np.abs(w[inside]), minlength=len(knots))
    jump_c = np.bincount(inverse, weights=v[inside] * b[inside] * np.sign(w[inside]), minlength=len(knots))
    slope = _compensated_prefix(m0, jump_m)
    intercept = _compensated_prefix(c0, jump_c)

    # Only units active somewhere in the domain enter the roundoff scale.
    ever_active = np.maximum(-w + b, w + b) > 0
    eps = 16 * np.finfo(np.float64).eps
    tol_m = eps * math.fsum(np.abs(vw[ever_active]))
    tol_c = eps * math.fsum(np.abs(vb[ever_active]))
    zero_piece = (np.abs(slope) <= tol_m) & (np.abs(intercept) <= tol_c)
    slope[zero_piece] = 0.0
    intercept[zero_piece] = 0.0
    nonflat = slope != 0
    crossing = np.full(slope.shape, np.inf)
    crossing[nonflat] = -intercept[nonflat] / slope[nonflat]
    fl = slope * left + intercept
    fr = slope * right + intercept
    crossing[nonflat & (np.abs(fl) <= tol_m * np.abs(left) + tol_c)] = left[nonflat & (np.abs(fl) <= tol_m * np.abs(left) + tol_c)]
    crossing[nonflat & (np.abs(fr) <= tol_m * np.abs(right) + tol_c)] = right[nonflat & (np.abs(fr) <= tol_m * np.abs(right) + tol_c)]
    interior_root = nonflat & (crossing > left) & (crossing < right)

    def positive_length(ll, rr):
        width = np.maximum(0.0, rr - ll)
        result = np.where(intercept > 0, width, 0.0)
        up = slope > 0
        down = slope < 0
        result[up] = np.clip(rr[up] - np.maximum(ll[up], crossing[up]), 0, width[up])
        result[down] = np.clip(np.minimum(rr[down], crossing[down]) - ll[down], 0, width[down])
        return result

    neg_right = np.minimum(right, 0.0)
    pos_left = np.maximum(left, 0.0)
    false_positive = positive_length(left, neg_right)
    positive_domain_length = np.maximum(0.0, right - pos_left)
    false_negative = positive_domain_length - positive_length(pos_left, right)
    wrong = float(np.sum(false_positive + false_negative))

    decisions = []
    for l, r, m, c, root, split in zip(left, right, slope, intercept, crossing, interior_root):
        if split:
            decisions.extend((bool(m < 0), bool(m > 0)))
        elif m == 0:
            decisions.append(bool(c > 0))
        else:
            # Position relative to the (possibly snapped) root is more
            # stable than subtracting nearly cancelling affine terms.
            middle = l + 0.5 * (r - l)
            decisions.append(bool(middle > root if m > 0 else middle < root))
    flips = int(np.count_nonzero(np.asarray(decisions[1:]) != np.asarray(decisions[:-1])))

    abs_slope = np.abs(slope)
    flip_length = np.sum(length[interior_root])
    result = {
        'Leb_wrong': float(np.clip(wrong, 0.0, 2.0)),
        'nb_regions': int(len(length)),
        'nb_flips': flips,
        'Lmax': float(np.max(abs_slope)),
        'Lavg': float(np.sum(abs_slope * length) / 2),
        'Lavg_flip': float(np.sum(abs_slope[interior_root] * length[interior_root]) / flip_length) if flip_length > 0 else 0.0,
    }
    for g in gammas:
        bands = np.where(np.abs(intercept) <= g, length, 0.0)
        a = (-g - intercept[nonflat]) / slope[nonflat]
        z = (g - intercept[nonflat]) / slope[nonflat]
        bands[nonflat] = np.maximum(0.0, np.minimum(right[nonflat], np.maximum(a, z)) - np.maximum(left[nonflat], np.minimum(a, z)))
        result[f'band_mass_{g}'] = float(np.sum(bands))
    return result

# Experiment protocol. All choices are fixed before the reported suite is run.
PROTOCOL = {
    'version': 1, 'fixed_p': 0.4, 'n_train': 32,
    'dataset_seeds': [100, 101, 102, 103, 104],
    'widths': [256, 1024], 'learning_rates': [0.003, 0.01],
    'initialization_scales': [0.5, 2.0],
    'sweep_seeds': [200, 201], 'sweep_ps': [i / 10 for i in range(11)],
    'sweep_width': 1024, 'sweep_lr': 0.003, 'sweep_init_scale': 1.0,
    'max_steps': 100000, 'check_every': 50, 'target_cross_entropy': 0.02,
    'relative_sigma': 0.001, 'perturbation_samples': 20,
    'sam_radius': 0.001, 'hessian_iterations': 30,
    'sampling': 'iid uniform inputs; independent Bernoulli(p) replacement by a fair label',
    'checkpoint': 'first checked zero-error CE<=target; otherwise minimum errors then minimum CE',
}
PROXIES = ['nb_flips', 'nb_regions', 'Lmax', 'Lavg', 'Lavg_flip',
           'band_mass_0.1', 'band_mass_0.2', 'band_mass_0.5', 'band_mass_1.0',
           'flat_rel', 'sam_sharp', 'hess_power']


def make_data(seed, p, n):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 1)).astype(np.float32)
    clean = (x[:, 0] > 0).astype(np.int64)
    mask = rng.random(n) < p
    replacement = rng.integers(0, 2, n)
    y = np.where(mask, replacement, clean)
    return torch.from_numpy(x), torch.from_numpy(y), int(mask.sum()), int((y != clean).sum())


def make_model(width, init_scale, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 2, bias=False))
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(init_scale)
    return model


def train_model(model, x, y, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = (len(y) + 1, float('inf'))
    best_state = None
    best_step = 0
    reached = False
    for step in range(1, PROTOCOL['max_steps'] + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if step % PROTOCOL['check_every'] == 0 or step == PROTOCOL['max_steps']:
            with torch.no_grad():
                logits = model(x)
                count = int((logits.argmax(1) != y).sum())
                ce = float(F.cross_entropy(logits, y))
            if not np.isfinite(ce):
                raise RuntimeError('Non-finite training loss')
            if (count, ce) < best:
                best = (count, ce)
                best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if count == 0 and ce <= PROTOCOL['target_cross_entropy']:
                reached = True
                break
    model.load_state_dict(best_state)
    return {'train_error_count': best[0], 'train_err': best[0] / len(y),
            'train_ce': best[1], 'checkpoint_step': best_step,
            'steps_run': step, 'target_reached': reached}


def parameter_proxies(model, x, y, seed):
    # Use float64 for small loss differences and Hessian-vector products.
    model = copy.deepcopy(model).double()
    x = x.double()
    torch.manual_seed(seed)
    params = list(model.parameters())
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    loss0 = F.cross_entropy(model(x), y).item()
    increases = []
    for _ in range(PROTOCOL['perturbation_samples']):
        model.load_state_dict(base)
        with torch.no_grad():
            for param in params:
                param.add_(PROTOCOL['relative_sigma'] * (param.abs() + 1e-12) * torch.randn_like(param))
            increases.append(F.cross_entropy(model(x), y).item() - loss0)
    model.load_state_dict(base)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params)
    norm = torch.sqrt(sum((g * g).sum() for g in grads))
    with torch.no_grad():
        if norm > 0:
            for param, grad in zip(params, grads):
                param.add_(PROTOCOL['sam_radius'] * grad / norm)
        sam = F.cross_entropy(model(x), y).item() - loss0
    model.load_state_dict(base)
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    vec = [torch.randn_like(param) for param in params]
    norm = torch.sqrt(sum((v * v).sum() for v in vec))
    vec = [v / norm for v in vec]
    for _ in range(PROTOCOL['hessian_iterations']):
        hv = torch.autograd.grad(sum((g * v).sum() for g, v in zip(grads, vec)), params, retain_graph=True)
        hn = torch.sqrt(sum((h * h).sum() for h in hv))
        if hn <= 1e-30:
            break
        vec = [h.detach() / hn for h in hv]
    hv = torch.autograd.grad(sum((g * v).sum() for g, v in zip(grads, vec)), params)
    rayleigh = sum((v * h).sum() for v, h in zip(vec, hv)).item()
    residual = float(torch.sqrt(sum(((h - rayleigh * v) ** 2).sum() for h, v in zip(hv, vec))))
    return {'flat_rel': float(np.mean(increases)), 'sam_sharp': sam,
            'hess_power': rayleigh, 'hess_residual_relative': residual / max(abs(rayleigh), 1e-30)}


def configurations():
    for seed in PROTOCOL['dataset_seeds']:
        for wi, width in enumerate(PROTOCOL['widths']):
            for si, scale in enumerate(PROTOCOL['initialization_scales']):
                for li, lr in enumerate(PROTOCOL['learning_rates']):
                    # Within a width, paired scales/LRs start from the same unscaled draw.
                    init_seed = 100000 + seed * 10 + wi
                    yield dict(suite='fixed_p', dataset_seed=seed, p=PROTOCOL['fixed_p'],
                               width=width, init_scale=scale, lr=lr, init_seed=init_seed,
                               proxy_seed=200000 + seed * 100 + wi * 10 + si * 2 + li)
    for seed in PROTOCOL['sweep_seeds']:
        for pi, p in enumerate(PROTOCOL['sweep_ps']):
            yield dict(suite='across_p', dataset_seed=seed, p=p,
                       width=PROTOCOL['sweep_width'], init_scale=PROTOCOL['sweep_init_scale'],
                       lr=PROTOCOL['sweep_lr'], init_seed=300000 + seed,
                       proxy_seed=400000 + seed * 100 + pi)


def config_key(c):
    return f"{c['suite']}-seed{c['dataset_seed']}-p{c['p']:.1f}-w{c['width']}-s{c['init_scale']}-lr{c['lr']}"


def run_suite(output_dir, checkpoint_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_file = output_dir / 'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != PROTOCOL:
            raise ValueError('Existing results use a different protocol; choose a fresh output directory')
    else:
        protocol_file.write_text(json.dumps(PROTOCOL, indent=2) + '\n')
    rows_file = output_dir / 'runs.json'
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    done = {r['id'] for r in rows}
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configs = list(configurations())
    for i, config in enumerate(configs):
        key = config_key(config)
        if key in done:
            continue
        t0 = time.perf_counter()
        x, y, replacements, flips = make_data(config['dataset_seed'], config['p'], PROTOCOL['n_train'])
        model = make_model(config['width'], config['init_scale'], config['init_seed'])
        fit = train_model(model, x, y, config['lr'])
        with torch.no_grad():
            exact_train_count = int((copy.deepcopy(model).double()(x.double()).argmax(1) != y).sum())
        assert exact_train_count == fit['train_error_count'], 'float32/float64 decision mismatch'
        w = model[0].weight.detach().numpy()[:, 0].astype(np.float64)
        b = model[0].bias.detach().numpy().astype(np.float64)
        v = (model[2].weight.detach().double()[1] - model[2].weight.detach().double()[0]).numpy()
        geom = geometry_stats(w, b, v)
        param = parameter_proxies(model, x, y, config['proxy_seed'])
        risk = config['p'] / 2 + (1 - config['p']) * geom['Leb_wrong'] / 2
        row = {'id': key, **config, **fit, **geom, **param,
               'population_error': risk, 'generalization_gap': risk - fit['train_err'],
               'replacement_count': replacements, 'label_flip_count': flips,
               'elapsed_seconds': time.perf_counter() - t0}
        rows.append(row)
        rows_file.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        if checkpoint_dir:
            torch.save({'config': config, 'state_dict': model.state_dict(), 'x': x, 'y': y}, checkpoint_dir / (key + '.pt'))
        print(f"{i+1:02d}/{len(configs)} {key}: errors={fit['train_error_count']}, "
              f"CE={fit['train_ce']:.4f}, steps={fit['steps_run']}, gap={row['generalization_gap']:.4f}, "
              f"seconds={row['elapsed_seconds']:.1f}", flush=True)
    (output_dir / 'environment.json').write_text(json.dumps({
        'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
        'device': 'cpu', 'training_dtype': 'float32', 'geometry_and_proxy_dtype': 'float64',
    }, indent=2) + '\n')
    with (output_dir / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def average_ranks(a):
    a = np.asarray(a)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inv]


def metrics(rows):
    result = {}
    for name in ['p', 'label_flip_count', 'Leb_wrong', *PROXIES, 'train_err']:
        x = [r[name] for r in rows]
        result[name] = {}
        for target in ['generalization_gap', 'population_error']:
            y = [r[target] for r in rows]
            result[name][target] = {'pearson': correlation(x, y),
                                   'spearman': correlation(average_ranks(x), average_ranks(y))}
    return result


def summarize(rows, output_dir):
    fixed = [r for r in rows if r['suite'] == 'fixed_p']
    interpolated = [r for r in fixed if r['train_error_count'] == 0]
    sweep = [r for r in rows if r['suite'] == 'across_p']
    groups = {'fixed_p_all': fixed, 'fixed_p_interpolated': interpolated,
              'fixed_p_target_reached': [r for r in fixed if r['target_reached']],
              'across_p_all': sweep}
    for axis in ['dataset_seed', 'width', 'lr', 'init_scale']:
        for value in sorted({r[axis] for r in fixed}):
            groups[f'fixed_p_interpolated_{axis}={value}'] = [r for r in interpolated if r[axis] == value]
    result = {}
    for name, group in groups.items():
        result[name] = {'n': len(group), 'interpolated': sum(r['train_error_count'] == 0 for r in group),
                        'target_reached': sum(r['target_reached'] for r in group),
                        'gap_min': min((r['generalization_gap'] for r in group), default=None),
                        'gap_max': max((r['generalization_gap'] for r in group), default=None),
                        'max_train_errors': max((r['train_error_count'] for r in group), default=None),
                        'metrics': metrics(group)}
    for group in [fixed, interpolated]:
        if len(group) > 2:
            assert abs(metrics(group)['Leb_wrong']['population_error']['pearson'] - 1) < 1e-10
    if len(interpolated) > 2:
        assert abs(metrics(interpolated)['Leb_wrong']['generalization_gap']['pearson'] - 1) < 1e-10
    output_dir = Path(output_dir)
    (output_dir / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('\nFixed p: %d/%d interpolated; %d/%d reached CE target' % (
        len(interpolated), len(fixed), sum(r['target_reached'] for r in fixed), len(fixed)))
    print('Interpolated gap range:', result['fixed_p_interpolated']['gap_min'], result['fixed_p_interpolated']['gap_max'])
    print('proxy                 fixed gap Pearson  Spearman  all-model risk Pearson')
    for name in ['p', 'label_flip_count', 'Leb_wrong', *PROXIES]:
        a = result['fixed_p_interpolated']['metrics'][name]['generalization_gap']
        b = result['fixed_p_all']['metrics'][name]['population_error']['pearson']
        fmt = lambda x: 'undefined' if x is None else f'{x:+.3f}'
        print(f"{name:22s} {fmt(a['pearson']):>9s} {fmt(a['spearman']):>9s} {fmt(b):>9s}")
    return result


def plot_results(rows, output_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed = [r for r in rows if r['suite'] == 'fixed_p' and r['train_error_count'] == 0]
    sweep = [r for r in rows if r['suite'] == 'across_p']
    fig, grid = plt.subplots(2, 2, figsize=(8.4, 6.7), constrained_layout=True)
    axes = grid.ravel()
    axes[0].scatter([r['p'] for r in sweep], [r['generalization_gap'] for r in sweep],
                    c=[r['p'] for r in sweep], cmap='viridis', vmin=0, vmax=1, s=42)
    axes[0].set(title='Across p: corruption itself', xlabel='Corruption probability p', ylabel='Exact generalization gap')
    pmap = axes[1].scatter([r['nb_flips'] for r in sweep], [r['generalization_gap'] for r in sweep],
                          c=[r['p'] for r in sweep], cmap='viridis', vmin=0, vmax=1, s=42)
    axes[1].set(title='Across p: decision flips', xlabel='Decision flips')
    fig.colorbar(pmap, ax=axes[1], label='p', shrink=.8)
    for ax, proxy, label in zip(axes[2:], ['nb_flips', 'Lavg'], ['Decision flips', 'Average absolute slope']):
        for lr, color in [(0.003, '#3a7ca5'), (0.01, '#d27a2c')]:
            group = [r for r in fixed if r['lr'] == lr]
            ax.scatter([r[proxy] for r in group], [r['generalization_gap'] for r in group],
                       s=40, alpha=.8, color=color, label=f'Adam lr={lr}')
        ax.set(title=f'Fixed p=0.4 ({len(fixed)} interpolators)', xlabel=label)
    axes[2].set_ylabel('Exact generalization gap')
    axes[3].set_xscale('log')
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/data/post-3')
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--figure', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/images/post-3/fixed-p-proxies.png')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.analyze_only:
        rows = json.loads((args.output / 'runs.json').read_text())
    else:
        rows = run_suite(args.output, args.checkpoints)
    summarize(rows, args.output)
    plot_results(rows, args.figure)


if __name__ == '__main__':
    main()
