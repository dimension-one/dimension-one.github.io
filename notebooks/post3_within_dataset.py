"""Post 3 follow-up: distinguish models trained on the same data.

Run from the repository root (requires the existing NumPy/PyTorch environment):
    .venv/bin/python -B notebooks/post3_within_dataset.py --dry-run
    .venv/bin/python -B notebooks/post3_within_dataset.py
    .venv/bin/python -B notebooks/post3_within_dataset.py --analyze-only

The default is 20 NEW dataset seeds (1000..1019), p=.4, 32 points, and
2 widths x 2 initialization scales x 2 learning rates x 2 initialization
draws = 320 fits. Scale and learning-rate comparisons share an unscaled
initialization within each dataset, width and draw. Every fit gets the same
200,000-step budget, increased in advance from the previous 100,000.

Primary analysis: one Spearman correlation per dataset among zero-error
fits, requiring at least 8 such fits. Average datasets equally; percentile
95% intervals resample informative datasets, never individual networks.
Constant proxies/targets are undefined, not zero. Coverage is reported for
each measure. CE-target fits and all-model population error are secondary.
Failures remain recorded; no dataset is replaced or silently dropped.
Interpolation uses float32 training predictions. Float64 count differences
are recorded as diagnostics and do not interrupt training or change selection.

Normalized bands use Delta / sqrt(E_uniform[-1,1] Delta^2), with the RMS
integrated analytically. This removes positive global logit scaling, not
all functional or parameterization dependence. The zero function has no
defined normalized margin. Signed training-margin quantiles are also saved.

Label changes are an observable dataset-only baseline. Wrong-label count
and nearest-neighbour disagreement are synthetic-oracle diagnostics using
the known clean rule. Pooled baseline correlations are descriptive context;
within-dataset discrimination is the main question.

Outputs go to assets/data/post-3-within-dataset, leaving the earlier results,
post and notebook untouched. Per-fit JSON and checkpoints support resumption;
the base helper is copied into source/ before it is imported. Training,
resumption and analysis use that saved copy, regardless of later edits to
the live helper. Protocol, runner and saved-helper mismatches are rejected.
Only one writer may use a directory.
The summary explicitly labels incomplete runs. Same-data min/max-gap fits
from the eligible dataset with the largest gap range illustrate boundary
positions in a figure; this selection is illustrative, not representative.

Tiny plumbing check, NOT a scientific run:
    .venv/bin/python -B notebooks/post3_within_dataset.py \
      --output /tmp/post3-within-smoke --dataset-count 2 --widths 8 \
      --max-steps 5 --check-every 5 --bootstrap-samples 100
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# Defer importing the helper until main has chosen the saved source. Direct
# analytical calls (e.g. unit tests) may lazily use the adjacent live helper.
base = None


GAMMAS = (0.1, 0.2, 0.5, 1.0)
BASELINES = ['label_changes', 'wrong_label_count', 'nn_Leb_wrong']
NORMALIZED = [f'normalized_band_mass_{g}' for g in GAMMAS] + [
    'normalized_Lmax', 'normalized_Lavg',
    'normalized_margin_min', 'normalized_margin_q10',
]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'assets/data/post-3-within-dataset'
LIVE_BASE_PATH = Path(__file__).resolve().with_name('post3_experiments.py')


def import_base(path):
    spec = importlib.util.spec_from_file_location('post3_saved_experiments', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_base():
    global base
    if base is None:
        base = import_base(LIVE_BASE_PATH)
    return base


def measure_names():
    return ['Leb_wrong', *BASELINES, *get_base().PROXIES, *NORMALIZED]


def dataset_baselines(x, y, p):
    """Exact 1-NN disagreement; endpoints extend to the edges of [-1,1]."""
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    if len(x) == 0 or len(x) != len(y) or not np.all(np.isin(y, [0, 1])):
        raise ValueError('Need matching nonempty inputs and binary labels')
    order = np.argsort(x)
    x, y = x[order].astype(float), y[order]
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0) or x[0] < -1 or x[-1] > 1:
        raise ValueError('Inputs must be distinct and inside [-1,1]')
    edges = np.r_[-1., (x[:-1] + x[1:]) / 2, 1.]
    left, right = edges[:-1], edges[1:]
    wrong = np.where(y == 1, np.maximum(0., np.minimum(right, 0.) - left),
                     np.maximum(0., right - np.maximum(left, 0.)))
    length = float(wrong.sum())
    return {'label_changes': int(np.count_nonzero(y[1:] != y[:-1])),
            'wrong_label_count': int(np.count_nonzero(y != (x > 0))),
            'nn_Leb_wrong': length, 'nn_population_error': p / 2 + (1 - p) * length / 2}


def affine_endpoints(w, b, v):
    """Values at all activation knots; sufficient for exact affine integrals."""
    w, b, v = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (w, b, v)]
    nz = w != 0
    knots = -b[nz] / w[nz]
    bounds = np.unique(np.r_[-1., knots[(knots > -1) & (knots < 1)], 1.])
    # Chunk evaluation to avoid quadratic peak memory for wider experiments.
    values = np.concatenate([
        np.maximum(bounds[i:i + 128, None] * w + b, 0.) @ v
        for i in range(0, len(bounds), 128)
    ])
    return bounds, values


def normalized_geometry(w, b, v, x, y, gammas=GAMMAS):
    base = get_base()
    bounds, values = affine_endpoints(w, b, v)
    peak = float(np.max(np.abs(values)))
    keys = [f'normalized_band_mass_{g}' for g in gammas] + NORMALIZED[-4:]
    if peak == 0:
        return {'logit_rms': 0., **dict.fromkeys(keys)}
    # On an affine interval, mean(f^2)=(a^2+ab+b^2)/3. Scale first for stability.
    a, z = values[:-1] / peak, values[1:] / peak
    rms = peak * np.sqrt(np.sum(np.diff(bounds) * (a*a + a*z + z*z)) / 6)
    geom = base.geometry_stats(w, b, np.asarray(v) / rms, gammas)
    delta = np.maximum(np.asarray(x).reshape(-1, 1) * w + b, 0.) @ (np.asarray(v) / rms)
    margins = (2 * np.asarray(y).reshape(-1) - 1) * delta
    return {'logit_rms': float(rms),
            **{f'normalized_band_mass_{g}': geom[f'band_mass_{g}'] for g in gammas},
            'normalized_Lmax': geom['Lmax'], 'normalized_Lavg': geom['Lavg'],
            'normalized_margin_min': float(np.min(margins)),
            'normalized_margin_q10': float(np.quantile(margins, .1))}


def within_dataset_summary(rows, proxies, target, min_models=8,
                           bootstrap_samples=10000, bootstrap_seed=20261010):
    base = get_base()
    per_dataset = []
    for seed in sorted({r['dataset_seed'] for r in rows}):
        group = [r for r in rows if r['dataset_seed'] == seed]
        entry = {'dataset_seed': seed, 'n_models': len(group),
                 'eligible': len(group) >= min_models, 'metrics': {}}
        for name in proxies:
            a = np.array([np.nan if r.get(name) is None else r[name] for r in group], float)
            b = np.array([r[target] for r in group], float)
            reason = None
            if not entry['eligible']:
                reason = 'too_few_models'
            elif not (np.isfinite(a).all() and np.isfinite(b).all()):
                reason = 'undefined_measure'
            elif np.ptp(b) == 0:
                reason = 'constant_target'
            elif np.ptp(a) == 0:
                reason = 'constant_proxy'
            rho = None if reason else base.correlation(base.average_ranks(a), base.average_ranks(b))
            entry['metrics'][name] = {'spearman': rho, 'reason': reason}
        per_dataset.append(entry)
    metrics = {}
    for name in proxies:
        informative = [d for d in per_dataset if d['metrics'][name]['spearman'] is not None]
        values = np.array([d['metrics'][name]['spearman'] for d in informative])
        ci = None
        if len(values) >= 2:
            rng = np.random.default_rng(bootstrap_seed)
            draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
            ci = np.quantile(draws, [.025, .975]).tolist()
        metrics[name] = {
            'mean_spearman': float(values.mean()) if len(values) else None,
            'ci95': ci,
            'n_eligible_datasets': sum(d['eligible'] for d in per_dataset),
            'n_informative_datasets': len(values),
            'informative_dataset_seeds': [d['dataset_seed'] for d in informative],
            **{f'n_{reason}_datasets': sum(d['metrics'][name]['reason'] == reason for d in per_dataset)
               for reason in ['constant_proxy', 'constant_target', 'undefined_measure', 'too_few_models']},
        }
    return {'target': target, 'min_models': min_models,
            'estimand': 'equal-weight mean over eligible datasets with defined Spearman correlation',
            'bootstrap': 'percentile 95%; resample informative datasets with replacement',
            'per_dataset': per_dataset, 'metrics': metrics}


def configurations(protocol):
    for seed in protocol['dataset_seeds']:
        for wi, width in enumerate(protocol['widths']):
            for rep in range(protocol['initialization_repeats']):
                init_seed = int(np.random.SeedSequence([20261010, seed, wi, rep]).generate_state(1)[0])
                for si, scale in enumerate(protocol['initialization_scales']):
                    for li, lr in enumerate(protocol['learning_rates']):
                        proxy_seed = int(np.random.SeedSequence([20261011, seed, wi, rep, si, li]).generate_state(1)[0])
                        yield {'dataset_seed': seed, 'p': protocol['p'], 'width': width,
                               'init_scale': scale, 'lr': lr, 'init_repeat': rep,
                               'init_seed': init_seed, 'proxy_seed': proxy_seed,
                               'id': f'd{seed}-w{width}-r{rep}-s{scale}-lr{lr}'}


def coefficients(model):
    return (model[0].weight.detach().double().numpy()[:, 0],
            model[0].bias.detach().double().numpy(),
            (model[2].weight.detach().double()[1] - model[2].weight.detach().double()[0]).numpy())


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.runner.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is using this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def code_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path(get_base().__file__)]}


def bind_saved_base(output, stored_protocol=None):
    """Freeze before first import; verify frozen bytes before every later import."""
    global base
    source_dir = output / 'source'
    saved_base = source_dir / 'post3_experiments.py'
    if stored_protocol is None:
        source_dir.mkdir(exist_ok=True)
        saved_base.write_bytes(LIVE_BASE_PATH.read_bytes())
        (source_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    else:
        expected = stored_protocol['code_sha256']
        for path in [Path(__file__), saved_base]:
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(path.name):
                raise ValueError(f'Runner or saved helper differs from saved protocol: {path}')
    base = import_base(saved_base)
    return base


def load_rows(output, configs):
    rows = []
    expected = {c['id']: c for c in configs}
    for path in sorted((output / 'runs').glob('*.json')):
        row = json.loads(path.read_text())
        config = expected.get(row['id'])
        if config is None or any(row.get(k) != v for k, v in config.items()):
            raise ValueError(f'Unexpected cached configuration: {path}')
        if not (output / 'checkpoints' / (row['id'] + '.pt')).exists():
            raise ValueError(f'Missing checkpoint for {row["id"]}')
        rows.append(row)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate saved run IDs')
    return rows


def prepare_datasets(output, protocol):
    base = get_base()
    dataset_dir = output / 'datasets'
    dataset_dir.mkdir(exist_ok=True)
    records = []
    for seed in protocol['dataset_seeds']:
        x, y, replacements, wrong = base.make_data(seed, protocol['p'], protocol['n_train'])
        record = {'dataset_seed': seed, 'x': x[:, 0].tolist(), 'y': y.tolist(),
                  'replacement_count': replacements,
                  **dataset_baselines(x.numpy(), y.numpy(), protocol['p'])}
        assert record['wrong_label_count'] == wrong
        path = dataset_dir / f'{seed}.json'
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError(f'Saved data changed for seed {seed}')
        atomic_json(path, record)
        records.append(record)
    return records


def run_suite(output, protocol, configs):
    base = get_base()
    (output / 'runs').mkdir(exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
    rows = load_rows(output, configs)
    done = {r['id'] for r in rows}
    # Existing helper functions use this module-level protocol. Override in this
    # process only; the old runner, notebook and saved protocol are never edited.
    old_protocol = base.PROTOCOL
    base.PROTOCOL = {**old_protocol, **protocol['training']}
    try:
        for i, config in enumerate(configs):
            if config['id'] in done:
                continue
            started = time.perf_counter()
            x, y, replacements, wrong = base.make_data(config['dataset_seed'], config['p'], protocol['n_train'])
            model = base.make_model(config['width'], config['init_scale'], config['init_seed'])
            fit = base.train_model(model, x, y, config['lr'])
            with torch.no_grad():
                count64 = int((copy.deepcopy(model).double()(x.double()).argmax(1) != y).sum())
            precision_mismatch = count64 != fit['train_error_count']
            w, b, v = coefficients(model)
            geom = base.geometry_stats(w, b, v)
            normalized = normalized_geometry(w, b, v, x.numpy(), y.numpy())
            param = base.parameter_proxies(model, x, y, config['proxy_seed'])
            risk = config['p'] / 2 + (1 - config['p']) * geom['Leb_wrong'] / 2
            row = {**config, **fit, **geom, **normalized, **param,
                   'train_error_count_float64': count64,
                   'precision_mismatch': precision_mismatch,
                   **dataset_baselines(x.numpy(), y.numpy(), config['p']),
                   'replacement_count': replacements, 'population_error': risk,
                   'generalization_gap': risk - fit['train_err'],
                   'elapsed_seconds': time.perf_counter() - started}
            checkpoint = output / 'checkpoints' / (config['id'] + '.pt')
            temp = checkpoint.with_suffix('.pt.tmp')
            torch.save({'config': config, 'state_dict': model.state_dict(), 'x': x, 'y': y}, temp)
            os.replace(temp, checkpoint)
            atomic_json(output / 'runs' / (config['id'] + '.json'), row)
            rows.append(row)
            print(f'{i + 1}/{len(configs)} {config["id"]}: errors={fit["train_error_count"]}, '
                  f'CE={fit["train_ce"]:.4f}, gap={row["generalization_gap"]:.4f}, '
                  f'seconds={row["elapsed_seconds"]:.1f}', flush=True)
    finally:
        base.PROTOCOL = old_protocol
    return rows


def pooled_metrics(rows, target):
    base = get_base()
    out = {}
    for name in measure_names():
        pairs = [(r[name], r[target]) for r in rows if r.get(name) is not None]
        a, b = np.array(pairs, float).T if pairs else ([], [])
        out[name] = {'n_models': len(pairs), 'pearson': base.correlation(a, b),
                     'spearman': base.correlation(base.average_ranks(a), base.average_ranks(b))}
    return out


def interpolation_rates(rows, configs):
    """Marginal success rates; denominator includes failed fits in each group."""
    rates = {}
    for axis in ['width', 'init_scale', 'lr']:
        rates[axis] = []
        for value in sorted({c[axis] for c in configs}):
            group = [r for r in rows if r[axis] == value]
            successes = sum(r['train_error_count'] == 0 for r in group)
            rates[axis].append({'value': value, 'n_fits': len(group),
                                'n_interpolators': successes,
                                'share_interpolating': successes / len(group) if group else None})
    return rates


def summarize(rows, output, protocol, configs):
    expected_per_dataset = len(configs) // len(protocol['dataset_seeds'])
    counts = []
    for seed in protocol['dataset_seeds']:
        group = [r for r in rows if r['dataset_seed'] == seed]
        interp = [r for r in group if r['train_error_count'] == 0]
        complete = len(group) == expected_per_dataset
        counts.append({'dataset_seed': seed, 'n_completed': len(group),
                       'n_expected': expected_per_dataset, 'complete': complete,
                       'n_interpolators': len(interp),
                       'precision_mismatch_count': sum(r.get('precision_mismatch', False) for r in group),
                       'n_ce_target': sum(r['target_reached'] for r in group),
                       'eligible_primary': complete and len(interp) >= protocol['min_models'],
                       'max_training_errors': max((r['train_error_count'] for r in group), default=None),
                       'flips_equal_label_changes_all_interpolators':
                           all(r['nb_flips'] == r['label_changes'] for r in interp) if interp else None,
                       'gap_range_interpolators': [min(r['generalization_gap'] for r in interp),
                                                   max(r['generalization_gap'] for r in interp)] if interp else None})
    complete_seeds = {d['dataset_seed'] for d in counts if d['complete']}
    eligible_seeds = {d['dataset_seed'] for d in counts if d['eligible_primary']}
    complete_rows = [r for r in rows if r['dataset_seed'] in complete_seeds]
    groups = {
        'primary_interpolators': ([r for r in complete_rows if r['train_error_count'] == 0], 'generalization_gap'),
        'secondary_ce_target': ([r for r in complete_rows if r['target_reached']], 'generalization_gap'),
        'secondary_all_models': (complete_rows, 'population_error'),
    }
    analyses = {}
    for name, (group, target) in groups.items():
        within = within_dataset_summary(group, measure_names(), target, protocol['min_models'],
                                        protocol['bootstrap_samples'], protocol['bootstrap_seed'])
        # Include zero-interpolator datasets explicitly in the analysis record.
        present = {d['dataset_seed'] for d in within['per_dataset']}
        within['excluded_empty_dataset_seeds'] = sorted(complete_seeds - present)
        analyses[name] = {'n_models': len(group), 'within_dataset': within,
                          'pooled_descriptive': pooled_metrics(group, target)}
    result = {'complete': len(rows) == len(configs), 'n_completed': len(rows), 'n_expected': len(configs),
              'precision_mismatch_count': sum(r.get('precision_mismatch', False) for r in rows),
              'primary_eligible_dataset_seeds': sorted(eligible_seeds),
              'interpolation_rates': {
                  'all_fits': interpolation_rates(rows, configs),
                  'primary_eligible_datasets': interpolation_rates(
                      [r for r in rows if r['dataset_seed'] in eligible_seeds], configs),
              },
              'datasets': counts, 'analyses': analyses,
              'notes': [
                  'Only completed dataset blocks enter correlations; all attempted fits remain recorded.',
                  'Intervals resample datasets with defined correlations; inspect coverage for each proxy.',
                  'No multiple-comparison correction; measures and thresholds fixed before the suite.',
                  'High pooled correlation can reflect dataset differences. Dataset-only measures cannot rank fits within a dataset.',
                  'Wrong-label count, nearest-neighbour disagreement and model disagreement use the known clean rule.',
                  'Positive normalized training margins suggest better separation: a useful margin may correlate negatively with error.',
                  'Interpolation and training error use float32 counts; float64 count mismatches are reported separately.',
              ]}
    atomic_json(output / 'runs.json', rows)
    atomic_json(output / 'summary.json', result)
    if rows:
        with (output / 'runs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = analyses['primary_interpolators']['within_dataset']['metrics']
    report = [f'Completed {len(rows)}/{len(configs)} fits. Complete suite: {result["complete"]}.',
              f'Float32/float64 training-count mismatches: {result["precision_mismatch_count"]}/{len(rows)} fits. '
              'Float32 counts define interpolation.',
              f'Primary: within-dataset gap Spearman, >= {protocol["min_models"]} interpolators.',
              'Equal dataset weights; 95% percentile bootstrap over informative datasets.',
              'Undefined correlations are not zero. Different measures may have different coverage.',
              'Raw margin bands (band_mass_*) are scale-dependent; normalized bands are listed first.',
              'In the reviewed pilot, all four raw thresholds gave the same within-dataset ranking. '
              'When logits are large away from boundaries, raw bands mostly measure narrow boundary neighborhoods.', '',
              'measure                          mean rho        95% interval   informative / eligible']
    normalized_bands = [f'normalized_band_mass_{g}' for g in GAMMAS]
    report_order = normalized_bands + [name for name in metrics if name not in normalized_bands]
    for name in report_order:
        stat = metrics[name]
        mean = 'undefined' if stat['mean_spearman'] is None else f'{stat["mean_spearman"]:+.3f}'
        ci = 'undefined' if stat['ci95'] is None else f'[{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}]'
        report.append(f'{name:32s} {mean:>9s} {ci:>21s} {stat["n_informative_datasets"]:4d} / {stat["n_eligible_datasets"]}')
    report += ['', 'Per-dataset completion / interpolation / CE target:']
    for d in counts:
        report.append(f'{d["dataset_seed"]}: {d["n_completed"]}/{d["n_expected"]} complete; '
                      f'{d["n_interpolators"]} interpolators; {d["n_ce_target"]} CE target; '
                      f'{d["precision_mismatch_count"]} precision mismatches; '
                      f'primary eligible={d["eligible_primary"]}')
    report += ['', 'Zero-training-error share by hyperparameter (float32; marginal groups).',
               'Denominators include both successful and failed completed fits, not just interpolators.']
    for group_name, label in [('all_fits', 'All completed fits (including incomplete dataset blocks)'),
                              ('primary_eligible_datasets', 'Fits on primary-eligible datasets only')]:
        report += ['', label, 'hyperparameter       value    zero-error / completed    share']
        for axis, entries in result['interpolation_rates'][group_name].items():
            for entry in entries:
                share = 'undefined' if entry['share_interpolating'] is None else f'{entry["share_interpolating"]:.1%}'
                report.append(f'{axis:18s} {entry["value"]:>8g} {entry["n_interpolators"]:8d} / '
                              f'{entry["n_fits"]:<8d} {share:>9s}')
    (output / 'report.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report), flush=True)
    return result


def comparison_figure(rows, output, summary):
    base = get_base()
    candidates = [d for d in summary['datasets'] if d['eligible_primary'] and
                  d['gap_range_interpolators'][1] > d['gap_range_interpolators'][0]]
    selection_file = output / 'boundary_comparison.json'
    figure_file = output / 'boundary_comparison.png'
    if not candidates:
        atomic_json(selection_file, {'available': False, 'reason': 'No eligible dataset with varying gap'})
        figure_file.unlink(missing_ok=True)
        return
    selected = max(candidates, key=lambda d: d['gap_range_interpolators'][1] - d['gap_range_interpolators'][0])
    group = [r for r in rows if r['dataset_seed'] == selected['dataset_seed'] and r['train_error_count'] == 0]
    pair = [min(group, key=lambda r: r['generalization_gap']), max(group, key=lambda r: r['generalization_gap'])]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, constrained_layout=True)
    for ax, row in zip(axes, pair):
        checkpoint = torch.load(output / 'checkpoints' / (row['id'] + '.pt'), weights_only=True)
        model = base.make_model(row['width'], row['init_scale'], row['init_seed'])
        model.load_state_dict(checkpoint['state_dict'])
        w, b, v = coefficients(model)
        bounds, values = affine_endpoints(w, b, v)
        for left, right, a, z in zip(bounds[:-1], bounds[1:], values[:-1], values[1:]):
            cuts = [left, right]
            if left < 0 < right:
                cuts.append(0.)
            if a * z < 0:
                cuts.append(left + (right - left) * (-a / (z - a)))
            cuts = sorted(cuts)
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                mid = (lo + hi) / 2
                delta = a + (z - a) * (mid - left) / (right - left)
                if (delta > 0) != (mid > 0):
                    ax.axvspan(lo, hi, color='#c65a47', alpha=.2, linewidth=0)
        displayed = values / row['logit_rms'] if row['logit_rms'] > 0 else np.zeros_like(values)
        ax.plot(bounds, displayed, color='#334e68', linewidth=1.4)
        x, y = checkpoint['x'].numpy().reshape(-1), checkpoint['y'].numpy()
        for label, color, marker in [(0, '#3569a8', 'o'), (1, '#ca7520', '^')]:
            ax.scatter(x[y == label], np.zeros(np.sum(y == label)), color=color, marker=marker,
                       s=26, edgecolors='white', linewidth=.4, zorder=5, label=f'training label {label}')
        ax.axhline(0, color='black', linewidth=.6)
        ax.axvline(0, color='black', linestyle=':', linewidth=.8)
        ax.set(ylabel='Logit difference / RMS', xlim=(-1, 1),
               title=f'Gap {row["generalization_gap"]:.3f}; {row["nb_flips"]} decision flips; '
                     f'width {row["width"]}, scale {row["init_scale"]}, lr {row["lr"]}, draw {row["init_repeat"]}')
    axes[0].legend(loc='best', fontsize=8)
    axes[-1].set_xlabel('x; shaded intervals disagree with the clean rule')
    fig.suptitle(f'Same training data (seed {selected["dataset_seed"]}); two interpolating fits')
    fig.savefig(figure_file, dpi=180)
    plt.close(fig)
    atomic_json(selection_file, {'available': True, 'dataset_seed': selected['dataset_seed'],
                                'run_ids': [r['id'] for r in pair],
                                'selection': 'largest gap range among primary-eligible datasets; its minimum/maximum gap fits',
                                'interpretation': 'Illustration selected for contrast, not an independent statistical test'})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset-count', type=int, default=20)
    parser.add_argument('--dataset-seed-start', type=int, default=1000)
    parser.add_argument('--widths', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--initialization-repeats', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=200000)
    parser.add_argument('--check-every', type=int, default=50)
    parser.add_argument('--min-models', type=int, default=8)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_protocol(args):
    base = get_base()
    protocol = {
        'version': 2, 'p': .4, 'n_train': 32,
        'dataset_seeds': list(range(args.dataset_seed_start, args.dataset_seed_start + args.dataset_count)),
        'widths': args.widths, 'initialization_repeats': args.initialization_repeats,
        'initialization_scales': [.5, 2.], 'learning_rates': [.003, .01],
        'training': {k: base.PROTOCOL[k] for k in ['target_cross_entropy', 'relative_sigma',
                    'perturbation_samples', 'sam_radius', 'hessian_iterations']},
        'min_models': args.min_models, 'bootstrap_samples': args.bootstrap_samples,
        'bootstrap_seed': 20261010, 'normalized_band_thresholds': list(GAMMAS),
        'normalization': 'RMS of logit difference under Uniform[-1,1], integrated analytically',
        'failure_policy': 'same raised budget for all; keep failures and never replace datasets',
        'interpolation_precision': 'float32; float64 count mismatches recorded without changing selection',
        'checkpoint': base.PROTOCOL['checkpoint'], 'sampling': base.PROTOCOL['sampling'],
        'code_sha256': code_hashes(),
    }
    protocol['training'].update(max_steps=args.max_steps, check_every=args.check_every)
    return protocol


def main():
    args = parse_args()
    if (min(args.dataset_count, args.initialization_repeats, args.max_steps, args.check_every,
            args.bootstrap_samples, *args.widths) < 1 or args.min_models < 3 or
            args.dataset_seed_start < 0 or len(set(args.widths)) != len(args.widths)):
        raise ValueError('Counts, steps and widths must be positive; min-models >=3; distinct widths; seed >=0')
    output = args.output.resolve()
    protocol_file = output / 'protocol.json'
    if args.dry_run:
        if protocol_file.exists():
            stored = json.loads(protocol_file.read_text())
            bind_saved_base(output, stored)
            protocol = stored if args.analyze_only else build_protocol(args)
            if protocol != stored:
                raise ValueError('Different protocol or source code; choose a fresh output directory')
        else:
            if args.analyze_only:
                raise FileNotFoundError(protocol_file)
            protocol = build_protocol(args)
        configs = list(configurations(protocol))
        print(json.dumps({'n_fits': len(configs), 'output': str(output), 'protocol': protocol}, indent=2))
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with output_lock(output):
        if protocol_file.exists():
            stored = json.loads(protocol_file.read_text())
            bind_saved_base(output, stored)
            protocol = stored if args.analyze_only else build_protocol(args)
            if stored != protocol:
                raise ValueError('Different protocol or source code; choose a fresh output directory')
        else:
            if args.analyze_only:
                raise FileNotFoundError(protocol_file)
            if any(p.name != '.runner.lock' for p in output.iterdir()):
                raise ValueError('Nonempty output directory without protocol')
            bind_saved_base(output)
            protocol = build_protocol(args)
            atomic_json(protocol_file, protocol)
        configs = list(configurations(protocol))
        environment = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                       'platform': platform.platform(), 'device': 'cpu', 'torch_threads': 1,
                       'training_dtype': 'float32', 'geometry_and_proxy_dtype': 'float64'}
        environment_file = output / 'environment.json'
        if environment_file.exists() and json.loads(environment_file.read_text()) != environment:
            raise ValueError('Runtime environment changed; use a fresh output directory')
        atomic_json(environment_file, environment)
        prepare_datasets(output, protocol)
        try:
            rows = load_rows(output, configs) if args.analyze_only else run_suite(output, protocol, configs)
        except KeyboardInterrupt:
            print('\nInterrupted; completed fits are saved. Summarizing complete dataset blocks.', flush=True)
            summarize(load_rows(output, configs), output, protocol, configs)
            return
        summary = summarize(rows, output, protocol, configs)
        comparison_figure(rows, output, summary)


if __name__ == '__main__':
    main()
