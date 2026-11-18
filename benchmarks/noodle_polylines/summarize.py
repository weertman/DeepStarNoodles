#!/usr/bin/env python3
"""Markdown tables from bench.py JSON reports (for ROUNDS.md).

    python benchmarks/noodle_polylines/summarize.py bench_round.json [--baseline bench_v0.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    r = json.loads(path.read_text())
    r['by'] = {(x['input'], x['kernel']): x for x in r['results']}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('report', type=Path)
    ap.add_argument('--baseline', type=Path, default=None)
    args = ap.parse_args()
    r = load(args.report)
    b = load(args.baseline) if args.baseline else None
    inputs = list(r['inputs'])
    sizes = {}
    for x in r['results']:
        sizes.setdefault(x['input'], (x['arms'], x['frames']))
    print('| input | arms | frames |\n|---|---:|---:|')
    for i in inputs:
        print(f'| {i} | {sizes[i][0]:,} | {sizes[i][1]:,} |')
    kernels = list(dict.fromkeys(x['kernel'] for x in r['results']))
    head = ' | '.join(f'{i} us/arm' + (' (x)' if b else '') for i in inputs)
    print(f'\n| kernel | {head} | peak RSS MB | CPU-h / camera-year |')
    print('|---|' + '---:|' * (len(inputs) + 2))
    for k in kernels:
        cells = []
        rss = 0.0
        for i in inputs:
            x = r['by'].get((i, k))
            if x is None:
                cells.append('-')
                continue
            rss = max(rss, x['peak_rss_mb'] or 0)
            cell = f"{x['us_per_arm']:.2f}"
            if b and (i, k) in b['by']:
                cell += f" ({b['by'][(i, k)]['us_per_arm'] / x['us_per_arm']:.1f}x)"
            cells.append(cell)
        proj = r['projection'].get(k, {}).get('cpu_h_per_camera_year')
        print(f"| {k} | {' | '.join(cells)} | {rss:.0f} | {'' if proj is None else f'{proj:.1f}'} |")


if __name__ == '__main__':
    main()
