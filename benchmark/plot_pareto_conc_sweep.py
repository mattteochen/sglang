#!/usr/bin/env python3
"""Plot a Pareto curve (Output/Total TPS-per-GPU vs per-user Output TPS) from
one or more `bench_serving` concurrency-sweep JSON files.

Each input is expected to be a JSON array where every element is the result of
a single `sglang.bench_serving` run (as produced by
`benchmark/bench_serving_glm5_conc_sweep.sh`). We use these fields:

  - max_concurrency           (annotated next to each point)
  - output_throughput         (total output tokens/sec across the server)
  - total_throughput          (input+output tokens/sec)
  - mean_tpot_ms              (mean time-per-output-token, ms)
  - server_info.tp_size       (number of GPUs the server runs on)

The two axes plotted are:
  - X: Output TPS/user  = 1000 / mean_tpot_ms
  - Y: Output TPS/GPU   = output_throughput / tp_size   (left subplot)
  - Y: Total  TPS/GPU   = total_throughput  / tp_size   (right subplot)

Usage:
  python benchmark/plot_pareto_conc_sweep.py \
      --input base_out/bench_glm5_fp8_conc_sweep_20260423_075336.json:baseline \
      --input compiled_out/bench_glm5_fp8_conc_sweep_20260423_072646.json:compiled \
      --output pareto.png

The `:<label>` suffix on each --input is optional; if omitted the file stem is
used as the series label.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class Point:
    concurrency: int
    output_tps_per_user: float
    output_tps_per_gpu: float
    total_tps_per_gpu: float


@dataclass
class Series:
    label: str
    path: str
    points: List[Point]


def _parse_input_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse `path[:label]`. A colon inside a Windows drive (`C:\\…`) is kept."""
    if ":" in spec and not (len(spec) > 1 and spec[1] == ":"):
        path, label = spec.rsplit(":", 1)
        return path, label
    return spec, None


def _load_runs(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # default json accepts Infinity / NaN literals
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a top-level JSON array of runs")
    return data


def _extract_points(runs: Sequence[dict]) -> List[Point]:
    points: List[Point] = []
    for run in runs:
        tpot = run.get("mean_tpot_ms")
        out_tp = run.get("output_throughput")
        tot_tp = run.get("total_throughput")
        conc = run.get("max_concurrency")
        tp_size = run.get("server_info", {}).get("tp_size")
        if None in (tpot, out_tp, tot_tp, conc, tp_size) or tpot <= 0 or tp_size <= 0:
            continue
        points.append(
            Point(
                concurrency=int(conc),
                output_tps_per_user=1000.0 / float(tpot),
                output_tps_per_gpu=float(out_tp) / float(tp_size),
                total_tps_per_gpu=float(tot_tp) / float(tp_size),
            )
        )
    points.sort(key=lambda p: p.concurrency)
    return points


def _load_series(specs: Sequence[str]) -> List[Series]:
    series: List[Series] = []
    for spec in specs:
        path, label = _parse_input_spec(spec)
        if label is None:
            label = os.path.splitext(os.path.basename(path))[0]
        runs = _load_runs(path)
        points = _extract_points(runs)
        if not points:
            raise ValueError(f"{path}: no valid benchmark rows found")
        series.append(Series(label=label, path=path, points=points))
    return series


def _plot(series: Sequence[Series], output_path: str, title: str) -> None:
    fig, (ax_out, ax_tot) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, s in enumerate(series):
        color = colors[idx % len(colors)]
        xs = [p.output_tps_per_user for p in s.points]
        y_out = [p.output_tps_per_gpu for p in s.points]
        y_tot = [p.total_tps_per_gpu for p in s.points]

        for ax, ys in ((ax_out, y_out), (ax_tot, y_tot)):
            ax.plot(xs, ys, linestyle="--", marker="o", color=color, label=s.label)
            for p, x, y in zip(s.points, xs, ys):
                ax.annotate(
                    str(p.concurrency),
                    xy=(x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

    for ax, ylabel in ((ax_out, "Output TPS/GPU"), (ax_tot, "Total TPS/GPU")):
        ax.set_xlabel("Output TPS/user (1000 / TPOT)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {output_path}")


def _print_table(series: Sequence[Series]) -> None:
    header = (
        f"{'series':<32} {'conc':>5} {'TPS/user':>10} "
        f"{'out TPS/GPU':>12} {'total TPS/GPU':>14}"
    )
    print(header)
    print("-" * len(header))
    for s in series:
        for p in s.points:
            print(
                f"{s.label[:32]:<32} {p.concurrency:>5d} "
                f"{p.output_tps_per_user:>10.2f} "
                f"{p.output_tps_per_gpu:>12.2f} "
                f"{p.total_tps_per_gpu:>14.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="PATH[:LABEL]",
        help="Benchmark JSON file; repeat for multiple series. "
        "Append `:label` to set the legend name.",
    )
    parser.add_argument(
        "--output",
        default="pareto.png",
        help="Output image path (default: pareto.png).",
    )
    parser.add_argument(
        "--title",
        default="Pareto Frontier",
        help="Figure suptitle (default: 'Pareto Frontier').",
    )
    args = parser.parse_args()

    series = _load_series(args.input)
    _print_table(series)
    _plot(series, args.output, args.title)


if __name__ == "__main__":
    main()
