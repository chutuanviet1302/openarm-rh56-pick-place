"""PickCell: fill one order with the two-arm cell -- table first, then the conveyor.

    python -m product.cell --order orders/demo_001.json                    # fixed layout, gt pose
    python -m product.cell --order orders/demo_002.json --pose-backend foundationpose
    python -m product.cell --order orders/demo_001.json --seed 3           # a random layout

Only what the order still needs is picked: the camera identifies every object on the
table and on the belt (all of them are shown in the replay), the cell takes the wanted
ones, leaves the rest on the table and lets the rest ride past on the belt. The belt
phase is skipped once the table filled the order, and ends early as soon as the
order is complete.

Writes runs/<order_id>_<yyyymmdd-hhmmss>/:
    order.json          the order as received
    frames.npz          the recording (replay: python -m product.cell --replay <run dir>)
    report.json         per-pick record, in-box audit, sim/wall seconds
    order_result.json   lines requested / picked / missing, mispicks, status
    metrics.json        scripts.task_metrics: timing, joint margin, smoothness, contacts
    kpi.json            product.kpi: the KPIs with their sources
    dashboard.html      product.dashboard (+ replay.gif, detections.png); runs/index.html lists runs
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from product.order import Order, OrderBook
from simulation.pick_place.bin_conveyor_task import BinConveyorTask

RUNS = Path("runs")


class OrderTask(BinConveyorTask):
    """The box + conveyor task restricted to what an order asks for."""

    def __init__(self, order: Order, **kwargs) -> None:
        self.book = OrderBook(order)
        super().__init__(**kwargs)

    def wanted(self, label: str | None) -> bool:
        return self.book.wanted(label)

    def on_pick(self, name: str, ok: bool) -> None:
        self.book.record(name, self.scene.object_types[name], ok)
        need = ", ".join(f"{k} x{self.book.remaining(k)}" for k in self.book.need if self.book.remaining(k) > 0)
        print(f"   order {self.book.order.order_id}: {'complete' if self.book.complete() else 'still needs ' + need}")

    def belt_done(self) -> bool:
        return self.book.complete()

    def after_table(self, result) -> None:
        if self.book.complete():
            print("\n=== order complete from the table: the belt is not needed ===")
            for demo in self.arm_over.values():
                demo.return_home()
            self.arm_over.clear()
            return
        super().after_table(result)


def run_order(order: Order, *, pose_backend: str = "gt", seed: int | None = None, motion: str = "waypoints",
              runs: Path = RUNS) -> Path:
    """Fill `order`; returns the run directory."""
    from scripts.task_metrics import measure

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = runs / f"{order.order_id}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "order.json").write_text(order.to_json(), encoding="utf-8")
    layout = None
    if seed is not None:
        from simulation.pick_place.layout import sample_layout

        print(f"drawing layout for seed {seed} ...", flush=True)
        layout = sample_layout(seed)
        (out / "layout.json").write_text(layout.to_json(), encoding="utf-8")
    task = OrderTask(order, pose_backend=pose_backend, motion=motion, layout=layout)
    frames = out / "frames.npz"
    try:
        result = task.run()
    finally:
        if layout is not None:
            task.recorder.extra["layout"] = np.asarray(layout.to_json())
        task.recorder.save(frames)
    report = asdict(result)
    report.update({"order_id": order.order_id, "seed": seed, "motion": motion})
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    outcome = task.book.result(result.in_box, dict(task.scene.object_types))
    (out / "order_result.json").write_text(json.dumps(asdict(outcome), indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = measure(frames, "bin_conveyor", out / "report.json")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    from product.dashboard import build

    print(f"dashboard: {build(out)}", flush=True)
    print(f"\norder {order.order_id}: {outcome.status.upper()}  "
          + ", ".join(f"{l.sku} {l.picked}/{l.requested}" for l in outcome.lines)
          + (f"; mispicks {outcome.mispicks}" if outcome.mispicks else "")
          + f"; sim {result.sim_seconds:.0f}s, wall {result.wall_seconds:.0f}s")
    print(f"run: {out}", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PickCell: fill an order with the two-arm cell (MuJoCo twin)")
    parser.add_argument("--order", type=Path, help="order JSON: {\"order_id\": ..., \"items\": {sku: qty}}")
    parser.add_argument("--pose-backend", default="gt", choices=("gt", "foundationpose"))
    parser.add_argument("--seed", type=int, help="random layout (default: the fixed demo layout)")
    parser.add_argument("--motion", default="waypoints", choices=("waypoints", "mink"))
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--replay", type=Path, help="play a run directory's recording in MuJoCo and exit")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--quality", default="high", choices=("high", "fast"))
    args = parser.parse_args(argv)
    if args.replay:
        from simulation.pick_place.bin_conveyor_task import _builder_for
        from simulation.pick_place.bin_task import replay

        replay(args.replay / "frames.npz", args.speed, scene_builder=_builder_for(args.replay / "frames.npz"),
               quality=args.quality)
        return
    if args.order is None:
        parser.error("--order is required (or --replay)")
    run_order(Order.load(args.order), pose_backend=args.pose_backend, seed=args.seed, motion=args.motion, runs=args.runs)


if __name__ == "__main__":
    main()
