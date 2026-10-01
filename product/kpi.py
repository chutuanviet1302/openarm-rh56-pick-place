"""KPIs of one PickCell run, each with where it comes from (file and field).

    kpis = compute(run_dir)        # reads order_result.json, report.json, metrics.json

Definitions (the definition is the metric):
    order lines filled   lines with missing == 0 / lines (audit)
    mispicks             objects in the tote the order did not ask for (audit)
    pick cycle           sim seconds from the start of a pick ("<arm> -> kind (name)" caption)
                         to the end of its outcome caption ("... in the box" / "missed":
                         settling and the way home) -- not the waiting for the belt after it
    items per hour       objects of the order in the tote / sim seconds x 3600
    safety               contact frames by kind and deepest penetration (task_metrics),
                         smallest arm-joint distance to a range end, jerk RMS, stops
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PICK = re.compile(r"->\s*(\S+)\s*\((\w+)\)")


def _load(run: Path, name: str) -> dict:
    return json.loads((run / name).read_text(encoding="utf-8"))


def pick_cycles(segments: list[dict], end_s: float) -> list[dict]:
    cycles = []
    for index, segment in enumerate(segments):
        found = PICK.search(segment["caption"])
        if not found:  # segments are runs of one caption: each "->" run starts one pick (a retry too)
            continue
        end = segment["start_s"] + segment["seconds"]
        after = segments[index + 1] if index + 1 < len(segments) else None
        if after is not None and found.group(2) in after["caption"] and not PICK.search(after["caption"]):
            end = after["start_s"] + after["seconds"]   # the outcome caption: settle, way home
        cycles.append({"name": found.group(2), "kind": found.group(1), "caption": segment["caption"],
                       "start_s": segment["start_s"], "seconds": round(min(end, end_s) - segment["start_s"], 1)})
    return cycles


def compute(run: Path) -> dict:
    run = Path(run)
    order, result = _load(run, "order.json"), _load(run, "order_result.json")
    report, metrics = _load(run, "report.json"), _load(run, "metrics.json")
    lines = result["lines"]
    filled = sum(1 for line in lines if line["missing"] == 0)
    in_tote = sum(line["picked"] for line in lines)
    sim_s = float(report["sim_seconds"])
    cycles = pick_cycles(metrics["segments"], float(metrics["sim_seconds"]))
    contacts = metrics["contacts"]
    joints = metrics["joints"]
    return {
        "order_id": order["order_id"],
        "status": {"value": result["status"], "source": "order_result.json:status"},
        "lines_filled": {"value": f"{filled}/{len(lines)}", "source": "order_result.json:lines[].missing"},
        "items_in_tote": {"value": f"{in_tote}/{sum(l['requested'] for l in lines)}", "source": "order_result.json:lines[].picked"},
        "mispicks": {"value": len(result["mispicks"]), "detail": result["mispicks"], "source": "order_result.json:mispicks"},
        "failed_attempts": {"value": len(result["failed_attempts"]), "detail": result["failed_attempts"],
                            "source": "order_result.json:failed_attempts"},
        "sim_seconds": {"value": round(sim_s, 1), "source": "report.json:sim_seconds"},
        "wall_seconds": {"value": round(float(report["wall_seconds"]), 1), "source": "report.json:wall_seconds"},
        "items_per_hour_sim": {"value": round(in_tote / sim_s * 3600.0, 1) if sim_s > 0 else None,
                               "source": "items_in_tote / report.json:sim_seconds"},
        "pick_cycles": {"value": cycles, "source": "metrics.json:segments (captions)"},
        "contacts": {"value": {k: {"frames": v["frames"], "deepest_mm": v["deepest_mm"]} for k, v in contacts.items()},
                     "source": "metrics.json:contacts"},
        "collision_free": {"value": all(v["frames"] == 0 for v in contacts.values()), "source": "metrics.json:contacts[].frames"},
        "min_joint_margin_deg": {"value": min(j["min_joint_margin_deg"] for j in joints.values()),
                                 "source": "metrics.json:joints[].min_joint_margin_deg"},
        "rms_jerk_rad_s3": {"value": {s: j["rms_jerk_rad_s3"] for s, j in joints.items()}, "source": "metrics.json:joints[].rms_jerk_rad_s3"},
        "stops": {"value": {s: j["stops"] for s, j in joints.items()}, "source": "metrics.json:joints[].stops"},
        "pose_backend": report.get("pose_backend"),
        "layout": "random, seed %s" % report["seed"] if report.get("seed") is not None else "fixed demo layout",
    }
