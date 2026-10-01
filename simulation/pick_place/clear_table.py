"""Table clearing: find the objects on the table, then pick them one by one into the basket.

    python -m simulation.pick_place.clear_table                       # headless, gt 6D pose
    python -m simulation.pick_place.clear_table --pose-backend foundationpose
    python -m simulation.pick_place.clear_table --viewer              # watch it in MuJoCo

Loop, until the camera sees nothing left on the table:
    1. look      RGB-D from the head camera -> ObjectDetector: every object on the
                 table, identified (no simulator state), with its pixel mask
    2. track     ObjectTracker: the same object keeps its id across looks
    3. choose    the next object: nearest the basket first (shortest carry); one the
                 planner cannot reach is skipped for now and tried again later
    4. pick      Demo on that object: 6D pose (pose backend, given the detector's
                 mask) -> grasp library -> plan -> grasp -> proof lift -> carry ->
                 drop into a free spot of the basket
    5. verify    look again: the object must be gone from the table (the detector
                 excludes the basket); the scene's own check says whether it landed
                 in the basket (reported, not used to decide)

Each object gets MAX_ATTEMPTS picks; the run stops when the table is clear, or when
every object left has used its attempts.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement
from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.scene import Scene

TASK_OBJECTS = ("can", "tuna_can", "gelatin_box", "orange")
PLATFORM = 0.10
BASKET = (0.28, 0.0)
# Default layout on the right arm's side of the work platform (footprint centres),
# >= 3 cm apart and clear of the basket and the resting fist. The first entry is the
# scene's initial pick object.
# Close together inside the right arm's reach on the work platform, each object on a
# spot (and heading) it was measured graspable from on its own (reach probe and
# grasp-library sweep, 2026-09-28); 2-3.5 cm between neighbouring footprints. The
# gelatin box is the most constrained: its jaw must cross the short side square.
LAYOUT = (
    Placement("gelatin_box", (0.317, -0.188), "upright", 0.0),
    Placement("tuna_can", (0.24, -0.29)),
    Placement("orange", (0.28, -0.40)),
    Placement("can", (0.40, -0.30)),
)
# Drop spots in the basket (offsets from its centre, m): the four quarters of the
# 180 mm floor, the arm's own side first.
BASKET_SLOTS = ((-0.04, -0.045), (0.04, -0.045), (-0.04, 0.045), (0.04, 0.045))
MAX_ATTEMPTS = 2


@dataclass
class PickRecord:
    track_id: int
    label: str
    attempt: int
    success: bool
    in_basket: bool
    gone_from_table: bool
    failure: str | None
    failed_phase: str | None
    grasp_name: str | None
    pose_error_mm: float | None
    pose_rotation_error_deg: float | None
    slot: tuple[float, float]
    sim_seconds: float
    wall_seconds: float


@dataclass
class TaskResult:
    pose_backend: str
    objects: list[str]
    cleared: list[str] = field(default_factory=list)
    left_on_table: list[str] = field(default_factory=list)
    picks: list[PickRecord] = field(default_factory=list)
    looks: list[dict] = field(default_factory=list)
    in_basket_final: dict[str, bool] = field(default_factory=dict)
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0

    def summary(self) -> str:
        attempts = len(self.picks)
        return (f"cleared {len(self.cleared)}/{len(self.objects)} objects in {attempts} pick(s); "
                f"in basket at the end: {sum(self.in_basket_final.values())}/{len(self.objects)}; "
                f"left on table: {self.left_on_table or 'none'}; sim {self.sim_seconds:.0f}s, wall {self.wall_seconds:.0f}s")


class TableClearing:
    def __init__(self, layout=LAYOUT, basket=BASKET, *, pose_backend: str = "gt", side: str = "right",
                 verbose: bool = False) -> None:
        layout = list(layout)
        self.scene = Scene(
            layout[0].xy, basket, work_platform_height=PLATFORM, pick_object=layout[0].key,
            pick_pose=layout[0].pose, pick_yaw_deg=layout[0].yaw_deg, extra_objects=layout[1:],
        )
        self.objects = [p.key for p in layout]
        self.pose_backend = pose_backend
        self.side = side
        self.verbose = verbose
        self.detector = ObjectDetector(self.scene.model, TASK_OBJECTS, basket_xy=basket)
        self.tracker = ObjectTracker()
        self.basket = basket
        self.viewer = None

    # ------------------------------------------------------------------ perception
    def enroll(self) -> None:
        """Reference features of each task object, seen alone at a typical spot."""
        def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
            return Scene((0.28, -0.25), self.basket, work_platform_height=PLATFORM, pick_object=key,
                         pick_pose=pose, pick_yaw_deg=yaw)

        self.detector.enroll(alone)

    def look(self, result: TaskResult) -> list:
        detections = self.detector.detect(self.scene.data)
        tracks = self.tracker.update(detections)
        by_track = {d.track_id: d for d in detections if d.track_id is not None}
        unknown = [d for d in detections if d.label is None]
        entry = {
            "t": round(float(self.scene.data.time), 2),
            "seen": [{"track": t.track_id, "label": t.label, "xyz": np.round(t.position, 3).tolist(),
                      "cost": round(by_track[t.track_id].cost, 2)} for t in tracks],
            "unidentified": [{"xyz": np.round(d.centroid, 3).tolist(), "cost": round(d.cost, 1)} for d in unknown],
        }
        result.looks.append(entry)
        print(f"[look t={entry['t']:.1f}s] on the table: "
              + (", ".join(f"#{s['track']} {s['label']} at {s['xyz'][:2]}" for s in entry["seen"]) or "nothing")
              + (f"; {len(unknown)} unidentified" if unknown else ""))
        return [(t, by_track[t.track_id]) for t in tracks]

    # ------------------------------------------------------------------ one pick
    def pick(self, track, detection, slot) -> tuple[bool, dict]:
        scene = self.scene
        scene.set_target(track.label)
        demo = Demo(scene=scene, side=self.side, pose_backend=self.pose_backend, release="drop",
                    place_offset=slot, detection_mask=detection.mask, verbose=self.verbose)
        started = time.perf_counter()
        trial = run_trial(demo, self.viewer)
        return trial, time.perf_counter() - started

    # ------------------------------------------------------------------ the task
    def run(self) -> TaskResult:
        started = time.perf_counter()
        result = TaskResult(self.pose_backend, list(self.objects))
        print("enrolling the task objects (reference features) ...")
        self.enroll()
        attempts: dict[int, int] = {}
        # Objects the planner found no reachable grasp for, in the table's current
        # state: skipped until something else has been cleared (a neighbour may have
        # been in the way); no physical attempt is spent on them.
        deferred: set[int] = set()
        slots = list(BASKET_SLOTS)
        while True:
            seen = self.look(result)
            candidates = [(t, d) for t, d in seen
                          if attempts.get(t.track_id, 0) < MAX_ATTEMPTS and t.track_id not in deferred]
            if not candidates:
                break
            # Shortest carry first.
            candidates.sort(key=lambda td: float(np.hypot(td[0].position[0] - self.basket[0], td[0].position[1] - self.basket[1])))
            track, detection = candidates[0]
            slot = slots[0] if slots else (0.0, 0.0)
            print(f"== pick #{track.track_id} {track.label} (attempt {attempts[track.track_id]}) -> basket spot {slot}")
            trial, wall = self.pick(track, detection, slot)
            if trial.failed_phase in ("perceive", "plan"):
                deferred.add(track.track_id)
                reason = (trial.failure_reason or "").splitlines()
                print(f"   -> no reachable grasp now; deferred ({reason[0] if reason else ''})")
                for line in reason[1:6]:
                    print(f"      {line.strip()[:160]}")
                result.picks.append(PickRecord(
                    track.track_id, track.label, attempts.get(track.track_id, 0), False, False, False,
                    reason[0][:200] if reason else None, trial.failed_phase, trial.grasp_name, None, None,
                    tuple(slot), round(trial.simulation_seconds, 1), round(wall, 1)))
                continue
            attempts[track.track_id] = attempts.get(track.track_id, 0) + 1
            after = self.look(result)
            gone = all(t.track_id != track.track_id for t, _ in after)
            in_basket = self.scene.object_in_basket(track.label)
            ok = gone and in_basket
            if ok:
                result.cleared.append(track.label)
                deferred.clear()  # the table changed: deferred objects get another look
                if slots:
                    slots.pop(0)
            record = PickRecord(
                track.track_id, track.label, attempts[track.track_id], ok, in_basket, gone,
                trial.failure_reason.splitlines()[0][:200] if trial.failure_reason else None, trial.failed_phase,
                trial.grasp_name, None if trial.perception_error_m is None else round(trial.perception_error_m * 1000, 1),
                trial.pose_rotation_error_deg, tuple(slot), round(trial.simulation_seconds, 1), round(wall, 1),
            )
            result.picks.append(record)
            print(f"   -> {'OK' if ok else 'FAILED'}: in basket {in_basket}, gone from table {gone}"
                  + (f"; {record.failure}" if record.failure else ""))
            # Put the arm back at attention if the episode ended elsewhere (a failed pick).
            self._home()
        result.left_on_table = [t.label for t, _ in self.look(result)]
        result.in_basket_final = {key: self.scene.object_in_basket(key) for key in self.objects}
        result.sim_seconds = float(self.scene.data.time)
        result.wall_seconds = time.perf_counter() - started
        return result

    def _home(self) -> None:
        """After a failed pick the arm may be anywhere: open the hand and blend back to
        the attention stance (slowly, from wherever it is)."""
        from simulation.pick_place import config as C
        from simulation.pick_place.executor import Executor

        scene, side = self.scene, self.side
        here = scene.data.qpos[scene.arm_qpos[side]]
        if np.max(np.abs(here - scene.attention_pose[side])) < 0.05:
            return
        executor = Executor(scene)
        executor.viewer = self.viewer
        executor.active_side = side
        executor.move_to({f"{side}_hand": scene.hand_ctrl(side, open_fingers=C.ALL_FINGERS)}, C.RELEASE_SECONDS)
        executor.move_to({f"{side}_arm": scene.attention_pose[side], f"{side}_hand": scene.rest_hand[side]}, 2.0 * C.RETURN_SECONDS)
        executor.hold(C.SETTLE_AT_START)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect the objects on the table and clear them into the basket")
    parser.add_argument("--pose-backend", default="gt", choices=("gt", "foundationpose"))
    parser.add_argument("--viewer", action="store_true", help="watch in the MuJoCo viewer")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("artifacts") / "clear_table.json")
    args = parser.parse_args(argv)
    task = TableClearing(pose_backend=args.pose_backend, verbose=args.verbose)
    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(task.scene.model, task.scene.data) as viewer:
            task.viewer = viewer
            result = task.run()
    else:
        result = task.run()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(result), indent=2, default=float), encoding="utf-8")
    print("\n" + result.summary())
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
