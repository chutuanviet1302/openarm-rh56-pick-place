"""Task layouts for the box + conveyor task: what is fixed, what is derived, what is drawn.

Fixed (the lab's fixtures): the work platform, the belt (x, width) and the robot.
Derived from the fixtures' geometry (no hand-placed numbers):
    box       stands on the platform (its robot-side wall at the platform's edge) and
              keeps PATH_CLEARANCE from the belt strip; inner x half-size follows.
    drop      candidate drop spots inside the box, ordered by distance from the spots
              this arm already dropped at (the robot remembers where it let go; the
              simulator's object positions are not read).
Drawn per seed (sample_layout):
    table     objects (kind and rest pose from the grasp library's known poses, yaw,
              position on the platform between the box and the belt-free edge),
              rejected unless spacing holds AND the real planner finds grasp -> carry
              -> drop for at least one arm (gt pose, the full scene as obstacles)
    belt      objects upstream on the belt, spaced so one pick cycle fits between two
              arrivals (cycle time measured, passed in)

    layout = sample_layout(seed=3)
    scene = layout.scene()
    layout.to_json() / Layout.from_json(text)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from simulation.five_finger_model import BASKET_WALL_THICKNESS, WORK_PLATFORM_X, Conveyor
from simulation.objects import Placement, footprint_radius
from simulation.pick_place import config as C
from simulation.pick_place.scene import Scene

PLATFORM_HEIGHT = 0.10
BELT = Conveyor(x=0.38, width=0.08)
BOX_WALL_HEIGHT = 0.08
# Inner half-length of the box along y: the size the six-object task has used, all
# six fitting in it (runs 2026-09-29: 6/6). Capacity, not a reach number.
BOX_HALF_Y = 0.20
# Kinds and rest poses the table objects are drawn from (each has grasp-library entries
# and a detector reference); belt objects stand upright.
TABLE_KINDS = (("can", "upright"), ("can", "lying"), ("peach", "upright"), ("orange", "upright"), ("apple", "upright"))
BELT_KINDS = (("can", "upright"), ("orange", "upright"), ("apple", "upright"), ("peach", "upright"))
DROP_GRID_STEP = 0.01
# Half the hand's narrowest extent per hand (m), measured on the model by
# hand_half_width() in each palm's own frame (tests/test_layout_sampler.py checks the
# numbers still match the model). The room the fingers need beside an object: kept
# between objects and between an object and the box wall -- 2 cm (PATH_CLEARANCE) let
# the right index finger, wrapping a lying can, touch the wall (0.56 mm, 2026-10-01).
HAND_HALF_WIDTH = {"right": 0.0407, "left": 0.0542}
# A plan check that runs longer is called a failure: the slowest *successful* plan in
# the object reach map took 83.4 s (p95 65 s; failures ended within 33.4 s) --
# artifacts/benchmarks/reach_map_objects.json, 109 + 12 cells, subprocess start included.
PLAN_CHECK_TIMEOUT_S = 90.0


@dataclass
class Layout:
    seed: int | None
    box_xy: tuple[float, float]
    box_half: tuple[float, float]
    table: list[Placement] = field(default_factory=list)
    belt: list[Placement] = field(default_factory=list)
    checks: dict = field(default_factory=dict)        # per table object: arms that plan it, seconds

    def scene(self) -> Scene:
        objects = list(self.table) + list(self.belt)
        first = objects[0]
        return Scene(first.xy, self.box_xy, work_platform_height=PLATFORM_HEIGHT, pick_object=first.key,
                     pick_pose=first.pose, pick_yaw_deg=first.yaw_deg, extra_objects=objects[1:],
                     basket_half_size=self.box_half, basket_wall_height=BOX_WALL_HEIGHT, conveyor=BELT)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Layout":
        raw = json.loads(text)

        def placements(rows):
            return [Placement(r["key"], tuple(r["xy"]), r["pose"], r["yaw_deg"], r["name"]) for r in rows]
        return cls(raw["seed"], tuple(raw["box_xy"]), tuple(raw["box_half"]), placements(raw["table"]),
                   placements(raw["belt"]), raw.get("checks", {}))


def hand_half_width(scene: Scene, side: str = "right") -> float:
    """Half the hand's narrowest extent (m), measured on the model in the palm frame:
    the room the fingers need beside an object, kept between footprint circles."""
    import mujoco

    model, data = scene.model, scene.data
    mujoco.mj_kinematics(model, data)
    base = scene.palm_body[side]
    rotation, origin = data.xmat[base].reshape(3, 3), data.xpos[base]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    points = []
    for g in range(model.ngeom):
        if scene.hand_side(g) != side or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        corners = signs * model.geom_aabb[g][3:] + model.geom_aabb[g][:3]
        world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        points.append((world - origin) @ rotation)
    points = np.concatenate(points)
    return 0.5 * float(np.min(points.max(axis=0) - points.min(axis=0)))


def platform_x_range(scene: Scene) -> tuple[float, float]:
    """World x span of the work platform's top."""
    import mujoco

    model, data = scene.model, scene.data
    mujoco.mj_kinematics(model, data)
    geom = model.geom("work_platform").id
    centre, half = float(data.geom_xpos[geom][0]), float(model.geom_size[geom][0])
    return centre - half, centre + half


def derive_box(platform_x: tuple[float, float], belt: Conveyor = BELT) -> tuple[tuple[float, float], tuple[float, float]]:
    """(box centre xy, inner half size): outer walls from the platform's robot-side edge
    to PATH_CLEARANCE short of the belt strip; centred on y = 0 between the arms."""
    outer_low = platform_x[0]
    outer_high = belt.x - 0.5 * belt.width - C.PATH_CLEARANCE
    half_x = 0.5 * (outer_high - outer_low) - BASKET_WALL_THICKNESS
    return (0.5 * (outer_low + outer_high), 0.0), (half_x, BOX_HALF_Y)


def table_region(box_xy, box_half, platform_x, belt: Conveyor = BELT, platform_half_y: float = 0.55):
    """Where a table object's footprint centre may go, as a function of its radius r
    and the arm that takes it: on the platform, off the belt strip (PATH_CLEARANCE),
    half that hand's width clear of the box walls."""
    def bounds(r: float, side: str = "right"):
        x = (platform_x[0] + r, belt.x - 0.5 * belt.width - C.PATH_CLEARANCE - r)
        y_out = (box_half[1] + BASKET_WALL_THICKNESS + HAND_HALF_WIDTH[side] + r, platform_half_y - r)
        return x, y_out
    return bounds


def box_clearance_ok(placement: Placement, box_half, side: str) -> bool:
    """Is the object's footprint half `side`'s hand width clear of the box wall?"""
    return abs(placement.xy[1]) - footprint_radius(placement) >= (
        box_half[1] + BASKET_WALL_THICKNESS + HAND_HALF_WIDTH[side] - 1e-6)


def drop_spot_candidates(box_half, side: str, radius: float, dropped: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop offsets from the box centre for one object of footprint radius `radius`:
    a DROP_GRID_STEP grid over the box's inside (the object's circle inside the walls;
    a circle wider than the box keeps to the centre line), on this arm's half (its own
    side of y = 0: the arms never reach across each other). Order: first the free
    spots -- at least 2 x radius from every spot already dropped at (by either arm),
    so the object does not land on another -- nearest the centre of this arm's half;
    then the rest, farthest from the dropped spots first.

    (Nearest the box centre first made the planner pick a heading for the lying can
    whose grip slipped 49 mm on the proof lift, 2026-10-01; the centre of the arm's
    half is where the plan-checked drops of the 6/6 runs were.)"""
    sign = -1.0 if side == "right" else 1.0
    half_x = max(0.0, box_half[0] - radius)
    steps = np.arange(0.0, half_x + 1e-9, DROP_GRID_STEP)
    xs = np.unique(np.round(np.concatenate([-steps, steps]), 6))  # symmetric: the centre line is a candidate
    reach_y = max(0.0, box_half[1] - radius)
    ys = np.arange(0.0, reach_y + 1e-9, DROP_GRID_STEP) * sign
    spots = [(float(x), float(y)) for x in xs for y in ys]
    home = (0.0, sign * 0.5 * box_half[1])

    def key(spot):
        far = min((float(np.hypot(spot[0] - d[0], spot[1] - d[1])) for d in dropped), default=np.inf)
        free = far >= 2.0 * radius
        return (not free, 0.0 if free else -round(far, 4), round(float(np.hypot(spot[0] - home[0], spot[1] - home[1])), 4))
    return sorted(spots, key=key)


def plan_check(layout: Layout, name: str, side: str, dropped: list[tuple[float, float]] | None = None,
               max_spots: int = 6) -> tuple[bool, str]:
    """Does the real planner find grasp -> lift -> carry -> drop for `name` with this arm,
    gt pose, every other object in place as an obstacle? (No physics.)"""
    from simulation.pick_place.demo import Demo

    scene = layout.scene()
    scene.set_target(name)
    placement = next(p for p in layout.table + layout.belt if p.label == name)
    spots = drop_spot_candidates(layout.box_half, side, footprint_radius(placement), dropped or [])[:max_spots]
    demo = Demo(scene=scene, side=side, pose_backend="gt", release="drop", place_candidates=spots,
                stay_over_basket=True)
    demo.log.verbose = False
    try:
        demo.phase_perceive()
        demo.phase_plan()
    except RuntimeError as error:
        return False, str(error).splitlines()[0][:160]
    return True, demo.scene.grasp_target.name if demo.scene.grasp_target else "planned"


_CHECK = r"""
import sys
from simulation.pick_place.layout import Layout, plan_check
layout = Layout.from_json(sys.stdin.read())
ok, detail = plan_check(layout, sys.argv[1], sys.argv[2])
print("CHECK", "OK" if ok else "NO", detail)
"""


def plan_check_isolated(layout: Layout, name: str, side: str, timeout: float = PLAN_CHECK_TIMEOUT_S) -> tuple[bool, str]:
    """plan_check in a fresh process, given up after `timeout` s (the planner has no
    time bound of its own) -- which also keeps the IK caches' memory out of this one."""
    import subprocess
    import sys

    try:
        out = subprocess.run([sys.executable, "-c", _CHECK, name, side], input=layout.to_json(), capture_output=True,
                             text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return False, f"no plan within {timeout:.0f}s"
    line = next((l for l in out.splitlines() if l.startswith("CHECK ")), "CHECK NO crashed")
    return line.startswith("CHECK OK"), line[9:]


def sample_layout(seed: int, n_table: int = 4, n_belt: int = 2, belt_spacing_m: float | None = None,
                  max_draws: int = 60, verbose: bool = True) -> Layout:
    """A random layout for `seed`: n_table objects that some arm can plan, n_belt
    objects upstream. `belt_spacing_m`: distance between belt objects (the measured
    pick cycle time x belt speed); None -> the largest belt footprint diameter + the
    table gap (objects not touching)."""
    rng = np.random.default_rng(seed)
    probe = Scene((0.3, -0.3), (0.25, 0.0), work_platform_height=PLATFORM_HEIGHT, conveyor=BELT)
    platform_x = WORK_PLATFORM_X
    box_xy, box_half = derive_box(platform_x)
    bounds = table_region(box_xy, box_half, platform_x)
    gap = max(HAND_HALF_WIDTH.values())
    layout = Layout(seed, box_xy, box_half)
    counts: dict[str, int] = {}

    def name_for(key: str) -> str:
        counts[key] = counts.get(key, 0) + 1
        return key if counts[key] == 1 else f"{key}_{counts[key]}"

    # Belt objects first (they never block a table plan: out of reach upstream).
    view_edge = 0.6   # the belt detector's workspace ends here (bin_conveyor_task)
    kinds = [BELT_KINDS[int(rng.integers(len(BELT_KINDS)))] for _ in range(n_belt)]
    radii = [footprint_radius(Placement(k, (0.0, 0.0), p)) for k, p in kinds]
    spacing = belt_spacing_m if belt_spacing_m is not None else 2.0 * max(radii) + gap
    y = view_edge + max(radii)
    for (key, pose), r in zip(kinds, radii):
        layout.belt.append(Placement(key, (BELT.x, round(y, 3)), pose, float(rng.uniform(0.0, 360.0)),
                                     name=f"{key}_belt_{len(layout.belt) + 1}"))
        y += spacing

    sides = ["right", "left"] * n_table
    draws = 0
    while len(layout.table) < n_table:
        draws += 1
        if draws > max_draws:
            raise RuntimeError(f"seed {seed}: {len(layout.table)}/{n_table} table objects after {max_draws} draws")
        key, pose = TABLE_KINDS[int(rng.integers(len(TABLE_KINDS)))]
        yaw = float(rng.uniform(0.0, 180.0 if pose == "lying" else 360.0))
        trial = Placement(key, (0.0, 0.0), pose, yaw)
        r = footprint_radius(trial)
        # Alternate the side of the table so both arms get work (each arm works its own side).
        side = sides[len(layout.table)]
        (x_low, x_high), (y_low, y_high) = bounds(r, side)
        if x_low >= x_high or y_low >= y_high:
            continue
        xy = (float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high)) * (-1.0 if side == "right" else 1.0))
        candidate = Placement(key, (round(xy[0], 3), round(xy[1], 3)), pose, round(yaw, 1), name=None)
        others = layout.table + layout.belt
        if any(np.hypot(candidate.xy[0] - o.xy[0], candidate.xy[1] - o.xy[1]) < r + footprint_radius(o) + gap
               for o in others):
            continue
        name = name_for(key)
        candidate = Placement(key, candidate.xy, pose, candidate.yaw_deg, name=name if name != key else None)
        trial_layout = Layout(seed, box_xy, box_half, layout.table + [candidate], layout.belt)
        started = time.perf_counter()
        ok, detail = plan_check_isolated(trial_layout, candidate.label, side)
        seconds = round(time.perf_counter() - started, 1)
        if verbose:
            print(f"  draw {draws}: {candidate.label} {pose} at {candidate.xy} yaw {candidate.yaw_deg:.0f} -> "
                  f"{side} {'plans' if ok else 'no plan'} ({seconds}s) {detail}", flush=True)
        if not ok:
            counts[key] -= 1
            continue
        layout.table.append(candidate)
        layout.checks[candidate.label] = {"arm": side, "grasp": detail, "plan_seconds": seconds}
    layout.checks["draws"] = draws
    return layout
