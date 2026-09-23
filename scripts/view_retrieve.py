"""Watch the basket retrieve task in the MuJoCo window.

One arm picks the can and puts it in the basket (optionally on a stand), then grasps
it back out and sets it down on the bare table; `--retrieve-arm left` makes the other
arm do the retrieval (no robust layout exists for that on the real-height robot --
see context_project.md). The executor aborts
the run on any arm/hand contact with the table, basket, stand or robot body, on the
two arms touching, and on the idle arm touching the can -- a finished run means none
of that happened.

    python -m scripts.view_retrieve                      # the committed layout
    python -m scripts.view_retrieve --stand 0.10 --basket 0.28 0.0

Keys in the window: Esc = free camera, [ / ] = cameras, Space = pause,
close the window to quit (run it again for a fresh episode).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

# The layout tests/test_retrieve.py checks (test_place_then_retrieve_from_basket).
DEFAULT_PICK = (0.08, -0.38)
DEFAULT_BASKET = (0.25, -0.25)
DEFAULT_RETRIEVE_TO = (0.15, -0.40)
DEFAULT_STAND = 0.0


def build_task(args: argparse.Namespace) -> RetrieveDemo:
    return RetrieveDemo(
        tuple(args.pick), tuple(args.basket), tuple(args.retrieve_to), side=args.retrieve_arm, place_side=args.place_arm,
        basket_stand_height=args.stand, place_offset=tuple(args.place_offset) if args.place_offset else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pick", type=float, nargs=2, default=list(DEFAULT_PICK))
    parser.add_argument("--basket", type=float, nargs=2, default=list(DEFAULT_BASKET))
    parser.add_argument("--retrieve-to", type=float, nargs=2, default=list(DEFAULT_RETRIEVE_TO))
    parser.add_argument("--place-arm", choices=("right", "left"), default="right")
    parser.add_argument("--retrieve-arm", choices=("right", "left"), default="right")
    parser.add_argument("--stand", type=float, default=DEFAULT_STAND, help="basket stand height (m), 0 = on the table")
    parser.add_argument("--place-offset", type=float, nargs=2, default=None, help="release point inside the basket (m)")
    args = parser.parse_args()

    task = build_task(args)
    with mujoco.viewer.launch_passive(task.place_in.model, task.place_in.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.20, 0.0, 0.25]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 200
        viewer.cam.elevation = -25
        print(f"=== {args.place_arm} hand places into the basket, {args.retrieve_arm} hand takes it out ===")
        result = run_retrieve_trial(task, viewer)
        print(f"  success: {result.success}")
        if result.failure_reason:
            print(f"  failure: {result.failure_reason}")
        print(
            f"  final can {[round(v, 3) for v in result.final_position]}, "
            f"set-down error {result.placement_error_m*1000:.1f}mm, tilt {result.bottle_tilt_deg:.1f}deg, "
            f"max penetration {result.max_penetration_m*1000:.1f}mm"
        )
        print("  scene left as-is; close the window to quit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)

if __name__ == "__main__":
    main()
