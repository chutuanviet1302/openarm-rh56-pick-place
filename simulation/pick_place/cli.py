"""Command line for the demo.

    python -m simulation.pick_place_demo                       # viewer, real time
    python -m simulation.pick_place_demo --headless --trials 3 # physics trials + report
    python -m simulation.pick_place_demo --plan-only           # print targets / IK table, no physics
    python -m simulation.pick_place_demo --stop-after grasp    # leave the viewer at that phase
    python -m simulation.pick_place_demo --verbose --trace 0.5 # per-phase observations + periodic state dump
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from simulation.five_finger_model import BASKET_POSITION_B, PICK_POSITION_A
from simulation.pick_place.demo import PHASES, Demo, run_trial, sample_layout
from simulation.pick_place.episode import write_report
from simulation.pick_place.handoff import HandoffDemo, run_handoff_trial
from simulation.pick_place.routing import Route, TaskRouter
from simulation.pick_place.scene import Scene
from simulation.vision_detector import VisionDetector

CAMERAS = ("isometric", "overhead", "d435_head", "right_wrist_camera", "front_view", "side_view", "close_grasp", "free")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scripted OpenArm bimanual five-finger pick-and-place demo")
    mode = parser.add_argument_group("mode")
    mode.add_argument("--headless", action="store_true", help="no window; run --trials episodes and write --report")
    mode.add_argument("--plan-only", action="store_true", help="derive targets and solve IK, print the table, run no physics")
    mode.add_argument("--trials", type=int, default=1)
    mode.add_argument("--report", type=Path, default=Path("artifacts/physics_trials.json"))

    layout = parser.add_argument_group("layout")
    layout.add_argument("--object", type=float, nargs=2, metavar=("X", "Y"), default=list(PICK_POSITION_A),
                        help=f"table-plane position of the object (pick point A), default {PICK_POSITION_A}")
    layout.add_argument("--basket", type=float, nargs=2, metavar=("X", "Y"), default=list(BASKET_POSITION_B),
                        help=f"table-plane position of the basket (place point B), default {BASKET_POSITION_B}")
    layout.add_argument("--randomize", action="store_true", help="sample a new IK-checked layout per trial (headless)")
    layout.add_argument("--seed", type=int, default=0)
    layout.add_argument("--perception", action="store_true", help="object position from the head camera (RGB-D), not sim state")
    layout.add_argument("--arm", choices=("auto", "right", "left"), default="auto",
                        help="arm selection; auto runs IK/collision preflight (default)")

    debug = parser.add_argument_group("debugging")
    debug.add_argument("--camera", choices=CAMERAS, default="isometric")
    debug.add_argument("--stop-after", choices=PHASES, help="stop the episode after this phase and leave the scene up")
    debug.add_argument("--verbose", action="store_true", help="print the IK table and object/wrist positions after each phase")
    debug.add_argument("--trace", type=float, metavar="SECONDS",
                       help="every SECONDS of sim time, print object pose, wrist pose and finger forces")
    return parser


def install_trace(demo: Demo, period: float) -> None:
    last = [-1.0]

    def on_step(executor) -> None:
        t = float(executor.data.time)
        if t - last[0] < period:
            return
        last[0] = t
        scene = executor.scene
        forces = scene.finger_contact_forces(demo.side)
        print(
            f"[trace t={t:6.2f}s] object={np.round(scene.object_position(), 3).tolist()} "
            f"wrist={np.round(scene.wrist_position(demo.side), 3).tolist()} "
            f"forces={{{', '.join(f'{k}:{v:.1f}' for k, v in forces.items())}}}"
        )

    demo.executor.on_step = on_step


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive")
    layout = dict(pick_position=tuple(args.object), basket_position=tuple(args.basket),
                  perception=args.perception, verbose=args.verbose)

    def route_for(trial_layout: dict):
        """(source_arm, reason) for a direct route, or the routing.RouteDecision
        itself for a handoff, so the caller can build the right episode type."""
        if args.arm != "auto":
            return (args.arm, "arm selected explicitly")
        scene = Scene(trial_layout["pick_position"], trial_layout["basket_position"])
        observed = None
        if args.perception:
            detection = VisionDetector(scene.model, "d435_head").detect_object(scene.data, render_annotation=False)
            if not detection.found:
                raise RuntimeError("route selection failed: object not found by d435_head")
            observed = detection.pos_world
        decision = TaskRouter(scene).select(observed)
        if decision.route == Route.REJECTED:
            raise RuntimeError(f"{decision.route.value}: {decision.reason}")
        print(f"route: {decision.route.value} — {decision.reason}")
        if decision.route in (Route.DIRECT_RIGHT, Route.DIRECT_LEFT):
            return (decision.source_arm, decision.reason)
        return decision

    def build_episode(trial_layout: dict) -> Demo | HandoffDemo:
        """A plain Demo for a direct route, or a HandoffDemo that relays the object
        through a staging point when no single arm can reach both pick and basket."""
        route = route_for(trial_layout)
        if isinstance(route, tuple):
            side, reason = route
            demo = Demo(**trial_layout, side=side)
            demo.route_reason = reason
            return demo
        decision = route
        handoff = HandoffDemo(
            trial_layout["pick_position"], trial_layout["basket_position"],
            source_arm=decision.source_arm, target_arm=decision.target_arm,
            handoff_position=decision.handoff_position,
            perception=trial_layout["perception"], verbose=trial_layout["verbose"],
        )
        handoff.route_reason = decision.reason
        return handoff

    def run_episode(episode: Demo | HandoffDemo, viewer=None, stop_after: str | None = None):
        if isinstance(episode, HandoffDemo):
            return run_handoff_trial(episode, viewer, stop_after=stop_after)
        return run_trial(episode, viewer, stop_after=stop_after)

    if args.plan_only:
        episode = build_episode(layout)
        if isinstance(episode, HandoffDemo):
            episode.leg1.phase_perceive()
            print(f"-- leg 1: {episode.source_arm} carries pick -> staging point {episode.handoff_position} --")
            print(episode.leg1.planner.describe(episode.leg1.planner.plan(episode.leg1.object_position())))
            leg2_probe = Demo(episode.handoff_position, episode.basket_position, side=episode.target_arm)
            print(f"-- leg 2: {episode.target_arm} carries staging point -> basket (planned from the nominal staging pose) --")
            print(leg2_probe.planner.describe(leg2_probe.planner.plan(leg2_probe.object_position())))
            return
        episode.phase_perceive()
        plan = episode.planner.plan(episode.object_position())
        print(episode.planner.describe(plan))
        return

    if args.headless:
        results = []
        rng = np.random.default_rng(args.seed)
        for index in range(1, args.trials + 1):
            trial_layout = dict(layout)
            if args.randomize:
                sample_side = args.arm if args.arm != "auto" else ("right" if index % 2 else "left")
                trial_layout.update(sample_layout(rng, side=sample_side, perception=args.perception))
            episode = build_episode(trial_layout)
            if args.trace:
                if isinstance(episode, HandoffDemo):
                    print("  (--trace is not supported for a handoff route; skipped)")
                else:
                    install_trace(episode, args.trace)
            result = run_episode(episode, stop_after=args.stop_after)
            print(f"  trial {index}: {result.summary()}")
            results.append(result)
        write_report(results, args.report)
        passed = sum(result.success for result in results)
        print(f"Physics trials: {passed}/{len(results)} passed; report: {args.report}")
        if passed != len(results):
            raise SystemExit(1)
        return

    episode = build_episode(layout)
    if args.trace and not isinstance(episode, HandoffDemo):
        install_trace(episode, args.trace)

    def watch(demo: Demo, stop_after: str | None, label: str) -> None:
        # Pressing R in the window restarts the episode: the scene is reset to its
        # start state and the whole sequence runs again.
        restart_requested = [False]

        def on_key(keycode: int) -> None:
            if keycode in (ord("R"), ord("r")):
                restart_requested[0] = True

        with mujoco.viewer.launch_passive(demo.model, demo.data, key_callback=on_key) as viewer:
            if args.camera == "free":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = demo.model.camera(args.camera).id
            print(
                f"[Camera] '{args.camera}' -- {label}.  Keys in the MuJoCo window:  [ / ] = previous / next "
                "camera   Esc = free camera (left-drag rotate, right-drag zoom, middle-drag pan)   Space = "
                "pause   R = run the episode again   Tab = side panel   close the window to continue"
            )
            run = 0
            while viewer.is_running():
                run += 1
                print(f"=== {label}, run {run} ===")
                result = run_trial(demo, viewer, stop_after=stop_after)
                print(f"  result: {result.summary()}")
                if not result.success:
                    print("   the scene is left as-is -- rotate the view to inspect the hand")
                print("   press R in the window to run again, or close it to continue")
                restart_requested[0] = False
                while viewer.is_running() and not restart_requested[0]:
                    mujoco.mj_step(demo.model, demo.data)
                    viewer.sync()
                    time.sleep(0.01)
                if not viewer.is_running():
                    break
                demo.restart()

    if isinstance(episode, HandoffDemo):
        watch(episode.leg1, None, f"leg 1: {episode.source_arm} -> staging point {episode.handoff_position}")
        handoff_xy = episode.leg1.scene.object_position()[:2].copy()
        episode.leg2 = Demo(
            (float(handoff_xy[0]), float(handoff_xy[1])), episode.basket_position,
            perception=episode.perception, verbose=episode.verbose, side=episode.target_arm,
        )
        watch(episode.leg2, args.stop_after, f"leg 2: {episode.target_arm} -> basket")
    else:
        watch(episode, args.stop_after, f"{episode.side} arm, direct")
