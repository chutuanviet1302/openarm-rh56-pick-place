"""Every tunable of the pick-and-place demo, in one place.

Grouped by which module consumes them. Nothing here is a hand-tuned wrist position:
the wrist targets are all *derived* at runtime (planner.py) from the object, the
basket and the hand's own geometry, and these numbers only set margins and gains.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- names
ARM_JOINTS = {
    "left": tuple(f"openarm_left_joint{i}" for i in range(1, 8)),
    "right": tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
}
ARM_ACTUATORS = {
    "left": tuple(f"left_joint{i}_ctrl" for i in range(1, 8)),
    "right": tuple(f"right_joint{i}_ctrl" for i in range(1, 8)),
}
EE_SITE = {"left": "left_ee_control_point", "right": "right_ee_control_point"}
BOTTLE_JOINT = "pick_bottle_joint"
OBJECT_GEOM = "pick_bottle_collision"
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# OpenArm v1 wrist: joint6 and joint7 are the two bend axes. "Hand in line with the
# forearm" means both are zero.
WRIST_PITCH_INDEX = 5
WRIST_BEND_INDICES = (5, 6)

# --------------------------------------------------------------------------- posture
# Reference posture is selected by scripts/sweep_postures.py from collision-free
# top-grasps 20/25/30 degrees off vertical. The exact value is updated from that sweep.
NATURAL_GRASP_JOINTS = np.array([0.065, 0.283, -1.042, 0.709, -0.395, -0.219, -0.658])
# Same tool pose has multiple 7-DOF IK branches. This seed keeps the elbow near the
# torso and lets the forearm reach outward, instead of lifting the elbow beside the
# wrist. The wrist orientation still comes from NATURAL_GRASP_JOINTS above.
RIGHT_SEED = np.array([-0.360, 0.286, 0.353, 0.729, -1.375, -0.067, 0.115])
MIRROR_JOINT_SIGNS = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
ARM_SEED = {"right": RIGHT_SEED, "left": RIGHT_SEED * MIRROR_JOINT_SIGNS}
TOP_GRASP_TILT_CANDIDATES_DEG = (20.0, 25.0, 30.0)
MIN_JOINT_MARGIN_DEG = 3.0
MIN_FLOOR_CLEARANCE = 0.005
# Attention stance ("nghiem"): both arms hanging straight at the sides, fists closed,
# fingers down, palms facing the body. That is the OpenArm v1 zero pose with the hand
# mounted along the flange axis; the left arm mirrors the right (see Scene.attention_pose).
ATTENTION_RIGHT = np.zeros(7)

# --------------------------------------------------------------------------- IK
IK_MAX_ITERATIONS = 6000
# Stop early when the pose error has not improved by IK_STALL_TOLERANCE for this many
# iterations (an unreachable target otherwise burns the whole budget: ~2.6s each).
IK_STALL_ITERATIONS = 40
IK_STALL_TOLERANCE = 1e-5
IK_POSITION_TOLERANCE = 0.004
IK_ROTATION_TOLERANCE = 0.05
IK_ROTATION_WEIGHT = 0.4
IK_DAMPING = 0.004
IK_MAX_STEP = 0.08
# Nullspace pull on the wrist pitch toward 0: the arm has one spare degree of freedom
# and this spends it on keeping the wrist straight during approach and carry.
WRIST_STRAIGHT_GAIN = 0.3

# --------------------------------------------------------------------------- grasp geometry
# Fraction of finger closure at which the jaw offsets are measured (the pre-shape).
GRASP_CLOSURE_FRACTION = 0.20
# Minimum free space per side between the open jaw and the object.
GRASP_CLEARANCE = 0.0005
# Raises the grip so the fingers close around the object rather than into the table.
GRASP_HEIGHT_BIAS = 0.04
# Bias along the thumb-to-fingers line to centre the can in the aperture.
JAW_AXIS_BIAS = -0.011
# 0 sits the wrist at the jaw midpoint, 0.5 puts the fingers themselves on the object.
JAW_BIAS_TOWARD_FINGERS = {"right": 0.0, "left": 0.30}
GRASP_POSITION_CORRECTION = np.array([0.0, 0.0, 0.0])
# Standoff opposite the fingers' full 3-D pointing direction.
APPROACH_STANDOFF = 0.08
# How high the hand rides before descending onto the standoff. 8cm let the hand sweep
# through the can on the way in from the attention stance; 15cm clears it.
HOVER_HEIGHT = 0.08
# Extra height of the 'raise' way point (above the hanging hand) over the hover.
RAISE_ABOVE_HOVER = 0.10
# Room the fist must keep from the basket/object/table along the unplanned joint
# blends (attention <-> raise <-> hover); the executed arm lags the command by a few mm.
PATH_CLEARANCE = 0.02
# Horizontal offsets (x, y) of the raise way point from the hanging hand, tried in
# order at each height: straight up first, then back, outward and inward.
RAISE_XY_OFFSETS = ((0.0, 0.0), (-0.06, 0.0), (0.0, -0.06), (0.0, 0.06), (-0.06, -0.06), (-0.06, 0.06))

# --------------------------------------------------------------------------- carry / place
# The object's *bottom* must ride at least this far above the basket walls.
CARRY_CLEARANCE_ABOVE_RIM = 0.05
# The servos sag under the can's weight, so the plan asks for this much extra; the
# requirement above is what gets checked.
CARRY_CLEARANCE_MARGIN = 0.01
# Planned height of the object's bottom above the basket floor at the end of the
# lowering path. From there the hand keeps descending in small steps until the object
# actually touches the floor (SET_DOWN_*), and only then opens: releasing a can that is
# still in the air let the opening thumb lever it 5cm up and it landed 3cm off.
PLACE_DROP_HEIGHT = 0.02
# No standing XY bias: the object is centred over the basket from its own measured
# position just before the set-down (Demo._centre_over_basket). The hand-tuned offsets
# that used to live here were calibrated against one carry timing -- at 19.3mm long they
# spent the whole 20mm placement budget before the trial even started.
PLACE_POSITION_CORRECTION = {
    "right": np.zeros(3),
    "left": np.zeros(3),
}
# Below this the object is already centred well enough to set down as planned.
SET_DOWN_CENTRING_TOLERANCE = 0.002
SET_DOWN_CENTRING_SECONDS = 0.5
SET_DOWN_STEP = 0.004
SET_DOWN_MAX_DEPTH = 0.04
# The set-down keeps descending until the object's centre is within this of its
# flat-resting height (it first touches on its rim when held tilted).
SET_DOWN_SEATED_TOLERANCE = 0.003
SET_DOWN_STEP_SECONDS = 0.15
# The carried object stays upright under any rotation about world z, so the place side
# may turn the hand about the vertical to wherever the arm reaches best. First
# candidate whose transfer *and* set-down poses both solve wins.
# The grasp itself may also turn about the vertical (a round can has no preferred
# heading); 0 is the reference posture, tried first.
GRASP_YAW_CANDIDATES_DEG = (0.0, -30.0, 30.0, 15.0, -15.0, -60.0, 60.0, -90.0, 90.0)
# How many physically rejected grasps (finger not pressing, proof lift failed) the
# episode lets go of and retries with another heading before giving up.
GRASP_RETRIES = 3
PLACE_YAW_CANDIDATES_DEG = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, -120.0, 120.0, -150.0, 150.0, 180.0)
# Number of Cartesian waypoints on the transfer and lowering segments. Joint-space
# interpolation between only the endpoints let the hand pitch on the way and the can
# rolled 30 degrees in the grip.
CARRY_PATH_STEPS = 8
TRANSFER_BASE_CLEARANCE = 0.22
TRANSFER_LONG_PATH_M = 0.45

# --------------------------------------------------------------------------- proof lift
PROOF_LIFT_HEIGHT = 0.05
# The hand itself must actually rise this much (servo sag: commanded 5cm -> ~4.2cm) ...
PROOF_LIFT_MIN_HAND_RISE = 0.03
# ... and the object may lag the hand by at most this much: "the bottle came with the
# hand" is a statement about slip, not about an absolute height.
PROOF_LIFT_MAX_SLIP = 0.01
PROOF_LIFT_MAX_TILT_DEG = 15.0

# --------------------------------------------------------------------------- finger closing
# Adaptive closing after correlllab/rh56_controller (grasp_executor._adaptive_force_phase):
# step each finger toward closure a little at a time and stop commanding it once it
# presses hard enough, so it settles ON the surface instead of being driven through it.
CONTACT_FORCE_TARGET_N = 8.0
# Minimum normal force for a finger to count as "pressing" before the lift.
GRASP_SECURE_MIN_FORCE_N = 0.5
GRASP_SECURE_MIN_THUMB_FORCE_N = 6.0  # secure grasps measured 9-24N on the thumb
GRASP_SECURE_MIN_OPPOSING_FORCE_N = 1.0  # index or middle must press back against the thumb
CLOSE_STEP_FRACTION = 0.01
CLOSE_SETTLE_SECONDS = 0.02
CLOSE_MAX_ITERATIONS = 120

# --------------------------------------------------------------------------- safety
# A contact only counts as a crash once it is deeper than this: MuJoCo reports contact
# slightly before the surfaces interpenetrate.
TABLE_CONTACT_TOLERANCE = 0.003
BASKET_CONTACT_TOLERANCE = 0.003  # same graze allowance as the table

# --------------------------------------------------------------------------- randomization
# Sampling boxes (table-plane x, y) covering the region the straight-wrist grasp reaches.
# Every sample is still verified by IK before physics runs (demo.sample_layout).
RANDOM_PICK_BOX = ((0.12, 0.18), (-0.38, -0.33))
RANDOM_BASKET_BOX = ((0.22, 0.26), (-0.06, -0.02))
MIN_PICK_TO_BASKET_M = 0.15
# The wrapped hand reaches ~8cm beyond the can's surface, so the basket's nearest wall
# must stay this far from the object's centre or the thumb clips it at grasp.
MIN_OBJECT_TO_BASKET_M = 0.06

# --------------------------------------------------------------------------- timing (s)
SETTLE_AT_START = 1.5
MOVE_TO_RAISE = 1.0
MOVE_TO_HOVER = 1.2
MOVE_TO_READY = 0.8
PRESHAPE_SETTLE = 0.3
MOVE_TO_PREGRASP = 1.2
PREGRASP_SETTLE = 0.2
MOVE_TO_GRASP = 1.0
PROOF_LIFT_SECONDS = 0.8
MOVE_TO_LIFT = 1.8
TRANSFER_SECONDS = 2.5
LOWER_SECONDS = 1.2
ALL_FINGERS = ("index", "middle", "ring", "pinky", "thumb")
RELAX_GRIP_SECONDS = 0.6  # grip force -> light contact before the fingers open
RELEASE_SECONDS = 1.0
RETREAT_SECONDS = 1.0
RETURN_SECONDS = 1.2
FINAL_SETTLE = 3.0

# --------------------------------------------------------------------------- viewer
VIEWER_FRAME_SECONDS = 1.0 / 30.0

# --------------------------------------------------------------------------- evaluation
BENCHMARK_TRIALS = 50
# Screening a benchmark fixture (scripts/make_benchmark_fixture.py). The set-down must
# still solve with the object this far off the nominal jaw centre in any direction --
# the grip shifts during the carry and measured held offsets vary by about this much.
FIXTURE_HELD_OFFSET_ENVELOPE_M = 0.01
BENCHMARK_REQUIRED_PASSES = 48  # 47/50 is only 94%; >=95% therefore means 48.
PLACEMENT_ERROR_LIMIT_M = 0.02
PERCEPTION_MAX_ERROR_M = 0.01
PERCEPTION_P95_ERROR_M = 0.005
FINAL_STABILITY_SECONDS = 0.5
