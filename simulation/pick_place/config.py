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
# OpenArm v2 wrist: joint5 = forearm roll, joint6 (axis y) and joint7 (axis x) are the
# two bend axes of the wrist. "Hand in line with the forearm" means both are zero.
WRIST_PITCH_INDEX = 5
WRIST_BEND_INDICES = (5, 6)

# --------------------------------------------------------------------------- posture
# Reference grasp posture for the right arm with the wrist *straight* (joint6 = joint7
# = 0), so hand and forearm form one line the way a person's do when picking a bottle
# up from the side. Found by sweeping straight-wrist configurations for one whose jaw
# lands on the table at can-waist height with the fingers near horizontal. The grasp
# orientation and the default pick point A are both derived from it by forward
# kinematics. With the hand mounted along the flange axis (five_finger_model.MOUNTS), this posture
# reaches forward over the table with the palm facing the midline, fingers pointing
# +x and tilted 22 degrees down, wrist bend 1.8 / -0.6 degrees.
NATURAL_GRASP_JOINTS = np.array([-0.041, 0.331, 0.14, 1.214, 0.314, 0.032, -0.011])
RIGHT_SEED = NATURAL_GRASP_JOINTS
# Symmetric attention stance; the left arm mirrors the right (see Scene.attention_pose).
# Fists held in front of the body over the table (elbow ~108 degrees), fingers forward,
# palms facing each other, ~13cm above the table, wrist straight.
ATTENTION_RIGHT = np.array([-0.383, 0.726, 0.051, 1.893, 0.108, 0.0, 0.0])

# --------------------------------------------------------------------------- IK
IK_MAX_ITERATIONS = 6000
IK_POSITION_TOLERANCE = 0.006
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
GRASP_HEIGHT_BIAS = 0.02
# Bias along the thumb-to-fingers line to centre the can in the aperture.
JAW_AXIS_BIAS = -0.011
# 0 sits the wrist at the jaw midpoint, 0.5 puts the fingers themselves on the object.
JAW_BIAS_TOWARD_FINGERS = 0.0
GRASP_POSITION_CORRECTION = np.array([0.0, 0.0, 0.0])
# Standoff behind the grasp, along the fingers' horizontal pointing direction.
APPROACH_STANDOFF = 0.08
# How high the hand rides before descending onto the standoff. 8cm let the hand sweep
# through the can on the way in from the attention stance; 15cm clears it.
HOVER_HEIGHT = 0.15

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
SET_DOWN_STEP = 0.004
SET_DOWN_MAX_DEPTH = 0.04
SET_DOWN_STEP_SECONDS = 0.15
# The carried object stays upright under any rotation about world z, so the place side
# may turn the hand about the vertical to wherever the arm reaches best. First
# candidate whose transfer *and* set-down poses both solve wins.
PLACE_YAW_CANDIDATES_DEG = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, -120.0, 120.0, -150.0, 150.0, 180.0)
# Number of Cartesian waypoints on the transfer and lowering segments. Joint-space
# interpolation between only the endpoints let the hand pitch on the way and the can
# rolled 30 degrees in the grip.
CARRY_PATH_STEPS = 8

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
RANDOM_PICK_BOX = ((0.36, 0.44), (-0.32, -0.22))
RANDOM_BASKET_BOX = ((0.34, 0.46), (-0.03, 0.05))
MIN_PICK_TO_BASKET_M = 0.15
# The wrapped hand reaches ~8cm beyond the can's surface, so the basket's nearest wall
# must stay this far from the object's centre or the thumb clips it at grasp.
MIN_OBJECT_TO_BASKET_M = 0.12

# --------------------------------------------------------------------------- timing (s)
SETTLE_AT_START = 1.5
MOVE_TO_HOVER = 1.2
MOVE_TO_READY = 0.8
PRESHAPE_SETTLE = 0.3
MOVE_TO_PREGRASP = 1.2
PREGRASP_SETTLE = 0.2
MOVE_TO_GRASP = 1.0
PROOF_LIFT_SECONDS = 0.8
MOVE_TO_LIFT = 1.2
TRANSFER_SECONDS = 2.0
LOWER_SECONDS = 1.2
RELEASE_SECONDS = 0.8
RETREAT_SECONDS = 1.0
RETURN_SECONDS = 1.2
FINAL_SETTLE = 3.0

# --------------------------------------------------------------------------- viewer
VIEWER_FRAME_SECONDS = 1.0 / 60.0
