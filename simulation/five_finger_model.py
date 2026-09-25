from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from simulation.openarm_mujoco import FLANGE_Z, configure_arm_servos, load_openarm_spec

_PROJECT_ROOT = Path(os.environ.get("OPENARM_PROJECT_ROOT", Path(__file__).parents[1]))
if not (_PROJECT_ROOT / "assets/ycb/ycb_tomato_soup_can/meshes/tomato_soup_can.obj").is_file():
    _PROJECT_ROOT = Path.cwd()
INSPIRE_ROOT = _PROJECT_ROOT / "assets/rh56_controller/h1_mujoco/archive/inspire"
# High-quality YCB object assets from P-161 project (mesh + texture).
# Each sub-folder has  meshes/<name>.obj  and  textures/<name>.png.
# Available objects: ycb_tomato_soup_can, ycb_apple, ycb_orange, ycb_peach,
#                    ycb_pear, ycb_plum, ycb_lemon, ycb_strawberry
YCB_ASSET_ROOT = _PROJECT_ROOT / "assets" / "ycb"
YCB_PICK_OBJECT_NAME = "ycb_tomato_soup_can"
# Measured AABB of tomato_soup_can.obj: 6.79cm x 6.77cm x 10.19cm, centred at origin.
# The collision cylinder is kept at the original tuned dimensions (slightly wider
# than the mesh) because the pick-place trajectories are calibrated against them.
OBJECT_RADIUS = 0.0354   # collision cylinder radius  (mesh half-width ≈ 3.40 cm)
OBJECT_HALF_HEIGHT = 0.05  # collision cylinder half-height (mesh ≈ 5.09 cm)
# Distance from the mesh origin down to its bottom face; the visual is raised by the
# difference so the drawn bottom coincides with the collision bottom on the table.
MESH_HALF_HEIGHT_BELOW = 0.0516
# World frame: origin on the table top under the robot. The robot stands ON the table
# (stock OpenArm v1 pedestal on its base plate), the object and basket share the top,
# and the room floor is one table height below. Lab measurements, 2026-09-18.
TABLE_TOP_Z = 0.0
TABLE_HEIGHT_ABOVE_FLOOR = 0.74     # measured table height
ROOM_FLOOR_Z = TABLE_TOP_Z - TABLE_HEIGHT_ABOVE_FLOOR
TABLE_X_RANGE = (-0.25, 0.75)       # runs from behind the pedestal to the far edge
TABLE_HALF_WIDTH = 0.55
TABLE_THICKNESS = 0.04
ROBOT_RISER_HEIGHT = 0.02901        # measured base plate under the pedestal (collision slab)
# Measured 2026-09-23 on the real robot, from the table top: 0.78 m to the top of the
# robot (pedestal and base plate included), 0.88 m to the top of the head camera on
# its bracket. These supersede the 2026-09-18 build-up (plate 29 mm + vendor 0.698 m
# to the shoulder), which put the robot top at 0.809 m -- the plate was counted on
# top of a vendor pedestal that already stands the full measured height.
ROBOT_TOP_ABOVE_TABLE = 0.78
CAMERA_TOP_ABOVE_TABLE = 0.88
# The vendor v1 torso housing rises 0.083 above the shoulder axis.
HEAD_TOP_ABOVE_SHOULDER_AXIS = 0.083
VENDOR_SHOULDER_ABOVE_PEDESTAL = 0.698   # vendor OpenArm v1 model: body_link0 -> shoulder axis
SHOULDER_AXIS_Z = TABLE_TOP_Z + ROBOT_TOP_ABOVE_TABLE - HEAD_TOP_ABOVE_SHOULDER_AXIS
CAMERA_BODY_HALF_HEIGHT = 0.0125
HEAD_FRONT_X = 0.066                # vendor torso housing front face (pedestal frame)
# D435i centre: its top face at the measured camera height (bracket above the head).
CAMERA_Z = TABLE_TOP_Z + CAMERA_TOP_ABOVE_TABLE - CAMERA_BODY_HALF_HEIGHT
CAMERA_ABOVE_SHOULDER_AXIS = CAMERA_Z - SHOULDER_AXIS_Z
HAND_PREFIX = "inspire_"
# Default pick point A and basket point B (table-plane x, y). A is out on the robot's
# right, B is forward of the pedestal with a physical gap from its base plate. The
# top grasp comes in at whatever heading the planner's
# grasp-yaw candidates reach. Both can be overridden per run
# (`pick_place_demo --object X Y --basket X Y`).
PICK_POSITION_A = (0.08, -0.38)
BASKET_POSITION_B = (0.25, -0.25)
BASKET_FLOOR_Z = TABLE_TOP_Z + 0.005
# Basket inner half-width and wall height. With the top grasp the hand comes down
# onto the can from above, so the basket only has to clear the can plus the fingers
# wrapped around it: 18cm inside. (The old 32cm basket was sized for a horizontal
# side grasp, whose palm reached 12cm behind the can; at 32cm it also ran into the
# pedestal's base plate, which spans x -0.16..0.10, y -0.10..0.10.) Position B is the
# nearest spot the lift pose still reaches (scripts/sweep_floor_layout.py: x <= 0.30)
# that keeps the basket clear of the base plate and the pick point outside its walls.
BASKET_HALF_WIDTH = 0.09
BASKET_WALL_HEIGHT = 0.05
BASKET_WALL_THICKNESS = 0.01
BASKET_SLOPE_FRICTION = 0.05  # V-floor insert plates (see basket_floor_tilt_deg)
BASKET_VALLEY_Y = 0.04       # V-floor groove, basket frame (+y = toward the left arm)
# Optional work platform (work_platform_height): a rectangular block on the table in
# front of the robot, as long as the table (y) and half as wide (x), clear of the
# pedestal's base plate (x <= 0.10). The object, basket and set-down spot sit on it.
WORK_PLATFORM_X = (0.17, 0.67)  # front edge 47mm clear of the resting fists (x 0.123)
# The stock floor-standing pedestal is placed on the measured base plate.
PEDESTAL_RAISE = SHOULDER_AXIS_Z - TABLE_TOP_Z - VENDOR_SHOULDER_ABOVE_PEDESTAL
# Flange -> Inspire hand base transform, derived from the two frames rather than tuned:
#
#   OpenArm v1 link7: the tool axis is +z (the chain runs along +z, the stock gripper's
#   hand body starts at z=FLANGE_Z=0.0955 where link7's mesh ends). With the arm hanging
#   at rest link7's axes are x = world forward, y = world -y, z = world down.
#   Inspire RH56 base frame: +z = fingers forward, +x = palm side (fingers curl toward
#   +x), y across the palm (right hand: index/thumb at +y, pinky at -y).
#
# The hand therefore continues the forearm (hand +z -> link7 +z) and the palm faces the
# robot's midline with the thumb forward when the arm hangs at rest: right hand palm
# -> world +y = link7 -y, i.e. a -90 degree turn about z; left hand mirrored (+90).
# The base sits on the flange face with a 1cm adapter plate in between.
HAND_ADAPTER_THICKNESS = 0.010
HAND_MOUNT_Z = FLANGE_Z + HAND_ADAPTER_THICKNESS
MOUNTS = {
    "right": ((0.0, 0.0, HAND_MOUNT_Z), (0.70710678, 0.0, 0.0, -0.70710678)),
    "left": ((0.0, 0.0, HAND_MOUNT_Z), (0.70710678, 0.0, 0.0, 0.70710678)),
}


def _attach_hand(arm: mujoco.MjSpec, side: str) -> None:
    scene = INSPIRE_ROOT / f"inspire_{side}.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"Inspire RH56DFX asset missing: {scene}")
    hand = mujoco.MjSpec.from_file(str(scene))
    root = hand.body("base")
    # Palm collision stays ON: the hand must not be able to pass through the basket or
    # the table. (It used to be disabled because the palm shell intersected link5 with
    # the old sideways mount; with the hand along the flange axis it no longer does --
    # checked by tests/test_mujoco.py::test_palm_collides_and_does_not_touch_the_arm.)
    root.pos = np.zeros(3)
    root.quat = np.array([1.0, 0.0, 0.0, 0.0])
    position, quaternion = MOUNTS[side]
    flange = arm.body(f"openarm_{side}_link7")
    mount = flange.add_frame(pos=position, quat=quaternion)
    mount.attach_body(root, prefix=f"{HAND_PREFIX}{side}_")
    # Visual adapter plate between the flange face and the hand base (no collision).
    flange.add_geom(
        name=f"{HAND_PREFIX}{side}_adapter",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, FLANGE_Z + 0.5 * HAND_ADAPTER_THICKNESS],
        size=[0.028, 0.5 * HAND_ADAPTER_THICKNESS, 0.0],
        rgba=[0.75, 0.75, 0.78, 1.0],
        contype=0,
        conaffinity=0,
    )


def _remove_stock_gripper(arm: mujoco.MjSpec, side: str) -> None:
    """Drop v1's parallel gripper (link8 -> hand -> two finger slides) and its actuators."""
    for name in (f"{side}_finger1_ctrl", f"{side}_finger2_ctrl"):
        actuator = arm.actuator(name)
        if actuator is not None:
            arm.delete(actuator)
    # The unnamed finger-coupling equality, the split tendon and the hand/finger
    # contact excludes all reference joints/bodies that are about to go.
    finger_joints = {f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"}
    for equality in list(arm.equalities):
        if equality.name1 in finger_joints or equality.name2 in finger_joints:
            arm.delete(equality)
    tendon = arm.tendon(f"split_{side}")
    if tendon is not None:
        arm.delete(tendon)
    gripper_bodies = {f"openarm_{side}_hand", f"openarm_{side}_right_finger", f"openarm_{side}_left_finger"}
    for exclude in list(arm.excludes):
        if exclude.bodyname1 in gripper_bodies or exclude.bodyname2 in gripper_bodies:
            arm.delete(exclude)
    for name in (f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"):
        joint = arm.joint(name)
        if joint is not None:
            arm.delete(joint)
    gripper = arm.body(f"openarm_{side}_link8")
    if gripper is not None:
        arm.delete(gripper)


def _camera_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> list[float]:
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    actual_up = np.cross(right, fwd)
    rot = np.column_stack([right, actual_up, -fwd])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat.tolist()


def build_five_finger_spec(
    *, pick_bottle: bool = False, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B,
    arm_half_separation: float | None = None, left_arm_mount_yaw_deg: float | None = None,
    right_arm_mount_yaw_deg: float | None = None, basket_stand_height: float = 0.0,
    basket_floor_tilt_deg: float = 0.0, work_platform_height: float = 0.0,
) -> mujoco.MjSpec:
    if not INSPIRE_ROOT.is_dir():
        raise FileNotFoundError("Inspire RH56DFX assets missing; clone correlllab/rh56_controller with h1_mujoco")

    arm = load_openarm_spec()
    if arm_half_separation is not None:
        if arm_half_separation <= 0.0:
            raise ValueError("arm_half_separation must be positive")
        # Simulation fixture only: move the two arm roots on the common torso.  The
        # default keeps OpenArm v1 geometry; handoff experiments opt in explicitly.
        arm.body("openarm_left_link0").pos[1] = arm_half_separation
        arm.body("openarm_right_link0").pos[1] = -arm_half_separation
    # Simulation fixture only, opt-in: turn an arm's mount about the vertical so its
    # grasp region -- which sits well outboard of its shoulder -- swings in toward
    # the table centre (so one arm can place into a basket the other takes back out
    # of). Link lengths, joint ranges and the grasp posture are the vendor's; only
    # the mount heading changes. Not a measured hardware configuration.
    for side, yaw_deg in (("left", left_arm_mount_yaw_deg), ("right", right_arm_mount_yaw_deg)):
        if not yaw_deg:
            continue
        base = arm.body(f"openarm_{side}_link0")
        half = np.deg2rad(yaw_deg) / 2.0
        turn = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, np.asarray(base.quat, dtype=float))
        base.quat = quat
    # The vendor scene's ground plane becomes the room floor; the table stands on it.
    arm.geom("floor").pos = [0.0, 0.0, ROOM_FLOOR_Z]
    table_x = 0.5 * (TABLE_X_RANGE[0] + TABLE_X_RANGE[1])
    table_half_x = 0.5 * (TABLE_X_RANGE[1] - TABLE_X_RANGE[0])
    # The top is drawn as a slab but collides as a plane: MuJoCo's cylinder-box contact
    # (convex MPR) gives the can a wobblier footing than cylinder-plane, enough to
    # turn a clean proof lift (0mm slip) into a 5mm-slip / 18-degree-tilt failure.
    # Nothing in the workspace reaches past the table's edges, so the infinite plane
    # is equivalent there.
    arm.worldbody.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[table_x, 0.0, TABLE_TOP_Z],
        size=[table_half_x, TABLE_HALF_WIDTH, 0.05],
        rgba=[0.0, 0.0, 0.0, 0.0],
        friction=[1.0, 0.005, 0.0001],
    )
    arm.worldbody.add_geom(
        name="table_top_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[table_x, 0.0, TABLE_TOP_Z - 0.5 * TABLE_THICKNESS],
        size=[table_half_x, TABLE_HALF_WIDTH, 0.5 * TABLE_THICKNESS],
        rgba=[0.96, 0.87, 0.70, 1.0],
        contype=0,
        conaffinity=0,
    )
    leg_half_height = 0.5 * (TABLE_HEIGHT_ABOVE_FLOOR - TABLE_THICKNESS)
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        arm.worldbody.add_geom(
            name=f"table_leg_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[table_x + sx * (table_half_x - 0.04), sy * (TABLE_HALF_WIDTH - 0.04), ROOM_FLOOR_Z + leg_half_height],
            size=[0.025, 0.025, leg_half_height],
            rgba=[0.55, 0.42, 0.28, 1.0],
            contype=0,
            conaffinity=0,
        )
    arm.worldbody.add_geom(
        name="robot_riser",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.03, 0.0, 0.5 * ROBOT_RISER_HEIGHT],
        size=[0.13, 0.10, 0.5 * ROBOT_RISER_HEIGHT],
        rgba=[0.58, 0.58, 0.60, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    arm.worldbody.add_camera(name="overhead", pos=[0.15, 0.0, 2.2], quat=[1, 0, 0, 0], fovy=50)
    arm.worldbody.add_camera(
        name="isometric",
        pos=[0.95, -0.85, 0.80],
        quat=_camera_quat(np.array([0.95, -0.85, 0.80]), np.array([0.25, -0.25, 0.25])),
        fovy=48,
    )
    arm.worldbody.add_camera(
        name="front_view",
        pos=[1.10, -0.22, 0.50],
        quat=_camera_quat(np.array([1.10, -0.22, 0.50]), np.array([0.25, -0.25, 0.20])),
        fovy=48,
    )
    arm.worldbody.add_camera(
        name="side_view",
        pos=[0.25, -1.15, 0.50],
        quat=_camera_quat(np.array([0.25, -1.15, 0.50]), np.array([0.25, -0.25, 0.20])),
        fovy=48,
    )
    # The close-up follows the pick point, so a relocated object stays framed.
    focus = np.array([pick_position[0], pick_position[1], TABLE_TOP_Z + OBJECT_HALF_HEIGHT])
    close_pos = focus + np.array([0.33, -0.23, 0.17])
    arm.worldbody.add_camera(name="close_grasp", pos=close_pos.tolist(), quat=_camera_quat(close_pos, focus), fovy=38)
    # Intel RealSense D435i head camera, measured 68.64mm above the shoulder axis.
    # Mounted on top of the head at its front edge (the housing's front face is at
    # x=0.066; further back the housing's own top fills the downward view), looking
    # down the centre line at the middle of the table so the whole work area -- pick
    # point, basket and the space around them -- is in the frame.
    d435_pos = np.array([HEAD_FRONT_X + CAMERA_BODY_HALF_HEIGHT, 0.0, CAMERA_Z])
    # Aim at the midline so the same D435 frame covers both mirrored workspaces.
    d435_target = np.array([0.30, 0.0, TABLE_TOP_Z])
    arm.worldbody.add_camera(
        name="d435_head",
        pos=d435_pos.tolist(),
        quat=_camera_quat(d435_pos, d435_target),
        fovy=55,
    )
    arm.worldbody.add_camera(
        name="realsense_d435",
        pos=d435_pos.tolist(),
        quat=_camera_quat(d435_pos, d435_target),
        fovy=55,
    )
    arm.worldbody.add_geom(
        name="realsense_d435_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0125, 0.045, 0.0125],
        pos=d435_pos.tolist(),
        rgba=[0.3, 0.3, 0.35, 1.0],
        contype=0,
        conaffinity=0,
    )
    # Green camera bracket (visual only): from the head top up to the camera.
    head_top = SHOULDER_AXIS_Z + HEAD_TOP_ABOVE_SHOULDER_AXIS
    bracket_bottom, bracket_top = head_top, CAMERA_Z - CAMERA_BODY_HALF_HEIGHT
    if bracket_top > bracket_bottom:
        arm.worldbody.add_geom(
            name="camera_bracket_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.03, 0.5 * (bracket_top - bracket_bottom)],
            pos=[HEAD_FRONT_X - 0.02, 0.0, 0.5 * (bracket_top + bracket_bottom)],
            rgba=[0.1, 0.6, 0.35, 1.0],
            contype=0,
            conaffinity=0,
        )
    # The pedestal body carries both arms, so raising it raises the shoulders too.
    pedestal = arm.body("openarm_body_link0")
    pedestal.pos = np.asarray(pedestal.pos) + [0.0, 0.0, PEDESTAL_RAISE]
    for side in MOUNTS:
        _remove_stock_gripper(arm, side)
        _attach_hand(arm, side)
    # attach_body merges the hand file's actuator defaults over the arm servos; put
    # the arm's own gains back (see configure_arm_servos).
    configure_arm_servos(arm)

    # Right arm wrist camera (Eye-in-Hand camera): mounted on the right wrist looking down along the fingers
    wrist_cam_pos = np.array([-0.04, -0.02, 0.09])
    wrist_cam_target = np.array([-0.25, -0.02, -0.02])
    wrist_cam_quat = _camera_quat(wrist_cam_pos, wrist_cam_target)
    arm.body("openarm_right_link7").add_camera(
        name="right_wrist_camera",
        pos=wrist_cam_pos.tolist(),
        quat=wrist_cam_quat,
        fovy=70,
    )

    if pick_bottle:
        riser_xy = np.array([-0.03, 0.0])
        riser_half = np.array([0.13, 0.10])
        pick_xy = np.asarray(pick_position, dtype=float)
        if np.all(np.abs(pick_xy - riser_xy) < riser_half + OBJECT_RADIUS):
            raise ValueError("pick object overlaps the robot base")

        # ── Resolve YCB mesh and texture from local assets/ycb/ ──────────
        obj_dir = YCB_ASSET_ROOT / YCB_PICK_OBJECT_NAME
        # Strip the "ycb_" prefix to get the bare asset name.
        bare_name = YCB_PICK_OBJECT_NAME.removeprefix("ycb_")
        mesh_file = obj_dir / "meshes" / f"{bare_name}.obj"
        texture_file = obj_dir / "textures" / f"{bare_name}.png"
        if not mesh_file.is_file():
            raise FileNotFoundError(
                f"YCB mesh asset missing: {mesh_file}  "
                f"(copy from P-161/objects/ycb/{YCB_PICK_OBJECT_NAME}/)"
            )
        # ── Load mesh with real texture ──────────────────────────────────
        # The soup can's mesh is already centred on its own origin (measured
        # AABB centre is within 0.5 mm of zero), so no refpos correction.
        arm.add_mesh(name="ycb_pick_object_mesh", file=str(mesh_file), scale=[1.0, 1.0, 1.0])

        # Texture & material: gives the visual geom the Campbell's label
        # instead of a flat colour. If the texture PNG is missing we fall
        # back gracefully to the geom rgba.
        _has_texture = texture_file.is_file()
        if _has_texture:
            arm.add_texture(
                name="ycb_can_texture",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(texture_file),
            )
            # MjSpec.add_material `textures` slot list: index-1 = diffuse map
            # (matches MuJoCo XML's `texture=` attribute which maps to slot 1).
            arm.add_material(
                name="ycb_can_material",
                textures=["", "ycb_can_texture", "", "", "", "", "", "", "", ""],
                rgba=[1.0, 1.0, 1.0, 1.0],
                specular=0.3,
                shininess=0.2,
            )

        # ── Pick object body ─────────────────────────────────────────────
        # Pick location A, standing on the table top.
        bottle = arm.worldbody.add_body(
            name="pick_bottle",
            pos=[float(pick_position[0]), float(pick_position[1]), OBJECT_HALF_HEIGHT + TABLE_TOP_Z + work_platform_height],
        )
        bottle.add_freejoint(name="pick_bottle_joint")
        bottle.add_geom(
            name="pick_bottle_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            # Collision cylinder matched to the real mesh AABB:
            # radius 3.40 cm, half-height 5.09 cm.
            size=[OBJECT_RADIUS, OBJECT_HALF_HEIGHT, 0.0],
            mass=0.2,
            friction=[1.2, 0.02, 0.002],
            rgba=[0.0, 0.0, 0.0, 0.0],
        )

        # Visual geom: textured mesh with the real Campbell's soup label.
        # When a texture is loaded, rgba must be white [1,1,1,1] because MuJoCo
        # multiplies the geom rgba with the texture colour – any tint would
        # obscure the label artwork.  Without texture we fall back to flat red.
        visual_rgba = [1.0, 1.0, 1.0, 1.0] if _has_texture else [0.80, 0.16, 0.12, 1.0]
        visual_kwargs = dict(
            name="ycb_mustard_bottle_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="ycb_pick_object_mesh",
            # The mesh is centred on its own origin (AABB z -0.0516..+0.0502), so it
            # must sit on the collision cylinder's origin. The old -OBJECT_HALF_HEIGHT
            # offset drew the can 5cm below its physics body: half sunk into the table
            # while the collision cylinder stood on top of it.
            pos=[0.0, 0.0, OBJECT_HALF_HEIGHT - MESH_HALF_HEIGHT_BELOW],
            mass=0.0,
            rgba=visual_rgba,
            contype=0,
            conaffinity=0,
            group=2,
        )
        if _has_texture:
            visual_kwargs["material"] = "ycb_can_material"
        bottle.add_geom(**visual_kwargs)
        # Release point B. Full physical collision on all 4 walls, so a hand that comes
        # in too low is caught by the trajectory's basket-contact check.
        distance = float(np.hypot(basket_position[0] - pick_position[0], basket_position[1] - pick_position[1]))
        if distance < 0.15:
            raise ValueError(f"basket must be at least 15cm from the pick point, got {distance*100:.1f}cm")
        basket_xy = np.asarray(basket_position, dtype=float)
        basket_outer_half = BASKET_HALF_WIDTH + BASKET_WALL_THICKNESS
        if np.all(np.abs(basket_xy - riser_xy) < riser_half + basket_outer_half):
            raise ValueError("basket overlaps the robot base")
        # Optional rectangular stand under the basket (same footprint as the basket's
        # outside), raising it toward the shoulders: the robot stands on the table
        # with its shoulders 0.70m up, so a basket on the bare top sits at the very
        # edge of both arms' reach. Full collision; the executor treats it like the
        # basket (any hand/arm contact aborts the trajectory).
        if work_platform_height < 0.0:
            raise ValueError("work_platform_height must be >= 0")
        if work_platform_height > 0.0:
            x0, x1 = WORK_PLATFORM_X
            for xy, label in ((pick_position, "pick point"), (basket_position, "basket")):
                if not (x0 + OBJECT_RADIUS <= float(xy[0]) <= x1 - OBJECT_RADIUS):
                    raise ValueError(f"{label} x={xy[0]} is off the work platform {WORK_PLATFORM_X}")
            arm.worldbody.add_geom(
                name="work_platform",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.5 * (x0 + x1), 0.0, TABLE_TOP_Z + 0.5 * work_platform_height],
                size=[0.5 * (x1 - x0), TABLE_HALF_WIDTH, 0.5 * work_platform_height],
                friction=[1.0, 0.005, 0.0001],
                rgba=[0.82, 0.74, 0.60, 1.0],
            )
        if basket_stand_height < 0.0:
            raise ValueError("basket_stand_height must be >= 0")
        if basket_stand_height > 0.0:
            arm.worldbody.add_geom(
                name="place_basket_stand",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[float(basket_position[0]), float(basket_position[1]), TABLE_TOP_Z + 0.5 * basket_stand_height],
                size=[basket_outer_half, basket_outer_half, 0.5 * basket_stand_height],
                rgba=[0.55, 0.45, 0.35, 1.0],
            )
        basket = arm.worldbody.add_body(
            name="place_basket",
            pos=[float(basket_position[0]), float(basket_position[1]), BASKET_FLOOR_Z + basket_stand_height + work_platform_height],
        )
        basket_color = [0.1, 0.55, 0.2, 1.0]
        bw = BASKET_HALF_WIDTH
        wh = 0.5 * (BASKET_WALL_HEIGHT - 0.005)
        wz = 0.005 + wh
        basket.add_geom(name="place_basket_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bw, bw, 0.005], rgba=basket_color)
        # Optional V-shaped insert on the floor: two low-friction plates sloping down
        # at `basket_floor_tilt_deg` into a groove along x at y = BASKET_VALLEY_Y (toward
        # the left arm). A can set down anywhere on the -y side slides into the groove
        # and stands there upright, with free room on its +y side for the left hand's
        # fingers (a single slope parked it against the +y wall, where the fingers
        # could not get round it). Priority 1: the contact uses the plates' friction,
        # not the can's 1.2 -- a smooth plastic insert.
        if basket_floor_tilt_deg:
            theta = np.deg2rad(basket_floor_tilt_deg)
            thick, groove_y, groove_z = 0.004, BASKET_VALLEY_Y, 0.005 + 0.0005
            for name, y0, y1, sign in (("a", -bw, groove_y, -1.0), ("b", groove_y, bw, 1.0)):
                length = y1 - y0
                rise = length * np.tan(theta)
                normal = np.array([0.0, -sign * np.sin(theta), np.cos(theta)])
                top_mid = np.array([0.0, 0.5 * (y0 + y1), groove_z + 0.5 * rise])
                centre = top_mid - normal * 0.5 * thick
                half = 0.5 * sign * theta
                basket.add_geom(
                    name=f"place_basket_slope_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=centre.tolist(), quat=[float(np.cos(half)), float(np.sin(half)), 0.0, 0.0],
                    size=[bw, 0.5 * length / np.cos(theta), 0.5 * thick],
                    friction=[BASKET_SLOPE_FRICTION, 0.005, 0.0001], priority=1, rgba=[0.3, 0.75, 0.4, 1.0],
                )
        basket.add_geom(name="place_basket_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bw + 0.5 * BASKET_WALL_THICKNESS, 0.0, wz], size=[0.5 * BASKET_WALL_THICKNESS, bw + 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-(bw + 0.5 * BASKET_WALL_THICKNESS), 0.0, wz], size=[0.5 * BASKET_WALL_THICKNESS, bw + 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, bw + 0.5 * BASKET_WALL_THICKNESS, wz], size=[bw, 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, -(bw + 0.5 * BASKET_WALL_THICKNESS), wz], size=[bw, 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
    return arm


def build_five_finger_model(
    *, pick_bottle: bool = False, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B,
    arm_half_separation: float | None = None, left_arm_mount_yaw_deg: float | None = None,
    right_arm_mount_yaw_deg: float | None = None, basket_stand_height: float = 0.0,
    basket_floor_tilt_deg: float = 0.0, work_platform_height: float = 0.0,
) -> mujoco.MjModel:
    model = build_five_finger_spec(
        pick_bottle=pick_bottle, pick_position=pick_position, basket_position=basket_position,
        arm_half_separation=arm_half_separation, left_arm_mount_yaw_deg=left_arm_mount_yaw_deg,
        right_arm_mount_yaw_deg=right_arm_mount_yaw_deg, basket_stand_height=basket_stand_height,
        basket_floor_tilt_deg=basket_floor_tilt_deg, work_platform_height=work_platform_height,
    ).compile()
    _stiffen_arm_actuators(model)
    _soften_hand_actuators(model)
    _soften_finger_contacts(model)
    return model


# Peak torque of the real OpenArm v1 joint motors (vendor openarm_bimanual.xml):
# DM8009 on joints 1-2, DM4340 on joints 3-4, DM4310 on joints 5-7. The servos used
# to be capped at 120 N*m on every joint, 3-17x what the motors can give.
ARM_MOTOR_TORQUE_LIMIT_NM = {1: 40.0, 2: 40.0, 3: 27.0, 4: 27.0, 5: 7.0, 6: 7.0, 7: 7.0}


def _stiffen_arm_actuators(model: mujoco.MjModel) -> None:
    """Keep planned 7-DOF poses under the added Inspire-hand payload, within the
    torque the real joint motors can deliver."""
    for actuator in range(model.nu):
        name = model.actuator(actuator).name or ""
        if not name.startswith(("left_joint", "right_joint")):
            continue
        joint = int(name.split("joint")[1][0])
        limit = ARM_MOTOR_TORQUE_LIMIT_NM[joint]
        model.actuator_gainprm[actuator, 0] = 800.0
        model.actuator_biasprm[actuator, 1] = -800.0
        model.actuator_biasprm[actuator, 2] = -12.0
        model.actuator_forcerange[actuator] = [-limit, limit]


def _soften_finger_contacts(model: mujoco.MjModel) -> None:
    """Give the finger collision geoms a normal contact time constant.

    The Inspire asset ships its finger collision geoms with solref[0] = 0.002s,
    only twice the 0.001s timestep. That is a near-rigid contact: a finger that
    closes onto the bottle resolves the overlap in a couple of steps, which
    measured out as 135N-10kN spikes against a 1.44kg bottle and flipped it on
    its side every time. 0.02s is MuJoCo's own default and models a fingertip
    that gives slightly under load, like the real hand's rubberised pads.
    """
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name or ""
        if not body.startswith(HAND_PREFIX):
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue
        model.geom_solref[geom, 0] = max(float(model.geom_solref[geom, 0]), 0.02)
        # The vendor left-hand collision meshes need the rubber-pad coefficient to
        # match the measured right-hand hold. Calibrate this again with RH56F1 data.
        if body.startswith(f"{HAND_PREFIX}left_"):
            model.geom_friction[geom, 0] = 2.0


def _soften_hand_actuators(model: mujoco.MjModel) -> None:
    """Give the finger position servos hardware-like compliance.

    The stock gains are stiff enough that a finger meeting the bottle keeps
    driving through the contact, and the solver resolves that by shoving the
    bottle away instead of stopping the finger -- the object squirts out of the
    hand while it closes. The real RH56DFX is tendon-driven and backdrivable, so
    a finger that meets an obstacle stalls. Dropping kp to 40 with light damping
    reproduces that: the finger stops on the surface and settles at a modest
    grip force, which is also what makes the measured per-finger forces in
    _finger_contact_forces() usable as a contact test.
    """
    for actuator in range(model.nu):
        if not (model.actuator(actuator).name or "").startswith(HAND_PREFIX):
            continue
        model.actuator_gainprm[actuator, 0] = 40.0
        model.actuator_biasprm[actuator, 1] = -40.0
        model.actuator_biasprm[actuator, 2] = -1.5
