from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from simulation.openarm_mujoco import official_model_path

INSPIRE_ROOT = Path(__file__).parents[1] / "assets/rh56_controller/h1_mujoco/archive/inspire"
# High-quality YCB object assets from P-161 project (mesh + texture).
# Each sub-folder has  meshes/<name>.obj  and  textures/<name>.png.
# Available objects: ycb_tomato_soup_can, ycb_apple, ycb_orange, ycb_peach,
#                    ycb_pear, ycb_plum, ycb_lemon, ycb_strawberry
YCB_ASSET_ROOT = Path(__file__).parents[1] / "assets" / "ycb"
YCB_PICK_OBJECT_NAME = "ycb_tomato_soup_can"
# Measured AABB of tomato_soup_can.obj: 6.79cm x 6.77cm x 10.19cm, centred at origin.
# The collision cylinder is kept at the original tuned dimensions (slightly wider
# than the mesh) because the pick-place trajectories are calibrated against them.
OBJECT_RADIUS = 0.0354   # collision cylinder radius  (mesh half-width ≈ 3.40 cm)
OBJECT_HALF_HEIGHT = 0.05  # collision cylinder half-height (mesh ≈ 5.09 cm)
# Distance from the mesh origin down to its bottom face; the visual is raised by the
# difference so the drawn bottom coincides with the collision bottom on the table.
MESH_HALF_HEIGHT_BELOW = 0.0516
TABLE_TOP_Z = 0.40
HAND_PREFIX = "inspire_"
# Default pick point A and basket point B (table-plane x, y). A is where a straight
# wrist (see pick_place_demo.NATURAL_GRASP_JOINTS) puts the hand's jaw; B is the
# nearest spot 15cm+ away that the same orientation still reaches. Both can be
# overridden per run (`pick_place_demo --object X Y --basket X Y`).
PICK_POSITION_A = (0.396, -0.275)
BASKET_POSITION_B = (0.40, -0.02)
BASKET_FLOOR_Z = 0.405
# Basket inner half-width and wall height. Sized for the hand, not the can: with a
# horizontal side grasp the palm's underside is only 3cm above the can's bottom at
# 8cm behind the can and ~7-10cm at 12cm behind it (measured hand profile), so a wall
# 5cm tall must be at least 12cm from the can's centre for the can to reach the floor
# without the palm resting on the rim. A 16cm basket left the palm sitting on the wall.
BASKET_HALF_WIDTH = 0.12
BASKET_WALL_HEIGHT = 0.05
BASKET_WALL_THICKNESS = 0.01
# How far the pedestal and both shoulders sit above the stock model's origin. This
# sets how far the arms have to reach *down* to work on the table, so it decides
# whether the object sits in the middle of the workspace or at its lower edge.
#
# 0.40 put the shoulders 70cm above the table. The wrist could then not get below
# z 0.72 anywhere over the table, but grasping the 10cm can at its waist needs the
# wrist at 0.635 -- so no grasp of it was reachable at all, at any tilt or position.
# At 0.30 every candidate object position on the right-hand side of the table solves.
PEDESTAL_RAISE = 0.10
# The stock pedestal is a floor-standing unit: a 20cm-tall base block, a bare square
# column and the torso housing the shoulders bolt to. Here the robot stands ON the
# table, so the base block is moved up onto the table top and the column shortened by
# the same amount, leaving the torso (and therefore the arms' reach) exactly where it
# was. Done on the mesh itself (vertices below PEDESTAL_CUT_Z shifted up by
# PEDESTAL_BASE_LIFT); the column has no intermediate vertices, so it simply shortens.
PEDESTAL_CUT_Z = 0.35          # world z separating the base block from the column
PEDESTAL_BASE_LIFT = 0.30      # foot 0.10 -> 0.40 = TABLE_TOP_Z
PEDESTAL_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "openarm"
# Flange -> Inspire hand base transform, derived from the two frames rather than tuned:
#
#   OpenArm v2 ee_base_link: the tool axis is -z (the chain runs 0 0 -L, the stock
#   gripper fingers sit at z=-0.068); link6's collision shell ends at z=-0.0285.
#   Inspire RH56 base frame:  +z = fingers forward, +x = palm side (fingers curl
#   toward +x), y across the palm (right hand: index/thumb at +y, pinky at -y).
#
# The hand therefore continues the forearm: hand +z -> flange -z, and the palm faces
# the robot's midline with the thumb forward when the arm hangs at rest (right hand:
# palm +y, thumb +x; left hand mirrored). Both are 180-degree rotations, about
# (1,1,0)/sqrt2 for the right and (1,-1,0)/sqrt2 for the left. The base sits just past
# link6's shell with a 1cm adapter plate in between (HAND_ADAPTER_THICKNESS).
#
# The earlier transform (180 degrees about (1,0,-1)) sent hand +z to flange -x, i.e.
# the fingers stuck out sideways at 80 degrees to the forearm, which is what made the
# hand look bolted on wrong and bent the wrist joints to compensate.
LINK6_SHELL_END = -0.0285
HAND_ADAPTER_THICKNESS = 0.010
HAND_MOUNT_Z = LINK6_SHELL_END - HAND_ADAPTER_THICKNESS
MOUNTS = {
    "right": ((0.0, 0.0, HAND_MOUNT_Z), (0.0, 0.70710678, 0.70710678, 0.0)),
    "left": ((0.0, 0.0, HAND_MOUNT_Z), (0.0, 0.70710678, -0.70710678, 0.0)),
}


def _read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    records = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]))
    return records["v"].reshape(-1, 3, 3).astype(np.float32).copy()


def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    records = np.zeros(len(triangles), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]))
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    records["n"] = normals
    records["v"] = triangles.reshape(-1, 9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(80) + len(triangles).to_bytes(4, "little") + records.tobytes())


def _pedestal_on_table(arm: mujoco.MjSpec) -> None:
    """Rewrite the pedestal meshes so the base block sits on the table top.

    The STL is in millimetres with the foot at z=0 (world z = 0.001*z + geom z). The
    generated files are cached in assets/openarm and regenerated when missing.
    """
    for mesh_name, geom_name in (("body_link0", "openarm_body_link0_visual"), ("body_link0_symp", "openarm_body_link0_collision")):
        mesh = arm.mesh(mesh_name)
        source = Path(arm.modelfiledir) / arm.meshdir / mesh.file if not Path(mesh.file).is_absolute() else Path(mesh.file)
        target = PEDESTAL_ASSET_DIR / f"{mesh_name}_on_table.stl"
        if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
            triangles = _read_binary_stl(source)
            geom_z = float(np.asarray(arm.geom(geom_name).pos)[2])
            world_z = triangles[:, :, 2] * 0.001 + geom_z
            triangles[:, :, 2] += np.where(world_z < PEDESTAL_CUT_Z, PEDESTAL_BASE_LIFT * 1000.0, 0.0).astype(np.float32)
            _write_binary_stl(target, triangles)
        mesh.file = str(target)


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
    flange = arm.body(f"openarm_{side}_ee_base_link")
    mount = flange.add_frame(pos=position, quat=quaternion)
    mount.attach_body(root, prefix=f"{HAND_PREFIX}{side}_")
    # Visual adapter plate between link6's shell and the hand base (no collision).
    flange.add_geom(
        name=f"{HAND_PREFIX}{side}_adapter",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, LINK6_SHELL_END - 0.5 * HAND_ADAPTER_THICKNESS],
        size=[0.028, 0.5 * HAND_ADAPTER_THICKNESS, 0.0],
        rgba=[0.75, 0.75, 0.78, 1.0],
        contype=0,
        conaffinity=0,
    )

    # Remove the bulky stock gripper-base and camera casing from the wrist
    # so the Inspire hand connects cleanly and directly to the forearm link.
    for geom_name in (
        f"ee_base_link_{side}_00",
        f"ee_base_link_{side}_01",
        f"ee_base_link_{side}_02",
        f"ee_base_link_{side}_collision_00",
    ):
        geom = arm.geom(geom_name)
        if geom is not None:
            arm.delete(geom)


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
    *, pick_bottle: bool = False, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B
) -> mujoco.MjSpec:
    if not INSPIRE_ROOT.is_dir():
        raise FileNotFoundError("Inspire RH56DFX assets missing; clone correlllab/rh56_controller with h1_mujoco")

    arm = mujoco.MjSpec.from_file(str(official_model_path()))
    arm.worldbody.add_camera(name="overhead", pos=[0.15, 0.0, 2.2], quat=[1, 0, 0, 0], fovy=50)
    arm.worldbody.add_camera(
        name="isometric",
        pos=[0.95, -0.85, 0.95],
        quat=_camera_quat(np.array([0.95, -0.85, 0.95]), np.array([0.25, -0.30, 0.45])),
        fovy=48,
    )
    arm.worldbody.add_camera(
        name="front_view",
        pos=[1.10, -0.22, 0.70],
        quat=_camera_quat(np.array([1.10, -0.22, 0.70]), np.array([0.25, -0.25, 0.45])),
        fovy=48,
    )
    arm.worldbody.add_camera(
        name="side_view",
        pos=[0.25, -1.15, 0.70],
        quat=_camera_quat(np.array([0.25, -1.15, 0.70]), np.array([0.25, -0.30, 0.45])),
        fovy=48,
    )
    # The close-up follows the pick point, so a relocated object stays framed.
    focus = np.array([pick_position[0], pick_position[1], TABLE_TOP_Z + OBJECT_HALF_HEIGHT])
    close_pos = focus + np.array([0.33, -0.23, 0.17])
    arm.worldbody.add_camera(name="close_grasp", pos=close_pos.tolist(), quat=_camera_quat(close_pos, focus), fovy=38)
    # Intel RealSense D435 overhead camera mounted on robot head looking down at the table
    # Matches camera_tf_publisher.py (Openarm-ROS2-robot-control): mounted at [0.08, 0.0, 0.70] looking at workspace [0.30, -0.25, 0.40]
    arm.worldbody.add_camera(
        name="d435_head",
        pos=[0.08, 0.0, 0.70],
        quat=_camera_quat(np.array([0.08, 0.0, 0.70]), np.array([0.30, -0.25, 0.40])),
        fovy=55,
    )
    arm.worldbody.add_camera(
        name="realsense_d435",
        pos=[0.08, 0.0, 0.70],
        quat=_camera_quat(np.array([0.08, 0.0, 0.70]), np.array([0.30, -0.25, 0.40])),
        fovy=55,
    )
    arm.worldbody.add_geom(
        name="realsense_d435_visual",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.0125, 0.045, 0.0125],
        pos=[0.08, 0.0, 0.70],
        rgba=[0.3, 0.3, 0.35, 1.0],
        contype=0,
        conaffinity=0,
    )
    table = arm.geom("table_top")
    # The table used to span x -0.20..0.90, i.e. it ran underneath the robot, which
    # stands at x=0. Arms resting at the sides were then over the table top, so once the
    # pedestal came down far enough to reach the can they hung *into* the table and every
    # run aborted on a table contact before it started. Moved forward to x 0.10..0.80 so
    # the robot stands behind its workbench, the way it would in reality; the object at
    # x=0.30 and the basket at x=0.36 both still sit well inside it.
    # T-shaped table: the work surface starts in front of the hanging arms (they occupy
    # x -0.04..0.12, |y| 0.105..0.19 when the robot stands at attention with the arms
    # straight down), and a narrow tongue behind it carries the pedestal base. So the
    # arms can hang freely beside the tongue without touching wood.
    table.size = np.array([0.285, 0.55, 0.04])   # x 0.13 .. 0.70
    table.pos = np.array([-0.055, 0.0, 0.0])
    # Pedestal mounting base on table: extends under the robot base so the robot is mounted on the table
    arm.body("table").add_geom(
        name="table_pedestal_mount",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.52, 0.0, 0.0],
        size=[0.15, 0.10, 0.04],   # x -0.20 .. 0.10, |y| <= 0.10: just the base block's footprint
        rgba=[0.82, 0.71, 0.55, 1.0],
    )
    for name in ("openarm_body_link0_visual", "openarm_body_link0_collision"):
        pedestal = arm.geom(name)
        pedestal.pos = np.asarray(pedestal.pos) + [0.0, 0.0, PEDESTAL_RAISE]
    _pedestal_on_table(arm)
    for side in MOUNTS:
        base = arm.body(f"openarm_{side}_base_link")
        base.pos = np.asarray(base.pos) + [0.0, 0.0, PEDESTAL_RAISE]
    for side in MOUNTS:
        arm.delete(arm.body(f"openarm_{side}_ee_inner_finger"))
        arm.delete(arm.body(f"openarm_{side}_ee_outer_finger"))
        _attach_hand(arm, side)

    # Right arm wrist camera (Eye-in-Hand camera): mounted on the right wrist looking down along the fingers
    wrist_cam_pos = np.array([-0.04, -0.02, 0.09])
    wrist_cam_target = np.array([-0.25, -0.02, -0.02])
    wrist_cam_quat = _camera_quat(wrist_cam_pos, wrist_cam_target)
    arm.body("openarm_right_ee_base_link").add_camera(
        name="right_wrist_camera",
        pos=wrist_cam_pos.tolist(),
        quat=wrist_cam_quat,
        fovy=70,
    )

    if pick_bottle:
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
        arm.delete(arm.body("bottle"))

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
            pos=[float(pick_position[0]), float(pick_position[1]), OBJECT_HALF_HEIGHT + TABLE_TOP_Z],
        )
        bottle.add_freejoint(name="pick_bottle_joint")
        bottle.add_geom(
            name="pick_bottle_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            # Collision cylinder matched to the real mesh AABB:
            # radius 3.40 cm, half-height 5.09 cm.
            size=[OBJECT_RADIUS, OBJECT_HALF_HEIGHT],
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
        basket = arm.worldbody.add_body(
            name="place_basket", pos=[float(basket_position[0]), float(basket_position[1]), BASKET_FLOOR_Z]
        )
        basket_color = [0.1, 0.55, 0.2, 1.0]
        bw = BASKET_HALF_WIDTH
        wh = 0.5 * (BASKET_WALL_HEIGHT - 0.005)
        wz = 0.005 + wh
        basket.add_geom(name="place_basket_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bw, bw, 0.005], rgba=basket_color)
        basket.add_geom(name="place_basket_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bw + 0.5 * BASKET_WALL_THICKNESS, 0.0, wz], size=[0.5 * BASKET_WALL_THICKNESS, bw + 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-(bw + 0.5 * BASKET_WALL_THICKNESS), 0.0, wz], size=[0.5 * BASKET_WALL_THICKNESS, bw + 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, bw + 0.5 * BASKET_WALL_THICKNESS, wz], size=[bw, 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
        basket.add_geom(name="place_basket_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, -(bw + 0.5 * BASKET_WALL_THICKNESS), wz], size=[bw, 0.5 * BASKET_WALL_THICKNESS, wh], rgba=basket_color)
    return arm


def build_five_finger_model(
    *, pick_bottle: bool = False, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B
) -> mujoco.MjModel:
    model = build_five_finger_spec(
        pick_bottle=pick_bottle, pick_position=pick_position, basket_position=basket_position
    ).compile()
    _stiffen_arm_actuators(model)
    _soften_hand_actuators(model)
    _soften_finger_contacts(model)
    return model


def _stiffen_arm_actuators(model: mujoco.MjModel) -> None:
    """Keep planned 7-DOF poses under the added Inspire-hand payload."""
    for actuator in range(model.nu):
        name = model.actuator(actuator).name or ""
        if not name.startswith(("left_joint", "right_joint")):
            continue
        model.actuator_gainprm[actuator, 0] = 800.0
        model.actuator_biasprm[actuator, 1] = -800.0
        model.actuator_biasprm[actuator, 2] = -12.0
        model.actuator_forcerange[actuator] = [-120.0, 120.0]


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
