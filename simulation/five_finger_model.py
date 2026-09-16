from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from simulation.openarm_mujoco import official_model_path

INSPIRE_ROOT = Path(__file__).parents[1] / "assets/rh56_controller/h1_mujoco/archive/inspire"
# The object to pick. Measured mesh sizes of the candidates in this dataset, against
# the hand's 9.3cm maximum jaw opening:
#   005_tomato_soup_can  7.1 x 7.0 x 10.0 cm   fits, and short enough to grip at its waist
#   006_mustard_bottle   6.3 x 10.4 x 18.8 cm  fits the jaw but is too tall to grip low
#   004_sugar_box        4.5 x 9.6 x 17.7 cm   same height problem
#   002_master_chef_can 10.2 x 10.4 x 14.0 cm  wider than the hand can open
YCB_PICK_OBJECT = (
    Path(__file__).parents[2]
    / "datasets_project/mujoco-ycb-dataset/asset/ycb/005_tomato_soup_can/poisson"
)
OBJECT_RADIUS = 0.0354
OBJECT_HALF_HEIGHT = 0.05
TABLE_TOP_Z = 0.40
HAND_PREFIX = "inspire_"
# How far the pedestal and both shoulders sit above the stock model's origin. This
# sets how far the arms have to reach *down* to work on the table, so it decides
# whether the object sits in the middle of the workspace or at its lower edge.
#
# 0.40 put the shoulders 70cm above the table. The wrist could then not get below
# z 0.72 anywhere over the table, but grasping the 10cm can at its waist needs the
# wrist at 0.635 -- so no grasp of it was reachable at all, at any tilt or position.
# At 0.30 every candidate object position on the right-hand side of the table solves.
PEDESTAL_RAISE = 0.10
# ponytail: provisional flange transforms; replace these two constants with measured CAD transforms.
MOUNTS = {
    "left": ((0.0, 0.0, 0.0), (0.0, 0.70710678, 0.0, -0.70710678)),
    "right": ((0.0, 0.0, 0.0), (0.0, 0.70710678, 0.0, -0.70710678)),
}


def _attach_hand(arm: mujoco.MjSpec, side: str) -> None:
    scene = INSPIRE_ROOT / f"inspire_{side}.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"Inspire RH56DFX asset missing: {scene}")
    hand = mujoco.MjSpec.from_file(str(scene))
    root = hand.body("base")
    # The supplied palm collision shell intersects OpenArm link 5 after mounting.
    # Palm contact is not used as grasp evidence, so keep the visible palm and let
    # only the articulated digits participate in collision.
    for geom in root.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    root.pos = np.zeros(3)
    root.quat = np.array([1.0, 0.0, 0.0, 0.0])
    position, quaternion = MOUNTS[side]
    mount = arm.body(f"openarm_{side}_ee_base_link").add_frame(pos=position, quat=quaternion)
    mount.attach_body(root, prefix=f"{HAND_PREFIX}{side}_")

    # The wrist flange is the stub the stock two-finger gripper bolted onto. With the
    # Inspire hand mounted over it the flange is entirely enclosed by the palm, and its
    # collision proxy overlaps the thumb's proximal link: the thumb sat 2cm *inside* it,
    # jammed at 18kN, so neither thumb joint could move at all (both rested outside their
    # own joint ranges, at -0.23 and -0.50 rad, while their actuators pushed uselessly).
    # That is what shrank the open jaw to 5.5cm -- narrower than the 6cm bottle -- and
    # left the frozen thumb parked inside the object on every approach. MuJoCo only
    # auto-excludes direct parent/child pairs, and the thumb is a grandchild of the
    # flange's body, so the exclusion has to be explicit. Nothing else can touch this
    # geom once the hand covers it, so disabling its collision is safe.
    flange = arm.geom(f"ee_base_link_{side}_collision_00")
    flange.contype = 0
    flange.conaffinity = 0


def build_five_finger_spec(*, pick_bottle: bool = False) -> mujoco.MjSpec:
    if not INSPIRE_ROOT.is_dir():
        raise FileNotFoundError("Inspire RH56DFX assets missing; clone correlllab/rh56_controller with h1_mujoco")

    arm = mujoco.MjSpec.from_file(str(official_model_path()))
    arm.worldbody.add_camera(name="overhead", pos=[0.15, 0.0, 2.2], quat=[1, 0, 0, 0], fovy=50)
    table = arm.geom("table_top")
    # The table used to span x -0.20..0.90, i.e. it ran underneath the robot, which
    # stands at x=0. Arms resting at the sides were then over the table top, so once the
    # pedestal came down far enough to reach the can they hung *into* the table and every
    # run aborted on a table contact before it started. Moved forward to x 0.10..0.80 so
    # the robot stands behind its workbench, the way it would in reality; the object at
    # x=0.30 and the basket at x=0.36 both still sit well inside it.
    table.size = np.array([0.35, 0.55, 0.04])
    table.pos = np.array([-0.12, 0.0, 0.0])
    for name in ("openarm_body_link0_visual", "openarm_body_link0_collision"):
        pedestal = arm.geom(name)
        pedestal.pos = np.asarray(pedestal.pos) + [0.0, 0.0, PEDESTAL_RAISE]
    for side in MOUNTS:
        base = arm.body(f"openarm_{side}_base_link")
        base.pos = np.asarray(base.pos) + [0.0, 0.0, PEDESTAL_RAISE]
    for side in MOUNTS:
        arm.delete(arm.body(f"openarm_{side}_ee_inner_finger"))
        arm.delete(arm.body(f"openarm_{side}_ee_outer_finger"))
        _attach_hand(arm, side)

    if pick_bottle:
        mesh_file = YCB_PICK_OBJECT / "textured.obj"
        if not mesh_file.is_file():
            raise FileNotFoundError(f"YCB {YCB_PICK_OBJECT.parents[1].name} asset missing: {YCB_PICK_OBJECT}")
        arm.delete(arm.body("bottle"))
        # The soup can's mesh is already centred on its own origin (measured AABB centre
        # is within 0.5mm of zero), so it needs no refpos correction.
        arm.add_mesh(name="ycb_pick_object_mesh", file=str(mesh_file), scale=[1.0, 1.0, 1.0])
        bottle = arm.worldbody.add_body(name="pick_bottle", pos=[0.30, -0.30, OBJECT_HALF_HEIGHT + TABLE_TOP_Z])
        bottle.add_freejoint(name="pick_bottle_joint")
        bottle.add_geom(
            name="pick_bottle_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            # A cylinder, matched to the mesh: radius 3.54cm, half-height 5.0cm. The can
            # replaces the mustard bottle because the bottle simply did not fit this hand:
            # 6.3cm across at best against a 9.3cm maximum jaw opening, and 18.8cm tall, so
            # any grip the arm could actually reach landed above the centre of mass and
            # rolled it over instead of holding it. The can is 7.0cm across (1.15cm of
            # clearance a side) and only 10cm tall, so the graspable band straddles its
            # centre of mass.
            size=[OBJECT_RADIUS, OBJECT_HALF_HEIGHT],
            mass=0.2,
            friction=[1.2, 0.02, 0.002],
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        bottle.add_geom(
            name="ycb_mustard_bottle_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="ycb_pick_object_mesh",
            mass=0.0,
            rgba=[0.80, 0.16, 0.12, 1.0],
            contype=0,
            conaffinity=0,
            group=2,
        )
        # Sits at the measured release point of the carried bottle. Kept only just wide
        # enough for the bottle's footprint: the arm's reachable pick and place zones
        # overlap heavily at this height, so a wider basket would run into the bottle's
        # own pick position instead of leaving a visible gap between the two.
        # B, 21cm from the object at A=(0.30, -0.15). Chosen from a reachability scan
        # rather than from the layout: with the wrist at the tilt the can needs, the
        # right arm reaches nothing on the left of the table at all, so a left-hand-side
        # basket has no release pose and the trial aborts before the robot moves. A
        # left-side B would need the left arm to do the placing.
        basket = arm.worldbody.add_body(name="place_basket", pos=[0.30, -0.15, 0.405])
        basket_color = [0.1, 0.55, 0.2, 1.0]
        basket.add_geom(name="place_basket_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.062, 0.062, 0.005], rgba=basket_color)
        basket.add_geom(name="place_basket_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.067, 0.0, 0.055], size=[0.005, 0.067, 0.05], rgba=basket_color)
        basket.add_geom(name="place_basket_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.067, 0.0, 0.055], size=[0.005, 0.067, 0.05], rgba=basket_color)
        basket.add_geom(name="place_basket_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.067, 0.055], size=[0.062, 0.005, 0.05], rgba=basket_color)
        basket.add_geom(name="place_basket_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, -0.067, 0.055], size=[0.062, 0.005, 0.05], rgba=basket_color)
    return arm


def build_five_finger_model(*, pick_bottle: bool = False) -> mujoco.MjModel:
    model = build_five_finger_spec(pick_bottle=pick_bottle).compile()
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
