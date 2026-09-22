# OpenArm + Inspire RH56F1 description

This ROS 2 package replaces the right OpenArm parallel gripper with the right-hand
Inspire RH56F1 model. The mount transform and six actuated joints match the MuJoCo
model in `assets/rh56_controller/h1_mujoco/archive/inspire/inspire_right.xml`.

The coupled distal joints use URDF `mimic` joints. MoveIt should plan the seven arm
joints only; the RH56 remains controlled separately.

The detailed STL files are visual-only for now. Simplified collision primitives and
their allowed-collision matrix must be calibrated before hardware execution.
