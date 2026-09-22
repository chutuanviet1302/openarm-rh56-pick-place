#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-$HOME/ros2_ws}"
project="${OPENARM_PROJECT_ROOT:?Set OPENARM_PROJECT_ROOT to the repository path}"
description="$workspace/src/openarm_description"
moveit="$workspace/src/openarm_ros2/openarm_bimanual_moveit_config"
overlay="$project/ros2/openarm_rh56_description"

ln -sfn "$overlay" "$workspace/src/openarm_rh56_description"
mkdir -p "$overlay/meshes"
cp "$project"/assets/rh56_controller/h1_mujoco/archive/inspire/assets/visual/R_hand_base_link.STL "$overlay/meshes/"
cp "$project"/assets/rh56_controller/h1_mujoco/archive/inspire/assets/visual/L_hand_base_link.STL "$overlay/meshes/"
cp "$project"/assets/rh56_controller/h1_mujoco/archive/inspire/assets/visual/right_*_visual.stl "$overlay/meshes/"
cp "$project"/assets/rh56_controller/h1_mujoco/archive/inspire/assets/visual/left_*_visual.stl "$overlay/meshes/"

python3 - "$description" "$moveit" <<'PY'
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import yaml

description, moveit = map(Path, sys.argv[1:])
top = description / "assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro"
text = top.read_text()
if '<xacro:arg name="ee_type"' not in text:
    text = text.replace('<xacro:arg name="ros2_control"', '<xacro:arg name="ee_type" default="parallel_link" />\n  <xacro:arg name="ros2_control"')
text = text.replace('ee_type="parallel_link"', 'ee_type="$(arg ee_type)"')
top.write_text(text)

robot_macro = description / "assets/robot/openarm_v1.0/urdf/robot/openarm_robot.xacro"
text = robot_macro.read_text()
text = text.replace('hand="true"\n            ee_type="${ee_type}"', 'hand="${ee_type == \'parallel_link\'}"\n            ee_type="${ee_type}"')
robot_macro.write_text(text)

dispatcher = description / "assets/robot/openarm_v1.0/urdf/ee/ee_dispatcher.xacro"
text = dispatcher.read_text()
if "openarm_rh56_description" not in text:
    text = text.replace(
        '<xacro:include filename="$(find openarm_description)/assets/robot/openarm_v1.0/urdf/ee/parallel_link/openarm_parallel_gripper.xacro" />',
        '<xacro:include filename="$(find openarm_description)/assets/robot/openarm_v1.0/urdf/ee/parallel_link/openarm_parallel_gripper.xacro" />\n  <xacro:include filename="$(find openarm_rh56_description)/urdf/rh56_right.xacro" />',
    )
    old = '''    <xacro:unless value="${ee_type == 'parallel_link'}">
      ${xacro.error('assets/robot/openarm_v1.0 only supports ee_type:=parallel_link.')}
    </xacro:unless>'''
    new = '''    <xacro:if value="${ee_type == 'rh56_bimanual'}">
      <xacro:if value="${arm_prefix == 'left_'}"><xacro:rh56_left parent="${connected_to}"/></xacro:if>
      <xacro:if value="${arm_prefix == 'right_'}"><xacro:rh56_right parent="${connected_to}"/></xacro:if>
    </xacro:if>
    <xacro:unless value="${ee_type in ('parallel_link', 'rh56_bimanual')}">
      ${xacro.error('Unsupported ee_type: ' + ee_type)}
    </xacro:unless>'''
    text = text.replace(old, new)
else:
    text = text.replace("ee_type == 'rh56_right'", "ee_type == 'rh56_bimanual'")
    text = text.replace("('parallel_link', 'rh56_right')", "('parallel_link', 'rh56_bimanual')")
    text = text.replace(
        '''      <xacro:if value="${arm_prefix == 'left_'}">
        <xacro:openarm_parallel_gripper connected_to="${connected_to}" arm_type="${arm_type}" arm_prefix="${arm_prefix}" ee_type="parallel_link" ee_kinematics="${ee_kinematics}" ee_kinematics_link="${ee_kinematics_link}" ee_inertials="${ee_inertials}" ee_joints_limits="${ee_joints_limits}"/>
      </xacro:if>''',
        '''      <xacro:if value="${arm_prefix == 'left_'}"><xacro:rh56_left parent="${connected_to}"/></xacro:if>''',
    )
dispatcher.write_text(text)

launch = moveit / "launch/demo.launch.py"
text = launch.read_text()
text = text.replace('"ee_type": "rh56_right"', '"ee_type": "rh56_bimanual"')
if '"ee_type": "rh56_bimanual"' not in text:
    text = text.replace('"ros2_control": "true",', '"ros2_control": "true",\n            "ee_type": "rh56_bimanual",')
text = text.replace('            TimerAction(period=1.0, actions=[gripper_spawner]),\n', '')
text = text.replace(
    'additional_env={"MESA_D3D12_DEFAULT_ADAPTER_NAME": "Intel"}',
    'additional_env={"LIBGL_ALWAYS_SOFTWARE": "1", "QT_QPA_PLATFORM": "xcb"}',
)
if 'additional_env={"LIBGL_ALWAYS_SOFTWARE": "1", "QT_QPA_PLATFORM": "xcb"}' not in text:
    text = text.replace(
        '            name="rviz2",\n            output="log",',
        '            name="rviz2",\n            output="log",\n            additional_env={"LIBGL_ALWAYS_SOFTWARE": "1", "QT_QPA_PLATFORM": "xcb"},',
    )
launch.write_text(text)

limits_path = moveit / "config/openarm_v1.0/joint_limits.yaml"
limits = yaml.safe_load(limits_path.read_text())
for joint in ("openarm_left_finger_joint1", "openarm_right_finger_joint1"):
    limits["joint_limits"].pop(joint, None)
limits_path.write_text(yaml.safe_dump(limits, sort_keys=False))

controllers_path = moveit / "config/openarm_v1.0/moveit_controllers.yaml"
controllers = yaml.safe_load(controllers_path.read_text())
manager = controllers["moveit_simple_controller_manager"]
manager["controller_names"] = [name for name in manager["controller_names"] if "gripper" not in name]
manager.pop("left_gripper_controller", None)
manager.pop("right_gripper_controller", None)
controllers_path.write_text(yaml.safe_dump(controllers, sort_keys=False))

srdf = moveit / "config/openarm_v1.0/openarm_bimanual.srdf"
original = subprocess.check_output(
    ["git", "show", "HEAD:openarm_bimanual_moveit_config/config/openarm_v1.0/openarm_bimanual.srdf"],
    cwd=moveit.parent,
    text=True,
)
root = ET.fromstring(original)
removed_groups = {"left_gripper", "right_gripper"}
removed_links = {
    "openarm_left_hand", "openarm_left_left_finger", "openarm_left_right_finger",
    "openarm_right_hand", "openarm_right_left_finger", "openarm_right_right_finger",
}
for element in list(root):
    if element.tag == "group" and element.get("name") in removed_groups:
        root.remove(element)
    elif element.tag == "group_state" and element.get("group") in removed_groups:
        root.remove(element)
    elif element.tag in {"end_effector", "passive_joint"}:
        root.remove(element)
    elif element.tag == "disable_collisions" and ({element.get("link1"), element.get("link2")} & removed_links):
        root.remove(element)
ET.indent(root, space="    ")
srdf.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n")
PY

set +u
source /opt/ros/humble/setup.bash
set -u
cd "$workspace"
colcon build --symlink-install --packages-select openarm_rh56_description openarm_description openarm_bimanual_moveit_config
