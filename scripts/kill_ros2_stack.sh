#!/usr/bin/env bash
# Kill any leftover openarm demo.launch.py process tree (rviz2, move_group,
# ros2_control_node, robot_state_publisher, controller_manager spawners).
#
# Symptom this fixes: a second `ros2 launch ... demo.launch.py` picks up a
# stale controller_manager node still running from a previous session (e.g.
# the terminal was closed instead of Ctrl+C'd), so two nodes named
# "controller_manager" race on the same DDS domain. Spawners then log
# "Controller already loaded, skipping load_controller" followed by
# "Failed to configure controller" against the wrong instance, and
# RViz/MoveIt shows nothing moving. Run this, then relaunch.
set -euo pipefail

for pattern in \
    'ros2 launch openarm_bimanual_moveit_config' \
    'rviz2' \
    'ros2_control_node' \
    'move_group' \
    'robot_state_publisher' \
    'controller_manager/spawner'; do
    pkill -9 -f "$pattern" 2>/dev/null || true
done

sleep 1
leftover=$(ps aux | grep -iE 'ros2|rviz2|control_node|move_group|spawner|robot_state_pub' | grep -v grep | grep -v kill_ros2_stack || true)
if [ -n "$leftover" ]; then
    echo "Still running:"
    echo "$leftover"
    exit 1
fi
echo "Clean."
