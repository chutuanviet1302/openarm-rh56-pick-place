# Docker: ROS 2 Humble environment for `openarm_pick_place`

Runs the ROS 2 package on Windows via Docker Desktop (WSL2 backend), without a
native ROS 2 install. Verified working end to end on 2026-09-18.

## Build the image

```bash
docker build -f docker/Dockerfile.ros2 -t openarm-ros2:humble .
```

~2.3 GB (`ros:humble-ros-base` + colcon + cv_bridge + tf2 + control/moveit/trajectory
msgs). MoveIt 2 and realsense2_camera itself are deliberately left out for now (see
the commented block in the Dockerfile) — add them when actually driving a live
camera or MoveIt planning scene; they roughly double the image size.

## Build + run the ROS 2 package

`custom_ros_messages` (the `MotorCmd`/`MotorCmds` types the hand driver uses) is a
lab-internal package not on any public index, so a **placeholder** stand-in lives
at `docker/stubs/custom_ros_messages/` — read its README before trusting anything
about the message shape. Replace that whole directory with the lab's real package
as soon as it's available.

```bash
# Git Bash on Windows: disable MSYS path mangling for the -v / -w flags
export MSYS_NO_PATHCONV=1

docker run --rm \
  -v "$(pwd -W)":/workspace/src/openarm_pick_place \
  -v "$(pwd -W)/docker/stubs/custom_ros_messages":/workspace/src/custom_ros_messages \
  -w //workspace \
  openarm-ros2:humble \
  bash -lc "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
```

Run the pure-Python test suite (no camera/MoveIt/hardware needed):

```bash
docker run --rm -v "$(pwd -W)":/workspace/src/openarm_pick_place \
  -w //workspace/src/openarm_pick_place \
  openarm-ros2:humble \
  bash -lc "source /opt/ros/humble/setup.bash && python3 -m unittest tests.test_mvp -v"
```

Try the actual nodes (they will sit in `rclpy.spin()` waiting for camera data /
action goals that never arrive without real hardware — that is success, not a
hang; Ctrl+C or `timeout N` to stop):

```bash
docker run --rm -it \
  -v "$(pwd -W)":/workspace/src/openarm_pick_place \
  -v "$(pwd -W)/docker/stubs/custom_ros_messages":/workspace/src/custom_ros_messages \
  -w //workspace \
  openarm-ros2:humble \
  bash -lc "source /opt/ros/humble/setup.bash && colcon build --symlink-install && \
    source install/setup.bash && cd src/openarm_pick_place && \
    ros2 run openarm_pick_place d435_perception --ros-args -p config_path:=config.example.json"
```

## What this has confirmed (2026-09-18)

- `colcon build` succeeds for `openarm_pick_place` — the package is packaged correctly.
- `python3 -m unittest tests.test_mvp` — 9/9 pass under real ROS 2 Humble (not just
  the Windows venv): `fakes`, `geometry`, `models`, `motion`, `perception`,
  `pipeline` are all sound.
- With the placeholder message package, both `d435_perception` and
  `motion_planning` **construct and spin cleanly** — every subscriber, publisher,
  action client (`FollowJointTrajectory`) and service client (`GetPositionIK`)
  is created without error. `cv_bridge`, `tf2_ros` and the message imports all
  resolve. The only reason they don't do anything is that there is no real D435
  publishing images and no real MoveIt/`ros2_control` server to talk to.

**Conclusion:** the ROS 2 code in this repo is not the blocker. What remains to
actually run the pipeline is external: (1) the lab's real `custom_ros_messages`
package, (2) a MoveIt 2 config for OpenArm v2, (3) `ros2_control` with a
`FollowJointTrajectory` controller, (4) a real or bagged D435 feed. See
`BAO_CAO_MENTOR.md` question 1.
