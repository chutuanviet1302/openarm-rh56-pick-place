from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulation.pick_place.config import ARM_JOINTS
from simulation.pick_place.executor import Executor
from simulation.pick_place.scene import Scene


def trajectory_to_executor(joint_names, points, expected_names):
    """Return reordered waypoints and per-segment durations for Executor.follow."""
    if set(joint_names) != set(expected_names) or len(joint_names) != len(expected_names):
        raise ValueError("trajectory joint names do not match the OpenArm arm")
    order = [joint_names.index(name) for name in expected_names]
    waypoints, durations, previous = [], [], 0.0
    for point in points:
        if len(point.positions) != len(joint_names):
            raise ValueError("trajectory point has the wrong number of positions")
        timestamp = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
        if timestamp <= previous:
            raise ValueError("trajectory times must be strictly increasing")
        waypoints.append(np.asarray(point.positions, dtype=float)[order])
        durations.append(timestamp - previous)
        previous = timestamp
    if not waypoints:
        raise ValueError("trajectory has no points")
    return waypoints, durations


def arm_action_topics(config: dict) -> dict[str, str]:
    """Read the bimanual schema while accepting the original right-only config."""
    ros = config["ros"]
    if "arm_actions" in ros:
        topics = ros["arm_actions"]
        if set(topics) != {"left", "right"} or not all(isinstance(value, str) and value for value in topics.values()):
            raise ValueError("ros.arm_actions must contain non-empty left and right topics")
        return dict(topics)
    if isinstance(ros.get("arm_action"), str) and ros["arm_action"]:
        return {"right": ros["arm_action"]}
    raise ValueError("configure ros.arm_actions for left/right (or legacy ros.arm_action for right)")


def main(args=None) -> None:
    try:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from custom_ros_messages.msg import MotorCmds
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as error:
        raise RuntimeError("Source ROS 2 and build custom_ros_messages before starting the bridge") from error

    class MujocoBridge(Node):
        def __init__(self):
            super().__init__("mujoco_openarm_bridge")
            self.declare_parameter("config_path", "config.json")
            config = json.loads(Path(self.get_parameter("config_path").value).read_text(encoding="utf-8"))
            self.scene = Scene()
            self.sim_executor = Executor(self.scene, on_step=self._on_step)
            self.publisher = self.create_publisher(JointState, config["ros"]["joint_states"], 10)
            self.create_subscription(MotorCmds, config["ros"]["hand_command"], self._on_hand, 10)
            self.actions = {
                side: ActionServer(
                    self,
                    FollowJointTrajectory,
                    topic,
                    execute_callback=lambda goal, side=side: self._execute(side, goal),
                    goal_callback=lambda request, side=side: self._goal(side, request),
                    cancel_callback=lambda _: CancelResponse.REJECT,
                )
                for side, topic in arm_action_topics(config).items()
            }

        def _goal(self, side, request):
            try:
                trajectory_to_executor(request.trajectory.joint_names, request.trajectory.points, ARM_JOINTS[side])
            except ValueError as error:
                self.get_logger().error(str(error))
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        def _execute(self, side, goal_handle):
            result = FollowJointTrajectory.Result()
            try:
                waypoints, durations = trajectory_to_executor(
                    goal_handle.request.trajectory.joint_names,
                    goal_handle.request.trajectory.points,
                    ARM_JOINTS[side],
                )
                self.sim_executor.follow({f"{side}_arm": waypoints}, durations)
            except Exception as error:
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                result.error_string = str(error)
                goal_handle.abort()
                return result
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
            return result

        def _on_hand(self, message):
            commands = list(message.motor_commands)
            if len(commands) != 12:
                self.get_logger().error("hand command must contain right[6] + left[6] motors")
                return
            for side, values in (("right", commands[:6]), ("left", commands[6:])):
                actuators = self.scene.hand_actuators[side]
                target = np.asarray([command.q for command in values], dtype=float)
                limits = self.scene.model.actuator_ctrlrange[actuators]
                self.scene.data.ctrl[actuators] = np.clip(target, limits[:, 0], limits[:, 1])

        def _on_step(self, _):
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            for side in ("left", "right"):
                message.name.extend(ARM_JOINTS[side])
                message.position.extend(self.scene.data.qpos[self.scene.arm_qpos[side]].tolist())
                for actuator in self.scene.hand_actuators[side]:
                    joint = int(self.scene.model.actuator_trnid[int(actuator), 0])
                    message.name.append(f"inspire_{side}_{self.scene.model.joint(joint).name}")
                    message.position.append(float(self.scene.data.qpos[self.scene.model.jnt_qposadr[joint]]))
            self.publisher.publish(message)

        def destroy_node(self):
            for action in self.actions.values():
                action.destroy()
            super().destroy_node()

    rclpy.init(args=args)
    node = MujocoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
