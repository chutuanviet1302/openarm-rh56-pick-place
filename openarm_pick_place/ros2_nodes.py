from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, GraspConfig, Pose, Workspace
from .motion import pick_place_waypoints, validate_target
from .perception import estimate_object_pose


def inspire_command_positions(raw_values, side: str) -> list[float]:
    values = np.asarray(raw_values, dtype=float)
    if side not in ("left", "right") or values.shape != (6,) or np.any((values < 0) | (values > 1000)):
        raise ValueError("Inspire command requires side left/right and six raw values in [0, 1000]")
    selected = (values / 1000 * np.pi).tolist()
    opened = [np.pi] * 6
    return (selected + opened) if side == "right" else (opened + selected)


def _ros_imports():
    try:
        import rclpy
        from builtin_interfaces.msg import Duration as DurationMsg
        from control_msgs.action import FollowJointTrajectory
        from custom_ros_messages.msg import MotorCmd, MotorCmds
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
        from rclpy.action import ActionClient
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        from trajectory_msgs.msg import JointTrajectoryPoint
    except ImportError as error:
        raise RuntimeError("Source ROS 2 and install realsense2_camera, cv_bridge, MoveIt 2 and ros2_control") from error
    return locals()


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        return ((matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale, scale / 4)
    index = int(np.argmax(np.diag(matrix)))
    j, k = (index + 1) % 3, (index + 2) % 3
    scale = 2 * np.sqrt(1 + matrix[index, index] - matrix[j, j] - matrix[k, k])
    quaternion = np.zeros(4)
    quaternion[index] = scale / 4
    quaternion[3] = (matrix[k, j] - matrix[j, k]) / scale
    quaternion[j] = (matrix[j, index] + matrix[index, j]) / scale
    quaternion[k] = (matrix[k, index] + matrix[index, k]) / scale
    return tuple(quaternion)


def perception_main(args=None) -> None:
    ros = _ros_imports()
    rclpy, Node = ros["rclpy"], ros["Node"]

    class D435PerceptionNode(Node):
        def __init__(self):
            super().__init__("d435_object_perception")
            self.declare_parameter("config_path", "config.json")
            self.config = _load(self.get_parameter("config_path").value)
            camera, topics = self.config["camera"], self.config["ros"]
            self.bridge, self.depth, self.info = ros["CvBridge"](), None, None
            self.publisher = self.create_publisher(ros["PoseStamped"], topics["object_pose"], 10)
            self.create_subscription(ros["CameraInfo"], topics["camera_info"], self._on_info, 10)
            self.create_subscription(ros["Image"], topics["aligned_depth"], self._on_depth, 10)
            self.create_subscription(ros["Image"], topics["color_image"], self._on_color, 10)

        def _on_info(self, message):
            self.info = message

        def _on_depth(self, message):
            self.depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

        def _on_color(self, message):
            if self.depth is None or self.info is None:
                return
            camera = self.config["camera"]
            intrinsics = CameraIntrinsics(self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5], camera["depth_scale"])
            rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            detected = estimate_object_pose(rgb, self.depth, intrinsics, camera["hsv_lower"], camera["hsv_upper"], timestamp=stamp)
            if detected is None:
                return
            output = ros["PoseStamped"]()
            output.header = message.header
            output.pose.position.x, output.pose.position.y, output.pose.position.z = detected.pose.position
            output.pose.orientation.x, output.pose.orientation.y, output.pose.orientation.z, output.pose.orientation.w = _quaternion_xyzw(detected.pose.rotation)
            self.publisher.publish(output)

    rclpy.init(args=args)
    node = D435PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def motion_main(args=None) -> None:
    ros = _ros_imports()
    rclpy, Node = ros["rclpy"], ros["Node"]

    class MotionPlanningNode(Node):
        def __init__(self):
            super().__init__("openarm_motion_planning")
            self.declare_parameter("config_path", "config.json")
            self.config = _load(self.get_parameter("config_path").value)
            cfg, topics = self.config["motion"], self.config["ros"]
            if len(cfg["joint_names"]) != 7:
                raise ValueError("OpenArm command must contain exactly 7 joint names")
            self.workspace = Workspace(np.array(self.config["workspace"]["minimum"]), np.array(self.config["workspace"]["maximum"]))
            self.grasp = GraspConfig(np.array(cfg["grasp_offset"]), np.array(cfg["approach_offset"]))
            self.place = Pose(np.array(cfg["place_position"]))
            self.buffer = ros["Buffer"]()
            self.listener = ros["TransformListener"](self.buffer, self)
            self.ik = self.create_client(ros["GetPositionIK"], topics["compute_ik"])
            self.arm = ros["ActionClient"](self, ros["FollowJointTrajectory"], topics["arm_action"])
            self.hand = self.create_publisher(ros["MotorCmds"], topics["hand_command"], 10)
            self.create_subscription(ros["PoseStamped"], topics["object_pose"], self._on_pose, 10)
            self.busy, self.solutions, self.names, self.seed = False, [], [], None

        def _on_pose(self, message):
            age = self.get_clock().now().nanoseconds * 1e-9 - (message.header.stamp.sec + message.header.stamp.nanosec * 1e-9)
            if self.busy or age > self.config["motion"]["max_pose_age"]:
                return
            try:
                transform = self.buffer.lookup_transform(self.config["frames"]["base"], message.header.frame_id, ros["Time"].from_msg(message.header.stamp), timeout=ros["Duration"](seconds=0.2))
                from tf2_geometry_msgs import do_transform_pose
                base_message = do_transform_pose(message.pose, transform)
                position = np.array([base_message.position.x, base_message.position.y, base_message.position.z])
                q = base_message.orientation
                rotation = self._rotation(q.x, q.y, q.z, q.w)
                waypoints = pick_place_waypoints(Pose(position, rotation), self.place, self.grasp)
                for pose in waypoints.values():
                    validate_target(pose, self.workspace, self.config["motion"]["table_height"], self.config["motion"]["clearance"])
            except Exception as error:
                self.get_logger().error(f"Rejected object pose: {error}")
                return
            self.busy, self.solutions, self.names, self.seed = True, [], list(waypoints), None
            self.waypoints = list(waypoints.values())
            self._request_ik()

        @staticmethod
        def _rotation(x, y, z, w):
            return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)), (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))

        def _request_ik(self):
            index = len(self.solutions)
            if index == len(self.waypoints):
                self._hand("open")
                self._execute([0, 1], lambda: (self._hand("grasp"), self._execute([2, 3, 4], lambda: (self._hand("open"), self._execute([5], self._done)))))
                return
            request = ros["GetPositionIK"].Request()
            request.ik_request.group_name = self.config["motion"]["move_group"]
            request.ik_request.avoid_collisions = True
            request.ik_request.pose_stamped = self._pose_message(self.waypoints[index])
            if self.seed is not None:
                request.ik_request.robot_state = self.seed
            self.ik.call_async(request).add_done_callback(self._on_ik)

        def _on_ik(self, future):
            response = future.result()
            if response is None or response.error_code.val != 1:
                self._abort("IK failed")
                return
            names, positions = response.solution.joint_state.name, response.solution.joint_state.position
            values = dict(zip(names, positions))
            try:
                self.solutions.append([values[name] for name in self.config["motion"]["joint_names"]])
            except KeyError as error:
                self._abort(f"IK response missing joint {error}")
                return
            self.seed = response.solution
            self._request_ik()

        def _pose_message(self, pose):
            message = ros["PoseStamped"]()
            message.header.frame_id = self.config["frames"]["base"]
            message.header.stamp = self.get_clock().now().to_msg()
            message.pose.position.x, message.pose.position.y, message.pose.position.z = pose.position
            message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = _quaternion_xyzw(pose.rotation)
            return message

        def _execute(self, indices, on_success):
            if not self.arm.wait_for_server(timeout_sec=2.0):
                self._abort("arm trajectory action unavailable")
                return
            goal = ros["FollowJointTrajectory"].Goal()
            goal.trajectory.joint_names = self.config["motion"]["joint_names"]
            for order, index in enumerate(indices, 1):
                point = ros["JointTrajectoryPoint"]()
                point.positions = self.solutions[index]
                point.time_from_start = ros["DurationMsg"](sec=order * self.config["motion"]["seconds_per_waypoint"])
                goal.trajectory.points.append(point)
            sent = self.arm.send_goal_async(goal)
            sent.add_done_callback(lambda future: self._goal_response(future, on_success))

        def _goal_response(self, future, on_success):
            handle = future.result()
            if handle is None or not handle.accepted:
                self._abort("trajectory rejected")
                return
            handle.get_result_async().add_done_callback(lambda future: on_success() if future.result().result.error_code == 0 else self._abort("trajectory failed"))

        def _hand(self, state):
            message = ros["MotorCmds"]()
            positions = inspire_command_positions(self.config["hand"][state], self.config["hand"]["side"])
            for position in positions:
                command = ros["MotorCmd"]()
                command.mode = 0
                command.q = position
                message.motor_commands.append(command)
            self.hand.publish(message)

        def _done(self):
            self.get_logger().info("Pick-and-place complete")
            self.busy = False

        def _abort(self, reason):
            self.get_logger().error(reason)
            self.busy = False

    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
