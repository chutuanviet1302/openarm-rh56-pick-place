cmake_minimum_required(VERSION 3.8)
project(custom_ros_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorCmd.msg"
  "msg/MotorCmds.msg"
)

ament_package()
