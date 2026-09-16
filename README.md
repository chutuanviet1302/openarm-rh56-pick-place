# OpenArm + Inspire Hand + D435 Pick & Place

Pipeline ROS 2 theo thiết kế của mentor:

`D435 RGB-D → /perception/object_pose (PoseStamped) → TF camera/base → MoveIt IK theo waypoint → FollowJointTrajectory 7-DOF → ros2_control`

Inspire RH56DFX nhận `custom_ros_messages/MotorCmds` qua `/hands/cmd`, đúng thứ tự driver `right[6] + left[6]`. Giá trị `open`/`grasp` trong `config.json` dùng thang raw 0–1000 của RH56.

## Chạy ROS 2

Yêu cầu ROS 2, `realsense2_camera`, `cv_bridge`, MoveIt 2 và controller hỗ trợ `FollowJointTrajectory`.

```bash
cp config.example.json config.json
colcon build --symlink-install
source install/setup.bash

# Terminal 1: D435, bật aligned depth
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# Terminal 2: perception phát position + orientation trong camera frame
ros2 run openarm_pick_place d435_perception --ros-args -p config_path:=$PWD/config.json

# Terminal 3: IK waypoint → trajectory 7 joint → Inspire Hand command
ros2 run openarm_pick_place motion_planning --ros-args -p config_path:=$PWD/config.json

# Terminal 4: driver Inspire RH56DFX tay trái (ID 2)
ros2 launch rh56_controller rh56_controller.launch.py serial_port:=/dev/ttyUSB0 hand_ids:=2
```

Trước khi chạy motion, TF `base_link ← camera_color_optical_frame`, MoveIt group và controller phải tồn tại đúng như `config.json`. Không dùng ma trận identity trong `calibration.example.json` trên robot thật.

## Chạy kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Chạy MuJoCo

Model OpenArm v2 chính thức được cung cấp bởi `openarm-mujoco`. Viewer thay cả hai gripper bằng Inspire RH56DFX trái/phải từ `correlllab/rh56_controller`.

```powershell
# Smoke test không mở cửa sổ
.\.venv\Scripts\python.exe -m simulation.run_simulation --headless

# Mở viewer tương tác
.\.venv\Scripts\python.exe -m simulation.run_simulation

# So sánh với gripper hai ngón nguyên bản
.\.venv\Scripts\python.exe -m simulation.run_simulation --stock-gripper

# Demo pick-and-place lon snack YCB 001 bằng cả hai tay
.\.venv\Scripts\python.exe -m simulation.pick_place_demo
```

`simulation/five_finger_model.py` gắn model Inspire RH56DFX 6-DOF/12-joint vào mỗi flange. Transform mount hiện là giá trị hiệu chỉnh mô phỏng; phải thay bằng transform CAD/đo thực tế trước khi đồng nhất sim-to-real.

Demo MuJoCo không ghi pose vật và không dùng weld/carry-assist; trial chỉ đạt khi Inspire Hand nâng vật bằng tiếp xúc vật lý và đưa tới vùng B.

## Trước khi nối phần cứng

1. Sao chép `config.example.json` thành `config.json`, điền đúng topic/action/frame và giới hạn workspace đã đo.
2. Sao chép `calibration.example.json` thành `calibration.json`; thay ma trận identity bằng `T_base_camera` đã kiểm tra trên lưới 3x3.
3. Cung cấp MJCF/URDF và mapping actuator của tay 5 ngón thực tế; OpenArm v2 đã dùng model chính thức.
4. Viết adapter ROS 2 quanh controller hiện có, giữ E-stop và giới hạn của lab làm lớp bảo vệ cuối.

## Gate an toàn

Không gửi motion command nếu calibration chưa xác nhận, pose quá cũ, target ngoài workspace, IK thất bại hoặc controller timeout. `FakeRobot`/`FakeHand` chỉ dùng cho kiểm thử state machine.
