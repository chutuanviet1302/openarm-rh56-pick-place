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

# Demo pick-and-place lon YCB bằng tay phải 5 ngón (viewer, chạy thời gian thực)
.
un_gui.bat --camera isometric        # isometric | close_grasp | front_view | side_view | overhead | free

# Headless: N trial vật lý, ghi artifacts/physics_trials.json
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 3

# Lấy vị trí vật từ camera đầu D435 (RGB-D → deprojection) thay vì trạng thái sim
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 3 --perception

# Đổi bố cục: vật A và rổ B (x y trên mặt bàn, mét)
.
un_gui.bat --object 0.25 -0.40 --basket 0.42 -0.18

# Thu thập demonstration data: mỗi trial một bố cục ngẫu nhiên đã kiểm tra IK/va chạm
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 8 --randomize --perception --seed 1 --report artifacts/randomized_trials.json
```

### Debug

```powershell
# Chỉ lập kế hoạch: bảng target / reached / sai số / wrist pitch từng phase, không chạy vật lý
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --plan-only

# Dừng sau một phase để soi cảnh (perceive|plan|ready|reach|grasp|carry|release)
.un_gui.bat --stop-after grasp --camera close_grasp

# Log chi tiết: vị trí vật + wrist sau mỗi phase; --trace in pose + lực ngón mỗi 0.5 s sim
.un_gui.bat --verbose --trace 0.5
```

Cấu trúc code (`simulation/pick_place/`): `config.py` (mọi tham số) → `kinematics.py` (IK/FK) → `scene.py` (model, index, hình học, contact) → `planner.py` (target + chuỗi IK) → `executor.py` (bước vật lý qua `ctrl`, abort khi va chạm) → `demo.py` (7 phase `phase_*`) → `episode.py` (log + `TrialResult`) → `cli.py`. `simulation/pick_place_demo.py` chỉ là lối vào tương thích.

### Tiêu chí một trial đạt (không có weld / carry-assist / ghi qpos vật)

1. Vật đứng trên mặt bàn (đáy mesh = đáy collision = `TABLE_TOP_Z`).
2. Cổ tay thẳng tại grasp: hướng nắm là FK của `NATURAL_GRASP_JOINTS` (joint6 ≈ 0°), bàn tay nối tiếp cẳng tay.
3. Cả 5 ngón có lực pháp tuyến ≥ 0.5 N trước khi nhấc.
4. Proof-lift: tay nâng ≥ 3 cm và vật trượt ≤ 1 cm so với tay.
5. Đáy vật cao hơn mép rổ ≥ 5 cm khi mang; hạ xuống cách đáy rổ 3 cm rồi mới mở tay.
6. Sai số đặt ≤ 2 cm, nghiêng ≤ 15° (góc trục z của vật với phương thẳng đứng).

Mỗi trial ghi đủ: bố cục A/B, vị trí perception + sai số so với ground truth, lực 5 ngón, wrist pitch, proof-lift, clearance, yaw đặt, wrist position và joint target theo từng phase.

`simulation/five_finger_model.py` gắn model Inspire RH56DFX 6-DOF/12-joint vào mỗi flange. Transform mount hiện là giá trị hiệu chỉnh mô phỏng; phải thay bằng transform CAD/đo thực tế trước khi đồng nhất sim-to-real.

`simulation/vision_detector.py` là pipeline perception trong sim (cùng cấu trúc với `openarm_pick_place/perception.py` trên robot thật): segment màu → depth → pinhole deprojection → camera→world → fit đường tròn bán kính đã biết; sai số đo được ≤ 5 mm trên 20 vị trí. Không đọc pose vật từ sim; không thấy vật thì trial fail vì perception.

## Trước khi nối phần cứng

1. Sao chép `config.example.json` thành `config.json`, điền đúng topic/action/frame và giới hạn workspace đã đo.
2. Sao chép `calibration.example.json` thành `calibration.json`; thay ma trận identity bằng `T_base_camera` đã kiểm tra trên lưới 3x3.
3. Cung cấp MJCF/URDF và mapping actuator của tay 5 ngón thực tế; OpenArm v2 đã dùng model chính thức.
4. Viết adapter ROS 2 quanh controller hiện có, giữ E-stop và giới hạn của lab làm lớp bảo vệ cuối.

## Gate an toàn

Không gửi motion command nếu calibration chưa xác nhận, pose quá cũ, target ngoài workspace, IK thất bại hoặc controller timeout. `FakeRobot`/`FakeHand` chỉ dùng cho kiểm thử state machine.
