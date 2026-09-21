# Context Project — OpenArm + Inspire RH56 + D435 Pick & Place

Cập nhật: 2026-09-21. File này tóm tắt project cho phiên làm việc mới (người hoặc AI) đọc để nắm ngữ cảnh nhanh, không cần đọc lại toàn bộ lịch sử commit/chat.

## 1. Mục tiêu và pipeline

Robot tay đơn OpenArm (7 DOF/tay, hai tay) gắn tay Inspire RH56 5 ngón, camera đầu D435, thực hiện pick-and-place theo pipeline mentor đề ra:

```
D435 RGB-D → /perception/object_pose (PoseStamped) → TF camera→base
  → MoveIt IK theo waypoint → FollowJointTrajectory (7 DOF) → ros2_control
  → Inspire RH56 nhận custom_ros_messages/MotorCmds qua /hands/cmd (right[6]+left[6])
```

Robot thật của lab đã được mentor xác nhận là **OpenArm v1** (18/09/2026), không phải v2. Tay Inspire thật ở lab là **RH56F1**; model dùng trong sim là **RH56DFX** (điều khiển tương thích, hình học có thể khác — xem mục 6).

Xem thêm [[project-scope-openarm-ros2]] (memory) cho bậc thang 12 milestone đầy đủ, [[project-target-pipeline-gtx1650]] cho pipeline đích cuối cùng trên máy GTX 1650 (FoundationPose 2 lần/task, thư viện grasp YAML, MoveIt 2 arm-only).

## 2. Trạng thái hiện tại — đã làm được gì

### 2.1 Mô phỏng MuJoCo (milestone 1–9, đã hoàn thành chu trình đầy đủ)

Xem chi tiết kỹ thuật đầy đủ trong [BAO_CAO_TRANG_THAI_PROJECT.md](BAO_CAO_TRANG_THAI_PROJECT.md) và [BAO_CAO_MENTOR.md](BAO_CAO_MENTOR.md) (báo cáo mentor, 18/09/2026 — vẫn là nguồn chi tiết nhất cho phần sim, phần ROS 2 dưới đây mới hơn báo cáo đó).

Tóm tắt kết quả đo được (bố cục mặc định trừ khi ghi chú khác):
- **Pick-and-place vật lý đầy đủ**: 3/3 trial đạt (tiếp xúc thật qua MuJoCo, không teleport vật).
- **Có perception thật** (không đọc trực tiếp state sim): 3/3 đạt, sai số vị trí vật đo được ≤ 4.7 mm (perception), sai số đặt cuối 12.5 mm.
- **Bố cục ngẫu nhiên + perception** (seed 7): 6/6 đạt, sai số đặt 12–17 mm.
- **Unit/integration test**: 38/38 đạt (`python -m unittest discover -s tests`).
- Nắm bằng **cả 5 ngón** có lực thật (điều kiện nhấc: mọi ngón ≥ 0.5 N; ngón cái xây lực đối lực ≥ 24 N; index/middle phải đối lực thật — nắm chỉ bằng ring/pinky bị coi là không an toàn và bị loại).
- Cổ tay giữ thẳng tại grasp (naturally-derived từ FK, không gõ tay số).
- Proof-lift đo bằng **trượt tay–vật** (không phải độ cao tuyệt đối, vì servo sag khiến lệnh nâng 5cm chỉ lên ~4.1–4.3cm).

Các quyết định kỹ thuật lớn đã tự đưa ra và đã kiểm chứng (không chặn sim, nhưng là giả định cần đối chiếu phần cứng thật):
- Mount tay Inspire lên flange OpenArm v1 suy từ 2 hệ trục CAD, không đoán mò → lệch trục cẳng tay chỉ 2.0°/2.9°. **Adapter dày 1cm là giả định**, chưa có CAD thật (P1, xem mục 6).
- Đế robot đặt trên bàn thật (không phải bệ dựng riêng) — bàn hình chữ T để hai tay buông thẳng không chạm bàn.
- Rổ phải rộng 32cm (thành 5cm) — đo được lòng bàn tay khi nắm chỉ cao hơn đáy lon 3.7–6.2cm tuỳ khoảng cách, rổ nhỏ hơn khiến lòng bàn tay tì lên thành rổ.
- Vùng với của tư thế cổ-tay-thẳng khá hẹp (x 0.36–0.44m, y −0.32…−0.22m) do vai OpenArm cao so với bàn.

### 2.2 Hình học lab thật đã đo (18/09/2026, commit `b6a7439`)

Thay số đo thật vào sim thay vì số dựng/tuned:
- Đế robot đứng trên tấm base vendor (29.01mm) trên bàn cao 74cm thật → trục vai cao **0.727m** trên mặt bàn (trước đó dùng 0.40m tuned).
- Camera đầu D435i cao thêm **+68.64mm** trên trục vai, nhìn dọc theo đường tâm bàn.
- Hệ quả: tầm với top-grasp chỉ còn x ≤ ~0.30m từ đế → planner phải thử nhiều hướng nắm (`GRASP_YAW_CANDIDATES_DEG`) và chọn hướng đầu tiên vừa IK-reachable vừa physically secure, có retry (`GRASP_RETRIES`).
- Xem [[lab-geometry-measurements]] — **lưu ý: sim đã commit hiện dùng 0.40m**, chưa chắc đã đồng bộ hoàn toàn với 0.727m đo thật; cần đối chiếu khi đọc code `simulation/openarm_mujoco.py`.

### 2.3 Perception RGB-D cải tiến (commit `dac2930`)

Camera đầu nâng cao/dốc hơn làm bias hằng số cũ (circle-fit + hiệu chỉnh lateral cố định) vượt ngưỡng chấp nhận (10.8mm) và thay đổi theo vị trí bàn. Thay bằng ước lượng hình học không cần tune theo camera: gom điểm depth trên mặt bàn thành vùng liên thông trên lưới mặt bàn, bỏ pixel biên silhouette (depth pha trộn với bề mặt lân cận như thành rổ), giữ vùng chạm pixel color-mask, lấy phân vị footprint làm tâm vật. Sai số đo được ≤ 2.6mm trên toàn bàn với tay, cả ở vị trí camera cũ và mới.

### 2.4 Công cụ ghi lại và xem lại episode (commit `bccb9ec`)

- `scripts/record_episode.py`: chạy 1 trial headless, quay từ mọi camera trong scene, lấy mẫu telemetry mỗi frame (pose vật/cổ tay, độ nghiêng, lực ngón tay) + timeline phase/note + ảnh chụp head-camera detection. Ghi vào `artifacts/episodes/<name>/`, index tại `artifacts/episodes/index.json` (gitignored, tạo lại khi cần).
- `viewer/index.html`: liệt kê episode đã ghi, phát lại — chuyển camera (hoặc lưới 2×2), tua theo timeline tô màu theo phase, đọc state sống tại thời điểm tua.
- `scripts/serve_viewer.py`: serve project qua HTTP (viewer cần origin http:// để fetch JSON/stream video, không chạy được qua file://); `.claude/launch.json` chạy nó như target preview Browser pane.

### 2.5 Tích hợp ROS 2 thật — RH56 + MuJoCo làm stand-in ros2_control (mới, **chưa commit**)

Đây là phần việc mới nhất, đang ở trạng thái uncommitted trong working tree (`git status`: `ros2/`, `openarm_pick_place/mujoco_bridge.py`, `scripts/install_rh56_ros2_overlay.sh`, `tests/test_mujoco_bridge.py`, sửa `README.md`, `config.example.json`, `setup.py`, `simulation/five_finger_model.py`, `simulation/openarm_mujoco.py`):

- **`ros2/openarm_rh56_description/`**: package ROS 2 mới — URDF/xacro (`urdf/rh56_right.xacro`) + mesh STL của tay Inspire RH56, để nạp vào URDF chính của OpenArm qua `ee_dispatcher.xacro` (ee_type mới: `rh56_bimanual`).
- **`scripts/install_rh56_ros2_overlay.sh`**: script cài đặt overlay tự động — symlink package RH56 vào `~/ros2_ws`, copy mesh, patch `openarm_description` (`openarm_v10.urdf.xacro`, `openarm_robot.xacro`, `ee_dispatcher.xacro`) để thêm `ee_type=rh56_bimanual`, patch `openarm_bimanual_moveit_config` (`demo.launch.py`, `joint_limits.yaml`, `moveit_controllers.yaml`, `openarm_bimanual.srdf`) để bỏ gripper controller mặc định (RH56 điều khiển riêng qua `/hands/cmd`, không qua MoveIt gripper action) và build bằng colcon.
- **`openarm_pick_place/mujoco_bridge.py`** (node `mujoco_bridge`, mới, có test `tests/test_mujoco_bridge.py`): cho phép **sim MuJoCo đóng vai ros2_control thật** — publish `/joint_states`, nhận lệnh tay qua topic `MotorCmds` (`/hands/cmd`, right[6]+left[6]), nhận `FollowJointTrajectory` action cho tay phải và chạy quỹ đạo đó bằng `Executor` của sim. Đây là bước bắc cầu: cho phép chạy `d435_perception`/`motion_planning` (node ROS 2 thật, xem `openarm_pick_place/ros2_nodes.py`) nói chuyện qua message thật với MuJoCo thay vì gọi Python trực tiếp như trong `pick_place_demo.py`.
- **Đã build và verify chạy trong WSL2 Ubuntu-22.04 thật** (không chỉ Docker) ngày 21/09/2026: `ros2 launch openarm_bimanual_moveit_config demo.launch.py arm_type:=openarm_v1.0 use_fake_hardware:=true` — MoveIt 2 + RViz2 khởi động, controller `left/right_joint_trajectory_controller` và `joint_state_broadcaster` load/configure/activate thành công khi môi trường sạch (xem mục 5).
- `config.example.json` đã cập nhật theo tên controller/move_group thật sinh ra từ overlay (`/right_joint_trajectory_controller/follow_joint_trajectory`, move_group `right_arm`, trước là placeholder `openarm_right`).
- `docker/Dockerfile.ros2`: image ROS 2 Humble **ros-base** (không phải desktop) chỉ để build/test package Python thuần (`colcon build` + `python3 -m pytest`) — **không có rviz2/Gazebo**, cố ý bỏ để giữ image nhỏ cho milestone "build/test được chưa"; MoveIt 2/RealSense bị comment sẵn, bật khi cần chạy node thật.

## 3. Cấu trúc thư mục chính

```
simulation/            Sim MuJoCo: openarm_mujoco.py (spec robot v1), five_finger_model.py
                        (spec tay RH56 5 ngón + vật YCB), pick_place_demo.py (CLI demo/trial),
                        run_simulation.py (viewer), vision_detector.py (perception RGB-D sim),
                        pick_place/ (config, scene, executor dùng chung sim + ROS bridge)
openarm_pick_place/     Package ROS 2: ros2_nodes.py (d435_perception, motion_planning — node
                        thật), mujoco_bridge.py (mới — MuJoCo làm ros2_control), fakes.py,
                        geometry.py, models.py, motion.py, perception.py, pipeline.py
ros2/                   Package ROS 2 mới: openarm_rh56_description (URDF/mesh tay RH56)
scripts/                install_rh56_ros2_overlay.sh, kill_ros2_stack.sh (dọn tiến trình
                        ros2 launch mồ côi), record_episode.py, serve_viewer.py,
                        measure_sweep.py, sweep_postures.py, sweep_floor_layout.py
viewer/                 Web viewer xem lại episode đã ghi (index.html)
docker/                 Dockerfile.ros2 (ROS 2 Humble ros-base, build/test package)
tests/                  38+ unit/integration test — sim, mujoco_bridge, manipulation protocol,
                        physics integrity, acceptance
assets/openarm_v1/      MJCF OpenArm v1 vendor (từ enactic/openarm_mujoco, v1 không có trong
                        gói pip — chỉ có v2)
assets/rh56_controller/ Submodule model tay RH56 (correlllab/rh56_controller)
config.example.json     Cấu hình chạy: tên topic ROS, move_group, joint names, ngưỡng grasp
```

Xem [[reference-openarm-project-repos]] cho việc repo nào cấp module nào và thứ tự đọc khi onboard.

## 4. Kết quả kiểm thử hiện tại (tóm lại từ mục 2.1, nguồn: `artifacts/*.json` + `unittest`)

| Hạng mục | Kết quả |
|---|---:|
| Trial vật lý, bố cục mặc định | 3/3 đạt |
| Trial + perception | 3/3 đạt, sai số perception ≤4.7mm |
| Trial ngẫu nhiên + perception (seed 7) | 6/6 đạt, sai số đặt 12–17mm |
| Unit/integration test | 38/38 đạt |
| Proof-lift | trượt 0mm, nghiêng 5° |
| Đặt vào rổ | sai số 12.5mm, nghiêng 0° |

Lệnh tái tạo (PowerShell, gốc repo, venv Python):
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 3 --perception
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 8 --randomize --perception --seed 1 --report artifacts/randomized_trials.json
.\run_gui.bat --camera isometric
```

Lệnh ROS 2 (WSL2 Ubuntu, xem README.md mục "Chạy ROS 2" để đầy đủ):
```bash
bash scripts/kill_ros2_stack.sh   # dọn tiến trình mồ côi trước khi chạy lại
source /opt/ros/humble/setup.bash && cd ~/ros2_ws && source install/setup.bash
ros2 launch openarm_bimanual_moveit_config demo.launch.py arm_type:=openarm_v1.0 use_fake_hardware:=true
ros2 run openarm_pick_place mujoco_bridge --ros-args -p config_path:=$PWD/config.example.json
```

## 5. Sự cố đã gỡ trong phiên làm việc gần nhất (21/09/2026)

1. **RViz2 segfault (exit code -11)** vài giây sau khi MoveIt bắt đầu cập nhật interactive marker. Nguyên nhân: `install_rh56_ros2_overlay.sh` ép rviz2 dùng adapter D3D12 `NVIDIA` (GTX 1650) qua WSLg — đường WARP đó không ổn định cho renderer Ogre của RViz. **Đã sửa**: đổi sang ép `Intel` (Iris Xe, GPU mà WSLg thực sự render qua) — verify chạy ổn định 30s+ không crash. File: `scripts/install_rh56_ros2_overlay.sh`, đã patch trực tiếp cả workspace `~/ros2_ws` đang chạy.
2. **Controller spawner báo lỗi giả** (`Controller already loaded, skipping load_controller` rồi `Failed to configure controller`, exit code 1) — **không phải lỗi code**. Nguyên nhân: tiến trình `ros2 launch` cũ (đóng terminal thay vì Ctrl+C) để lại `ros2_control_node`/`move_group`/`rviz2` mồ côi, hai node `controller_manager` cùng tên tranh nhau trên cùng DDS domain. Verify: dọn sạch tiến trình mồ côi rồi chạy lại → cả `left/right_joint_trajectory_controller` và `joint_state_broadcaster` load/configure/activate sạch, không lỗi. **Đã thêm**: `scripts/kill_ros2_stack.sh` để dọn nhanh khi gặp lại.

## 6. Còn mở / rủi ro cần đối chiếu phần cứng thật (không chặn sim)

| Mức | Việc | Ghi chú |
|---|---|---|
| P1 | Độ dày adapter flange→đế tay (1cm) là giả định | Cần CAD/đo thật của adapter RH56F1 trước sim-to-real |
| P1 | Giới hạn lực servo cánh tay trong sim ±120N | Gấp 3 motor DM8009 thật (40N) — hạ về giá trị thật rồi kiểm tra lại proof-lift/sag |
| P1 | Vùng với tư thế cổ-tay-thẳng hẹp (A: x 0.36–0.44, y −0.32…−0.22) | Cân nhắc tư thế tham chiếu thứ hai hoặc dùng tay trái cho nửa bàn bên trái |
| P1 | Hình học lab 0.727m (vai)/+68.64mm (camera) đo 18/09 — cần xác nhận sim hiện tại (0.40m) đã đồng bộ số đo này chưa | Đối chiếu `simulation/openarm_mujoco.py` với [[lab-geometry-measurements]] |
| P2 | Rổ 32cm thành 5cm là kích thước tối thiểu cho side-grasp với bàn tay này | Quyết định theo rổ thật của lab; rổ nhỏ hơn cần grasp từ trên xuống |
| P2 | Tay trái chỉ giữ tư thế chờ, chưa demo hai tay | |
| P2 | Camera `overhead` không phát hiện lon (mặt trên bạc, không đỏ) | Dùng `d435_head`, hoặc thêm segment theo depth-plane |
| P2 | Chưa có policy học (VLA/RL) | Harness headless + log episode đã sẵn sàng làm dữ liệu/benchmark |
| — | Orientation vật trong perception chưa làm (chỉ có position, lon tròn) | |
| — | D435 thật + calibration `T_base_camera` (hiện dùng identity mẫu trong `calibration.example.json`) chưa chạy | **Không dùng ma trận identity trên robot thật** |
| — | RH56F1 thật vs RH56DFX sim — chưa xác nhận khác biệt hình học/khớp | Câu hỏi đang chờ mentor trả lời (xem BAO_CAO_MENTOR.md mục 5) |

## 7. Câu hỏi đang chờ mentor (từ BAO_CAO_MENTOR.md, 18/09, có thể đã được trả lời — cần kiểm tra lại)

1. Adapter flange→đế tay và model tay: lab có CAD/số đo thật của adapter RH56F1 không? Hướng lắp (lòng bàn tay vào thân, ngón cái phía trước) đúng không? RH56F1 có khác RH56DFX về hình học/khớp không, có URDF/MJCF của F1 không?
2. Phần cứng thật để set tham số sim: D435 gắn ở đâu, kích thước rổ, chiều cao bàn, đế robot đặt trên bàn hay bệ riêng? *(Phần này đã được trả lời một phần — xem mục 2.2, đo thật ngày 18/09.)*
3. Ưu tiên bước tiếp theo: (a) chuyển pipeline sim sang chạy qua ROS 2 msg thật, (b) thu dataset demo hàng loạt, hay (c) nối phần cứng thật? *(Mục 2.5 cho thấy (a) đã bắt đầu — mujoco_bridge + demo.launch.py chạy được trong WSL2.)*

## 8. Nguồn tham khảo khác

- Memory: [[project-scope-openarm-ros2]], [[inspire-rh56-joint-values]], [[reference-openarm-project-repos]], [[project-target-pipeline-gtx1650]], [[lab-geometry-measurements]]
- [README.md](README.md) — lệnh chạy đầy đủ (sim + ROS 2), bao gồm 2 sự cố mới gỡ ở mục 5
- [BAO_CAO_TRANG_THAI_PROJECT.md](BAO_CAO_TRANG_THAI_PROJECT.md) — báo cáo kỹ thuật chi tiết phần sim (18/09/2026)
- [BAO_CAO_MENTOR.md](BAO_CAO_MENTOR.md) — báo cáo gửi mentor, đối chiếu pipeline + câu hỏi (18/09/2026)
