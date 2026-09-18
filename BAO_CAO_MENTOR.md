# Báo cáo tiến độ — Project Intern: OpenArm + Inspire Hand + D435, điều khiển qua ROS 2

**Ngày:** 18/09/2026
**Repo (private):** https://github.com/chutuanviet1302/openarm-rh56-pick-place
**Phạm vi đến nay:** mới làm thử toàn bộ pipeline trong **mô phỏng MuJoCo** (đã chuyển sang **OpenArm v1** sau khi anh xác nhận; MJCF từ `enactic/openarm_mujoco/v1`). Package ROS 2 đã build và chạy được trong Docker (ROS 2 Humble) với message giả — xem `docker/README.md`.

## 1. Đối chiếu với pipeline anh đề ra

Pipeline: Perception (vision → object position / orientation) → ROS 2 msg → Motion planning (IK) → Control (command joint, 7 DOF) → Inspire hand.

**Perception**
- Sim: đã làm. Camera D435 mô phỏng ở đầu robot: RGB + depth → deprojection → camera → world → fit đường tròn theo bán kính lon. Sai số ≤ 4.7 mm trên 20 vị trí. Không thấy vật thì báo fail, không fallback sang pose từ simulator.
- Chưa làm: orientation của vật (lon tròn nên mới ước lượng position); chưa chạy với D435 thật.
- Package ROS 2 (`openarm_pick_place`) đã có node `d435_perception` phát PoseStamped, chưa kiểm tra với dữ liệu thật.

**ROS 2 msg**
- Chưa chạy thật. Đã có 2 node (`d435_perception`, `motion_planning`), topic `/perception/object_pose`, TF camera → base, lệnh tay qua `/hands/cmd` (MotorCmds). Mới test bằng FakeRobot / FakeHand.
- Trong sim: perception → planner gọi trực tiếp bằng Python, chưa qua message.

**Motion planning / IK**
- Sim: đã làm. IK damped-least-squares + nullspace giữ cổ tay thẳng. Wrist target các phase (hover, pregrasp, grasp, lift, transfer, lower) suy từ pose vật + hình học rổ + hình học bàn tay (đo bằng FK), không có số gõ tay. Mang vật theo 8 waypoint Cartesian.
- Thật: node `motion_planning` gọi MoveIt IK rồi gửi FollowJointTrajectory — chưa chạy vì chưa có MoveIt config của OpenArm.

**Control (7 DOF)**
- Sim: đã làm. Position servo qua `data.ctrl`, không ghi trực tiếp qpos; quỹ đạo quintic; mỗi bước vật lý kiểm tra va chạm bàn / rổ, vi phạm > 3 mm thì abort.
- Thật: chưa nối ros2_control.

**Inspire hand**
- Sim: model RH56DFX, 6 actuator / 12 khớp mỗi tay. Nắm theo cách driver thật: ngón cái đối diện trước, rồi đóng từng ngón từng bước tới khi lực ≥ 8 N. Điều kiện nhấc: cả 5 ngón có lực ≥ 0.5 N.
- Thật: đã map giá trị sang thang 0–1000 của driver, chưa chạy với tay thật (lab dùng RH56F1, sim là DFX).

**Kết quả sim hiện tại — OpenArm v1** (bố cục mặc định, có perception): cổ tay gập 0.0° / 1.8° tại grasp; 5 ngón lực 24.2 / 9.8 / 8.5 / 10.2 N + ngón út sau khi siết thêm; proof-lift trượt 0 mm; đáy lon cao hơn mép rổ 5.4 cm khi mang; lon chạm đáy rổ rồi mới mở tay; đặt sai số 12.5 mm, nghiêng 0°. Bố cục ngẫu nhiên + perception (seed 7): 6/6 đạt, sai số 12–17 mm. Test tự động: 38/38.

## 2. Đã sửa theo góp ý của anh (mount bàn tay)
- Lỗi cũ đo được: transform flange → tay đưa trục ngón về −x của flange → ngón lệch 80° so với cẳng tay, cổ tay phải bẻ để bù.
- Sửa: suy transform từ hai hệ trục (OpenArm v1: trục dụng cụ = +z của `link7`, mặt flange z = 0.0955 nơi gripper gốc bắt vào; Inspire +z = ngón, +x = lòng bàn tay). Tay phải quay −90° quanh z, tay trái +90°; đế tay trên mặt flange với adapter 1 cm (giả định).
- Kết quả: ngón lệch trục cẳng tay 2.0° (phải) / 2.9° (trái); lòng bàn tay hướng vào thân, ngón cái phía trước. Có test tự động chốt < 5°. Tư thế nắm, tư thế chờ, A/B đều suy lại từ mount mới.

## 3. Khó khăn / quyết định tự đưa ra trong sim
- Rổ phải đổi 16 cm → 32 cm: bật collision lòng bàn tay thì đo được mặt dưới tay (v1) chỉ cao hơn đáy lon 3.7 cm ở 12 cm sau lon, 5.6–6.2 cm ở 14–16 cm; thành rổ 5 cm phải cách tâm lon ≥ 14–16 cm thì lon mới chạm đáy mà lòng bàn tay không tì lên mép.
- Vùng với của tư thế cổ-tay-thẳng hẹp (A trong x 0.36–0.44, y −0.32…−0.22) vì vai OpenArm cao so với bàn; phía đặt cho xoay tay quanh trục đứng để rổ nằm trong vùng với.
- Servo sag: lệnh nâng 5 cm chỉ lên 4.1–4.3 cm → tiêu chí proof-lift đổi thành trượt tay–vật ≤ 1 cm.
- Thả vật: mở tay khi lon còn lơ lửng thì ngón cái bẩy lon rơi lệch 3 cm → hạ tới khi lon chạm đáy mới mở, và lập lại kế hoạch đặt từ offset thật của lon trong tay.
- Lỗi metric: tilt = 2·acos|w| là tổng góc quay → thay bằng góc giữa trục z của lon và phương thẳng đứng.
- Đế robot đặt trên bàn (viết lại STL pedestal, giữ độ cao vai); bàn chữ T để tay buông thẳng không chạm bàn.

## 4. Còn thiếu so với pipeline
- Chạy thật D435 + calibration T_base_camera (hiện là identity mẫu).
- Chạy thật ROS 2: cần MoveIt 2 config cho OpenArm, controller FollowJointTrajectory, driver rh56_controller.
- Orientation của vật trong perception; tay trái mới giữ tư thế chờ; chưa thu dataset hàng loạt (mỗi episode đã ghi đủ dữ liệu).

## 5. Câu hỏi xin anh góp ý
1. **Adapter flange → đế tay và model tay:** lab có CAD / số đo thật của adapter RH56F1 lên OpenArm không, hướng lắp (lòng bàn tay vào thân, ngón cái phía trước) có đúng không? RH56F1 có khác RH56DFX về hình học / khớp không, lab có URDF hoặc MJCF của F1 không?
2. **Phần cứng thật để set tham số sim cho khớp:** D435 gắn ở đâu, kích thước rổ, chiều cao bàn, đế robot đặt trên bàn hay bệ riêng? (Sim hiện: rổ 32 cm thành 5 cm; vai ở z = 0.798 m; giới hạn lực servo sim ±120 N, gấp 3 motor thật.)
3. **Bước tiếp theo nên ưu tiên:** (a) chuyển pipeline sim sang chạy qua ROS 2 msg thật (perception node → motion node) ngay trong sim, (b) thu dataset demo hàng loạt, hay (c) nối phần cứng (D435 → MoveIt → ros2_control → rh56 driver)?

## 6. Kế hoạch đề xuất
- Tách sim thành hai node ROS 2 thật, cùng message với robot thật, để khi có hardware chỉ đổi driver.
- Thay adapter / rổ / bàn bằng số đo thật, chạy lại trial.
- Chạy D435 thật + calibration, rồi MoveIt + ros2_control + rh56 driver.

Ảnh kèm (`artifacts/`): `mount_check_right_hand_side.png`, `natural_grasp_grasp_iso.png`, `setdown_top.png`, `perception_d435_head.png`.
