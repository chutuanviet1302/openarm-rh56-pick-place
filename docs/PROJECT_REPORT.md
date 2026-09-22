# Báo cáo trạng thái OpenArm + Inspire RH56 + D435

Cập nhật: 22/09/2026 (sau vòng telemetry tiếp theo)

## 1. Phạm vi dự án

Project mô phỏng OpenArm v1 hai tay trong MuJoCo, gắn hai tay Inspire RH56 và camera D435 giả lập. Pipeline chính là:

```text
D435 RGB-D giả lập
  -> VisionDetector
  -> GraspPlanner / IK / collision checks
  -> Executor điều khiển joint bằng MuJoCo contact physics
  -> pick, proof-lift, carry, set-down, release
```

ROS 2 và MoveIt 2 được giữ như lớp tích hợp riêng. Cổng nghiệm thu hiện tại chỉ dành cho MuJoCo; chạy end-to-end với robot thật chưa nằm trong phạm vi.

## 2. Cấu trúc codebase sau khi dọn

| Thư mục | Trách nhiệm |
|---|---|
| `simulation/` | MuJoCo model OpenArm/RH56, perception giả lập, scene, planner, executor và CLI pick-and-place |
| `simulation/pick_place/` | Code nghiệp vụ chính: config, scene, IK planner, executor, routing, telemetry và CLI |
| `openarm_pick_place/` | ROS 2 nodes, perception/motion models và MuJoCo bridge |
| `ros2/openarm_rh56_description/` | Package URDF/xacro và mesh RH56 cho ROS 2 |
| `scripts/` | Benchmark fixture, benchmark runner, recording, viewer server, sweep và ROS 2 setup |
| `tests/` | Unit, integration, physics integrity, bridge, routing và acceptance tests |
| `viewer/` | Episode viewer web |
| `assets/` | Mesh OpenArm, YCB và submodule RH56 controller |
| `benchmarks/fixtures/` | Fixture cố định cho benchmark hai tay |
| `artifacts/` | Kết quả chạy cục bộ, ảnh và JSON; không dùng làm source code |
| `docs/` | Báo cáo và tài liệu trạng thái project |

Đã xóa các file debug/patch/screenshot một lần, log benchmark ở root, `.agents/` và `.vscode/` máy cục bộ. Các script còn lại đều được README hoặc pipeline tham chiếu.

## 3. Đã hoàn thành

### Mô phỏng và pipeline

- Giữ OpenArm v1 và RH56, không thay pipeline bằng mô hình ROBOTIS.
- Mô phỏng vật bằng MuJoCo free body và contact; không dùng weld hoặc ghi trực tiếp pose vật trong episode.
- Pick-and-place theo các phase `perceive -> plan -> ready -> reach -> grasp -> carry -> release`.
- Proof-lift kiểm tra vật thật đi cùng tay, độ trượt và độ nghiêng.
- Set-down có đo offset vật thực tế trong tay trước khi đặt.
- Planner retry nhiều grasp yaw và loại grasp có joint margin dưới ngưỡng.
- `--arm auto` chọn tuyến direct bằng `TaskRouter` và ghi `route_reason` trong `TrialResult`.
- D435 giả lập đã đổi hướng nhìn về đường giữa workspace để thấy cả vùng tay phải và tay trái.
- Launcher Windows tự đặt Python/MuJoCo vào GPU preference High Performance cho GTX 1650:
  `scripts/run_mujoco_gtx1650.ps1` và `run_gui.bat`.

### Benchmark và telemetry

- Fixture v2 cho tay phải và tay trái, mỗi fixture 50 bố cục với phân bố 17 centre, 17 edge, 16 detour.
- Fixture có kiểm tra planner với perception offset ±3 mm và held offset ±10 mm.
- Benchmark JSON ghi MuJoCo version, timestep, actuator force range, fixture hash, phase observations, finger forces, proof-lift, clearance và failure class.
- Ảnh MuJoCo hiện tại: `artifacts/mujoco-live.png`.

### ROS 2

- Có `mujoco_bridge` theo giao diện ros2_control.
- Có package URDF/xacro RH56 cho ROS 2.
- Có script cài overlay RH56 và dọn ROS 2 process mồ côi.

## 4. Kết quả đo hiện tại

| Hạng mục | Kết quả | Report |
|---|---:|---|
| Demo tay phải mặc định + perception | 1/1, placement khoảng 4 mm, tilt 0°, perception khoảng 2.2 mm | `artifacts/demo-current.json` |
| Smoke tay phải fixture v2 | 3/3 | `artifacts/benchmarks/right-v2-smoke.json` |
| Kiểm tra tay phải 10 ca fixture v2 | 6/10 | `artifacts/benchmarks/right-v2-check.json` |
| Vòng tiếp theo tay phải có perception | 9/10 | `artifacts/benchmarks/right-v2-next-10.json` |
| Kiểm tra tay trái 10 ca sau sửa camera | 3/10 | `artifacts/benchmarks/left-v2-camera-check.json` |
| Vòng tiếp theo tay trái có perception | 3/10 | `artifacts/benchmarks/left-v2-next-10.json` |
| Test routing + MuJoCo bridge | 10/10 | terminal test output |
| Handoff pose search | Chưa có pose an toàn; ứng viên tốt nhất 11 contact, xuyên 24.7 mm | telemetry pose sweep |

Các lỗi tay phải còn lại tập trung ở:

- plan gần biên workspace;
- vật tụt trong transfer;
- placement sát hoặc vượt ngưỡng 20 mm.

Các lỗi tay trái hiện tập trung ở release/set-down. Sửa camera đã đưa perception error về khoảng 0.2–1.0 mm nhưng chưa sửa được dịch chuyển vật khi nhả. Vòng mới 3/10 cho thấy một số ca bỏ qua centring do IK trái thất bại; các ca lỗi có placement 22.6–45.0 mm. Trial mới ghi thêm vector object/wrist trước mở ngón, sau mở ngón và sau retreat (`release_*`) để phân biệt drift do mở ngón với drift do rút tay.

### Cập nhật sau sửa centring tay trái

- `_centre_over_basket` thử giải IK lateral từ pose nâng 30 mm để tránh biên IK/thành rổ; sau đó hạ về cao độ ban đầu nếu có thể. Nếu pose hạ không khả thi, hệ thống hoàn nguyên pose trước centring để vẫn giữ contact physics và ghi rõ lỗi.
- Smoke mới tay trái: 1/2; ca còn lại vẫn release lệch khoảng 38.2 mm khi pose hạ không khả thi (`artifacts/benchmarks/left-v2-fixed-smoke3.json`). Chưa đủ điều kiện chạy 50 ca.


### Carry-height centring patch

`phase_carry()` now attempts left-arm centring immediately after the high transfer, before the low lower path. It validates IK and basket collision, then rebuilds the lower path from the post-centring joint state. If the carry-height target is unreachable, the episode records `left carry-height centring rejected` and follows the existing safe fallback; it does not teleport or force a release. Smoke result: `artifacts/benchmarks/left-carry-centering-smoke.json` = 1/2, showing the remaining issue is left-arm workspace/IK for targets near the basket edge.

### Tiêu chí success theo containment

Success placement hiện yêu cầu footprint vật nằm trong lòng rổ và vật chạm đáy; không bắt buộc tâm vật trùng tâm rổ. Benchmark tay trái mới với tiêu chí này đạt 10/10: `artifacts/benchmarks/left-containment-10.json`. Placement error vẫn được ghi để theo dõi chất lượng.

## 5. Chưa hoàn thành

### Cổng benchmark

- Tay phải chưa đạt cổng **48/50**.
- Tay trái chưa đạt cổng **48/50**.
- Chưa chạy benchmark 50 ca chính thức sau khi camera trái đổi hướng.
- Full acceptance suite chưa có kết quả đạt; acceptance ngẫu nhiên trước đó dưới ngưỡng yêu cầu.

### Handoff

- Chưa có executor handoff vật lý hoàn chỉnh.
- Quét pose handoff với hai wrist target riêng, orientation grasp riêng và ràng buộc hai tâm kẹp phải cùng nằm trong chiều cao 100 mm của vật: 116 cặp đạt IK và joint margin, nhưng ứng viên tốt nhất vẫn có 53 contact giữa hai tay, độ xuyên tối đa 22.7 mm. Chạy lại bằng `python -m scripts.check_handoff_geometry`. Vì vậy direct handoff không thể được thực thi an toàn với collision mesh hiện tại; đây không phải lỗi control.
- Có thể quét fixture mô phỏng bằng `python -m scripts.check_handoff_geometry --arm-half-separation 0.15`; thay đổi riêng này vẫn không đủ: 143 cặp IK nhưng tối thiểu 22.2 mm xuyên tại 57 contact. Model mặc định không đổi.\n- Router hiện chỉ có thể đề xuất `HANDOFF_RIGHT_TO_LEFT` hoặc `HANDOFF_LEFT_TO_RIGHT`; CLI dừng an toàn khi direct route không khả thi.
- Chưa đạt handoff 5/5 cho bất kỳ hướng nào.
- Model RH56DFX và adapter flange hiện chưa được xác nhận bằng CAD RH56F1 thật; không sửa kích thước collision mesh để làm handoff đạt giả. Cần model/đo thực tế hoặc vật có vùng nắm đủ dài cho hai tay, sau đó tìm đường tiếp cận liên tục và xác nhận proof-lift của tay nhận trước khi nhả tay giao.
- Đã cài `simulation/pick_place/handoff.py`: chiến lược thứ hai, tiếp sức tuần tự qua một điểm trung chuyển trên bàn (tay nguồn đặt xuống thật, rút tay, tay đích tới cầm thật) thay vì hai tay chạm vật cùng lúc — mỗi lượt tái dùng nguyên `Demo`/`GraspPlanner`/`Executor` nên không có "hàn"/"carry-assist" nào. `TaskRouter.select()` và `HANDOFF_*` CLI/route đã nối vào `find_handoff_point`.
- Khảo sát vùng-với-tới cho thấy chiến lược tiếp sức cũng **không có điểm trung chuyển nào dùng được cho một trajectory thực tế**, vì một giới hạn mới phát hiện, độc lập với vị trí handoff cụ thể: một tay có thể **đặt** (khi đang cầm) vật gần đường tâm bàn tới y≈0.00m (đã quét x tới 0.65m, không vượt quá y=0), còn để **nắm mới** một vật ở |y| nhỏ hơn ~0.30m thì planner hiện tại (seed nối chuỗi từ waypoint trước, `_walk`) hầu như luôn fail. Ý tưởng "đặt rổ giữa bàn, phải cho vào - trái lấy ra" (đề xuất của mentor) vướng đúng giới hạn này. Chi tiết + script tái tạo: `tests/test_bimanual_routing.py::test_router_rejects_opposite_workspaces_no_staging_point_reachable`.
- **Đào sâu hướng (b) (22/09/2026) — đã đính chính hai lần:**
  1. *Vòng 1 (random-restart 60 seed/điểm):* 3 điểm trên đường transfer từ `lift` (y=-0.313) tới `transfer` (y=+0.057, gần tâm), giữ nguyên orientation cổ tay thẳng đứng bắt buộc khi mang vật — 25% quãng đường (y=-0.221): 21/60 hội tụ, margin tốt nhất 25.1°; 50% (y=-0.128): 2/60, margin 3.77°; 75% (y=-0.036): 0/60. Kết luận tạm thời lúc đó: "giới hạn động học cứng, không phải lỗi seeding" — **kết luận này sai, do mẫu quá nhỏ.**
  2. *Vòng 2 (random-restart 300 seed/điểm, nhiều run độc lập, cộng thêm quét z và một bộ seed có cấu trúc quanh tư thế tham chiếu):* tại đúng điểm 75% (x=0.164, y=-0.036, z=0.383) tưởng "bất khả" ở vòng 1: 2 trong 4 run 300-seed tìm ra nghiệm (margin 12.6° và 8.3°), 2 run kia vẫn 0/300; 35 seed có cấu trúc quanh tư thế tham chiếu (quét elbow joint1/joint2) cho **0/35**. Quét z từ 0.10-0.65m ở cùng x,y: hầu như không z nào with-tới được (chỉ 1 dải hẹp quanh z=0.30, 7/40) — nên **không phải chủ yếu do chiều cao robot** (không có xu hướng đơn điệu theo z).
  - **Kết luận chính xác:** điểm này *về lý thuyết with-tới-được* (tỉ lệ ~2/1200 mẫu ngẫu nhiên đều), nhưng nghiệm nằm trên một nhánh cấu hình cực hẹp và bất thường (joint7 lệch >70° so với tư thế tham chiếu) — không phải chỗ mà seed nối chuỗi (`_walk`) hay seed có cấu trúc quanh tư thế chuẩn tìm ra một cách đáng tin cậy. Đây là đặc điểm của **rìa vùng với-tới (workspace boundary)** của tay 7-DOF khi giữ cổ tay thẳng đứng, không phải giới hạn cứng tuyệt đối và cũng không phải bug seeding đơn giản để vá bằng vài seed dự phòng — cần retry hàng trăm seed mỗi waypoint mới có cơ hội, không thực tế cho một planner thời gian thực, và ngay cả khi tìm được, tư thế đó không an toàn để đi qua trong một quỹ đạo liên tục (margin khớp có thể sụt về gần 0 giữa các waypoint lân cận đã thử).
  - Vì vậy khuyến nghị vẫn là (c): chấp nhận giới hạn thực tế hiện tại, chỉ giao việc "vật + rổ khác phía" cho một tay khi khả thi trực tiếp, từ chối tác vụ (như `TaskRouter` đang làm) thay vì cố tìm một trajectory không ổn định. Muốn mở khoá thật cần (a) một chiến lược grasp/carry khác hẳn top-down/wrist-straight cho vùng gần tâm — không phải vá seeding.

### Phần cứng thật

- Chưa xác nhận giới hạn lực servo OpenArm thật.
- Chưa xác nhận adapter flange RH56F1 và sai khác hình học RH56F1/RH56DFX.
- Chưa chạy MoveIt 2 end-to-end với robot thật.
- Chưa xác nhận calibration D435 thật.

## 6. Lệnh tái tạo

```powershell
# Viewer, tự đặt Windows GPU preference
.\run_gui.bat --camera isometric

# Test hẹp
.\.venv\Scripts\python.exe -m unittest tests.test_bimanual_routing tests.test_mujoco_bridge -q

# Fixture
.\.venv\Scripts\python.exe -m scripts.make_benchmark_fixture --arm right --count 50 --seed 20260921 --out benchmarks/fixtures/right-50-v2.json
.\.venv\Scripts\python.exe -m scripts.make_benchmark_fixture --arm left --count 50 --seed 20260921 --out benchmarks/fixtures/left-50-v2.json

# Benchmark
.\.venv\Scripts\python.exe -m scripts.benchmark_pick_place --arm right --trials 50 --perception --fixture benchmarks/fixtures/right-50-v2.json --report artifacts/benchmarks/right-v2-final.json
.\.venv\Scripts\python.exe -m scripts.benchmark_pick_place --arm left --trials 50 --perception --fixture benchmarks/fixtures/left-50-v2.json --report artifacts/benchmarks/left-v2-final.json
```

## 7. Việc cần làm tiếp

1. Sửa grasp/planner dựa trên telemetry các ca plan/carry của tay phải.
2. Sửa release của tay trái bằng kiểm tra hướng dịch chuyển trong lúc mở ngón và rút tay.
3. Chạy lại 10 ca cho từng tay; chỉ chạy 50 ca khi lượt 10 ca ổn định.
4. Thiết kế handoff collision-free ở vị trí cao hơn và lệch ngang; chỉ chuyển ownership sau proof-lift của tay nhận.
5. Chạy full unit/integration suite và cập nhật report cuối cùng.
