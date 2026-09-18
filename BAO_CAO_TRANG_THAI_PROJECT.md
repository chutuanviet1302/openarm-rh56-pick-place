# Báo cáo trạng thái project OpenArm Pick-and-Place

**Ngày kiểm tra:** 18/09/2026 (cập nhật: chuyển sang OpenArm v1 theo xác nhận của mentor)
**Phạm vi:** MuJoCo, OpenArm **v1** hai tay (MJCF vendor tại `assets/openarm_v1/`), Inspire RH56 5 ngón (tay phải thao tác), lon YCB tomato soup, camera đầu D435
**Trạng thái tổng thể:** Hoàn thành chu trình pick-and-place bằng tiếp xúc vật lý, có perception, có randomization, có log episode đầy đủ.
Thay thế `BAO_CAO_LOI_PROJECT.md` (16/09) — mọi mục P0/P1 trong đó đã được xử lý hoặc không còn áp dụng.

## 1. Kết quả kiểm tra hiện tại

| Hạng mục | Kết quả | Bằng chứng |
|---|---:|---|
| Trial pick-and-place vật lý, bố cục mặc định | **3/3 đạt** | `artifacts/physics_trials.json` |
| Trial với perception (vị trí vật từ camera D435, không đọc sim) | **3/3 đạt**, sai số perception 3.0 mm | `--perception` |
| Trial bố cục ngẫu nhiên + perception (seed 7, v1) | **6/6 đạt**, sai số đặt 12–17 mm | `artifacts/randomized_trials.json` |
| Unit/integration test | **38/38 đạt** | `python -m unittest discover -s tests` |
| Cổ tay thẳng tại grasp | joint6 = 0.0°, joint7 = +1.8° (v1, bố cục mặc định) | log `wrist bend at grasp` |
| 5 ngón chạm trước khi nhấc | thumb 24.2 / index 9.8 / middle 8.5 / ring 10.2 N, ngón út sau khi siết thêm | `grasp_forces` |
| Proof-lift | tay +4.1 cm, vật +4.1 cm, trượt 0 mm, nghiêng 5° | `proof_lift_*` |
| Đáy vật trên mép rổ khi mang | +5.4 cm (yêu cầu ≥ 5 cm) | `carry_clearance_above_rim_m` |
| Đặt vào rổ | sai số 12.5 mm, nghiêng 0.0° | `placement_error_m`, `bottle_tilt_deg` |
| Tốc độ viewer | thời gian thực (60 Hz redraw) | `_render()` |

## 2. Những gì đã sửa trong đợt này

### Chuyển sang OpenArm v1 (18/09)
- Sim trước đây dựng trên v2 (gói pip). v1 khác: chuỗi link chạy +z, joint6/joint7 = trục gập x/y (đảo so với v2), không có `ee_base_link` — trục dụng cụ là +z của `link7` (mặt flange z = 0.0955), actuator là motor mô-men. Chiều dài link giống v2 nên tầm với tương đương.
- `openarm_mujoco.load_openarm_spec()`: thay motor bằng position servo với gain **và** damping/armature/friction của v2 (v1 không có armature, timestep 2 ms → cùng gain bị mất ổn định: joint2 chạy tới 190°); timestep 1 ms; site flange `*_ee_control_point`. Hàm `configure_arm_servos` idempotent vì attach tay Inspire ghi đè default actuator; `_stiffen_arm_actuators` cũ (kp 800, lực ±120 N) vẫn áp dụng sau compile.
- Mount tay suy lại: phải quay −90° quanh z, trái +90°, đế trên mặt flange + adapter 1 cm; ngón lệch trục cẳng tay 2.0°/2.9°.
- `scripts/sweep_postures.py` (mới, trong repo): quét lại tư thế nắm (cổ tay −0.0°/+1.8° tại grasp) và tư thế chờ; A = (0.421, −0.332), B = (0.42, −0.04).
- Đế v1 trên bàn: khối đế dịch lên mặt bàn, cột scale theo z tới hộp thân (geom tách rời nên không cần ghi lại mesh).
- Planner thêm kiểm tra **tay–rổ tại pose đặt** bằng va chạm MuJoCo khi chọn yaw (yaw +60 trước đây cho lòng bàn tay tì lên thành sau).
- Profile mặt dưới bàn tay v1 khi nắm: +3.7 cm ở 12 cm sau lon, +5.6–6.2 cm ở 14–16 cm → rổ 24 cm không đủ với thành 5 cm; rổ đổi thành **32 cm**.

### Mount bàn tay (góp ý mentor: "ghép bàn tay với cánh tay sai, phá hủy các khớp")
- Đo được: transform cũ (quay 180° quanh (1,0,−1)) đưa trục ngón tay (+z của bàn tay) về −x của flange, tức ngón tay chĩa **ngang 80°** so với trục cẳng tay; các khớp cổ tay phải bẻ để bù, và "cổ tay thẳng" trước đây thực ra là tay gắn vuông góc.
- Sửa: transform suy ra từ hai hệ trục (v2 lúc đó: trục dụng cụ −z của `ee_base_link`) → quay 180° quanh (1,1,0)/√2 (phải) và (1,−1,0)/√2 (trái). Với v1 xem mục trên. Ngón tay lệch trục cẳng tay 2.0° (phải) / 2.9° (trái); lòng bàn tay hướng vào thân, ngón cái phía trước.
- Suy lại toàn bộ phía sau: `NATURAL_GRASP_JOINTS` (j6 = 1.8°, j7 = −0.6°, tay vươn trước, ngón nghiêng xuống 22°), tư thế chờ (nắm tay trước thân, khuỷu 108°), A = (0.396, −0.275), B = (0.40, −0.02), hộp randomization; IK nullspace kéo cả j6 và j7 về 0.
- Đặt vật: sau khi nắm, đo vị trí thật của vật so với cổ tay và **lập lại kế hoạch đặt** từ offset đo được (`plan_place(held_offset)`); hạ 2 cm trên đáy rổ. Sai số đặt 10–16 mm.

### Đế robot và rổ
- Tư thế nghiêm (mọi khớp = 0) không chạm bàn: bàn hình chữ T — mặt làm việc x 0.13…0.70, lưỡi sau |y| ≤ 0.10 chỉ đỡ khối đế; hai tay buông thẳng nằm hai bên lưỡi, 0 tiếp xúc (test `test_arms_hanging_straight_down_do_not_touch_the_table`).
- Đế robot đặt trên mặt bàn: khối đế của pedestal (cao 20 cm) được dịch lên mặt bàn bằng cách viết lại STL (`assets/openarm/*_on_table.stl`), cột ngắn lại 30 cm, thân/vai giữ nguyên độ cao nên tầm với không đổi.
- Bàn tay **không được xuyên rổ**: bật lại collision của lòng bàn tay (trước bị tắt vì mount cũ làm vỏ lòng bàn tay cắt link5; mount mới không còn — test `test_palm_collides_and_does_not_touch_the_arm`). Đo profile mặt dưới bàn tay khi nắm: chỉ cao hơn đáy lon 3 cm ở 8 cm sau lon, 7–10 cm ở 12 cm sau lon → rổ 16 cm làm lòng bàn tay tì lên thành; rổ đổi thành 24 cm (thành 5 cm). Độ xuyên sâu nhất tay–rổ trong cả chu trình: 0.3 mm (ngưỡng abort 3 mm).

### Grasp và quỹ đạo
- Lon vẽ chìm 5 cm trong bàn: visual mesh bị dịch `-OBJECT_HALF_HEIGHT` dù mesh đã căn tâm. Đáy mesh giờ trùng đáy collision trên mặt bàn.
- Điều kiện nhấc: **cả 5 ngón** phải có lực ≥ 0.5 N (trước chỉ cần ngón cái + 1 ngón); ngón chưa chạm được siết thêm riêng.
- Độ cao mang vật suy ra từ mép rổ + 5 cm (+1 cm margin cho servo sag), kiểm tra bằng đo thực tế ở lift và transfer; bỏ cơ chế "IK không tới thì hạ bớt".
- Đặt vật: hạ xuống cách đáy rổ 2 cm rồi mở tay (trước thả rơi từ 6 cm).
- Tư thế tự nhiên: hướng nắm = FK của `NATURAL_GRASP_JOINTS` (cổ tay thẳng), IK có nullspace kéo joint6 → 0; hướng tiếp cận theo hướng ngón tay.
- Phía đặt được xoay tay quanh trục đứng (`PLACE_YAW_CANDIDATES_DEG`) để rổ nằm trong vùng với; đường mang vật giữ 8 waypoint Cartesian.
- Proof-lift đo **trượt** giữa tay và vật thay vì độ cao tuyệt đối (servo sag làm tay chỉ lên 4.1–4.3 cm dù lệnh 5 cm; ngưỡng cũ 4.0 cm fail oan 3/6 trial ngẫu nhiên).

### Metric
- `tilt` cũ = `2·acos|w|` là tổng góc quay, nên xoay tay 30° quanh trục đứng báo "nghiêng 30°". Thay bằng góc giữa trục z của vật và phương thẳng đứng (`upright_tilt_degrees`) trong demo và test.

### Perception
- `VisionDetector` cũ segment màu rồi **copy `data.xpos` của vật** làm kết quả — không phải perception. Viết lại: RGB + depth từ camera MuJoCo → intrinsics từ `fovy` → deprojection → camera→world → fit đường tròn bán kính đã biết (camera chỉ thấy một mặt lon). Sai số ≤ 4.7 mm trên 20 vị trí; không thấy vật → `RuntimeError`, không fallback.
- `--perception` đưa vị trí này vào planner; sai số so với ground truth được ghi vào report.

### Thu thập demo data
- `--randomize --seed`: mỗi trial một bố cục A/B trong `RANDOM_PICK_BOX` / `RANDOM_BASKET_BOX`, chỉ chấp nhận khi toàn bộ chuỗi waypoint giải được IK, A–B ≥ 15 cm và tâm vật cách thành rổ ≥ 12 cm (tay quấn vươn ~8 cm ngoài mặt lon).
- `TrialResult` ghi: bố cục, perception, lực 5 ngón, wrist pitch, proof-lift (tay/vật/trượt), clearance, yaw đặt, wrist position và joint target theo phase.

### Vận hành
- Viewer chạy thời gian thực: redraw 60 Hz thay vì sync + sleep mỗi bước 1 ms (Windows sleep tối thiểu ~1.6 ms → chậm 0.4×).
- CLI `--object X Y --basket X Y`; camera `close_grasp` bám theo A.

## 3. Còn mở (không chặn sim)

| Mức | Việc | Ghi chú |
|---|---|---|
| P1 | Độ dày adapter flange→đế tay (1 cm) là giả định | Thay bằng CAD adapter thật trước sim-to-real |
| P2 | Rổ 32 cm (thành 5 cm) là kích thước tối thiểu cho side-grasp ngang với bàn tay này; rổ nhỏ/thành cao hơn cần grasp từ trên xuống hoặc đặt lon lệch tâm | Quyết định theo rổ thật của lab |
| P1 | Giới hạn lực servo cánh tay trong sim là ±120 N (`_stiffen_arm_actuators`, có từ trước) — gấp 3 motor DM8009 thật (40 N) | Hạ về giá trị thật rồi kiểm tra lại proof-lift/sag trước sim-to-real |
| P1 | Vùng với của tư thế cổ-tay-thẳng hẹp (A: x 0.36–0.44, y −0.32…−0.22) | Nếu cần A rộng hơn: thêm tư thế tham chiếu thứ hai hoặc dùng tay trái cho nửa bàn bên trái |
| P2 | Tay trái chỉ giữ tư thế chờ | Demo hai tay như video tham chiếu chưa làm |
| P2 | Camera `overhead` không phát hiện được lon (mặt trên lon màu bạc, không đỏ) | Dùng `d435_head`; hoặc thêm segment theo depth-plane thay vì màu |
| P2 | Chưa có policy học (VLA/RL) | Harness headless + log episode đã sẵn sàng làm dữ liệu và benchmark |

## 4. Lệnh tái tạo

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 3 --perception
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 8 --randomize --perception --seed 1 --report artifacts/randomized_trials.json
.\run_gui.bat --camera isometric
```
