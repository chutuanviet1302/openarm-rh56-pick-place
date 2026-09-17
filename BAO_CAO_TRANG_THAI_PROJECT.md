# Báo cáo trạng thái project OpenArm Pick-and-Place

**Ngày kiểm tra:** 17/09/2026
**Phạm vi:** MuJoCo, OpenArm hai tay, Inspire RH56 5 ngón (tay phải thao tác), lon YCB tomato soup, camera đầu D435
**Trạng thái tổng thể:** Hoàn thành chu trình pick-and-place bằng tiếp xúc vật lý, có perception, có randomization, có log episode đầy đủ.
Thay thế `BAO_CAO_LOI_PROJECT.md` (16/09) — mọi mục P0/P1 trong đó đã được xử lý hoặc không còn áp dụng.

## 1. Kết quả kiểm tra hiện tại

| Hạng mục | Kết quả | Bằng chứng |
|---|---:|---|
| Trial pick-and-place vật lý, bố cục mặc định | **3/3 đạt** | `artifacts/physics_trials.json` |
| Trial với perception (vị trí vật từ camera D435, không đọc sim) | **3/3 đạt**, sai số perception 3.0 mm | `--perception` |
| Trial bố cục ngẫu nhiên + perception (seed 1) | **8/8 đạt** | `artifacts/randomized_trials.json` |
| Unit/integration test | **36/36 đạt** | `python -m unittest discover -s tests` |
| Cổ tay thẳng tại grasp | joint6 = +3.6° (bố cục mặc định), < 10° mọi bố cục ngẫu nhiên | log `wrist pitch at grasp` |
| 5 ngón chạm trước khi nhấc | thumb 15.9 / index 3.4 / middle 5.3 / ring 9.2 / pinky 9.8 N | `grasp_forces` |
| Proof-lift | tay +4.3 cm, vật +4.0 cm, trượt 2 mm, nghiêng 2° | `proof_lift_*` |
| Đáy vật trên mép rổ khi mang | +6.0 cm (yêu cầu ≥ 5 cm) | `carry_clearance_above_rim_m` |
| Đặt vào rổ | sai số 14.5 mm, nghiêng 0.0° | `placement_error_m`, `bottle_tilt_deg` |
| Tốc độ viewer | thời gian thực (60 Hz redraw) | `_render()` |

## 2. Những gì đã sửa trong đợt này

### Grasp và quỹ đạo
- Lon vẽ chìm 5 cm trong bàn: visual mesh bị dịch `-OBJECT_HALF_HEIGHT` dù mesh đã căn tâm. Đáy mesh giờ trùng đáy collision trên mặt bàn.
- Điều kiện nhấc: **cả 5 ngón** phải có lực ≥ 0.5 N (trước chỉ cần ngón cái + 1 ngón); ngón chưa chạm được siết thêm riêng.
- Độ cao mang vật suy ra từ mép rổ + 5 cm (+1 cm margin cho servo sag), kiểm tra bằng đo thực tế ở lift và transfer; bỏ cơ chế "IK không tới thì hạ bớt".
- Đặt vật: hạ xuống cách đáy rổ 3 cm rồi mở tay (trước thả rơi từ 6 cm; 1–2 cm thì ngón út chạm thành rổ).
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
| P1 | Transform mount flange→palm và offset ngón là giá trị sim | Thay bằng CAD/đo thực trước sim-to-real |
| P1 | Vùng với của tư thế cổ-tay-thẳng hẹp (x 0.19–0.34, y −0.46…−0.36 cho A) | Nếu cần A rộng hơn: thêm tư thế tham chiếu thứ hai hoặc dùng tay trái cho nửa bàn bên trái |
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
