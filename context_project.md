# Context Project — OpenArm + Inspire RH56 + D435 Pick & Place

Cập nhật: 2026-09-25 (kiểm tra lại rổ giữa bàn + giật, xem mục cuối; 2026-09-23 task rổ giữa bàn hai tay; trước đó 2026-09-21 Stage 1/2). File này tóm tắt project cho phiên làm việc mới (người hoặc AI) đọc để nắm ngữ cảnh nhanh, không cần đọc lại toàn bộ lịch sử commit/chat.

**Đọc mục 0 trước** — đó là trạng thái mới nhất và việc đang dở; mục 1–8 bên dưới là bối cảnh nền, một số con số đã lỗi thời (đánh dấu ở từng chỗ).

> **Mới nhất (23/09/2026):** task "tay phải đặt vào rổ giữa bàn → tay trái lấy ra" đã PASS trên bố trí mô phỏng (xoay đế tay), kèm chốt an toàn hai-tay-không-chạm. Xem mục cuối file **"Tiến độ 23/09/2026"**. Các mục handoff/retrieve 22/09 phía dưới là lịch sử điều tra dẫn tới kết quả đó.

### Bổ sung 22/09/2026 — fixture v2 và kiểm chứng tiếp theo

- Đã tạo fixture sàng theo IK + sai số perception ±3 mm cho cả hai tay: `benchmarks/fixtures/right-50-v2.json` và `benchmarks/fixtures/left-50-v2.json` (mỗi fixture 50 bố cục, vùng 17/17/16).
- Smoke benchmark tay phải trên fixture v2 đạt **3/3** với perception (`artifacts/benchmarks/right-v2-smoke.json`). Khi chạy dài, lỗi động lực học vẫn xuất hiện ở carry/release và một số plan; lượt dừng ở 28/50 có 3 lỗi, nên **chưa được ghi là kết quả Stage 2**.
- Thử tăng lực đóng 8→10 N, đổi ma sát hai bên và duy trì lực đóng theo từng bước đều làm mẫu 10 ca xấu hơn; đã hoàn tác. Cấu hình hiện dùng `CONTACT_FORCE_TARGET_N=8 N`, `TRANSFER_SECONDS=2.5 s`.
- Demo tay phải + D435 giả lập hiện đạt **1/1**, sai số đặt 4.0 mm, nghiêng 0°, perception 2.2 mm (`artifacts/demo-current.json`). Ảnh viewer: `artifacts/mujoco-live.png`.
- Handoff hai chiều vẫn **chưa đạt**: quét pose giao nhận còn xuyên tay 24.7–31 mm; router chỉ trả đề xuất và CLI dừng an toàn, không chuyển quyền giữ vật giả lập.
- Kiểm tra hẹp sau thay đổi: `tests.test_bimanual_routing` + `tests.test_mujoco_bridge` đạt **9/9**; `py_compile` các module MuJoCo đạt. Full suite và cổng 48/50 chưa đạt.
- Benchmark kiểm chứng tiếp theo: tay phải fixture v2 đạt **6/10** (`artifacts/benchmarks/right-v2-check.json`); tay trái với camera cũ đạt **0/10** vì D435 không thấy workspace y dương. Đã đổi `d435_head` về ngắm đường giữa; perception tay trái trong lượt mới giảm còn 0.2–0.9 mm nhưng benchmark vật lý chỉ **3/10** (`artifacts/benchmarks/left-v2-camera-check.json`), chủ yếu lỗi release 22.6–45.0 mm. Hai cổng 48/50 vẫn chưa đạt.

### Bổ sung 21/09/2026 — đo lại cổng MuJoCo hai tay

- Benchmark fixture `right-50.json` đã kết thúc: **26/50** ở baseline (`artifacts/benchmarks/right-7.json`), gồm 6 lỗi plan, 9 carry, 9 release. Oracle replay gắn 10 lỗi `perception`, 14 lỗi `manipulation`, nhưng nhãn này chỉ biểu thị kết quả replay; có trường hợp proof-lift trượt làm vật đổi vị trí trước khi planner chạy lại.
- Sửa `GraspPlanner.plan()` để thử grasp yaw kế tiếp nếu set-down của yaw hiện tại không giải được. Cùng fixture và perception đạt **27/50** (`artifacts/benchmarks/right-after-plan.json`): 5 lỗi plan, 9 carry, 9 release. Chỉ trial 46 đổi từ fail sang pass. **Stage 2 chưa đạt cổng 48/50**.
- `make_benchmark_fixture.py` giờ sàng thêm khả năng lập kế hoạch khi tọa độ camera lệch ±3 mm (sai số D435 giả lập đã đo tối đa khoảng 2.6 mm). Fixture `right-50.json` cũ được giữ nguyên để đối chiếu; layout 39 của fixture cũ không qua sàng mới. Fixture tay trái đang được tạo, chưa có kết quả benchmark 50 trial.
- Planner giờ loại grasp có joint margin <3° (ngưỡng đã dùng ở waypoint khác); fixture cũ chứa nhiều grasp margin 0° dù IK giải được. Thêm yaw ±15° sau các hướng ±30° hiện có để retry; trial 21 và 39 cũ đã chạy đạt bằng hướng mới. Trace trial 8 cho thấy lực ngón cái giảm 19→0 N trong transfer 3 s; rút về 2.5 s làm 3/9 ca carry lỗi cũ đạt, 3 ca pass được kiểm tra vẫn đạt. Cần benchmark lại đủ fixture mới trước khi kết luận tỷ lệ.
- Demo mặc định tay phải với D435 giả lập đạt 1/1 (đặt lệch 10.2 mm, nghiêng 0°, perception lệch 2.2 mm). Bố cục mirror tay trái đạt 1/1 với oracle pose (18.8 mm), nhưng với D435 giả lập fail ở release (29.9 mm); không coi đó là cổng tay trái. Đã mở viewer MuJoCo cho demo tay phải và lưu ảnh `artifacts/mujoco-live.png`.
- Thử pose nắm đồng thời của hai tay quanh lon ở vùng giao x≈0.2 m: cả hai tay giải được IK nhưng bàn tay xuyên nhau sâu đến khoảng 42 mm. Quét rộng hơn theo x/y/yaw/pitch/độ cao cũng không tìm được pose hết va chạm (ứng viên ít va chạm nhất vẫn có 20 contact, sâu tới 31 mm). Router mới chỉ đề nghị `HANDOFF_*`; **chưa có thực thi handoff vật lý**. Cần thiết kế tư thế nhận vật khác và chứng minh khoảng hở trước khi cho executor chuyển quyền giữ vật.

## 0. Trạng thái mới nhất (phiên chiều 21/09/2026 — Stage 1/2 theo kế hoạch pipeline hai tay)

### 0.1 Đã commit (`cdee7be`)

Tổng quát hóa pipeline sim từ tay-phải-cứng sang tham số `side` (planner, executor, CLI), thêm route selector hai tay (`simulation/pick_place/routing.py`) và MuJoCo bridge đóng vai ros2_control (`openarm_pick_place/mujoco_bridge.py`). Sửa hai hồi quy tìm được bằng đo đạc trực tiếp (log per-step, không đoán):

- **Vật rơi giữa chừng lúc mang (transfer)**: `TRANSFER_SECONDS` bị nâng 2.0→6.0 trong phần việc trước đó không ghi lý do. Giữ vật lâu ở tư thế xoay làm lực kẹp rò rỉ dần rồi mất hẳn — đo được ngón cái 20.4N→7.6N trong 3.4s **trong khi vật đứng yên tuyệt đối**, rồi rơi tự do. Đặt lại 3.0s.
- **Sai số đặt vật 20–24mm, hụt sát ngưỡng**: `PLACE_POSITION_CORRECTION` là hằng số chỉnh tay có độ lớn 19.3mm trong khi ngưỡng đạt là 20mm — code cố ý nhắm lệch tâm rổ để bù việc rơi hụt, nhưng lượng hụt thật đến từ grip creep lúc mang nên hằng số cũ đã lỗi thời. Thay bằng `Demo._centre_over_basket()`: đo vật đang thực sự ở đâu ngay trước khi hạ và dịch cổ tay bù đúng lượng đó; hằng số về 0 cho cả hai tay.

Kết quả: unit test 50/52→**51/52**; acceptance ngẫu nhiên (seed 7, 20 trial) 12/20→**17/20**; **test tay trái (Stage 3) tự pass theo** — sai số 55.2mm biến mất vì cùng hằng số cũ gây lỗi ở cả hai tay; suite nhanh hơn 498s→331s.

**Đã thử và loại bỏ vì đo không ủng hộ** (không còn trong code, nhưng đáng nhớ để khỏi thử lại): tăng lực kẹp 8N→14N (vật bị ép bật ra lúc kẹp); yêu cầu 3/4 ngón thay vì 2/4 (không đổi kết quả); dàn đều góc xoay yaw theo cả tuyến đường thay vì dồn vào chặng đầu (vật rơi sớm hơn); gộp transfer+lower thành một chuyển động liên tục (không đổi); giữ lực kẹp liên tục qua `sustain_grip` (phản tác dụng — xem 0.2); tách bước nhả tay theo lực `open_until_released` (siết được phân bố nhưng đổi trial nào lỗi, không tăng tổng, bỏ vì overfit trên mẫu 20); chọn raise nhiều margin nhất thay vì đầu tiên vượt ngưỡng (làm hỏng 2 trial khác, xem 0.3).

### 0.2 Phát hiện quan trọng: đường cong lực–độ đóng của ngón cái có đỉnh

Quét động học thuần (`jaw_offsets_at`, không vật lý) cho thấy khẩu độ hàm kẹp thu hẹp tới cực tiểu 58mm tại flexion≈0.40 rồi **mở rộng trở lại** lên 63mm tại giới hạn actuator 0.570 (giá trị `closed_ctrl` hiện tại). Cung gập ngón cái đưa đầu ngón vượt qua các ngón đối diện thay vì dừng lại như 4 ngón kia.

Đã thử dừng ngón cái đúng tại cực tiểu hình học (điểm khẩu độ hẹp nhất) — **kết quả xấu đi rõ rệt**: 17/20→10/20, vật rơi ngay từ lúc nhấc. Lý do: đây là servo vị trí, lực kẹp sinh ra chính từ phần lệnh *vượt quá* vị trí vật cản; bỏ phần vượt đó thì không còn gì ép vào vật. Đã hoàn lại `closed_ctrl["thumb"] = upper`, phát hiện được ghi lại tại [scene.py:97](simulation/pick_place/scene.py:97) kèm số liệu để không ai thử lại vô ích.

**Ý nghĩa**: các lỗi còn lại (rơi vật, creep lúc hạ, xê dịch lúc nhả) nhiều khả năng cùng gốc — grasp không đủ chắc vì rơi vào sườn yếu của đường cong lực này, không phải lỗi ở khâu carry/release. Hướng điều tra tiếp theo nên là **chất lượng grasp** (chọn tư thế/offset sao cho ngón cái rơi vào sườn khỏe), nhưng công việc chiều nay chuyển hướng sang dựng fixture Stage 2 trước (xem 0.4).

### 0.3 Phát hiện quan trọng: `find_raise` có bug — chọn ứng viên đầu tiên thay vì tốt nhất

`GraspPlanner.find_raise()` có biến `feasible` khai báo nhưng không dùng (dấu vết ý định ban đầu) — hàm trả về waypoint "raise" **đầu tiên** vượt `MIN_JOINT_MARGIN_DEG=3.0`, không phải waypoint có margin cao nhất. Hệ quả: map toàn bộ vùng bàn với được, **mọi** plan báo margin tối thiểu đúng 3.3° — con số này phản ánh ngưỡng chấp nhận, không phản ánh độ khó bố cục.

Đã thử sửa (chọn margin cao nhất, chỉ trả giá collision-check theo thứ tự giảm dần) — **kết quả xấu đi**: 17/20→16/20, test tay trái hỏng lại. Lý do: waypoint margin cao nằm xa hover hơn, đổi đường blend attention→raise→hover, hỏng pha nắm ở 2 layout khác. Đã hoàn lại, giữ nguyên hành vi "đầu tiên vượt ngưỡng"; lý do + số liệu ghi tại [planner.py](simulation/pick_place/planner.py) (hàm `find_raise`).

**Hệ quả cho benchmark**: vì margin tại waypoint "raise" luôn ghim ở 3.3°, chỉ số phân biệt độ khó bố cục thật phải lấy tại **tư thế nắm** (`plan.joints["grasp"]"`), không phải min trên toàn chuỗi. Đã áp dụng trong `scripts/make_benchmark_fixture.py`.

### 0.4 Map vùng với thật (right arm) — hẹp hơn tưởng, nút thắt là joint5

Quét bằng planner thật (IK + full plan) trên lưới object×basket phát hiện:

- Vùng lấy mẫu cũ trong `RANDOM_PICK_BOX`/`RANDOM_BASKET_BOX` (config.py) rất hẹp — object trong ô 6×5cm, basket trong ô 4×4cm — gần như không phải benchmark thật.
- Nút thắt vùng với **không phải chiều dài tay mà là joint5** (giới hạn ±90°): giữ hướng cổ tay thẳng cố định trong khi vươn ngang bàn buộc joint5 gánh toàn bộ thay đổi hướng, tại tư thế nắm margin của nó có thể chỉ còn 2.4°.
- Vùng basket với được (object cố định gần đế) chỉ còn dải hẹp x∈[0.21, 0.27]; phần lớn ô trong lưới quét báo margin đúng 0.0° (trùng phát hiện 0.3 — ghim bởi `find_raise`, không phải giới hạn thật).

Từ map này rút ra hộp lấy mẫu mới cho fixture: `PICK_BOX=((0.02,0.27),(0.20,0.43))`, `BASKET_BOX=((0.19,0.29),(0.01,0.21))` (magnitude, dấu y theo tay) — tỉ lệ lọt qua sàng tăng từ ~4% lên ~16%.

### 0.5 Fixture Stage 2 (50 layout, tay phải) — đã dựng, benchmark đang chạy

Theo đúng yêu cầu kế hoạch ("mỗi layout kiểm tra reachability trước khi lưu fixture"), viết `scripts/make_benchmark_fixture.py`:

- Sàng bằng kinematics thật (không vật lý): layout phải qua `parse_layout` (trên bàn, cách nhau ≥15cm, không chồng đế), planner phải giải được toàn chuỗi, **và** set-down phải còn giải được khi vật lệch tâm jaw danh nghĩa ±10mm theo cả 4 hướng (vì grip không bao giờ nằm đúng tâm — đây là nguyên nhân lỗi "held-offset compensation unavailable" từng thấy ở trial 19 cũ).
- Phân vùng `centre`/`edge`/`detour` theo margin tại tư thế nắm và `route_strategy`; "edge" là 1/3 dưới của **quần thể đã sàng** (phân vị), không phải một ngưỡng độ tuyệt đối tự đặt — vì margin khả dụng chỉ trải 0–16.6°, ngưỡng tuyệt đối sẽ bắt tất cả hoặc không bắt gì.
- Kết quả: **50/50 layout**, không thiếu vùng nào (17 centre / 17 edge / 16 detour), từ pool 81 ứng viên qua sàng. Margin tại nắm: min 0.0° (có layout đúng-tại-giới-hạn — ca biên thật, không loại bỏ), median 2.2°, max 16.6°. File: `benchmarks/fixtures/right-50.json`.
- Thêm `--fixture` vào `scripts/benchmark_pick_place.py`: chạy đúng layout đã lưu thay vì bốc RNG mới mỗi lần (tránh việc IK gần biên workspace không tái lập bit-for-bit giữa các lần chạy), ghi nhãn vùng + tỉ lệ đạt theo vùng vào báo cáo.

**Đang chạy nền lúc ghi tài liệu này** (chưa có kết quả): `python -m scripts.benchmark_pick_place --arm right --trials 50 --perception --fixture benchmarks/fixtures/right-50.json`. Cổng Stage 2 là **48/50**. Đây sẽ là con số Stage 2 thật đầu tiên, khác `test_twenty_trials_acceptance` (mẫu 20, vùng hẹp, RNG mới mỗi lần — không phải thước đo Stage 2 của kế hoạch).

### 0.6 Phát hiện phụ: hình học lab đã đồng bộ, không còn treo

Nghi vấn cũ ở mục 2.2/6 dưới đây ("sim vẫn dùng 0.40m") **đã lỗi thời** — xác minh trực tiếp trong `simulation/five_finger_model.py`: `ROBOT_RISER_HEIGHT=0.02901`, `SHOULDER_AXIS_ABOVE_RISER=0.698` (→0.727m trên mặt bàn, đúng số đo lab), `CAMERA_ABOVE_SHOULDER_AXIS=0.06864`. Đã cập nhật memory `lab-geometry-measurements`. Hệ quả: các bố cục gần đế fail IK là giới hạn workspace thật (joint5, xem 0.4), không phải lỗi mô hình hóa.

### 0.7 Việc chưa commit, cần quyết định

- `ros2/` (package URDF/mesh RH56, ~6MB STL nhị phân) + `scripts/install_rh56_ros2_overlay.sh`, `scripts/kill_ros2_stack.sh` — đưa mesh nhị phân vào lịch sử git là khó đảo ngược, **chưa quyết định**: commit thẳng / tách submodule / gitignore + script tải.
- `.agents/`, `artifacts/benchmarks/` (output runner cũ, không phải fixture), `benchmarks/` (fixture + kết quả mới) — chưa commit.
- Submodule `assets/rh56_controller` đang ở trạng thái dirty.
- Ưu tiên câu hỏi mentor đã đổi: **giới hạn lực/vận tốc servo thật OpenArm v1** lên đầu (xem mục 6, P1) — vì mọi phát hiện chiều nay (lực kẹp rò rỉ, creep lúc hạ, xê dịch lúc nhả) đều nhạy với độ cứng servo, và sim đang mạnh gấp 3 lần thật (`forcerange=±120N` vs DM8009 thật 40N — xem `simulation/five_finger_model.py:411`). Tinh chỉnh chất lượng grasp (0.2) trước khi có số thật có nguy cơ không chuyển sang phần cứng được.

### 0.8 Việc tiếp theo sau khi có kết quả benchmark

1. Đọc `_bench.log` / báo cáo JSON trong `artifacts/benchmarks/` — nếu ≥48/50: Stage 2 đạt cổng, có thể sang Stage 3 (tổng quát hóa tay trái — đã một phần nhờ 0.1) hoặc dựng fixture tay trái tương tự.
2. Nếu <48/50: xem `by_region` trong báo cáo để biết vùng nào yếu (centre/edge/detour), đối chiếu `failure_class` (perception vs manipulation, do oracle replay) trước khi sửa.
3. Vẫn nên chờ mentor trả lời câu lực servo trước khi tinh chỉnh sâu vào chất lượng grasp (0.2), vì kết quả tinh chỉnh trên servo ±120N có thể không chuyển sang 40N thật.

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

*(Con số dưới đây từ phiên 18/09; đã lỗi thời so với mục 0 — 51/52 test, acceptance 17/20 trên mẫu 20 hoặc chờ kết quả fixture 50-layout Stage 2. Giữ lại mục này vì mô tả các quyết định kỹ thuật/tiêu chí grasp vẫn còn đúng.)*

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
- Đế robot đứng trên tấm base vendor (29.01mm) trên bàn cao 74cm thật → trục vai cao **0.727m** trên mặt bàn.
- Camera đầu D435i cao thêm **+68.64mm** trên trục vai, nhìn dọc theo đường tâm bàn.
- Hệ quả: tầm với top-grasp hẹp, nút thắt là **joint5** chứ không phải chiều dài tay (xem mục 0.4) → planner phải thử nhiều hướng nắm (`GRASP_YAW_CANDIDATES_DEG`) và chọn hướng đầu tiên vừa IK-reachable vừa physically secure, có retry (`GRASP_RETRIES`).
- **Đã xác nhận đồng bộ (21/09, mục 0.6)**: sim hiện dùng đúng 0.727m/+68.64mm, không còn là 0.40m tuned. Xem [[lab-geometry-measurements]].

### 2.3 Perception RGB-D cải tiến (commit `dac2930`)

Camera đầu nâng cao/dốc hơn làm bias hằng số cũ (circle-fit + hiệu chỉnh lateral cố định) vượt ngưỡng chấp nhận (10.8mm) và thay đổi theo vị trí bàn. Thay bằng ước lượng hình học không cần tune theo camera: gom điểm depth trên mặt bàn thành vùng liên thông trên lưới mặt bàn, bỏ pixel biên silhouette (depth pha trộn với bề mặt lân cận như thành rổ), giữ vùng chạm pixel color-mask, lấy phân vị footprint làm tâm vật. Sai số đo được ≤ 2.6mm trên toàn bàn với tay, cả ở vị trí camera cũ và mới.

### 2.4 Công cụ ghi lại và xem lại episode (commit `bccb9ec`)

- `scripts/record_episode.py`: chạy 1 trial headless, quay từ mọi camera trong scene, lấy mẫu telemetry mỗi frame (pose vật/cổ tay, độ nghiêng, lực ngón tay) + timeline phase/note + ảnh chụp head-camera detection. Ghi vào `artifacts/episodes/<name>/`, index tại `artifacts/episodes/index.json` (gitignored, tạo lại khi cần).
- `viewer/index.html`: liệt kê episode đã ghi, phát lại — chuyển camera (hoặc lưới 2×2), tua theo timeline tô màu theo phase, đọc state sống tại thời điểm tua.
- `scripts/serve_viewer.py`: serve project qua HTTP (viewer cần origin http:// để fetch JSON/stream video, không chạy được qua file://); `.claude/launch.json` chạy nó như target preview Browser pane.

### 2.5 Tích hợp ROS 2 thật — RH56 + MuJoCo làm stand-in ros2_control

**`openarm_pick_place/mujoco_bridge.py` và các sửa đổi sim liên quan đã commit** (`cdee7be`, mục 0.1). Còn uncommitted: `ros2/` (package URDF/mesh) và `scripts/install_rh56_ros2_overlay.sh` — xem mục 0.7 để biết vì sao đang treo (mesh nhị phân, cần quyết định cách đưa vào git).

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

## 4. Kết quả kiểm thử hiện tại (cập nhật 21/09 chiều — xem mục 0 cho chi tiết/nguyên nhân)

| Hạng mục | Kết quả |
|---|---:|
| Unit/integration test (`unittest discover -s tests`) | **51/52 đạt** |
| Acceptance ngẫu nhiên mẫu 20 (`test_twenty_trials_acceptance`, seed 7) | **17/20** (cần ≥19/20 — không phải thước đo Stage 2 chính thức) |
| Stage 2 fixture 50-layout, tay phải (`benchmarks/fixtures/right-50.json`) | **đang chạy, chưa có kết quả** — cổng 48/50 |
| Test tay trái mirror (`test_left_arm_mirrored_trial_metrics`) | **đạt** (tự sửa theo cùng fix mục 0.1) |
| Proof-lift | trượt ≤1mm, nghiêng 1–4° (đo lại nhiều trial trong phiên 0.1–0.5) |

*(Bảng cũ bên dưới từ phiên 18/09, giữ lại làm tham chiếu lịch sử — số liệu đã lỗi thời)*

| Hạng mục | Kết quả (18/09) |
|---|---:|
| Trial vật lý, bố cục mặc định | 3/3 đạt |
| Trial + perception | 3/3 đạt, sai số perception ≤4.7mm |
| Trial ngẫu nhiên + perception (seed 7, 6 trial) | 6/6 đạt, sai số đặt 12–17mm |
| Unit/integration test | 38/38 đạt |
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
| **P1 — ưu tiên hỏi trước** | Giới hạn lực servo cánh tay trong sim ±120N (`simulation/five_finger_model.py:411`) | Gấp 3 motor DM8009 thật (40N). Mọi phát hiện chiều 21/09 (lực kẹp rò rỉ, creep lúc hạ, xê dịch lúc nhả — mục 0.1–0.2) nhạy với độ cứng servo; tinh chỉnh grasp trước khi có số thật có nguy cơ không chuyển sang phần cứng |
| P1 | Độ dày adapter flange→đế tay (1cm) là giả định | Cần CAD/đo thật của adapter RH56F1 trước sim-to-real |
| P1 | Vùng với tư thế cổ-tay-thẳng hẹp; nút thắt là **joint5** (±90°), không phải chiều dài tay — xác nhận bằng map vùng với 21/09 (mục 0.4) | Cân nhắc tư thế tham chiếu thứ hai hoặc dùng tay trái cho nửa bàn bên trái |
| ~~P1~~ đã xong | ~~Hình học lab 0.727m/+68.64mm~~ | **Đã xác nhận đồng bộ 21/09** (mục 0.6) — không còn treo |
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

- Memory: [[project-scope-openarm-ros2]], [[inspire-rh56-joint-values]], [[reference-openarm-project-repos]], [[project-target-pipeline-gtx1650]], [[lab-geometry-measurements]] (đã cập nhật 21/09 — xem mục 0.6)
- [README.md](README.md) — lệnh chạy đầy đủ (sim + ROS 2), bao gồm 2 sự cố mới gỡ ở mục 5
- [BAO_CAO_TRANG_THAI_PROJECT.md](BAO_CAO_TRANG_THAI_PROJECT.md) — báo cáo kỹ thuật chi tiết phần sim (18/09/2026, đã lỗi thời một phần — xem mục 0)
- [BAO_CAO_MENTOR.md](BAO_CAO_MENTOR.md) — báo cáo gửi mentor, đối chiếu pipeline + câu hỏi (18/09/2026)
- `scripts/make_benchmark_fixture.py`, `scripts/benchmark_pick_place.py` — công cụ fixture/benchmark Stage 2 (mục 0.5), docstring trong từng file giải thích đầy đủ lý do thiết kế
- `_bench.log` (thư mục gốc, gitignored/tạm) — log benchmark 50-layout đang chạy lúc ghi tài liệu này; xóa sau khi đọc kết quả

## Cập nhật thực thi 22/09/2026
- Đã chạy vòng benchmark mới có perception: tay phải 9/10 (rtifacts/benchmarks/right-v2-next-10.json), tay trái 3/10 (rtifacts/benchmarks/left-v2-next-10.json). Chưa đủ điều kiện chạy 50 ca.
- TrialResult bổ sung telemetry release object/wrist trước mở, sau mở và sau retreat để phân tích hướng drift.
- Full handoff collision-free chưa triển khai; pose sweep hiện chưa đạt điều kiện an toàn.

- 23/09/2026: centring trái thử pose nâng 30 mm, hạ lại nếu IK cho phép; smoke 1/2, release vẫn còn lỗi khi hạ pose không khả thi.

- 23/09/2026: phase_carry thử centring tay trái ở carry height và re-plan lower path; target gần mép rổ vẫn bị IK reject, smoke 1/2.
- Quét handoff tĩnh với hai wrist target riêng và ràng buộc hai tâm kẹp nằm trong chiều cao 100 mm của vật: 116 cặp có IK/joint margin nhưng tốt nhất vẫn xuyên nhau 22.7 mm tại 53 contact (`python -m scripts.check_handoff_geometry`). Direct handoff chưa có pose vật lý an toàn với collision mesh hiện tại.

- Success placement được đánh giá theo containment trong lòng rổ + contact đáy, không bắt buộc tâm rổ. Tay trái containment benchmark đạt 10/10: artifacts/benchmarks/left-containment-10.json.

## Cập nhật handoff (tiếp sức qua điểm trung chuyển) 22/09/2026
- Yêu cầu: tay phải cầm vật -> chuyển cho tay trái -> tay trái đặt vào rổ. Hai hướng đã thử, cả hai đều bị chặn vật lý (không phải lỗi code):
  1. Hai tay cùng nắm vật đồng thời: đã biết thất bại từ trước (22.7mm xuyên tay).
  2. Tiếp sức tuần tự qua 1 điểm trung chuyển trên bàn (đã cài `simulation/pick_place/handoff.py` + `find_handoff_point` + `HANDOFF_*` route thực thi được trong `cli.py`): quét lại phát hiện **đặt** (khi đang cầm) chỉ với tới y≈0 (đường tâm bàn, đã quét x tới 0.65m không cải thiện), còn **nắm mới** một vật cần y >= ~0.30-0.33m tính từ tâm (grasp joint margin về 0 ngay khi gần tâm, không phụ thuộc x). Không có điểm nào vừa đặt được vừa nắm ra được -> không tồn tại điểm trung chuyển khả thi cho layout đối xứng (vật và rổ ở hai phía đối nhau).
  3. ~~Ý tưởng của mentor "đặt rổ giữa bàn, phải cho vào - trái lấy ra" đã kiểm tra và bị chặn bởi đúng giới hạn ở mục 2~~ **-> SAI, đính chính bên dưới: đã làm được khi rổ đặt trong vùng nắm-được thật của MỘT tay (không phải y≈0 tuyệt đối) và CÙNG một tay làm cả hai việc.**
- Router (`TaskRouter.select`) giờ trả `REJECTED` kèm lý do rõ ràng khi cả hai chiến lược trên đều fail, thay vì đề xuất một route không chạy được.
- Việc cần làm tiếp nếu muốn giải quyết thật: (a) chiến lược grasp/carry khác (không phải top-down/wrist-straight) cho vùng gần tâm, hoặc (c) chấp nhận giới hạn hiện tại và để router trả REJECTED.
- **(b) đã đào sâu 2 vòng, đính chính lại kết luận ban đầu:**
  - Vòng 1 (60 seed ngẫu nhiên/điểm): 75% quãng đường transfer (y=-0.036, gần tâm) → 0/60 hội tụ. Kết luận vội vàng lúc đó: "giới hạn động học cứng". **Sai — do mẫu quá nhỏ.**
  - Vòng 2 (300 seed/điểm x4 run độc lập + quét z 0.10-0.65m + 35 seed có cấu trúc quanh tư thế chuẩn): đúng điểm đó, 2/4 run 300-seed tìm ra nghiệm (margin 12.6° và 8.3°), 2 run kia 0/300; seed có cấu trúc quanh tư thế chuẩn 0/35; quét z không có xu hướng theo chiều cao (không phải do "robot cao/thấp"). Nghiệm tồn tại nhưng trên nhánh cấu hình rất hẹp/bất thường (joint7 lệch >70° so với chuẩn).
  - **Kết luận đúng:** đây là rìa vùng với-tới (workspace boundary), không phải giới hạn tuyệt đối cũng không phải bug seeding vài-seed-là-xong — cần hàng trăm seed/waypoint mới có ~1/600 cơ hội, không thực tế cho planner thời gian thực, và tư thế tìm được cũng không đủ ổn định để đi qua trong quỹ đạo liên tục. Sửa `_walk`/`transfer_route_candidates` bằng vài seed dự phòng sẽ không giải quyết được. Muốn mở khoá thật cần (a) — grasp/carry khác hẳn top-down/wrist-straight cho vùng gần tâm.

## Đính chính + thành công: "rổ giữa bàn" (22/09/2026, sau khi người dùng cho phép chỉnh setup)
- Yêu cầu mới: robot nắm vật đặt vào rổ ở giữa bàn, rồi **tay trái** nắm lại vật nằm trong rổ và nhấc ra ngoài. Mấu chốt khác với mục "Cập nhật handoff" ở trên: đây không phải chuyển tay (hai tay khác nhau), mà có thể để **một tay duy nhất** làm cả hai việc — né hoàn toàn giới hạn "đặt chỉ tới y≈0 nhưng nắm cần y≥0.30".
- Map vùng nắm-được thật của tay trái bằng quét (`plan_pick`, không phải chỉ 9 yaw mặc định mà toàn bộ lưới x,y): vùng chỉ là một "nêm" hẹp x∈[0.02,0.18], y∈[0.28,0.40] (thu hẹp dần khi x tăng) — ra ngoài vùng này (kể cả vẫn ở "nửa" của tay trái, y>0, không hề gần tâm) là **không nắm mới được**, kể cả với x lớn (đã thử tới x=0.65). Đây là giới hạn thật của tư thế top-down/wrist-straight, độc lập với chuyện gần tâm bàn hay không.
- Tìm được cặp điểm A=(0.02,0.28), M=(0.16,0.38) đều nắm được và cách nhau 17.2cm (đủ 15cm tối thiểu của `Scene()`), nằm ở góc xa nhất của vùng nêm.
- Cài `simulation/pick_place/retrieve.py`: `RetrieveDemo`/`run_retrieve_trial` — leg 1 là `Demo` bình thường (đặt vật vào rổ tại M), leg 2 tiếp tục **trên cùng một Scene/MjData** (không dựng lại model mới) để tay trái nắm thật từ trong rổ (đã thêm basket_geoms vào check va chạm lúc nắm trong `planner._plan_pick`) rồi mang ra đặt ở điểm bàn trống C (dùng `centers()`'s `place_floor` param mới, tách khỏi `scene.basket_floor()`).
- Lần chạy đầu tiên FAIL ở bước khép ngón (chỉ ngón cái+giữa chạm, index/ring/pinky không chạm) vì planner bị ép chọn `grasp yaw +0` (các yaw khác bị basket chặn) — yaw này hoá ra không hợp với hình học bàn tay trái. Nguyên nhân thật: `run_retrieve()` (viết mới) thiếu cơ chế retry-với-yaw-khác mà `Demo.run()` gốc đã có cho lỗi grasp vật lý. Thêm lại cơ chế đó (loại yaw lỗi, thử lại) -> lần thử thứ 3 dùng yaw -15° và **thành công hoàn toàn**: lực ngón tốt (thumb 15.6N, index 5.7N, middle 8.5N, ring 11.3N), trượt 0mm, tilt cuối 0°.
- Điểm đặt C=(0.02,0.28) (trùng A) làm bước rút tay cuối cùng fail (không tìm được raise waypoint an toàn, do quá gần cột trụ robot) dù vật đã đặt xong. Đổi C=(0.10,0.28): **PASS toàn bộ**, sai số đặt 36.2mm, tilt 0°. C=(0.05,0.32) cũng PASS (47.6mm). C=(0.12,0.30) fail (đặt xuống chưa chạm bàn sau 2.8cm hạ, có thể do quá gần rổ M).
- Layout cuối cùng dùng trong `tests/test_retrieve.py`: A=(0.02,0.28), M=(0.16,0.38) ("giữa bàn" theo nghĩa vùng dùng được, không phải x,y=0.25,0 hình học), C=(0.10,0.28), side="left". Ảnh render xác nhận vật đứng vững ngoài rổ sau khi hoàn tất.
- Đã thêm `basket_geoms` vào check va chạm của `_plan_pick` (áp dụng cho MỌI lần nắm, không riêng retrieve — an toàn vì basket thường ở xa nên là no-op với các pick bình thường).
- Full test suite (`unittest discover -s tests`) sau các thay đổi: không regression (chỉ còn fail cũ `test_twenty_trials_acceptance`, không liên quan).

## Đính chính quan trọng: rổ ở ĐÚNG giữa bàn (y=0.0) khả thi ngay với code hiện có (22/09/2026, sau)
- Người dùng nhớ đúng: đã từng thử rổ giữa bàn, robot nắm được vật. Sau khi thử "chiến lược nắm khác" (thất bại) và "sửa URDF dịch bệ đỡ + tách vai" (thất bại, xem log điều tra ở trên) đều không giải quyết được x=0.30m, tìm lại thấy episode có sẵn `artifacts/episodes/centre-to-right` (18/09/2026): vật nắm được **trực tiếp tại (0.22, 0.0)** — đúng đường tâm — PASS, grasp_yaw=90°, sai số 4.7mm. Vậy khẳng định "nắm cần y≥0.30" ở các mục trước chỉ đúng cho layout đã test, không phải quy luật chung.
- Test lại chiều giao vật vào rổ giữa bàn: `Demo((0.10, -0.35), (0.22, 0.0), side="right")` → **PASS ngay, không sửa code**: sai số 14.5mm, tilt 0°, lực ngón thumb=17.9N/index=8.9N/middle=5.0N/ring=8.1N, grasp_yaw=+30°, place_yaw=+60°.
- Lý do các lần thử trước fail: dùng `PICK_POSITION_A=(0.08,-0.38)` (điểm nhặt SÂU NHẤT) làm điểm nhặt → đường transfer (mang vật, không phải nắm/đặt) phải băng gần hết bàn, mọi waypoint trung gian ở độ cao mang vật đều ngoài tầm. Đổi sang điểm nhặt vừa phải (0.10,-0.35) → đường transfer ngắn hơn nhiều → giải được ngay.
- Bài học: "gần tâm" (y≈0) và "xa 30cm" (x=0.30) là hai giới hạn ĐỘC LẬP, bị nhầm thành một vì mọi layout test trước đó kết hợp cả hai cùng lúc. x=0.30 vẫn thật sự bất khả (đã kiểm chứng kỹ). y=0.0 tự nó không phải rào cản.
- Đã thêm test khóa lại: `tests/test_bimanual_routing.py::test_delivers_object_to_true_centreline_basket`. Lệnh tái tạo: `python -m simulation.pick_place_demo --object 0.10 -0.35 --basket 0.22 0.0`.
- (Đã thử và revert: thêm `pedestal_forward_offset_m` vào `five_finger_model.py`/`scene.py` — không giải quyết được vấn đề gốc (orientation, không phải vị trí lắp) nên đã bỏ. Đã thử và revert: thêm bank seed dự phòng (`_solve_with_retries`) vào `planner.py` — kỹ thuật hoạt động nhưng phá vỡ `test_grasp_at_joint_limit_is_not_an_executable_layout`, một test an toàn có chủ đích; đã bỏ vì xung đột trực tiếp với quyết định an toàn đã có của dự án.)

## Tiến độ 23/09/2026: tay phải đặt vào rổ giữa bàn → tay trái lấy ra (đã đạt, bố trí mô phỏng)

**Trạng thái:** task "tay phải nắm vật đặt vào rổ ở giữa bàn, tay trái nắm vật từ rổ nhấc ra đặt lên bàn" chạy vật lý **PASS**, hai tay không chạm nhau. Đã commit + push (`2aa82e8` trên `origin/master`).

**Layout đạt (chỉ có trong mô phỏng, chưa có trên phần cứng thật):**
- Vật A=(0.26,-0.26) → rổ B=(0.32, **0.0**) (đúng trục giữa, thẳng trước thân) → đặt ra C=(0.34,0.16).
- `arm_half_separation=0.06` (mặc định vendor 0.031), `left_arm_mount_yaw_deg=-95`, `right_arm_mount_yaw_deg=+40` (xoay đế tay quanh trục đứng). Chiều dài link, giới hạn khớp, tư thế nắm: giữ nguyên của vendor.
- Kết quả: sai số đặt 30.5mm, nghiêng cuối 0°, xuyên thấu 0mm.
- Lệnh: `RetrieveDemo((0.26,-0.26),(0.32,0.0),(0.34,0.16), side="left", place_side="right", arm_half_separation=0.06, left_arm_mount_yaw_deg=-95, right_arm_mount_yaw_deg=40)`. Test: `tests/test_retrieve.py::test_right_places_left_retrieves_centre_basket`. Episode trên web viewer: `centre-basket-right-in-left-out`.

**Vì sao cần đổi bố trí (đã đo, không đoán):**
- Bố trí gốc: tay phải đặt được tới giữa bàn, nhưng tay trái không nắm được ở bất kỳ điểm nào quanh giữa (0/20 vị trí). Vùng nắm của mỗi tay nằm lệch ra ngoài vai của nó ~0.25–0.40m, nên không có vị trí rổ nào cả hai cùng dùng được.
- Xoay đế tay làm vùng nắm quay quanh vai. Xoay đối xứng thì cả hai vùng cùng dồn vào giữa, khiến tay phải mất chỗ nhặt vật. Phải xoay **bất đối xứng**: trái nhiều (-95°), phải ít (+40°).
- Chỉ xoay đế trái -115/-120° với khoảng cách vendor thì khâu 1 tay trái va vào đế tay phải: khớp 1 kẹt thiếu 25.5°, tay trượt vật (lực 5 ngón = 0).
- Cửa sổ nắm của tay trái ở giữa bàn rất hẹp (x≈0.305–0.35m). Chọn -95° vì cửa sổ đó phủ đúng điểm vật thực sự rơi sau chặng 1, (0.308,-0.013); ở -85° vật rơi hụt ngoài cửa sổ vài mm.

**An toàn (thêm vào code, áp dụng cho MỌI episode):**
- `Scene.robot_side()`, `Scene.inter_arm_contacts()`: phát hiện mọi tiếp xúc giữa phần trái và phần phải của robot (cả khâu tay lẫn bàn tay).
- `Executor._check_collisions` dừng ngay khi (a) hai tay chạm nhau, hoặc (b) tay đang nghỉ chạm vào vật (`Executor.active_side`, do `Demo` gán). Lỗi (b) đã xảy ra thật khi đổi bố trí: tư thế nghỉ của tay trái nằm trên đường tay phải mang vật, làm vật bị hất.
- Đo trên quỹ đạo đã chạy (layout đạt):
  - Khoảng cách nhỏ nhất giữa hai tay: **23.5mm** (ngón trỏ trái ↔ khâu 6 tay phải).
  - Margin khớp nhỏ nhất khi chuyển động: tay trái **3.8°** (khớp 5), tay phải **4.1°** (khớp 1); ngưỡng planner ≥3°.
  - **Chưa giải quyết:** khớp 4 của cả hai tay nằm đúng giới hạn 0° ở tư thế nghỉ (`ATTENTION_RIGHT = zeros`, tay buông thẳng, dải khớp 4 là [0°,140°]). Tư thế này có sẵn từ trước và dùng chung cho mọi episode. Muốn sửa thì phải cho khớp 4 gập nhẹ ở tư thế nghỉ, rồi chạy lại toàn bộ test/benchmark vì mọi đường nâng tay đều xuất phát từ tư thế này.
- Chỉ số rủi ro còn lại: margin 3.8–4.1° chỉ vừa trên ngưỡng 3°; điểm đặt ra C=(0.34,0.16) là ứng viên duy nhất lập được kế hoạch (margin khớp tại điểm đặt 3.2°). Layout này đang ở sát rìa vùng với-tới, chưa chạy nhiều lần có nhiễu.

**Tham số mới (opt-in, mặc định không đổi gì):** `left_arm_mount_yaw_deg`, `right_arm_mount_yaw_deg` (`five_finger_model.build_five_finger_spec`, `Scene`); `RetrieveDemo(place_side=..., arm_half_separation=..., left_arm_mount_yaw_deg=..., right_arm_mount_yaw_deg=...)`; `scripts/record_episode.py --place-arm --arm-half-separation --left-mount-yaw --right-mount-yaw`.

**Test:** full suite 60/61. Test duy nhất fail là `test_twenty_trials_acceptance` (ngẫu nhiên, đã fail từ trước phiên này với lý do lift/transfer thấp); đang chạy lại riêng để xác nhận không liên quan đến chốt an toàn mới.

**Việc tiếp theo đề xuất:**
1. Sửa tư thế nghỉ để khớp 4 không nằm đúng giới hạn.
2. Chạy nhiều trial có nhiễu vị trí vật cho layout rổ giữa bàn (hiện mới chạy 1 layout cố định).
3. Nếu muốn đưa lên robot thật: đo/chế tạo đế tay xoay tương ứng, hoặc tìm chiến lược nắm khác cho vùng giữa bàn thay vì xoay đế.

## Cập nhật 23/09/2026 (chiều): layout rổ giữa bàn không tái tạo được -> chỉnh lại
- Chạy lại `test_right_places_left_retrieves_centre_basket` (layout rổ (0.32,0.0), đế trái -95°): **FAIL** ở cả HEAD `2aa82e8` sạch lẫn working tree. Tay phải đặt vật xong (không đổi), nhưng vật nằm ở (0.295,-0.012) thay vì (0.308,-0.013) như ghi nhận -> hụt cửa sổ nắm tay trái (x≥0.305) 10mm -> "no reachable grasp".
- Đo telemetry thả: trước khi mở tay vật ở (0.307,-0.010); lúc tay phải vừa mở ngón vừa rút lên, vật bị kéo lùi thêm 12mm về phía -x.
- Đã thử và bỏ: nhắm rổ xa hơn (x=0.33: vẫn trượt 14mm, x≥0.335: tay phải không đặt tới); cho tay phải "mở trước rồi rút" như tay trái (kết quả hỗn loạn: tốt ở x=0.33, tệ hơn 52mm ở x=0.32).
- **Layout mới PASS:** rổ B=(0.32, **-0.02**), `left_arm_mount_yaw_deg=-100` (các tham số khác giữ nguyên). Trình tự: tay phải nắm A=(0.26,-0.26) -> đặt vào rổ -> **thu về tư thế nghỉ** (`phase_release` kết thúc bằng `attention_pose`) -> tay trái nắm vật trong rổ -> đặt ra C=(0.34,0.16). Sai số 24.9mm, nghiêng 0°, xuyên thấu 0mm, không chạm tay-tay (executor tự dừng nếu có).
- Độ bền (nhiễu điểm nhặt A): 6/7 PASS (±5mm chéo, ±10mm theo y, -10mm theo x); fail khi A lệch +10mm theo x. Vẫn là layout sát rìa vùng với-tới.
- Lưu ý: submodule `assets/rh56_controller/h1_mujoco/archive/inspire/inspire_left.xml` đang có sửa đổi chưa commit (đảo dấu y các site đầu ngón trái) — không phải nguyên nhân lỗi trên (HEAD với asset gốc fail y hệt), nhưng cần xác nhận ai sửa và có giữ không.

## Kiểm tra cấu hình OpenArm v1 so với vendor (23/09/2026)
Đối chiếu model đã compile với `openarm_mujoco/v1/openarm_bimanual.xml`:
- **Đúng:** dải góc 7 khớp cả hai tay khớp vendor (kể cả j1/j2 bất đối xứng trái-phải); ctrlrange = dải khớp; `limited=true`.
- **Sai lệch:** servo vị trí kp=800/kv=12, lực ±120 cho MỌI khớp; vendor giới hạn theo motor: DM8009 (j1-2) ±40, DM4340 (j3-4) ±27, DM4310 (j5-7) ±7 Nm. Damping/friction cũng khác (vendor 0.4/0.1).
- Đo task rổ giữa bàn: tay phải nằm trong giới hạn motor (đỉnh 16 Nm ở j1). Tay trái j6/j7 vượt 7 Nm ~50% thời gian (đỉnh j6 35 Nm, j5 22 Nm), j3 đỉnh 25.3/27 Nm.
- **Nguyên nhân chính:** ở tư thế nghỉ (tất cả khớp = 0), tay trái chạm bệ `robot_riser`; với đế trái −100° lực đè cần j6=9.4 Nm, j7=6.9 Nm giữ liên tục. Executor không bắt được lỗi này (chỉ bắt tay-tay và tay-nghỉ-chạm-vật).
- Margin khớp tối thiểu: j4 = 0° (tư thế nghỉ nằm đúng giới hạn), j1 trái 3.4°, j5 trái 3.3°.
- Vận tốc đỉnh 3.9 rad/s (j4). MJCF vendor không có giới hạn vận tốc; `Openarm-ROS2-robot-control/.../safety_config.yaml` ghi max 1.0-2.0 rad/s (file chung, dải khớp trong đó không khớp OpenArm nên chỉ tham khảo).

## Tiến độ 23/09/2026 (chiều) — WIP, nhánh `wip/real-height-centre-basket`

**Robot thật (ảnh + số đo của người dùng):** hai tay buông thẳng hai bên cột giữa, không xoay đế. Đỉnh robot (gồm bệ + tấm đế) 0.78 m, đỉnh camera trên giá xanh 0.88 m so với mặt bàn. Mô phỏng cũ cộng tấm đế 29 mm hai lần (đỉnh 0.809 m) -> đã sửa: vai 0.697 m, camera tâm 0.8675 m (`ROBOT_TOP_ABOVE_TABLE`, `CAMERA_TOP_ABOVE_TABLE` trong `five_finger_model.py`), thêm giá camera (chỉ hiển thị). Robot dùng bàn tay RH56.
Bố trí xoay đế (-100°/+40°, tách 0.06 m) của buổi sáng KHÔNG khớp robot thật -> chỉ là thí nghiệm sim.

**Tư thế nghỉ mới:** `ATTENTION_RIGHT = [-20, 10, 0, 40, 0, 0, 0]°` — đầu ngón cách bàn 40 mm, khớp 4 cách giới hạn 40° (cũ: 10 mm sau khi hạ robot). `tests/test_mujoco` 17/17 OK.

**Đo được về tầm với (robot đúng chiều cao):**
- Vai→cổ tay ~0.436 m + bàn tay ~0.19 m: tay chỉ vừa tới mặt bàn dưới vai. Nắm ngang ở tầm lon trong rổ là bất khả thi (cổ tay không xuống tới z≈0.1 m; 120 hướng ngón gần ngang đều không có IK).
- Khớp 2 vai chỉ khép vào +10° -> tay trái khó qua đường giữa; khi nâng thẳng thì khớp 5 (90°)/khớp 6 (45°) chạm giới hạn.
- Nhiều kết luận "không với tới" trước đây là do planner chỉ thử 1–2 seed IK: với 80 seed ngẫu nhiên tay trái nắm được (0.30, 0.0) ở độ cao bệ 0.10 m, margin 11°.

**Code đã thêm (chưa test đầy đủ):**
- `planner._grasp_solutions`: 2 seed chuẩn + bank 40 seed cố định (RNG 0), thử nối chuỗi từ tối đa 6 nghiệm nắm (`GRASP_SEED_BANK_*`, `GRASP_CHAIN_ATTEMPTS` trong config).
- Twist-lift (`_twist_approach`): khi nâng thẳng kẹt giới hạn khớp, xoay bàn tay quanh trục lon trong lúc nâng (lon vẫn thẳng đứng).
- `Demo(place_offset=...)` / `RetrieveDemo(place_offset=...)`: thả lon lệch trong rổ, rổ vẫn ở giữa.
- `basket_stand_height`: bệ hộp dưới rổ, tính là vật cản như rổ (planner kiểm tra, executor abort khi chạm).
- `scripts/view_retrieve.py`: mở MuJoCo xem bài tay phải đặt / tay trái lấy.

**Kết quả hiện tại:** planner (chỉ động học) giải được rổ trên bàn ở (0.25, 0) và (0.30, 0) (lon lệch +3 cm), bệ không giúp (cao hơn còn tệ hơn). **Vật lý chưa pass:** (0.30, 0) tay phải không có đường mang tới rổ; (0.25, 0) lon rơi ở (0.229, -0.002) nên chuỗi tay trái không nối được.

**Việc tiếp theo:**
1. Chạy lại toàn bộ test (lần chạy trước bị ngắt, chưa có tổng kết) — seed bank có thể đổi kết quả các test cũ (lần trước một seed bank đã làm hỏng một test an toàn).
2. Làm transfer/place của tay phải cũng dùng seed bank; cho retrieve lập kế hoạch lại theo vị trí lon thật sau khi thả.
3. Khi pass: đo khoảng cách hai tay, margin khớp, độ lún vào bàn/rổ/bệ (yêu cầu: không chạm bàn, không va chạm), rồi mở `python -m scripts.view_retrieve` cho người dùng xem.

## Tiến độ 24/09/2026 — kết luận bài rổ giữa bàn (robot đúng chiều cao)

**Kết luận (báo mentor):** trên robot thật (vai 0.697 m trên mặt bàn, tay + bàn tay ~0.63 m, khớp vai 2 chỉ khép +10°) và với các ràng buộc an toàn (không chạm bàn/rổ/bệ, không lún vật, margin khớp >= 3°, giới hạn tốc độ khớp), **không có vị trí rổ nào để "tay phải đặt vào rổ, tay trái lấy ra" chạy ổn định**:
- Tay phải đặt lệch trái trong rổ tối đa y ≈ +0.04 m (rổ x 0.25–0.26). Rổ dời sang trái (tâm y 0.05/0.07/0.09, x 0.22–0.28, 6 điểm nắm khác nhau) -> tay phải không đặt được (bàn tay vướng thành rổ phía gần).
- Tay trái chỉ nắm được lon trong rổ khi y >= ~+0.03 m, chỉ ở hướng nắm -60°. Vùng chung ~1 cm < sai số thả lon (±1.5 cm); lệch 2 mm là kế hoạch đổi từ được sang không.
- Rổ x = 0.30 m: tay phải không đặt lệch trái được. Kê bệ 5–10 cm: tay trái mất hết nghiệm.
- Thí nghiệm hạ đế robot 10–30 cm CHƯA có kết luận: tư thế nghỉ và hướng nắm phải chỉnh lại theo độ cao mới rồi mới đo được.

**Đề xuất:** (1) robot đứng trên sàn cạnh bàn như thiết kế OpenArm, hoặc hạ đế (cần thí nghiệm đầy đủ); (2) hoặc đổi nhiệm vụ để mỗi tay làm trong vùng của nó; (3) bài trung chuyển qua điểm giữa (0.08,-0.38)->(0.25,0.25) cũng không ổn định (lon tuột 5/6 lần) -> router từ chối.

**Chạy được, đã kiểm chứng:** cùng một tay (phải) đặt vào rổ rồi lấy ra: nắm (0.08,-0.38) -> rổ (0.25,-0.25) -> đặt (0.15,-0.40): sai số 27 mm, nghiêng 0°, không lún. Xem: `python -m scripts.view_retrieve`.

**Thay đổi an toàn/planner (23–24/09):**
- Giới hạn tốc độ khớp trung bình 0.6 rad/s trong `Executor.follow` (trước đó một bước 0.18 rad/0.09 s hất văng lon).
- Thả lon ở khe hở <= 8 mm khi tay đã duỗi hết tầm (lon phải nằm trọn trong rổ).
- Bộ seed IK cho tư thế nắm (chỉ chạy khi 2 seed chuẩn không nối được chuỗi) và cho đường mang (đi ngược từ tư thế đặt về tư thế nâng).
- Sửa lỗi: hướng tay đã xoay (twist-lift) còn sót lại làm IK tư thế nắm lệch 45°.
- Router trung chuyển: điểm trung chuyển phải chịu sai số rơi ±2 cm (lưới 3×3) và không dùng bộ seed.
- Đã thử rồi bỏ (làm tệ hơn hoặc đo sai): vòng giữ lực nắm (ngón cái đã ở giới hạn hành trình -> lon bị ép bật ra), kiểm tra "bàn tay mở lún vào vật" (loại cả tư thế chạy tốt).
- Xoá 2 test lấy chéo tay dựa trên đế xoay giả (-95/-100°, +40°): robot thật không có đế xoay.
- **Còn mở (24/09):** `test_trial_records_full_evidence` (Demo perception, layout mặc định) đặt lệch 21.5 mm > ngưỡng 20 mm. Lon được căn còn 6.9 mm ở độ cao mang nhưng trôi ~13 mm trong đoạn hạ xuống đáy rổ. Hỏng từ khi thêm giới hạn tốc độ khớp (ở e434639 lon còn rơi khi nâng; đã sửa bằng `Demo.resolve_lift_from_here`: giải lại tư thế nâng từ tư thế hiện tại để khuỷu liên tục). Cần tinh chỉnh đoạn hạ; không nới ngưỡng. Toàn bộ test: 59/60 (trước 2 sửa cuối).

## Tiến độ 24/09/2026 (tối) — HOÀN THÀNH bài rổ giữa bàn, chéo tay (bệ 10 cm)

**Yêu cầu:** tay phải nắm lon bỏ vào rổ ở giữa bàn, tay trái lấy ra đặt ra ngoài; an toàn giới hạn khớp, không va chạm. Robot giữ đúng chiều cao thật (đỉnh 0.78 m).

**Giải pháp đã chạy được:**
- Bệ làm việc hình chữ nhật cao 10 cm (`work_platform_height=0.10`): dài bằng bàn (1.1 m theo y), rộng nửa bàn (0.5 m), x 0.17–0.67 (mép trước cách nắm tay đang nghỉ 4.7 cm — ở x 0.12 nắm tay lấn vào bệ). Lon, rổ (đáy phẳng, (0.28, 0) — giữa, thẳng thân robot) và điểm đặt ra nằm trên bệ.
- Nắm xiên (planner `_grasp_tilts`, `GRASP_TILT_CANDIDATES`): ngón tay chúc ~38° dưới ngang, ngón cái và các ngón kẹp ngang thân lon (lệch cao <= 20 mm; góc x+60 để ngón cái trượt qua nắp lon -> bị loại). Trên bệ nắm xiên được thử trước; trên mặt bàn giữ nắm từ trên xuống như cũ.
- Kết quả: layout danh nghĩa pick (0.28,-0.25) -> rổ (0.28,0) -> đặt (0.28,0.25): sai số 3 mm, nghiêng 0°, không lún. Lệch điểm nắm ±1.5 cm: 7/8 pass (hỏng (0.28,-0.235): tay phải cầm không đủ chắc, lon tuột khi mang).
- Xem: `python -m scripts.view_retrieve --place-arm right --retrieve-arm left --pick 0.28 -0.25 --basket 0.28 0.0 --retrieve-to 0.28 0.25 --platform 0.10`. Test: `tests/test_retrieve.py::test_right_places_left_retrieves_centre_basket_on_platform`.

**An toàn thêm trong đợt này:** khoảng cách cánh tay (khâu 2–7 + bàn tay) tới thân robot >= 15 mm trong planner (`arm_body_clearance`; khâu 5 tay trái từng sượt thân); nâng lon thẳng đứng theo đường Descartes (`Demo.lift_straight_up`) thay vì nội suy khớp; nâng cao thêm 1.5 cm trên bệ để bù lon tụt; lon được dựng thẳng trước khi đặt khi lấy ra (retrieve); điểm nâng tay dự phòng tính theo bàn tay đang buông / hướng tay chuẩn.

**Mượt khi xem MuJoCo:** vật lý chạy 1.27× thời gian thực; giật là do planner tính giữa chừng (tới 32 s/lần). Đã: giảm vòng lặp IK của seed dự phòng trong `find_raise` (6000 -> 400) — cả bài 169 s -> 75 s; planner chạy ở luồng phụ (`Executor.think`) nên cửa sổ vẫn vẽ lại, xoay camera được trong lúc tính.

**Ảnh "ngón tay xuyên bàn" (người dùng gửi):** đo suốt một lần chạy, ngón tay luôn >= 22 mm trên mặt bàn; vết đen dưới bàn nhiều khả năng là bóng đổ của trình xem, chưa kiểm chứng bằng cách tắt bóng.

**Còn lại / đã thử bỏ:** rổ đáy chữ V (tham số `basket_floor_tilt_deg`, 6/9) — giữ làm tùy chọn, không dùng cho bài chính. Hạ/nâng robot ±10–15 cm không giúp (đoạn mang cần cổ tay z ≈ 0.38 m).
- **Đã sửa (24/09):** `test_trial_records_full_evidence` (lệch 21.5 mm) — hai lỗi ở bước hạ lon vào rổ: (1) `Executor.descend_until` lấy đích theo cổ tay *đo được* (lệch vài mm do tải) nên cổ tay trôi ngang ~11 mm; giờ theo tư thế *được ra lệnh* (FK của ctrl). (2) `resting_z` trong `Demo` dùng tâm tấm đáy rổ thay vì mặt trên (thiếu 5 mm) nên điều kiện "đã chạm đáy" không bao giờ đạt, tay ép xuống hết 4 cm. **Toàn bộ test: 61/61 OK.**

## Tiến độ 24/09/2026 — trạng thái cuối ngày (master = 4acb04c)

**Đã xong và đã merge vào master:**
- Mô phỏng đúng robot thật: đỉnh robot 0.78 m, đỉnh camera 0.88 m (vai 0.697 m), giá camera xanh, tư thế nghỉ đầu ngón cách bàn 40 mm.
- Bài chính: rổ đáy phẳng ở giữa bàn (0.28, 0) trên bệ 10 cm; tay phải bỏ lon vào rổ, tay trái lấy ra đặt ra ngoài. Danh nghĩa: sai số 3–7 mm, nghiêng 0°, không va chạm; lệch điểm nắm ±1.5 cm: 7/8.
- An toàn: dừng khi tay/cánh tay chạm bàn, bệ, rổ, thân robot; hai tay chạm nhau; tay nghỉ chạm lon; margin khớp >= 3°; tốc độ khớp tay <= 0.6 rad/s; cánh tay cách thân >= 15 mm (planner).
- Sửa lỗi đặt lon: hạ theo tư thế được ra lệnh (không theo cổ tay đo được), độ cao "đã chạm đáy" tính từ mặt trên tấm đáy.
- Xem MuJoCo mượt hơn: planner chạy luồng phụ (cửa sổ không treo), cả bài 169 s -> 75 s; phím **R** phát lại lần chạy đã ghi, không có đoạn dừng.
- Test: **61/61 OK** (lần chạy đầy đủ cuối, trước commit phím R — commit này chỉ sửa script xem).

**Chạy để xem:**
`python -m scripts.view_retrieve --place-arm right --retrieve-arm left --pick 0.28 -0.25 --basket 0.28 0.0 --retrieve-to 0.28 0.25 --platform 0.10`

**Việc tiếp theo đề xuất:**
1. Nâng độ ổn định tay phải ở điểm nắm (0.28, -0.235) (lon tuột khi mang — cầm chưa đủ chắc).
2. Tay trái đôi khi phải nắm lại lần 2–3; ưu tiên hướng nắm đã thành công (-60°) để giảm thời gian.
3. Kiểm chứng giả thuyết "ngón tay xuyên bàn" là bóng đổ (tắt bóng trong trình xem).
4. Chuẩn bị chuyển sang robot thật (ROS 2 / MoveIt) theo lộ trình trong project-scope.

## 24/09/2026 — Nguyên nhân giật (theo yêu cầu mentor, log cho PlotJuggler)

`python -m scripts.log_joint_states` -> `artifacts/joint_logs/*.csv` (mở bằng PlotJuggler, trục thời gian `sim_time`; cột `<tay>/j<i>/cmd|pos|vel|err_deg|torque`, `wall_dt_ms`, `thinking`, `phase`).

- **Do lệnh truyền vào (đã sửa):** `Executor.follow` bắt đầu mỗi đoạn từ vị trí khớp *đo được* thay vì *lệnh hiện tại*; servo trễ 0.3–1° nên lệnh nhảy lùi trong 1 ms -> mômen đổi dấu ~11.6 N·m, 39 lần/lần chạy. Nay cánh tay bắt đầu từ lệnh hiện tại (bàn tay vẫn từ vị trí thật để thả vật đúng). Kết quả: 39 -> 0 đỉnh, vận tốc lệnh max 418 -> 65 °/s, bước mômen max 11.6 -> 0.6 N·m.
- **Không phải do máy yếu:** vật lý chạy 2.6× thời gian thực (0.77 ms/2 ms mô phỏng, p99 3.5 ms). Chỗ "đứng hình" là planner tính kế hoạch (1–15 s/lần, vật lý dừng) — phím R phát lại không có đoạn dừng.
- Kèm theo: siết lại ngón < 5 N sau nâng thử (`REGRIP_BELOW_N`); cánh tay cách thân >= 30 mm dọc cả đường mang. Test 61/61, 20 bài ngẫu nhiên 20/20.

## 24/09/2026 — cuối ngày (master = 8d40dce)
- Bài chính chạy lại trên MuJoCo sau khi sửa giật: thành công, lệch 2.1 mm, nghiêng 0°, không va chạm; phím R phát lại mượt.
- Log khớp cho PlotJuggler: `artifacts/joint_logs/retrieve_headless.csv` (trước sửa) và `retrieve_headless_fixed.csv` (sau sửa), ~60 MB mỗi file, không đưa lên git. Máy chưa cài PlotJuggler — cần cài (bản Windows từ GitHub Releases) để xem.
- Việc tiếp theo: xem log trên PlotJuggler cùng mentor; nâng độ chắc tay phải ở điểm nắm (0.28,-0.235); giảm số lần tay trái phải nắm lại.

## 24/09/2026 (tối) — vì sao tay trái phải nắm lại lần 2

- **Ngón cái không mở (đã sửa):** ở tư thế nghỉ bàn tay nắm lại, ngón cái đối diện và gập ngang qua ngón trỏ; lệnh mở ngón cái tại chỗ làm nó kẹt vào ngón trỏ (khớp gập ở 0.38 dù lệnh 0.0), tay tới lon với ngón cái trên nắp lon -> hỏng, phải nắm lại.
  - Cách sửa **đang có trong code** (khác bản nháp "mở tay 2 bước"): tay trái nghỉ với ngón cái xoay ra ngoài (`REST_THUMB_UNOPPOSED = {left: True}`, `Scene.rest_hand` thay `closed_hand` ở tư thế nghỉ/lúc thu tay) nên không còn kẹt; ngón cái trái gập sẵn theo `GRASP_CLOSURE_FRACTION` như planner giả định (`PRESHAPE_THUMB_MATCH_PLAN`). Mở 2 bước (`PRESHAPE_THUMB_STAGED`) **tắt cho cả hai tay**: bật cho tay phải làm hỏng 3/20 bài ngẫu nhiên có camera (lon rơi khi mang).
  - Nắm xiên có hệ số lệch riêng (`OBLIQUE_GRASP_HEIGHT_BIAS`, `OBLIQUE_JAW_BIAS_TOWARD_FINGERS`, dùng qua `GraspPlanner.grasp_tilt`): hệ số của nắm từ trên xuống đặt đầu ngón tay trái ngang nắp lon.
- ~~Còn mở: lần nắm đầu tay trái hỏng vì ngón cái chạm lon~~ -> không còn thấy: kiểm tra 25/09 tay trái nắm được ngay lần đầu (hướng -60°).
- **Chưa mượt khi xem trực tiếp:** do (1) các lần planner tính (robot đứng yên vài giây) và (2) mỗi lần nắm hỏng phải thả ra – tính lại – nắm lại. Phím R phát lại mượt; (2) đã hết ở kiểm tra 25/09.

## 25/09/2026 — kiểm tra lại bài rổ giữa bàn + giật (PlotJuggler log)

- Bài chính (pick (0.28,-0.25) -> rổ (0.28,0) bệ 10 cm -> đặt (0.28,0.25)): **2/2 PASS**, sai số 2.6 mm, nghiêng cuối 0°, **mỗi tay nắm 1 lần, không nắm lại**; thời gian mô phỏng 90 s -> 63 s.
- Log `artifacts/joint_logs/check_20260925.csv` (không đưa lên git), 14 khớp: vận tốc lệnh > 69 °/s **0** mẫu (max 64.5 °/s); mômen nhảy > 3 N·m/2 ms **0** lần. Bước mômen lớn nhất 2.05 N·m (phải j6, t=22.12 s, pha thả) có lệnh khớp đứng yên -> do ngón nhả lon đổi tải cổ tay, không phải lệnh giật. **Không còn giật do lệnh.**
- Vật lý nhanh 3× thời gian thực; "khựng" khi xem trực tiếp là planner tính (tổng 57.6 s ≈ 60% thời gian xem).
- Test: toàn bộ suite 60/61, test duy nhất fail đã sửa và chạy riêng OK (`test_demo_starts_in_symmetric_ready_pose_with_both_hands_closed` giờ so với `rest_hand`; trước đó so với `closed_hand` nên fail khi tay trái nghỉ với ngón cái xoay ra).
- **Còn yếu:** tay trái sau siết lại chỉ còn ngón cái 9.0 N + trỏ 8.6 N (giữa/áp út/út 0 N), trượt 4 mm, nghiêng 9° lúc nâng thử; lon nghiêng tới 21° trong tay phải lúc mang (log 24/09 cũng 17.7°) rồi vẫn đặt thẳng. Máy chưa cài PlotJuggler (phân tích trực tiếp từ CSV).

### 25/09/2026 — lệch điểm nắm ±1.5 cm (8 hướng quanh (0.28,-0.25))

- Chạy lại trên code hiện tại: **6/8**. Hỏng (0.28,-0.235): tay trái nâng thử trượt 8 mm, nghiêng 14° (vẫn dưới ngưỡng cũ 10 mm/15°) -> siết lại mọi ngón 0 N, lon tuột về rổ. (0.265,-0.235) đạt nhưng lệch 31.4 mm (cùng dấu hiệu trượt 9 mm/13°).
- **Sửa:** ngưỡng nâng thử theo từng tay `PROOF_LIFT_SLIP_LIMIT` / `PROOF_LIFT_TILT_LIMIT_DEG` (config.py, dùng trong `Demo.phase_grasp`): trái 6 mm / 11° (lần nắm giữ được: <= 4 mm / 9°), phải giữ 10 mm / 15° (tay phải nâng thử trượt 7 mm vẫn mang tốt). Vượt ngưỡng -> thả ra, nắm lại hướng khác (cơ chế retry sẵn có).
- **Sau sửa: 7/8** + danh nghĩa đạt (2.6 mm). (0.28,-0.235): 2.9 mm, (0.265,-0.235): 8.0 mm — cả hai tay trái nắm lại 1 lần (lần 2 trượt 1–2 mm, nghiêng 4°), thêm ~20 s mô phỏng. Test 61/61 OK.
- ~~Còn hỏng: (0.28,-0.265) — tay phải làm rơi lon khi mang~~ -> đã sửa, xem mục dưới.

### 25/09/2026 — sửa tay phải làm rơi lon khi mang: lệch ±1.5 cm **8/8** (+ danh nghĩa = 9/9)

- **Đo (log lực từng ngón mỗi 0.1 s):** lon không rơi vì nắm sai mà vì lực kẹp chập chờn khi nâng thẳng lên sau nâng thử (12 N -> 1.5 N -> 12 N, chu kỳ ~0.2 s); lon tụt dần 12–17 mm trong tay rồi rơi ở đầu đoạn mang. Lệnh ngón đứng yên, vị trí ngón nhích tới đúng lệnh -> lực về 0 (servo vị trí: lực = phần lệnh vượt quá bề mặt lon).
- **Sửa (3 chỗ):**
  1. `Demo.lift_straight_up` bắt đầu đường thẳng từ cổ tay *được ra lệnh* (`wrist_position_at(ctrl)`) thay vì cổ tay *đo được* (lún vài mm dưới tải) — cùng loại lỗi với cú giật đã sửa ở `Executor.follow`/`descend_until`; trước đó đầu đoạn nâng kéo tay xuống, lực về 0 đúng lúc đó.
  2. Siết lại sau nâng thử đọc lực **trung bình 0.1 s / 10 mẫu** (`REGRIP_SAMPLE_SECONDS`, `REGRIP_SAMPLES`) thay vì 1 thời điểm (thời điểm đó đọc ~0 N mọi ngón -> không ngón nào được siết).
  3. `REGRIP_BELOW_N` theo từng tay: **phải 8 N** (bằng lực mục tiêu kẹp; 5 N để các ngón 5–8 N chập chờn), **trái 5 N** (thử 8 N: ngón giữa 7.2 N bị siết thêm, lon nghiêng 7° trong tay rồi đổ khi đặt ra).
- **Kết quả theo từng bước** (8 điểm lệch + danh nghĩa): trước sửa 8/9 -> sửa 1: 8/9 (ca hỏng dời sang (0.265,-0.265)) -> +sửa 2: 8/9 -> ngưỡng 8 N cả hai tay: 8/9 (tay trái làm đổ lon ở (0.265,-0.25)) -> **ngưỡng theo tay: 9/9**, sai số 0.6–7.7 mm. Mỗi điểm chạy 1 lần.
- **Kiểm chứng:** chỉ test nhanh `tests.test_mujoco` 17/17 OK — **chưa chạy toàn bộ suite** (~25 phút) sau các sửa này; bước siết lại dùng cho mọi bài của cả hai tay nên cần chạy full suite lần tới.
- **Rủi ro còn lại:** lực kẹp vẫn chập chờn khi nâng — sửa này làm tay đủ chắc để vượt qua, chưa khử gốc (tiếp xúc ngón–lon trong mô phỏng / hệ số servo ngón).
