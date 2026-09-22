# Context Project — OpenArm + Inspire RH56 + D435 Pick & Place

Cập nhật: 2026-09-21 (phiên chiều — Stage 1/2 theo kế hoạch pipeline hai tay). File này tóm tắt project cho phiên làm việc mới (người hoặc AI) đọc để nắm ngữ cảnh nhanh, không cần đọc lại toàn bộ lịch sử commit/chat.

**Đọc mục 0 trước** — đó là trạng thái mới nhất và việc đang dở; mục 1–8 bên dưới là bối cảnh nền, một số con số đã lỗi thời (đánh dấu ở từng chỗ).

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
