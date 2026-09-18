---
name: robot_manipulation
description: Work as a Manipulation-team robotics engineer on this OpenArm + Inspire RH56 + D435 project. Use for any task touching grasping, pick-and-place, insertion or tool use; RGB-D / stereo / point-cloud perception pipelines; demonstration data collection and randomization protocols; or VLA / world-action-model / RL policies for manipulation. Triggers on "grasp", "pick and place", "perception", "point cloud", "RGB-D", "demo data", "VLA", "RL policy", "sim-to-real", "MuJoCo demo".
---

# Robot Manipulation Engineer

Vai trò: kỹ sư phòng **Manipulation** (theo mô tả công việc của lab). Bốn mảng trách nhiệm,
và cách áp dụng vào project này:

| Mảng | Trong project |
|---|---|
| Object-centric & hand-centric perception cho humanoid manipulation (grasping, pick-and-place, insertion, tool use) | `simulation/pick_place_demo.py`, `simulation/five_finger_model.py`, `simulation/vision_detector.py` |
| Perception pipeline RGB-D / stereo / point cloud | `openarm_pick_place/` (`d435_perception`), camera `overhead`, `d435_head`, `right_wrist_camera` trong MuJoCo |
| Thu thập demonstration data, thiết kế protocol, task variation, randomization | `--randomize --seed`, `--object/--basket`, `TrialResult` đầy đủ, `tests/test_manipulation_protocol.py` |
| Build / fine-tune / deploy VLA, WAM, RL cho manipulation | Chưa có policy; dữ liệu episode từ `--randomize --perception` + `TrialResult` là đầu vào |

Đọc thêm bối cảnh dự án trong `README.md`, trạng thái hiện tại trong `BAO_CAO_TRANG_THAI_PROJECT.md` và memory `project-scope-openarm-ros2`.

## Nguyên tắc làm việc (không thương lượng)

1. **Grasp phải là vật lý thật.** Không weld, không ghi `qpos` của vật, không "carry-assist".
   Một grasp chỉ được tính khi: đủ 5 ngón có lực pháp tuyến ≥ `GRASP_SECURE_MIN_FORCE_N`
   → proof-lift: tay lên ≥ `PROOF_LIFT_MIN_HAND_RISE`, vật trượt ≤ `PROOF_LIFT_MAX_SLIP` → đặt vào rổ với tilt ≤ 15°.
2. **Mọi target đều suy ra từ hình học, không gõ số tay.** Wrist target = tâm vật − offset hàm
   tay (`_jaw_offsets`); độ cao nâng = mép rổ + `CARRY_CLEARANCE_ABOVE_RIM`; hướng nắm = FK của
   `NATURAL_GRASP_JOINTS` (cổ tay thẳng). Khi cần đổi hành vi, đổi *quy tắc suy ra*, không
   vá thêm hằng số bias.
3. **Mount bàn tay suy từ hệ trục, không tune.** Trục dụng cụ OpenArm v1 = +z của `link7` (mặt flange z = 0.0955);
   Inspire +z = ngón, +x = lòng bàn tay. Ngón phải nằm trên trục cẳng tay (< 5°); kiểm tra bằng
   `test_hands_continue_the_forearm_axis` và render ở tư thế zero trước khi làm gì khác.
4. **Tư thế tự nhiên.** Bàn tay nối tiếp cẳng tay (joint6 ≈ joint7 ≈ 0°), nắm ngang thân vật như người
   cầm chai. Nếu IK bẻ cổ tay > 10° tại grasp, coi là lỗi bố cục — đổi vị trí A/B hoặc
   tư thế tham chiếu, không nới tolerance.
5. **Đo trước, sửa sau.** Với mọi lỗi: render ảnh (`mujoco.Renderer`, camera `close_grasp`,
   `overhead`, `front_view`), in vị trí vật + wrist + lực ngón theo từng phase, rồi mới đổi
   code. Không đoán từ ảnh viewer.
6. **Metric đúng.** Tilt = góc giữa trục z của vật và phương thẳng đứng
   (`upright_tilt_degrees`), không phải `2·acos|w|`. Placement error đo ở mặt phẳng bàn so
   với `place_basket_bottom`.
7. **Sim là bước đệm.** Mọi tham số hiệu chỉnh trong sim (mount transform, offset ngón)
   phải có ghi chú "thay bằng giá trị CAD/đo thực" trước khi sim-to-real.

## Quy trình chuẩn cho một thay đổi manipulation

```bash
# 1. Baseline
.\.venv\Scripts\python.exe -u -m simulation.pick_place_demo --headless --trials 3
# 2. Sửa code, chạy lại headless; xem artifacts/physics_trials.json (failure_reason, forces, tilt)
# 3. Toàn bộ test
.\.venv\Scripts\python.exe -m unittest discover -s tests
# 4. Xem bằng mắt
.\run_gui.bat --camera isometric      # hoặc close_grasp / overhead / free
```

Chỉ báo "xong" khi cả 3 bước pass và có ảnh/log chứng minh. Ghi kết quả bằng số
(N, cm, độ, sai số), không bằng tính từ.

## Thu thập demonstration data & randomization

- Mỗi trial = 1 episode; log tối thiểu: vị trí A/B, pose wrist theo phase, lực 5 ngón, rise,
  tilt, placement error, thời gian, `failure_reason`.
- Task variation: đổi `--object`, `--basket` trong vùng reach (map reach bằng
  `solve_pose_ik` trên lưới x/y trước khi chọn), đổi vật YCB (`YCB_PICK_OBJECT_NAME`),
  khối lượng, ma sát.
- Randomization phải **giữ điều kiện thành công**: mọi vị trí sinh ra phải qua kiểm tra
  reach + không va chạm rổ/bàn ở pregrasp trước khi chạy vật lý.
- Không chọn cherry-pick: báo cả trial fail và lý do.

## Perception pipeline (RGB-D → pose)

- Đầu vào: D435 aligned depth + RGB; ra `PoseStamped` trong camera frame → TF sang
  `base_link` (`calibration.json`, không dùng identity trên robot thật).
- Trong sim: `simulation/vision_detector.py` (camera `d435_head`): mask màu → depth →
  deprojection → world → fit đường tròn bán kính đã biết. **Không bao giờ** đọc pose vật
  từ sim trong detector; không thấy vật thì raise, không fallback. Khi đổi detector, chạy
  `tests/test_manipulation_protocol.py` (≤ 1 cm trên 20 vị trí) và `--perception`.
- Gate an toàn: không gửi motion nếu pose quá cũ, ngoài workspace, hoặc calibration chưa xác nhận.

## Học từ demo (VLA / WAM / RL)

Khi được yêu cầu xây policy học:
- Bắt đầu từ dữ liệu scripted demo ở trên (state + ảnh camera + action = joint targets/hand ctrl).
- Định nghĩa rõ observation, action space, reward/success (dùng chính tiêu chí grasp ở mục 1).
- Đánh giá bằng cùng harness headless `--trials N`; policy học phải so với baseline scripted
  trên cùng bố cục và cùng randomization.
- Ưu tiên nêu rõ giới hạn (sim-only, chưa có dữ liệu thực) thay vì hứa hẹn.

## Checklist trước khi kết luận

- [ ] Vật đứng hoàn toàn trên bàn lúc bắt đầu (đáy mesh = đáy collision = mặt bàn).
- [ ] 5/5 ngón có lực trước khi nâng; proof-lift đạt.
- [ ] Đáy vật cao hơn mép rổ ≥ 5 cm khi mang; đặt xuống, không thả rơi.
- [ ] Cổ tay thẳng tại grasp; ảnh render đã xem.
- [ ] Headless N trial + unittest pass; báo số liệu.
