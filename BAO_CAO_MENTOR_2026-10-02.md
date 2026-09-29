# Báo cáo tiến độ — OpenArm + Inspire RH56 + D435 (mô phỏng MuJoCo)

**Ngày:** 02/10/2026
**Repo (private):** https://github.com/chutuanviet1302/openarm-rh56-pick-place — nhánh `viet_dev`
**Theo góp ý 28/09:** (1) nhận dạng 6D bằng FoundationPose, (2) vật đứng / nằm, (3) 2–4 vật YCB, (4) vật đang chuyển động; và góp ý 29/09: thay chuỗi IK/waypoint "truyền thống" bằng phương pháp hiện đại.

## 1. Pipeline hiện tại

```
D435 (sim, RGB-D)  ->  detector + tracker  ->  FoundationPose 6D (WSL2, GTX 1650)
      -> grasp library (YAML, theo loại vật và tư thế đứng/nằm)
      -> di chuyển tay: mink (IK vi phân dạng QP, giới hạn khớp + tránh va chạm)
      -> tiếp cận / nắm / nhấc: policy học được (LeRobot ACT)   [so sánh với bản viết tay]
      -> mang sang rổ (đường Cartesian của planner) -> thả
```

| Thành phần | Trước (18/09) | Bây giờ |
|---|---|---|
| Perception | vị trí lon (fit đường tròn) | **6D FoundationPose**, sai số 0.4–2 mm; hiển thị hộp + trục 6D trên ảnh camera và trong replay |
| Vật | 1 lon | lon (đứng + nằm), táo, cam, đào — 4 loại YCB, 5 tư thế |
| Di chuyển tay | IK từng waypoint + nội suy khớp | **mink**: mỗi chu kỳ giải một bài QP (bám pose cổ tay + tư thế, ràng buộc giới hạn khớp, khoảng cách tay–bàn/rổ/vật) |
| Nắm | chuỗi viết tay, vòng hở | **ACT (LeRobot)** vòng kín theo lực 5 ngón, học từ demo |
| Vật chuyển động | — | băng tải: tracker ước lượng vận tốc, chặn đầu, tay đi theo băng khi đóng ngón |

## 2. Kết quả

**Task bàn + băng tải** (4 vật trên bàn + 2 vật trên băng tải, 2 tay):

| Phiên bản | Vào rổ | Thời gian (sim) |
|---|---|---|
| 28/09 | 5/6 | 302 s |
| 29/09, gắp liên tiếp không về tư thế nghỉ, pose gt | **6/6** | **199 s** |
| với FoundationPose | *(đang chạy)* | |
| với mink | *(đang chạy)* | |

**Policy học được vs waypoint viết tay** — 50 vị trí/góc/vật ngẫu nhiên, seed cố định khác seed lúc sinh demo; thành công = vật được nhấc ≥ 3 cm cùng tay:

| Phương pháp | Thành công | Thời gian nắm (s) |
|---|---|---|
| Waypoint viết tay | *(đang chạy)* | |
| LeRobot ACT | *(đang train trên Kaggle)* | |

Dữ liệu: *(N)* demo tự sinh bằng pipeline viết tay (~75% episode đạt), 10 Hz, state-only (38 chiều quan sát → 15 chiều hành động: pose cổ tay đích + 6 lệnh ngón). ACT 40M tham số, 40k bước, GPU T4 (Kaggle).

## 3. Vì sao chọn cách này
- **Không vứt IK:** mọi phương pháp hiện đại vẫn dùng IK bên trong. Cái đổi là cách ra lệnh: từ chuỗi waypoint cố định + hằng số chỉnh tay theo từng vật sang (a) tối ưu hoá có ràng buộc mỗi chu kỳ (mink) và (b) policy vòng kín học từ dữ liệu.
- **Pipeline viết tay thành "chuyên gia" sinh dữ liệu** — cách nhiều nhóm làm để có demo nhanh trong sim.
- **Vừa GTX 1650:** policy state-only (không ảnh) chạy real-time trên CPU laptop; train trên GPU Kaggle miễn phí.

## 4. Hạn chế (nói thẳng)
- Policy mới học đoạn tiếp cận–nắm–nhấc bằng tay phải, vật đứng yên; vật trên băng tải vẫn dùng bản viết tay.
- Đầu vào policy dùng pose vật ước lượng một lần lúc bắt đầu (FoundationPose ~30 s/lần trên GTX 1650), không nhìn lại trong khi nắm.
- mink: tay trái còn lỗi khi mang (đã sửa: mink chỉ lái tay không, đoạn mang vật giữ đường của planner) — *(kết quả chạy lại)*.
- Toàn bộ trong mô phỏng; chưa chạy ROS 2 / robot thật.

## 5. Bước tiếp theo
1. DAgger: thêm demo ở những vị trí policy hỏng, train lại.
2. Policy nhìn lại vật trong lúc nắm (FoundationPose tracking mode thay vì register mỗi lần).
3. Đưa policy + mink vào ROS 2 (node điều khiển) → MoveIt 2 / cuRobo cho đoạn mang → robot thật.

Ảnh / video kèm: `artifacts/pose_overlays/*.png` (hộp + trục 6D), replay `artifacts/bin_conveyor_frames.npz`.
