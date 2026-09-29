# Báo cáo tuần 28/09 – 02/10/2026 — OpenArm + Inspire RH56 + D435 (mô phỏng MuJoCo)

**Repo (private):** https://github.com/chutuanviet1302/openarm-rh56-pick-place — nhánh `viet_dev`
**Góp ý của anh buổi 28/09:** (1) nhận dạng 6D bằng FoundationPose; (2) gắp vật đứng / nằm ở nhiều tư thế; (3) 2–4 vật YCB; (4) *(mở rộng)* gắp vật đang chuyển động. Bổ sung 29/09: thay chuỗi IK / waypoint "truyền thống" bằng phương pháp hiện đại.

---

## 1. Tóm tắt

| Mục tiêu | Kết quả |
|---|---|
| 6D FoundationPose | ✅ Chạy trong pipeline (WSL2, GTX 1650): sai số **0.3–2.1 mm, ~1.4°** so với ground truth. Hiển thị hộp + trục 6D trên ảnh camera và trong replay |
| Vật đứng / nằm | ✅ Lon đứng, lon nằm; grasp library chọn kiểu nắm theo tư thế đo được |
| 2–4 vật YCB | ✅ 4 loại: lon cà chua, táo, cam, đào |
| Vật chuyển động | ✅ Băng tải: tracker ước lượng vận tốc, tay chặn đầu và đi theo băng khi khép ngón |
| **Task tổng hợp** (4 vật trên bàn + 2 vật trên băng, 2 tay, thả vào 1 rổ) | ✅ **6/6 vào rổ, 199 s** (đầu tuần: 5/6, 302 s) |
| Di chuyển bằng tối ưu hoá (mink) | ✅ **6/6, 196 s** |
| Policy học được (LeRobot ACT) | ⚠️ Chạy được end-to-end, **32%** nắm thành công, còn thua bản viết tay (**62%**) |

---

## 2. Pipeline

```
Camera D435 (sim, RGB-D)
  -> detector + tracker (không dùng trạng thái simulator)
  -> FoundationPose 6D  (WSL2, GTX 1650, FP32)
  -> grasp library (YAML: tâm nắm, hướng hàm, pre-shape, thứ tự khép ngón theo vật + tư thế)
  -> di chuyển tay không:  mink (IK vi phân dạng QP: giới hạn khớp + tránh va chạm bàn/rổ/vật)
  -> tiếp cận / nắm / nhấc thử:  (a) chuỗi viết tay   hoặc   (b) policy LeRobot ACT
  -> mang sang rổ (đường Cartesian của planner) -> thả
```

`get_object_pose()` là điểm nối duy nhất giữa perception và điều khiển: đổi `gt` / `foundationpose` không đụng phần còn lại.

---

## 3. Việc đã làm theo ngày

**28/09 — 6D + nhiều vật**
- Registry vật YCB: collision, khối lượng, đối xứng, các tư thế nghỉ.
- Grasp library YAML theo vật + tư thế; planner thử lần lượt các phương án.
- Cầu nối FoundationPose Windows ↔ WSL. Trên GTX 1650 phải chạy FP32, vì FP16 cho ra pose NaN.
- Detector RGB-D + tracker có vận tốc; task băng tải; task thả vào rổ bằng 2 tay: 4/4 với cả gt lẫn FoundationPose.

**29/09 — hoàn thiện task, tăng tốc**
- **Gắp liên tiếp:** thả xong không về tư thế nghỉ mà đi thẳng tới vật kế tiếp.
- Sửa độ tin cậy:
  - vị trí đào và điểm thả lon (tránh hướng nắm làm lon tuột);
  - nâng theo độ cao thật của vật trong tay;
  - thay cam trên băng tải bằng lon (cam tuột ở mọi lần chạy).
- Kết quả: **5/6, 302 s → 6/6, 199 s**.
- FoundationPose làm mặc định; hiển thị pose 6D (hộp đỏ = ước lượng, hộp xanh = thật, sai số mm) giống demo CenterPose / FoundationPose.
- Lỗi FoundationPose ước lượng lon **lộn ngược** (trục 180°) → với vật đối xứng trục, tự lật lại pose.
- Replay: in ra GPU đang render; timer 1 ms cho hình mượt (máy đang render bằng Intel Iris, cần bật High performance cho `python.exe`).

**29–30/09 — phương pháp hiện đại**
- **mink** (`mink_motion.py`): mỗi chu kỳ giải một QP (bám pose cổ tay + tư thế, ràng buộc giới hạn khớp và khoảng cách tay–vật cản). Dùng cho đoạn tay không đi tới vật.
  - Thử cho mink cả đoạn mang vật: táo rơi. Thử cho mink cả đoạn về nghỉ: ngón chạm thành rổ.
  - Hai đoạn đó giữ đường đã kiểm chứng. Task đầy đủ: **6/6, 196 s**.
- **Dữ liệu demo:** pipeline viết tay làm "chuyên gia", tự sinh episode (vật, tư thế, vị trí, góc ngẫu nhiên), ghi 10 Hz, chỉ giữ episode nhấc thử thành công. Được **326 demo / 26 950 frame**, tỉ lệ giữ ~75%.
- **LeRobot ACT** (40M tham số, state-only 38 chiều → 15 chiều): train trên **Kaggle T4**, 10k bước ≈ 23 phút; chạy suy luận trên CPU laptop.

---

## 4. Kết quả đánh giá policy

50 layout ngẫu nhiên, seed cố định khác seed lúc sinh demo. Cả hai phương pháp **không được thử lại**. Thành công = vật được nhấc ≥ 3 cm cùng tay.

| Vật | Viết tay | LeRobot ACT |
|---|---|---|
| Lon đứng | 12/12 | 0/12 |
| Lon nằm | 3/14 | 3/14 |
| Táo | 6/9 | 3/9 |
| Cam | 7/9 | 4/9 |
| Đào | 3/6 | **6/6** |
| **Tổng** | **31/50 (62%)** | **16/50 (32%)** |
| Thời gian nắm TB | 8.3 s | 14.1 s |

Nhận xét:
- ACT nắm được, nhưng **nhấc chậm**: có 7 lần chỉ lên được 2.5–3 cm, ngay dưới ngưỡng.
- **Lon đứng hỏng toàn bộ** là lỗi hệ thống, chưa tìm ra nguyên nhân.
- ACT tốt hơn bản viết tay ở **đào** (6/6).
- **Lon nằm** khó với cả hai (21%).

---

## 5. Vấn đề gặp phải và cách giải quyết (bài học)

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| Policy ACT bản đầu 0/12, tay bò rất chậm | Hành động ghi "lệnh so với vị trí đo được", thực chất chỉ là độ trễ servo (~3.5 mm), không phải chuyển động (~7.8 mm / 0.1 s) | Đổi thành "lệnh kế tiếp so với lệnh hiện tại" |
| Phát lại đúng hành động demo vẫn 0/4 | Sai số IK cộng dồn, lệnh trôi lên 4 cm | Tích luỹ lệnh theo hành động, mink bám trong 1 mm → **4/4** (phép kiểm tra "oracle") |
| Train LeRobot trên Kaggle lỗi 2 lần | Kaggle tự giải nén zip; `transformers` có sẵn xung đột `huggingface_hub` | Sửa notebook, ghim phiên bản |
| Train chậm (83 phút) | 40k bước ≈ 560 epoch cho dữ liệu nhỏ | 10k bước, 4 luồng nạp dữ liệu → 23 phút |
| Sinh demo chết giữa chừng | Rò bộ nhớ (3.5 GB / tiến trình) | Chạy theo lô 25 episode rồi thoát |

---

## 6. Hạn chế

- Toàn bộ trong mô phỏng; chưa chạy ROS 2 / MoveIt 2 / robot thật.
- Policy mới học đoạn nắm của tay phải, vật đứng yên, và **chưa vượt bản viết tay**. Demo chính vẫn dùng nắm viết tay.
- Policy dùng pose vật ước lượng một lần lúc bắt đầu, không nhìn lại trong lúc nắm. FoundationPose mất ~30 s/lần trên GTX 1650.
- Tay trái chưa dùng policy; vật trên băng tải dùng chuỗi viết tay.

## 7. Bước tiếp theo đề xuất

1. Tìm nguyên nhân lon đứng 0/12; cho policy nhấc nhanh hơn (chunk dài hơn / lọc demo).
2. DAgger: thêm demo ở những vị trí policy hỏng, train lại.
3. Diffusion Policy trên cùng dữ liệu để so với ACT.
4. Đưa pipeline vào ROS 2: perception node → điều khiển (mink / MoveIt 2) → robot thật.

---

**Chạy lại:**
```
python -m simulation.pick_place.bin_conveyor_task                       # FoundationPose (mặc định)
python -m simulation.pick_place.bin_conveyor_task --pose-backend gt --motion mink
python -m simulation.pick_place.bin_conveyor_task --replay artifacts/bin_conveyor_frames.npz
python -m scripts.eval_grasp_policy --trials 50 --methods scripted
ACT_ENSEMBLE=none .venv-lerobot/Scripts/python -m scripts.eval_grasp_policy --methods policy --policy artifacts/lerobot_runs/act_v6
```
Ảnh pose 6D: `artifacts/pose_overlays/`. Số liệu: `artifacts/eval_scripted.json`, `artifacts/eval_act_v6.json`, `artifacts/bin_conveyor_mink.json`.
