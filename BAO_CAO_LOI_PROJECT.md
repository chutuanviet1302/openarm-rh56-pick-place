# Báo cáo lỗi project OpenArm Pick-and-Place

**Ngày kiểm tra:** 16/09/2026  
**Phạm vi:** MuJoCo, OpenArm hai tay, Inspire RH56, vật YCB mustard, camera trên cao  
**Trạng thái tổng thể:** Chưa hoàn thành task pick-and-place bằng tiếp xúc vật lý.

## 1. Kết quả kiểm tra hiện tại

| Hạng mục | Kết quả | Bằng chứng |
|---|---:|---|
| Trial pick-and-place vật lý | **0/1 đạt** | `artifacts/physics_trials.json` |
| Unit/integration test | **24/25 đạt** | `python -m unittest discover -s tests -v` |
| Khởi tạo MuJoCo và OpenArm | Đạt | Model load và step được |
| Hai Inspire Hand thay gripper gốc | Đạt | Đủ 24 joint và 12 actuator tay |
| Camera cố định nhìn từ trên xuống | Đạt | Camera `overhead` và test hướng camera đạt |
| Vật bên phải, điểm thả bên trái | Đạt | A = `[0.30, -0.15]`, B = `[0.30, 0.15]` |
| Khối lượng mustard | Đạt | Body mass = `0.2 kg`; visual mesh mass = `0` |
| Nắm, nhấc và chuyển vật A→B | **Chưa đạt** | Chu trình dừng trước khi robot chuyển động |

Lỗi trực tiếp của trial gần nhất:

```text
IK failed for right target
[0.15449132143498834, -0.209434182184683, 0.6469954939713435]
```

Pose cuối của vật vẫn bằng pose khởi tạo `[0.30, -0.15, 0.495]` và thời gian mô phỏng của trial là `0.0 s`. Robot chưa tiếp cận, chưa nắm và chưa nhấc vật.

## 2. Các lỗi đang chặn project

### P0 — IK không giải được pose nắm

`Demo._solve_poses()` thất bại ngay tại target grasp của tay phải. Target được suy ra từ tâm chai và offset đầu ngón nhưng nằm ngoài vùng mà solver hiện tại tìm được với orientation 45°.

**Ảnh hưởng:** Toàn bộ state machine không bắt đầu; không thể đánh giá grasp, lift, transfer hoặc release.

**Nguyên nhân có khả năng cao:**

- Tâm nắm yêu cầu cổ tay xuống quá thấp so với pedestal đã nâng `0.40 m`.
- Orientation bị ràng buộc đủ 3 trục, làm giảm vùng IK khả dụng.
- Solver chỉ dùng một seed và có thể kẹt ở nhánh nghiệm không phù hợp.
- Offset bàn tay được đo tại một pose probe nhưng áp dụng trực tiếp cho pose chai thấp hơn.
- Điểm A mới được chọn theo bố cục trái/phải, chưa được chọn bằng phép quét vùng IK.

**Cần sửa:** Quét workspace cho A/B và orientation, thử nhiều seed, chọn pose nắm reachable rồi mới lưu scene. Thêm kiểm tra FK sai số và joint-limit cho từng waypoint.

### P0 — Chưa chứng minh được grasp bằng ma sát

Project chưa có trial nào đạt chuỗi `thumb contact + finger contact → lift 5 cm → hold 2 s`. Trước khi siết điều kiện vật lý, các bản demo cũ từng ghim pose vật hoặc cập nhật vật theo tay; kết quả đó không được tính là grasp thành công.

Hiện `REQUIRE_THUMB_OPPOSITION = True`, nhưng điều kiện này chưa được chạy tới do IK fail.

**Cần sửa:** Tạo test riêng với cổ tay cố định, đưa chai vào đúng vùng giữa ngón cái và bốn ngón, đóng chậm theo lực, sau đó proof-lift không weld và không ghi `qpos` vật.

### P0 — Chưa có kiểm thử end-to-end trung thực

Các test hiện tại kiểm tra cấu trúc model, camera, mass và việc code không ghi trực tiếp trạng thái vật lý trong các method runtime quan trọng. Chưa có test chạy thành công toàn chu trình và xác nhận:

- vật nâng ít nhất 4 cm;
- giữ 2 giây;
- di chuyển A→B ít nhất 15 cm;
- đặt sai số không quá 20 mm;
- đứng ổn định 3 giây sau release.

**Cần sửa:** Thêm acceptance test headless dựa trên `TrialResult`; test phải fail nếu IK, contact, lift, transfer hoặc place fail.

## 3. Lỗi model và điều khiển

### P1 — Hai bàn tay chưa đối xứng đúng

Test `test_hands_have_opposite_chirality` đang fail:

```text
left_thumb[1] - left_middle[1] = -0.1039
expected > 0
```

Điều này cho thấy mount hoặc quy ước chirality của tay trái chưa khớp với giả định test. Cần kiểm tra transform flange→palm bằng marker trong local frame, không sửa test chỉ để làm xanh.

### P1 — Arm position controller bám target kém khi bỏ ép qpos

Trong kiểm tra controller trước đó, tay phải còn sai số joint tới khoảng `0.56 rad` ở một khớp sau lệnh tới ready pose. Khi không còn ghi trực tiếp `qpos`, sai số này làm bàn tay không đến vùng nắm dù IK target đúng về mặt hình học.

**Cần sửa:** Đo desired/actual joint theo thời gian, tăng thời lượng hợp lý, tune gain/damping trong giới hạn ổn định và đặt timeout theo sai số endpoint.

### P1 — Collision mustard còn là một box duy nhất

Mustard có thân, vai và nắp không đồng nhất nhưng collision hiện dùng một box. Vật hiển thị đã xoay 90° còn collision không mô tả đúng toàn bộ silhouette.

**Ảnh hưởng:** Có thể nhìn thấy khoảng trống nhưng MuJoCo báo contact, hoặc mesh nhìn chạm nhưng collision chưa chạm.

**Cần sửa:** Dùng 2–3 primitive cho thân, vai và nắp; đo sai lệch vùng grasp dưới 2 mm.

### P1 — Cơ chế weld/carry cũ vẫn còn trong model và state

Model vẫn tạo equality `grasp_left_box` và `grasp_right_box`; `Demo` vẫn khởi tạo `grasp_equality`, `carry_side` và `hold_pose`. Các nhánh runtime dùng chúng đã được bỏ khỏi luồng chính, nhưng code và equality còn tồn tại.

**Rủi ro:** Dễ vô tình bật lại cơ chế giữ vật giả trong lần sửa sau; làm người kiểm tra hiểu nhầm kết quả.

**Cần sửa:** Xóa equality và state không còn dùng. Thêm test khẳng định không có weld giữa hand và object.

### P1 — Điều kiện đặt vật chưa được kiểm chứng

Rổ hiện nằm cách vật 30 cm theo trục Y và camera overhead hiển thị đúng bố cục. Tuy nhiên lower pose chưa có nghiệm IK và chưa kiểm tra va chạm thành rổ.

**Cần sửa:** Ban đầu thay rổ bằng vùng đặt phẳng. Sau khi place ổn định mới thêm thành rổ và kiểm tra clearance.

## 4. Lỗi chất lượng và khả năng bảo trì

### P2 — Code demo chứa nhiều comment và biến từ các thử nghiệm cũ

Một số comment mô tả kết quả đo cũ không còn tương ứng hoàn toàn với model hiện tại. `GRASP_Y_BIAS` được khai báo nhưng không tham gia tính target. Các state weld/carry không còn dùng vẫn tồn tại.

**Cần sửa:** Xóa dead code, chuyển các thông số thực sự cần tune vào một `GraspConfig`, và để log trial lưu giá trị cấu hình đã dùng.

### P2 — Báo cáo trial còn thiếu dữ liệu chẩn đoán

`TrialResult` hiện chỉ lưu success, failure reason, final position và simulation time.

**Cần bổ sung:** target/actual joint, pose A/B, IK residual, nhóm ngón tiếp xúc, lực từng ngón, độ nâng, độ nghiêng, độ trượt và sai số đặt.

### P2 — Workspace chưa phải Git repository

Lệnh `git status` trả về `not a git repository`. Không có lịch sử thay đổi hoặc điểm quay lại rõ ràng cho các lần tune model.

**Cần sửa:** Khởi tạo repository hoặc đặt project vào repository quản lý phiên bản trước khi tiếp tục thay đổi lớn.

## 5. Các lỗi đã được sửa và cần giữ regression

- Visual mesh mustard không còn cộng thêm khối lượng; tổng body mass là `0.2 kg`.
- Mustard dùng scale `1:1`.
- Camera `overhead` nhìn theo phương `-Z`.
- A và B nằm ở hai phía đối nhau và cách nhau ít nhất 15 cm.
- Runtime methods chính không ghi trực tiếp `qpos/qvel` để kéo robot hoặc vật.
- Grasp bắt buộc ngón cái và ít nhất một ngón đối diện.
- Model có tay Inspire trái và phải riêng, thay gripper hai ngón gốc.

## 6. Thứ tự xử lý đề xuất

1. Sửa đối xứng mount tay và xác nhận từng joint/fingertip trong local frame.
2. Tạo bài test bàn tay cố định để đạt opposed grasp và proof-lift 5 cm.
3. Quét workspace tìm A/B và orientation có nghiệm IK liên tục.
4. Tune arm controller để endpoint error nằm trong tolerance mà không ghi qpos.
5. Chạy `READY → APPROACH → GRASP → PROOF_LIFT` bằng vật lý.
6. Thêm transfer và place trên vùng phẳng.
7. Thêm rổ, 5 trial cố định, rồi 20 trial có nhiễu.

## 7. Lệnh tái hiện

```powershell
.\.venv\Scripts\python.exe -m simulation.pick_place_demo --headless --trials 1
Get-Content artifacts\physics_trials.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m simulation.pick_place_demo
```

## 8. Kết luận

Scene, camera, object mass và bố trí A/B đã có nền tảng đúng hơn, nhưng project chưa đạt nhiệm vụ cầm vật từ A tới B. Blocker đầu tiên là IK grasp; sau khi giải quyết, project vẫn phải vượt qua opposed-contact proof-lift và acceptance test vật lý trước khi được xác nhận hoàn thành.
