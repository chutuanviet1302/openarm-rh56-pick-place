# Kiến trúc hệ thống — OpenArm + Inspire RH56 pick & place (MuJoCo)

Tài liệu này giải thích hệ thống mô phỏng gồm những khối nào, mỗi khối làm gì, và một lần
chạy đi qua các khối theo thứ tự nào. Ví dụ xuyên suốt là **bài chính**: tay phải nắm lon bỏ
vào rổ ở giữa bàn, tay trái lấy lon ra đặt ra ngoài (rổ trên bệ 10 cm).

```bash
python -m scripts.view_retrieve --place-arm right --retrieve-arm left \
    --pick 0.28 -0.25 --basket 0.28 0.0 --retrieve-to 0.28 0.25 --platform 0.10
```

---

## 1. Tổng quan: 3 tầng

Hình dung: **tầng 1 là sân khấu, tầng 2 là bộ não và cơ bắp, tầng 3 là đạo diễn.**

```mermaid
flowchart TB
    subgraph T3["Tầng 3 – Điều phối nhiệm vụ: làm việc gì, tay nào làm"]
        CLI["cli.py / run_gui.bat"]
        VIEW["scripts/view_retrieve.py<br/>(xem MuJoCo, phím R xem lại)"]
        REC["scripts/record_episode.py<br/>+ viewer/ (web)"]
        ROUTER["routing.py<br/>TaskRouter"]
        DEMO["demo.py<br/>Demo: 1 tay, A → rổ"]
        RETR["retrieve.py<br/>RetrieveDemo: bỏ vào rổ rồi lấy ra"]
        HAND["handoff.py<br/>trung chuyển qua điểm giữa"]
        EP["episode.py<br/>EpisodeLog, TrialResult"]
    end
    subgraph T2["Tầng 2 – Não & cơ bắp"]
        PLAN["planner.py<br/>GraspPlanner: lập kế hoạch"]
        EXEC["executor.py<br/>Executor: chạy + an toàn"]
        KIN["kinematics.py<br/>IK / FK"]
        VIS["vision_detector.py<br/>camera RGB-D (tùy chọn)"]
    end
    subgraph T1["Tầng 1 – Thế giới mô phỏng"]
        MODEL["five_finger_model.py<br/>dựng robot, bàn, bệ, lon, rổ"]
        SCENE["scene.py<br/>Scene: đọc trạng thái, va chạm"]
        CFG["config.py<br/>mọi thông số"]
        MJ[("MuJoCo<br/>vật lý")]
    end

    CLI --> ROUTER --> DEMO
    CLI --> HAND --> DEMO
    VIEW --> RETR --> DEMO
    REC --> RETR
    DEMO --> PLAN
    DEMO --> EXEC
    DEMO --> VIS
    DEMO --> EP
    PLAN --> KIN
    PLAN --> SCENE
    EXEC --> SCENE
    EXEC --> KIN
    SCENE --> MODEL --> MJ
    PLAN -.-> CFG
    EXEC -.-> CFG
    DEMO -.-> CFG
```

---

## 2. Từng khối làm gì

### Tầng 1 — Thế giới mô phỏng

| File | Vai trò |
|---|---|
| `simulation/five_finger_model.py` | Dựng mô hình MuJoCo: OpenArm (đỉnh 0.78 m, vai 0.697 m, camera 0.88 m — đúng robot thật) + 2 bàn tay Inspire RH56 + bàn + lon (trụ 10 cm) + rổ (18×18 cm, thành 5 cm). Tùy chọn: **bệ làm việc** (`work_platform_height`), bệ kê rổ, đáy rổ chữ V, xoay/tách đế tay (thí nghiệm). |
| `simulation/pick_place/scene.py` | `Scene`: vị trí/độ nghiêng lon, lon có trong rổ không, lực từng ngón, tư thế nghỉ, và các phép kiểm tra va chạm (bàn, bệ, rổ, thân robot, hai tay chạm nhau). |
| `simulation/pick_place/config.py` | Mọi con số (tốc độ, dung sai, độ cao nâng, danh sách hướng nắm…) kèm lý do và ngày đo. |

### Tầng 2 — Não & cơ bắp

| File | Vai trò |
|---|---|
| `kinematics.py` | **FK**: góc 7 khớp → vị trí/hướng cổ tay. **IK** (DLS): vị trí/hướng cổ tay mong muốn → góc 7 khớp. |
| `planner.py` | `GraspPlanner`: *tính trên giấy, không chạy vật lý*. Từ vị trí lon và rổ → chuỗi tư thế khớp cho mọi pha, đã kiểm tra va chạm và giới hạn khớp. |
| `executor.py` | `Executor`: đưa tay qua các tư thế trong vật lý MuJoCo, khép ngón theo lực, nâng thử, hạ tới khi chạm đáy; **kiểm tra an toàn ở mọi bước 1 ms**; `think()` chạy planner ở luồng phụ để cửa sổ không treo. |
| `vision_detector.py` | (tùy chọn) Tìm lon bằng ảnh RGB-D từ camera đầu. |

### Tầng 3 — Điều phối

| File | Vai trò |
|---|---|
| `demo.py` | `Demo`: **một tay, một việc** — nắm vật ở A, đặt vào rổ B, qua 7 pha. |
| `retrieve.py` | `RetrieveDemo`: **hai màn nối tiếp trong cùng một thế giới vật lý** — màn 1 (tay phải) bỏ vào rổ, màn 2 (tay trái) lấy ra đặt ra ngoài. |
| `routing.py` | `TaskRouter`: chọn tay phải / tay trái / trung chuyển / từ chối. |
| `handoff.py` | Trung chuyển qua điểm giữa bàn (hiện bị từ chối vì chưa ổn định). |
| `episode.py` | Nhật ký từng pha và kết quả `TrialResult`. |
| `cli.py`, `scripts/*` | Điểm vào: chạy dòng lệnh, xem MuJoCo, ghi video, quét tầm với. |

---

## 3. Lớp và quan hệ giữa chúng

```mermaid
classDiagram
    class RetrieveDemo {
        place_in : Demo  (tay phải)
        retrieve : Demo  (tay trái)
        run()
    }
    class Demo {
        side
        scene : Scene
        planner : GraspPlanner
        executor : Executor
        log : EpisodeLog
        plan : Plan
        phase_perceive()
        phase_plan()
        phase_ready()
        phase_reach()
        phase_grasp()
        phase_carry()
        phase_release()
    }
    class GraspPlanner {
        orientation
        plan(object, place_floor)
        plan_place(plan, held_offset)
        find_raise()
        hand_contacts()
        arm_body_clearance()
    }
    class Plan {
        joints : pha → 7 góc khớp
        centers : pha → vị trí cổ tay
        paths : transfer, lower
        grasp_yaw_deg
        grasp_tilt
        place_yaw_deg
    }
    class Executor {
        move_to()
        follow()
        close_until_contact()
        proof_lift()
        descend_until()
        think()
        _check_collisions()
    }
    class Scene {
        model, data
        object_position()
        finger_contact_forces()
        support_contacts()
        basket_contacts()
        robot_body_contacts()
        inter_arm_contacts()
    }
    RetrieveDemo "1" --> "2" Demo
    Demo --> GraspPlanner
    Demo --> Executor
    Demo --> Scene
    GraspPlanner --> Plan : tạo ra
    GraspPlanner --> Scene
    Executor --> Scene
```

---

## 4. Luồng chạy bài chính

### 4.1 Toàn cảnh

```mermaid
flowchart TD
    A["① Dựng thế giới<br/>RetrieveDemo → Scene → five_finger_model<br/>robot 0.78 m, bệ 10 cm, lon (0.28,-0.25), rổ (0.28,0)"] --> B["Mở cửa sổ MuJoCo<br/>gắn RecordingViewer (ghi khung hình)"]
    B --> C["② Màn 1 – TAY PHẢI<br/>Demo(side=right): 7 pha<br/>nắm lon → bỏ vào rổ"]
    C --> D["③ Màn 2 – TAY TRÁI<br/>Demo(side=left) trên CÙNG thế giới<br/>lấy lon trong rổ → đặt ra (0.28, 0.25)"]
    D --> E{"④ Chấm điểm<br/>lon ngoài rổ? đứng trên bệ?<br/>nghiêng < 15°?"}
    E -->|đạt| F["success + sai số, độ nghiêng, độ lún"]
    E -->|không đạt| G["failure + lý do"]
    F --> H["⑤ Bấm R: phát lại các khung hình đã ghi"]
    G --> H
```

### 4.2 Bảy pha của một tay (`Demo`)

```mermaid
stateDiagram-v2
    [*] --> PERCEIVE
    PERCEIVE --> PLAN : vị trí lon (mô phỏng hoặc camera)
    PLAN --> READY : có Plan hợp lệ
    PLAN --> [*] : không có kế hoạch nào → báo lỗi
    READY --> REACH : raise → hover
    REACH --> GRASP : mở tay, tiến vào dọc hướng ngón
    GRASP --> CARRY : đủ ngón chạm + nâng thử đạt
    GRASP --> RETRY : ngón không chạm / lon tuột / nghiêng
    RETRY --> PLAN : thả lon, loại hướng nắm này (tối đa 3 lần)
    CARRY --> RELEASE : lon đã chạm đáy rổ / mặt bệ
    RELEASE --> [*] : về tư thế nghỉ

    note right of PLAN
        thử kiểu nắm (xiên / từ trên xuống)
        × 9 hướng quay tay × nhiều seed IK
    end note
    note right of CARRY
        nâng thẳng qua miệng rổ
        đo lon lệch trong tay → tính lại điểm đặt
        (màn 2: dựng thẳng lon nếu nghiêng)
    end note
```

### 4.3 Ai gọi ai trong một màn (tuần tự)

```mermaid
sequenceDiagram
    autonumber
    participant D as Demo (đạo diễn)
    participant P as GraspPlanner (não)
    participant K as kinematics (IK)
    participant E as Executor (cơ bắp)
    participant S as Scene + MuJoCo

    D->>S: object_position()
    D->>E: think(planner.plan)
    E->>P: plan(vị trí lon, sàn đặt)  [luồng phụ]
    loop mỗi kiểu nắm × hướng tay × seed
        P->>K: solve_pose_ik(đích cổ tay)
        K-->>P: 7 góc khớp
        P->>S: kiểm tra va chạm, giới hạn khớp, cách thân ≥ 15 mm
    end
    P-->>D: Plan (joints, centers, paths)
    D->>E: move_to(raise, hover, pregrasp, grasp)
    E->>S: mj_step × N  +  _check_collisions() mỗi bước
    D->>E: close_until_contact() → proof_lift()
    E-->>D: lực ngón, độ nâng, độ nghiêng
    D->>E: lift_straight_up() → think(plan_place với lệch thực tế)
    D->>E: follow(transfer) → follow(lower) → descend_until(chạm đáy)
    D->>E: nới lực → mở ngón + rút tay → về tư thế nghỉ
    Note over E,S: Va chạm bất kỳ → RuntimeError "trajectory aborted" → dừng ngay
```

---

## 5. Bên trong planner: tìm một kế hoạch

```mermaid
flowchart TD
    START["plan(vị trí lon)"] --> T{"Có bệ làm việc?"}
    T -->|có| ORDER1["Thứ tự: nắm xiên trước<br/>(y-30°, x±30°, y-15°), rồi từ trên xuống"]
    T -->|không| ORDER2["Thứ tự: từ trên xuống trước, rồi nắm xiên"]
    ORDER1 --> LOOP
    ORDER2 --> LOOP
    LOOP["Với mỗi kiểu nắm × 9 hướng quay tay"] --> IK1["IK tư thế nắm<br/>2 seed chuẩn → nếu không nối được: 40 seed dự phòng"]
    IK1 --> CHAIN["Nối chuỗi: pregrasp · hover · lift<br/>(nếu kẹt khớp khi nâng: twist-lift)"]
    CHAIN --> RAISE["find_raise: điểm nhấc tay từ tư thế nghỉ"]
    RAISE --> CHECK{"Kiểm tra<br/>• margin khớp ≥ 3°<br/>• tay không chạm bàn/bệ/rổ<br/>• cánh tay cách thân ≥ 15 mm<br/>• đầu ngón cách mặt ≥ 5 mm"}
    CHECK -->|không đạt| LOOP
    CHECK -->|đạt| PLACE["plan_place: đường mang + hạ<br/>12 hướng đặt × 3 lộ trình<br/>(seed chuẩn → seed dự phòng, đi ngược từ điểm đặt)"]
    PLACE -->|không có| LOOP
    PLACE -->|có| OK["Plan hoàn chỉnh"]
    LOOP -->|hết phương án| FAIL["RuntimeError: no reachable grasp"]
```

---

## 6. An toàn: ba lớp

```mermaid
flowchart LR
    subgraph L1["Lớp 1 – Planner (trước khi chạy)"]
        A1["margin khớp ≥ 3°"]
        A2["bàn tay không chạm bàn / bệ / rổ<br/>(chừa 2 cm trên đường nối)"]
        A3["cánh tay cách thân robot ≥ 15 mm"]
        A4["đầu ngón cách mặt làm việc ≥ 5 mm"]
    end
    subgraph L2["Lớp 2 – Executor (mỗi bước 1 ms)"]
        B1["tay / cánh tay chạm bàn hoặc bệ > 3 mm → DỪNG"]
        B2["chạm rổ > 3 mm → DỪNG"]
        B3["chạm thân robot → DỪNG"]
        B4["hai tay chạm nhau → DỪNG"]
        B5["tay đang nghỉ chạm lon → DỪNG"]
        B6["tốc độ khớp tay ≤ 0.6 rad/s"]
    end
    subgraph L3["Lớp 3 – Kiểm chứng kết quả"]
        C1["nâng thử: lon theo tay, trượt ≤ 1 cm, nghiêng ≤ 15°"]
        C2["mang: đáy lon cao hơn miệng rổ ≥ 5 cm"]
        C3["cuối: lon đúng chỗ, đứng thẳng"]
    end
    L1 --> L2 --> L3
```

---

## 7. Xem MuJoCo mượt: luồng phụ và phát lại

```mermaid
sequenceDiagram
    participant M as Luồng chính (cửa sổ)
    participant W as Luồng phụ (planner)
    participant V as Cửa sổ MuJoCo

    M->>W: Executor.think(planner.plan)
    loop mỗi 30 ms tới khi planner xong
        M->>V: viewer.sync()  (xoay/zoom vẫn được)
    end
    W-->>M: Plan
    M->>V: chạy vật lý + sync mỗi khung hình
    Note over M: RecordingViewer lưu qpos từng khung hình
    M->>V: xong → "success – press R to replay"
    V->>M: phím R
    loop mỗi khung hình đã ghi
        M->>V: đặt qpos, mj_kinematics, sync, chờ đúng nhịp thời gian
    end
```

---

## 8. Vì sao bài chính cần bệ 10 cm và nắm xiên

```mermaid
flowchart TD
    P1["Vai cao 0.70 m trên bàn<br/>tay + bàn tay chỉ ~0.63 m"] --> Q1["Tay vừa đủ chạm bàn ngay dưới vai<br/>→ khó với ra xa, khó nắm ngang"]
    P2["Khớp vai 2 chỉ khép vào +10°"] --> Q2["Mỗi tay khó với qua đường giữa"]
    Q1 --> R["Rổ ở giữa bàn trên mặt bàn:<br/>vùng hai tay cùng làm được chỉ ~1 cm → không ổn định"]
    Q2 --> R
    R --> S1["Bệ 10 cm: lon lên gần tầm tay"]
    S1 --> S2["Nắm từ trên xuống trên bệ: kẹt khớp gần vai"]
    S2 --> S3["Nắm xiên: ngón chúc ~38°,<br/>ngón cái & các ngón kẹp ngang thân lon"]
    S3 --> OK["Hai tay đều làm được ở tâm rổ,<br/>margin khớp ~22–27°"]
```

---

## 9. Kết quả hiện tại (24/09/2026)

| Hạng mục | Kết quả |
|---|---|
| Bài chính (danh nghĩa) | thành công; sai số đặt 3–7 mm, nghiêng 0°, không lún |
| Lệch điểm nắm ±1.5 cm | 7/8 thành công (hỏng: (0.28,-0.235), tay phải cầm chưa chắc) |
| Toàn bộ test | 61/61 OK |
| Thời gian một lần chạy | ~75 s (trước tối ưu: 169 s); vật lý nhanh hơn thời gian thực 1.27× |

Chi tiết từng ngày, số liệu đo và các hướng đã thử: xem `context_project.md`.
