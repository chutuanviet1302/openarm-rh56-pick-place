"""Build docs/BAO_CAO_TIEN_DO_2026-10-02.pdf (mentor progress report, Vietnamese).

    python scripts/make_report_pdf.py
"""

from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
addMapping("Arial", 0, 0, "Arial")
addMapping("Arial", 1, 0, "Arial-Bold")
addMapping("Arial", 0, 1, "Arial-Italic")

BLUE = colors.HexColor("#1f4e79")
H1 = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=17, leading=22, textColor=BLUE, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("p", fontName="Arial", fontSize=10, leading=14, spaceAfter=3)
S = ParagraphStyle("s", parent=P, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
C = ParagraphStyle("c", parent=P, fontSize=9, leading=12, spaceAfter=0)
CB = ParagraphStyle("cb", parent=C, fontName="Arial-Bold")


def table(rows, widths):
    data = [[Paragraph(str(c), CB if i == 0 else C) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def bullets(items):
    return [Paragraph("• " + i, P) for i in items]


s = [Paragraph("Báo cáo tiến độ tuần 28/09 – 02/10/2026", H1),
     Paragraph("Project: OpenArm 7-DOF + Inspire RH56 + RealSense D435 — pick &amp; place trong mô phỏng MuJoCo<br/>"
               "Người thực hiện: Chu Tuấn Việt · Repo: github.com/chutuanviet1302/openarm-rh56-pick-place (nhánh viet_dev)", S),
     Spacer(1, 6),
     Paragraph("<b>Góp ý của anh (28/09):</b> (1) nhận dạng 6D bằng FoundationPose; (2) gắp vật đứng / nằm; "
               "(3) 2–4 vật YCB; (4) mở rộng: gắp vật chuyển động. <b>Bổ sung 29/09:</b> thay chuỗi IK/waypoint "
               "“truyền thống” bằng phương pháp hiện đại.", P)]

s += [Paragraph("1. Tóm tắt kết quả", H2), table([
    ["Mục tiêu", "Kết quả"],
    ["6D FoundationPose", "Đạt — chạy trong pipeline (WSL2, GTX 1650); sai số 0.3–2.1 mm, ~1.4°; hiển thị hộp + trục 6D"],
    ["Vật đứng / nằm", "Đạt — lon đứng, lon nằm; grasp library chọn kiểu nắm theo tư thế đo được"],
    ["2–4 vật YCB", "Đạt — lon cà chua, táo, cam, đào"],
    ["Vật chuyển động", "Đạt — băng tải: tracker ước lượng vận tốc, tay chặn đầu và đi theo băng khi khép ngón"],
    ["Task tổng hợp (4 vật bàn + 2 vật băng, 2 tay, 1 rổ)", "<b>6/6 vào rổ, 199 s</b> (đầu tuần: 5/6, 302 s)"],
    ["Di chuyển bằng tối ưu hoá (mink QP)", "<b>6/6, 196 s</b>"],
    ["Policy học được (LeRobot ACT)", "Chạy end-to-end; nắm thành công 32% — còn thua bản viết tay (62%)"],
], [5.2 * cm, 11.8 * cm])]

s += [Spacer(1, 8),
      Table([[Image("docs/bin_conveyor_frame.png", 8.2 * cm, 5.1 * cm), Image("docs/pose6d_peach.png", 6.8 * cm, 5.1 * cm)]]),
      Paragraph("Trái: task bàn + băng chuyền (GIF x10: docs/bin_conveyor_x10.gif). "
                "Phải: pose 6D FoundationPose trên ảnh camera đầu (đỏ = ước lượng, xanh = thật).", S)]

s += [Paragraph("2. Pipeline", H2)] + bullets([
    "Camera D435 (sim, RGB-D) → detector + tracker (không dùng trạng thái simulator)",
    "→ FoundationPose 6D (WSL2, GTX 1650, FP32) → T_world_object",
    "→ grasp library YAML (tâm nắm, hướng hàm, pre-shape, thứ tự khép ngón theo vật + tư thế)",
    "→ di chuyển tay không: mink (IK vi phân dạng QP, ràng buộc giới hạn khớp + tránh va chạm bàn/rổ/vật)",
    "→ tiếp cận / nắm / nhấc thử: chuỗi viết tay <i>hoặc</i> policy LeRobot ACT",
    "→ mang sang rổ (đường Cartesian của planner) → thả; gắp liên tiếp, không về tư thế nghỉ giữa các vật",
])

s += [Paragraph("3. Việc đã làm", H2), table([
    ["Ngày", "Nội dung"],
    ["28/09", "Registry vật YCB (collision, khối lượng, đối xứng, tư thế nghỉ); grasp library YAML; cầu nối "
              "FoundationPose Windows ↔ WSL (GTX 1650 phải FP32 — FP16 cho pose NaN); detector RGB-D + tracker; "
              "task băng tải; task thả rổ 2 tay (4/4 với gt và FoundationPose)."],
    ["29/09", "Gắp liên tiếp; sửa điểm thả, độ cao nâng, bố cục; thay cam trên băng bằng lon (cam tuột mọi lần) → "
              "6/6, 199 s. FoundationPose mặc định; hiển thị pose 6D; tự lật pose lon bị ước lượng lộn ngược."],
    ["29–30/09", "mink QP cho đoạn tay không (6/6, 196 s). Sinh tự động 326 demo / 26 950 frame bằng pipeline "
                 "viết tay; chuyển sang LeRobotDataset; train LeRobot ACT (40M tham số) trên Kaggle T4 (~23 phút); "
                 "đánh giá trên 50 layout."],
], [2.2 * cm, 14.8 * cm])]

s += [Paragraph("4. Đánh giá policy: ACT so với bản viết tay", H2),
      Paragraph("50 layout ngẫu nhiên (vật, tư thế, vị trí, góc), seed khác lúc sinh demo; không thử lại; "
                "thành công = vật được nhấc ≥ 3 cm cùng tay.", P),
      table([["Vật", "Viết tay", "LeRobot ACT"],
             ["Lon đứng", "12/12", "0/12"], ["Lon nằm", "3/14", "3/14"], ["Táo", "6/9", "3/9"],
             ["Cam", "7/9", "4/9"], ["Đào", "3/6", "<b>6/6</b>"],
             ["<b>Tổng</b>", "<b>31/50 (62%)</b>", "<b>16/50 (32%)</b>"], ["Thời gian nắm TB", "8.3 s", "14.1 s"]],
            [5 * cm, 5 * cm, 5 * cm])]
s += bullets(["ACT nắm được nhưng nhấc chậm: 7 lần chỉ lên 2.5–3 cm, ngay dưới ngưỡng.",
              "Lon đứng hỏng toàn bộ với ACT — lỗi hệ thống, chưa tìm ra nguyên nhân.",
              "ACT tốt hơn ở đào (6/6); lon nằm khó với cả hai (21%)."])

s += [Paragraph("5. Vấn đề gặp phải và cách xử lý", H2), table([
    ["Vấn đề", "Nguyên nhân", "Xử lý"],
    ["ACT bản đầu 0/12, tay bò rất chậm", "Hành động = lệnh so với vị trí đo được → chỉ học độ trễ servo "
     "(~3.5 mm), không phải chuyển động (~7.8 mm / 0.1 s)", "Hành động = lệnh kế tiếp so với lệnh hiện tại"],
    ["Phát lại đúng hành động demo vẫn 0/4", "Sai số IK cộng dồn, lệnh trôi 4 cm", "Tích luỹ lệnh; mink bám 1 mm → 4/4"],
    ["mink làm rơi táo / chạm thành rổ", "Giải IK từng đoạn khi mang vật; đường thẳng về nghỉ cắt qua rổ",
     "mink chỉ lái đoạn tay không"],
    ["Train Kaggle lỗi 2 lần", "Kaggle tự giải nén zip; transformers xung đột huggingface_hub",
     "Sửa notebook, ghim phiên bản"],
    ["Train chậm (83 phút)", "40k bước ≈ 560 epoch cho dữ liệu nhỏ", "10k bước, 4 luồng nạp → 23 phút"],
    ["Sinh demo chết giữa chừng", "Rò bộ nhớ (3.5 GB / tiến trình)", "Chạy theo lô 25 episode"],
], [4.6 * cm, 6.4 * cm, 6 * cm])]

s += [Paragraph("6. Hạn chế", H2)] + bullets([
    "Toàn bộ trong mô phỏng; chưa chạy ROS 2 / MoveIt 2 / robot thật.",
    "Policy mới học đoạn nắm của tay phải với vật đứng yên và chưa vượt bản viết tay — demo chính dùng nắm viết tay.",
    "Pose vật ước lượng một lần lúc bắt đầu; FoundationPose ~30 s / lần trên GTX 1650.",
])
s += [Paragraph("7. Bước tiếp theo", H2)] + bullets([
    "Tìm nguyên nhân lon đứng 0/12; cho policy nhấc nhanh hơn.",
    "DAgger: thêm demo ở vị trí policy hỏng; thử Diffusion Policy trên cùng dữ liệu.",
    "Đưa pipeline vào ROS 2: perception node → điều khiển (mink / MoveIt 2) → robot thật.",
])

SimpleDocTemplate("docs/BAO_CAO_TIEN_DO_2026-10-02.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.6 * cm, bottomMargin=1.6 * cm, title="Báo cáo tiến độ 28/09 – 02/10/2026",
                  author="Chu Tuấn Việt").build(s)
print("ok")
