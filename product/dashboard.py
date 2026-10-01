"""Per-run dashboard (runs/<id>/dashboard.html) and the run list (runs/index.html).

    python -m product.dashboard runs/demo_001_20261001-150000      # (re)build one run's page
    python -m product.dashboard --index                            # rebuild runs/index.html

Every number on the page is a KPI from product.kpi (its source file and field shown on
hover). The capacity box takes the business's own figures as input; it has no defaults.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from product.kpi import compute

RUNS = Path("runs")
STATUS_TEXT = {"complete": "Đủ đơn", "short": "Thiếu hàng", "failed": "Lỗi soạn"}
CONTACT_TEXT = {
    "table_platform_belt": "Tay chạm bàn / bệ / băng",
    "box": "Tay chạm thùng",
    "robot_body": "Tay chạm thân robot",
    "inter_arm": "Hai tay chạm nhau",
    "non_target_object": "Tay chạm hàng khác",
    "arm_link_object": "Cánh tay (không phải bàn tay) chạm hàng",
}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#5b6475;--line:#e3e6ec;--ok:#1a7f37;--warn:#b35900;--bad:#c62828;--bar:#2f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--ink:#e6e9ef;--muted:#9aa3b2;--line:#2a303c;--ok:#3fb950;--warn:#e3a33b;--bad:#ff6b6b;--bar:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:20px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 10px}
.sub{color:var(--muted);margin:0 0 18px}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}
.k{color:var(--muted);font-size:13px}.v{font-size:24px;font-weight:650}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:600;font-size:13px}
td.n{text-align:right;font-variant-numeric:tabular-nums}img{max-width:100%;border-radius:8px;border:1px solid var(--line)}
.two{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}label{display:block;font-size:13px;color:var(--muted);margin-top:8px}
input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
code{background:var(--bg);padding:1px 5px;border-radius:4px;word-break:break-all}a{color:var(--bar)}.card{overflow-x:auto}
"""


def _cell(kpi: dict, key: str, text: str | None = None, cls: str = "") -> str:
    item = kpi[key]
    value = item["value"] if text is None else text
    return f'<div class="v {cls}" title="{html.escape(item["source"])}">{html.escape(str(value))}</div>'


def _cycle_chart(cycles: list[dict]) -> str:
    if not cycles:
        return "<p class='k'>Không có lượt gắp.</p>"
    top = max(c["seconds"] for c in cycles) or 1.0
    row, width = 26, 620
    parts = [f'<svg viewBox="0 0 {width + 190} {row * len(cycles) + 8}" width="100%" role="img" '
             f'aria-label="Thời gian chu kỳ mỗi lượt gắp">']
    for i, c in enumerate(cycles):
        y = 4 + i * row
        w = max(2.0, c["seconds"] / top * width)
        parts.append(f'<text x="0" y="{y + 16}" font-size="12" fill="currentColor">{html.escape(c["name"])}</text>'
                     f'<rect x="120" y="{y + 3}" width="{w:.1f}" height="16" rx="3" fill="var(--bar)"/>'
                     f'<text x="{126 + w:.1f}" y="{y + 16}" font-size="12" fill="currentColor">{c["seconds"]:.1f} s</text>')
    parts.append("</svg>")
    return "".join(parts)


def detection_snapshot(run: Path) -> Path | None:
    """The look that saw the most objects, as PNG (the annotated head-camera image)."""
    import cv2

    recording = np.load(run / "frames.npz")
    if "det_look" not in recording.files or not len(recording["det_look_times"]):
        return None
    counts = np.bincount(recording["det_look"], minlength=len(recording["det_look_times"]))
    best = int(np.argmax(counts))
    out = run / "detections.png"
    cv2.imwrite(str(out), cv2.cvtColor(recording["det_images"][best], cv2.COLOR_RGB2BGR))
    return out


def build(run: Path, gif: bool = True) -> Path:
    run = Path(run)
    kpi = compute(run)
    (run / "kpi.json").write_text(json.dumps(kpi, indent=2, ensure_ascii=False), encoding="utf-8")
    result = json.loads((run / "order_result.json").read_text(encoding="utf-8"))
    if gif and not (run / "replay.gif").exists():
        from scripts.make_replay_gif import render_gif

        render_gif(run / "frames.npz", run / "replay.gif", speed=10.0, fps=12, size=(560, 350))
    snapshot = detection_snapshot(run)
    status = kpi["status"]["value"]
    status_cls = {"complete": "ok", "short": "warn"}.get(status, "bad")
    lines = "".join(
        f"<tr><td>{html.escape(l['sku'])}</td><td class='n'>{l['requested']}</td><td class='n'>{l['picked']}</td>"
        f"<td class='n {'bad' if l['missing'] else ''}'>{l['missing']}</td></tr>" for l in result["lines"])
    contacts = "".join(
        f"<tr><td>{CONTACT_TEXT.get(k, k)}</td><td class='n {'ok' if v['frames'] == 0 else 'bad'}'>{v['frames']}</td>"
        f"<td class='n'>{v['deepest_mm']}</td></tr>" for k, v in kpi["contacts"]["value"].items())
    jerk = ", ".join(f"{s}: {v}" for s, v in kpi["rms_jerk_rad_s3"]["value"].items())
    stops = ", ".join(f"{s}: {v}" for s, v in kpi["stops"]["value"].items())
    margin = kpi["min_joint_margin_deg"]["value"]
    order_sim_s = kpi["sim_seconds"]["value"]
    media = []
    if (run / "replay.gif").exists():
        media.append('<div class="card"><h2>Replay (x10)</h2><img src="replay.gif" alt="Replay của lần chạy, tăng tốc 10 lần"></div>')
    if snapshot is not None:
        media.append('<div class="card"><h2>Camera: mọi vật detect được</h2>'
                     '<img src="detections.png" alt="Ảnh camera đầu với khung nhận dạng các vật trên bàn và băng">'
                     '<p class="k">Xanh: trên bàn. Cam: trên băng (mũi tên = vận tốc). Xám: không nhận dạng được.</p></div>')
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PickCell {html.escape(kpi['order_id'])}</title>
<style>{CSS}</style></head><body><main>
<p class="k"><a href="../index.html">← Tất cả đơn</a></p>
<h1>Đơn {html.escape(kpi['order_id'])}</h1>
<p class="sub">Mô phỏng MuJoCo · pose: {html.escape(str(kpi['pose_backend']))} · {html.escape(kpi['layout'])} ·
di chuột lên số để xem nguồn</p>
<div class="grid">
  <div class="card"><div class="k">Trạng thái</div>{_cell(kpi, 'status', STATUS_TEXT.get(status, status), status_cls)}</div>
  <div class="card"><div class="k">Dòng đơn đủ</div>{_cell(kpi, 'lines_filled')}</div>
  <div class="card"><div class="k">Soạn sai (mispick)</div>{_cell(kpi, 'mispicks', cls='ok' if kpi['mispicks']['value'] == 0 else 'bad')}</div>
  <div class="card"><div class="k">Thời gian đơn (sim)</div>{_cell(kpi, 'sim_seconds', f"{order_sim_s:.0f} s")}</div>
  <div class="card"><div class="k">Món/giờ (sim)</div>{_cell(kpi, 'items_per_hour_sim')}</div>
  <div class="card"><div class="k">Không va chạm</div>{_cell(kpi, 'collision_free', 'Đạt' if kpi['collision_free']['value'] else 'Không đạt', 'ok' if kpi['collision_free']['value'] else 'bad')}</div>
</div>
<div class="two">
  <div class="card"><h2>Dòng đơn</h2><table><tr><th>SKU</th><th>Yêu cầu</th><th>Trong thùng</th><th>Thiếu</th></tr>{lines}</table>
  <p class="k">Robot đã thả: {html.escape(', '.join(result['robot_dropped']) or '—')} · Lượt hỏng: {html.escape(', '.join(result['failed_attempts']) or '—')}</p></div>
  <div class="card"><h2>Thời gian mỗi lượt gắp (sim)</h2>{_cycle_chart(kpi['pick_cycles']['value'])}</div>
</div>
<div class="two">
  <div class="card"><h2>An toàn chuyển động</h2><table><tr><th>Loại tiếp xúc</th><th>Số frame</th><th>Sâu nhất (mm)</th></tr>{contacts}</table>
  <p class="k">Biên khớp nhỏ nhất: <b>{margin}°</b> · jerk RMS (rad/s³): {jerk} · số lần dừng: {stops} · thời gian tính (wall): {kpi['wall_seconds']['value']} s</p></div>
  <div class="card"><h2>Ước tính công suất (nhập số của doanh nghiệp)</h2>
  <label>Số đơn/ngày (cùng cỡ đơn này)<input id="orders" type="number" min="0" step="1"></label>
  <label>Số giờ vận hành/ngày<input id="hours" type="number" min="0" step="0.5"></label>
  <p id="cap" class="k">Nhập số để tính. Dựa trên thời gian đơn đo được: {order_sim_s:.0f} s (sim).</p></div>
</div>
{''.join(media)}
<p class="k">File: <code>order_result.json</code>, <code>kpi.json</code>, <code>metrics.json</code>, <code>report.json</code>.
Replay 3D: <code>python -m product.cell --replay {html.escape(str(run))}</code></p>
</main><script>
const T={order_sim_s};
function upd(){{const o=+document.getElementById('orders').value,h=+document.getElementById('hours').value;
const el=document.getElementById('cap');if(!o||!h){{return}}
const need=o*T/3600, cells=Math.ceil(need/h);
el.textContent=`Cần ${{need.toFixed(1)}} giờ-robot/ngày → ${{cells}} cell cho ${{h}} giờ vận hành (theo thời gian mô phỏng; robot thật cần đo lại).`;}}
document.getElementById('orders').oninput=upd;document.getElementById('hours').oninput=upd;
</script></body></html>"""
    out = run / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    build_index(run.parent)
    return out


def build_index(runs: Path = RUNS) -> Path:
    rows = []
    for run in sorted((p for p in runs.iterdir() if (p / "kpi.json").exists()), reverse=True):
        k = json.loads((run / "kpi.json").read_text(encoding="utf-8"))
        status = k["status"]["value"]
        cls = {"complete": "ok", "short": "warn"}.get(status, "bad")
        rows.append(f"<tr><td><a href='{run.name}/dashboard.html'>{html.escape(k['order_id'])}</a></td>"
                    f"<td class='k'>{html.escape(run.name)}</td><td class='{cls}'>{STATUS_TEXT.get(status, status)}</td>"
                    f"<td class='n'>{k['items_in_tote']['value']}</td><td class='n'>{k['mispicks']['value']}</td>"
                    f"<td class='n'>{k['sim_seconds']['value']:.0f}</td><td>{'Đạt' if k['collision_free']['value'] else 'Không'}</td></tr>")
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PickCell runs</title><style>{CSS}</style></head><body><main><h1>PickCell: các đơn đã chạy</h1>
<div class="card"><table><tr><th>Đơn</th><th>Lần chạy</th><th>Trạng thái</th><th>Món trong thùng</th><th>Mispick</th><th>Sim (s)</th><th>Không va chạm</th></tr>
{''.join(rows) or '<tr><td colspan=7 class=k>Chưa có lần chạy.</td></tr>'}</table></div></main></body></html>"""
    out = runs / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args(argv)
    if args.run:
        print(build(args.run, gif=not args.no_gif))
    if args.index or not args.run:
        print(build_index())


if __name__ == "__main__":
    main()
