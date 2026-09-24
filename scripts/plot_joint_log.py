"""Draw the jerk-analysis figures from joint-state logs as SVG (no plotting library).

    python -m scripts.plot_joint_log
    -> docs/jerk_analysis/*.svg  (used by docs/jerk_analysis.md)

Inputs are the PlotJuggler CSVs written by scripts/log_joint_states.py:
artifacts/joint_logs/retrieve_headless_plot.csv       (before the fix)
artifacts/joint_logs/retrieve_headless_fixed_plot.csv (after the fix)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LOGS = Path("artifacts") / "joint_logs"
OUT = Path("docs") / "jerk_analysis"
BEFORE = LOGS / "retrieve_headless_plot.csv"
AFTER = LOGS / "retrieve_headless_fixed_plot.csv"
COLORS = {"cmd": "#1f6feb", "pos": "#e36209", "torque": "#8250df", "wall": "#cf222e", "ideal": "#6e7781"}


def load(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return np.array(rows[1:], dtype=float), {name: i for i, name in enumerate(rows[0])}


def _num(v: float, span) -> str:
    """Tick label with enough decimals for the axis span; a real minus sign."""
    rng = abs(span[1] - span[0])
    text = f"{v:.3f}" if rng < 1 else f"{v:.2f}" if rng < 10 else f"{v:.0f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return text.replace("-", "−")


class Panel:
    """One x/y chart inside an SVG canvas."""

    def __init__(self, x: float, y: float, w: float, h: float, xr, yr, title: str, xlabel: str, ylabel: str) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xr, self.yr = xr, yr
        self.parts: list[str] = []
        self._frame(title, xlabel, ylabel)

    def px(self, v): return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w
    def py(self, v): return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def _frame(self, title, xlabel, ylabel) -> None:
        p = self.parts
        p.append(f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" fill="#ffffff" stroke="#d0d7de"/>')
        for v in np.linspace(*self.xr, 6):
            X = self.px(v)
            p.append(f'<line x1="{X:.1f}" y1="{self.y}" x2="{X:.1f}" y2="{self.y + self.h}" stroke="#eaeef2"/>')
            p.append(f'<text x="{X:.1f}" y="{self.y + self.h + 16}" font-size="11" text-anchor="middle" fill="#57606a">{_num(v, self.xr)}</text>')
        for v in np.linspace(*self.yr, 5):
            Y = self.py(v)
            p.append(f'<line x1="{self.x}" y1="{Y:.1f}" x2="{self.x + self.w}" y2="{Y:.1f}" stroke="#eaeef2"/>')
            p.append(f'<text x="{self.x - 6}" y="{Y + 4:.1f}" font-size="11" text-anchor="end" fill="#57606a">{_num(v, self.yr)}</text>')
        p.append(f'<text x="{self.x}" y="{self.y - 8}" font-size="13" font-weight="bold" fill="#24292f">{title}</text>')
        p.append(f'<text x="{self.x + self.w / 2}" y="{self.y + self.h + 32}" font-size="11" text-anchor="middle" fill="#57606a">{xlabel}</text>')
        p.append(f'<text x="{self.x - 44}" y="{self.y + self.h / 2}" font-size="11" text-anchor="middle" fill="#57606a" '
                 f'transform="rotate(-90 {self.x - 44} {self.y + self.h / 2})">{ylabel}</text>')

    def line(self, xs, ys, color: str, width: float = 1.6, dash: str = "", step: bool = False, stride: int = 1) -> None:
        mask = (xs >= self.xr[0]) & (xs <= self.xr[1])
        xs, ys = xs[mask], np.clip(ys[mask], *self.yr)
        if stride > 1 and len(xs) > 2:
            # thin out, but keep every sample next to a jump so steps stay visible
            jump = np.abs(np.diff(ys)) > (self.yr[1] - self.yr[0]) * 0.02
            keep = np.zeros(len(xs), bool); keep[::stride] = True; keep[-1] = True
            keep[:-1] |= jump; keep[1:] |= jump
            xs, ys = xs[keep], ys[keep]
        if len(xs) < 2:
            return
        pts = []
        for i, (a, b) in enumerate(zip(xs, ys)):
            if step and i:
                pts.append(f"{self.px(a):.1f},{self.py(ys[i - 1]):.1f}")
            pts.append(f"{self.px(a):.1f},{self.py(b):.1f}")
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def legend(self, items: list[tuple[str, str]]) -> None:
        for i, (label, color) in enumerate(items):
            X, Y = self.x + 10, self.y + 16 + 16 * i
            self.parts.append(f'<line x1="{X}" y1="{Y - 4}" x2="{X + 18}" y2="{Y - 4}" stroke="{color}" stroke-width="2.5"/>')
            self.parts.append(f'<text x="{X + 24}" y="{Y}" font-size="11" fill="#24292f">{label}</text>')

    def note(self, xv, yv, text: str, color: str = "#cf222e") -> None:
        X, Y = self.px(xv), self.py(yv)
        self.parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="5" fill="none" stroke="{color}" stroke-width="2"/>')
        self.parts.append(f'<text x="{X + 9:.1f}" y="{Y - 7:.1f}" font-size="11" font-weight="bold" fill="{color}">{text}</text>')


def save(path: Path, width: int, height: int, panels: list[Panel], title: str) -> None:
    body = "\n".join(part for panel in panels for part in panel.parts)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="Segoe UI, Arial, sans-serif">\n<rect width="100%" height="100%" fill="#f6f8fa"/>\n'
           f'<text x="20" y="26" font-size="15" font-weight="bold" fill="#24292f">{title}</text>\n{body}\n</svg>\n')
    path.write_text(svg, encoding="utf-8")


def fig_zoom(before, after, path: Path) -> None:
    """cmd vs pos and torque of right j4 at the grasp -> carry boundary, before | after."""
    t0, t1 = 14.60, 14.72
    panels = []
    for col, (label, (A, c)) in enumerate((("TRƯỚC khi sửa", before), ("SAU khi sửa", after))):
        t = A[:, c["sim_time"]]
        cmd, pos, tau = (np.degrees(A[:, c["right/j4/cmd"]]), np.degrees(A[:, c["right/j4/pos"]]), A[:, c["right/j4/torque"]])
        m = (t >= t0) & (t <= t1)
        lo, hi = min(cmd[m].min(), pos[m].min()) - 0.3, max(cmd[m].max(), pos[m].max()) + 0.3
        x = 90 + col * 560
        top = Panel(x, 70, 470, 230, (t0, t1), (lo, hi), f"{label}: khớp 4 tay phải – lệnh vs vị trí",
                    "thời gian mô phỏng (s)", "góc (°)")
        top.line(t, cmd, COLORS["cmd"], 2.0, step=True)
        top.line(t, pos, COLORS["pos"], 1.6, dash="5,3")
        top.legend([("cmd (lệnh)", COLORS["cmd"]), ("pos (vị trí thật)", COLORS["pos"])])
        bottom = Panel(x, 370, 470, 230, (t0, t1), (-6, 10), f"{label}: mômen khớp 4 tay phải",
                       "thời gian mô phỏng (s)", "mômen (N·m)")
        bottom.line(t, tau, COLORS["torque"], 2.0)
        bottom.line(np.array([t0, t1]), np.array([0.0, 0.0]), COLORS["ideal"], 1.0, dash="3,3")
        if col == 0:
            i = int(np.argmax(np.abs(np.diff(cmd[m])))) + 1
            tm = t[m][i]
            top.note(tm, cmd[m][i], "lệnh nhảy bậc -0.84° trong 1 bước")
            j = int(np.argmin(tau[m]))
            bottom.note(t[m][j], tau[m][j], "mômen +7.5 → -4.0 N·m")
        else:
            bottom.note(t[m][len(t[m]) // 2], tau[m][len(t[m]) // 2], "không đổi dấu đột ngột", "#1a7f37")
        panels += [top, bottom]
    save(path, 1140, 650, panels, "Hình 1 – Cú giật ở đầu một đoạn chuyển động (lúc bắt đầu nâng lon, t ≈ 14.64 s)")


def fig_overview(before, after, path: Path) -> None:
    """right j4 torque over the whole place leg, before and after."""
    panels = []
    for row, (label, (A, c)) in enumerate((("TRƯỚC khi sửa", before), ("SAU khi sửa", after))):
        t, tau = A[:, c["sim_time"]], A[:, c["right/j4/torque"]]
        p = Panel(90, 70 + row * 300, 1000, 210, (0, 30), (-8, 14), f"{label}: mômen khớp 4 tay phải, cả chặng đặt lon (0–30 s)",
                  "thời gian mô phỏng (s)", "mômen (N·m)")
        p.line(t, tau, COLORS["torque"], 1.3, stride=4)
        dtau = np.abs(np.diff(tau))
        for i in np.where(dtau > 3)[0]:
            X = p.px(t[i + 1])
            p.parts.append(f'<line x1="{X:.1f}" y1="{p.y}" x2="{X:.1f}" y2="{p.y + p.h}" stroke="#cf222e" stroke-opacity="0.35"/>')
        count = int((dtau > 3).sum())
        p.parts.append(f'<text x="{p.x + p.w - 8}" y="{p.y + 18}" font-size="12" font-weight="bold" text-anchor="end" '
                       f'fill="{"#cf222e" if count else "#1a7f37"}">{count} lần mômen nhảy &gt; 3 N·m (vạch đỏ)</text>')
        panels.append(p)
    save(path, 1140, 640, panels, "Hình 2 – Mômen khớp 4 tay phải trong cả chặng đặt lon: trước và sau khi sửa")


def fig_wall_time(before, path: Path) -> None:
    """wall-clock time vs simulated time: slope < 1 = faster than real time; vertical jumps = planning."""
    A, c = before
    t, wall, think = A[:, c["sim_time"]], A[:, c["wall_time"]], A[:, c["thinking"]]
    p = Panel(90, 70, 1000, 420, (0, float(t[-1])), (0, float(wall[-1]) * 1.05),
              "Thời gian thực trên máy theo thời gian mô phỏng", "thời gian mô phỏng (s)", "thời gian thực (s)")
    p.line(np.array([0, t[-1]]), np.array([0, t[-1]]), COLORS["ideal"], 1.2, dash="6,4")
    p.line(t, wall, COLORS["wall"], 2.0, stride=40)
    p.legend([("máy thật (đo được)", COLORS["wall"]), ("đúng thời gian thực (y = x)", COLORS["ideal"])])
    jumps = np.where(np.diff(wall) > 1.0)[0]
    for i in jumps[:8]:
        p.note(t[i + 1], wall[i + 1], f"planner {wall[i + 1] - wall[i]:.1f} s")
    save(path, 1140, 540, [p], "Hình 3 – Máy chạy nhanh hơn thời gian thực; các bậc đứng là lúc planner tính (robot đứng yên)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before, after = load(BEFORE), load(AFTER)
    fig_zoom(before, after, OUT / "fig1_zoom_j4.svg")
    fig_overview(before, after, OUT / "fig2_torque_overview.svg")
    fig_wall_time(before, OUT / "fig3_wall_time.svg")
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
