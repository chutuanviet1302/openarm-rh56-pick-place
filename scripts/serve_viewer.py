"""Serve the MuJoCo trajectory studio on http://localhost:8000/viewer/control.html."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import math
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from simulation.five_finger_model import (
    BASKET_HALF_WIDTH,
    BASKET_WALL_THICKNESS,
    OBJECT_RADIUS,
    TABLE_HALF_WIDTH,
    TABLE_X_RANGE,
)
from simulation.pick_place.routing import Route, TaskRouter
from simulation.pick_place.scene import Scene

ROOT = Path(__file__).resolve().parents[1]
TABLE_X = TABLE_X_RANGE
TABLE_Y = (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH)
RISER_CENTER = (-0.03, 0.0)
RISER_HALF = (0.13, 0.10)
MIN_PICK_TO_BASKET_M = 0.15
JOBS: dict[str, dict] = {}


def parse_layout(payload: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Validate the browser input before it reaches the simulator."""
    try:
        obj = tuple(float(value) for value in payload["object"])
        basket = tuple(float(value) for value in payload["basket"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("object và basket phải là hai tọa độ số [x, y]") from None
    if len(obj) != 2 or len(basket) != 2 or not all(math.isfinite(v) for v in (*obj, *basket)):
        raise ValueError("tọa độ phải gồm đúng hai số hữu hạn")
    for label, (x, y) in (("Vật", obj), ("Rổ", basket)):
        if not (TABLE_X[0] <= x <= TABLE_X[1] and TABLE_Y[0] <= y <= TABLE_Y[1]):
            raise ValueError(f"{label} nằm ngoài mặt bàn")
    if math.dist(obj, basket) < MIN_PICK_TO_BASKET_M:
        raise ValueError("Vật và rổ phải cách nhau ít nhất 15 cm")
    for label, point, half_size in (
        ("Vật", obj, OBJECT_RADIUS),
        ("Rổ", basket, BASKET_HALF_WIDTH + BASKET_WALL_THICKNESS),
    ):
        if all(abs(point[i] - RISER_CENTER[i]) < RISER_HALF[i] + half_size for i in range(2)):
            raise ValueError(f"{label} chồng lên đế robot")
    return obj, basket


def run_job(job_id: str, obj: tuple[float, float], basket: tuple[float, float], side: str, route: str) -> None:
    JOBS[job_id] = {"status": "running", "message": "Đang tạo trajectory và chạy MuJoCo…", "route": route}
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.record_episode",
        "--object",
        *map(str, obj),
        "--basket",
        *map(str, basket),
        "--arm",
        side,
        "--name",
        job_id,
        "--fps",
        "15",
        "--cameras",
        "isometric",
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-1200:])
        episode_path = ROOT / "artifacts" / "episodes" / job_id / "episode.json"
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        failure = episode["result"]["failure"]
        JOBS[job_id] = {
            "status": "complete",
            "passed": episode["result"]["passed"],
            "message": failure or "Trajectory hoàn thành",
            "episode": job_id,
            "route": route,
        }
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        JOBS[job_id] = {"status": "failed", "message": str(error)}


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/api/trajectory":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            obj, basket = parse_layout(payload)
            decision = TaskRouter(Scene(obj, basket)).select()
            if decision.route == Route.REJECTED:
                raise ValueError(decision.reason)
            if decision.route not in (Route.DIRECT_RIGHT, Route.DIRECT_LEFT):
                raise ValueError(f"{decision.route.value} đã được chọn nhưng handoff vật lý chưa vượt cổng nghiệm thu")
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
            return
        job_id = datetime.now().strftime("web-%Y%m%d-%H%M%S-%f")[:-3]
        JOBS[job_id] = {"status": "queued", "message": "Đã xếp hàng"}
        threading.Thread(
            target=run_job,
            args=(job_id, obj, basket, decision.source_arm, decision.route.value),
            daemon=True,
        ).start()
        self.send_json(202, {"job": job_id, "route": decision.route.value})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        prefix = "/api/trajectory/"
        if self.path.startswith(prefix):
            job = JOBS.get(self.path.removeprefix(prefix))
            self.send_json(200 if job else 404, job or {"error": "Không tìm thấy tác vụ"})
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/viewer/control.html"
    print(f"trajectory studio: {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
