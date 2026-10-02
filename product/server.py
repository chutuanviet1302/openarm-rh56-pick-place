"""PickCell web app: place an order, watch the cell work on it, read its dashboard.

    python -m product.server                 # http://localhost:8010
    python -m product.server --port 8010 --no-browser

One order at a time (a run takes ~10 min wall and ~4 GB RAM on the laptop, measured
2026-10-01): orders wait in a queue, a single worker runs `python -m product.cell` for
each and keeps its log. Only the app page (product/web) and the run folders (runs/) are
served; the server listens on 127.0.0.1 only.

API
    GET  /api/catalog                 SKUs the cell knows
    POST /api/orders                  {"order_id"?, "items": {sku: qty}, "seed"?, "pose_backend"?} -> {"job": id}
    GET  /api/orders/<job>            status, queue position, phase, log tail, run folder when done
    GET  /api/runs                    finished runs with their KPIs
    POST /api/runs/<run>/replay       open the MuJoCo replay window on this machine
"""

from __future__ import annotations

import argparse
import http.server
import json
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from product.order import CATALOG, Order

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "product" / "web"
RUNS = ROOT / "runs"
JOBS_DIR = RUNS / "_jobs"
JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()
LOCK = threading.Lock()
PHASE = re.compile(r"\d/7 ([A-Z]+):")
RUN_LINE = re.compile(r"^run: (.+)$", re.M)
SAFE_RUN = re.compile(r"^[A-Za-z0-9_-]+$")


def stock() -> dict[str, dict]:
    """What the cell has in stock in the fixed layout, per SKU: on the table, on the belt.
    (From the task's own layout constants, so it cannot drift from what the cell does.)"""
    counts = {sku: {"table": 0, "belt": 0} for sku in CATALOG}
    try:
        from simulation.pick_place.bin_conveyor_task import BELT_OBJECTS, TABLE_OBJECTS

        for where, objects in (("table", TABLE_OBJECTS), ("belt", BELT_OBJECTS)):
            for p in objects:
                for sku, entry in CATALOG.items():
                    if entry["kind"] == p.key:
                        counts[sku][where] += 1
    except Exception as error:  # noqa: BLE001 - the page then shows no stock, not an error
        print(f"(stock unavailable: {error})")
        return {}
    return counts


def presets() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "orders").glob("*.json")):
        try:
            order = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"order_id": order["order_id"], "items": order["items"]})
        except (OSError, ValueError, KeyError):
            continue
    return rows


def _tail(path: Path, lines: int = 30) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def worker() -> None:
    while True:
        job_id = QUEUE.get()
        job = JOBS[job_id]
        log = JOBS_DIR / f"{job_id}.log"
        order_file = JOBS_DIR / f"{job_id}.order.json"
        command = [sys.executable, "-u", "-m", "product.cell", "--order", str(order_file),
                   "--pose-backend", job["pose_backend"]]
        if job["seed"] is not None:
            command += ["--seed", str(job["seed"])]
        with LOCK:
            job.update(status="running", started=time.time())
        with log.open("w", encoding="utf-8") as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
        found = RUN_LINE.findall(log.read_text(encoding="utf-8", errors="replace"))
        with LOCK:
            job["finished"] = time.time()
            if code == 0 and found:
                run = Path(found[-1].strip())
                job.update(status="done", run=run.name)
                result = RUNS / run.name / "order_result.json"
                if result.exists():
                    job["order_status"] = json.loads(result.read_text(encoding="utf-8"))["status"]
            else:
                job.update(status="error", error=f"product.cell exited with {code}")
        QUEUE.task_done()


def job_view(job_id: str) -> dict | None:
    with LOCK:
        job = dict(JOBS.get(job_id) or {})
    if not job:
        return None
    lines = _tail(JOBS_DIR / f"{job_id}.log", 400)
    phases = [m.group(1) for line in lines for m in [PHASE.search(line)] if m]
    waiting = [j for j in list(QUEUE.queue)]
    job.update(
        job=job_id,
        queue_position=waiting.index(job_id) + 1 if job_id in waiting else 0,
        phase=phases[-1].lower() if phases else None,
        log=lines[-30:],
        elapsed_s=round((job.get("finished") or time.time()) - job["started"], 0) if job.get("started") else None,
        dashboard=f"/runs/{job['run']}/dashboard.html" if job.get("run") else None,
    )
    return job


def list_runs() -> list[dict]:
    rows = []
    if RUNS.exists():
        for run in sorted(RUNS.iterdir(), reverse=True):
            kpi = run / "kpi.json"
            if kpi.exists():
                k = json.loads(kpi.read_text(encoding="utf-8"))
                rows.append({"run": run.name, "order_id": k["order_id"], "status": k["status"]["value"],
                             "items_in_tote": k["items_in_tote"]["value"], "mispicks": k["mispicks"]["value"],
                             "sim_seconds": k["sim_seconds"]["value"], "collision_free": k["collision_free"]["value"],
                             "dashboard": f"/runs/{run.name}/dashboard.html"})
    return rows


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # quiet: the browser polls every 2 s

    def translate_path(self, path: str) -> str:
        """Serve only product/web (at /) and runs/ (at /runs/)."""
        path = unquote(urlparse(path).path)
        if path.startswith("/runs/"):
            base, rest = RUNS, path[len("/runs/"):]
        else:
            base, rest = WEB, path.lstrip("/") or "index.html"
        target = (base / rest).resolve()
        if base.resolve() not in target.parents and target != base.resolve():
            return str(WEB / "__forbidden__")
        return str(target)

    def send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/api/catalog":
            self.send_json(200, {"catalog": CATALOG, "stock": STOCK, "presets": presets()})
        elif path == "/api/runs":
            self.send_json(200, list_runs())
        elif path.startswith("/api/orders/"):
            job = job_view(path.rsplit("/", 1)[-1])
            self.send_json(200 if job else 404, job or {"error": "Không tìm thấy đơn"})
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}") if length < 100_000 else {}
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "JSON không hợp lệ"})
            return
        if path == "/api/orders":
            self.create_order(payload)
        elif path.startswith("/api/runs/") and path.endswith("/replay"):
            run = path[len("/api/runs/"):-len("/replay")]
            if not SAFE_RUN.match(run) or not (RUNS / run / "frames.npz").exists():
                self.send_json(404, {"error": "Không tìm thấy lần chạy"})
                return
            subprocess.Popen([sys.executable, "-m", "product.cell", "--replay", str(RUNS / run)], cwd=ROOT)
            self.send_json(202, {"ok": True, "message": "Cửa sổ MuJoCo đang mở trên máy chạy server"})
        else:
            self.send_json(404, {"error": "not found"})

    def create_order(self, payload: dict) -> None:
        job_id = time.strftime("web-%Y%m%d-%H%M%S")
        with LOCK:
            while job_id in JOBS:
                job_id += "x"
        try:
            order_id = str(payload.get("order_id") or job_id)
            order = Order.from_dict({"order_id": order_id, "items": payload.get("items") or {}})
            seed = payload.get("seed")
            seed = None if seed in (None, "") else int(seed)
            backend = payload.get("pose_backend") or "gt"
            if backend not in ("gt", "foundationpose"):
                raise ValueError("pose_backend: gt hoặc foundationpose")
        except (ValueError, TypeError) as error:
            self.send_json(400, {"error": str(error)})
            return
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        (JOBS_DIR / f"{job_id}.order.json").write_text(order.to_json(), encoding="utf-8")
        with LOCK:
            JOBS[job_id] = {"status": "queued", "order": order.items, "order_id": order.order_id, "seed": seed,
                            "pose_backend": backend, "queued_at": time.time()}
        QUEUE.put(job_id)
        self.send_json(202, {"job": job_id})


STOCK: dict = {}


def main(argv=None) -> None:
    global STOCK
    STOCK = stock()
    parser = argparse.ArgumentParser(description="PickCell web app")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    threading.Thread(target=worker, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}/"
    print(f"PickCell: {url}   (Ctrl+C to stop)", flush=True)
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
