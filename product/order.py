"""Orders for the PickCell order-kitting cell: what to put in the tote, and what came of it.

    order = Order.load(Path("orders/demo_001.json"))      # {"order_id": ..., "items": {"apple": 1, "can": 1}}
    book = OrderBook(order)
    book.wanted("apple")       # True while an apple is still missing
    book.record("apple_1", "apple", ok=True)
    book.result(audit_in_tote={...}, object_kinds={...})

SKUs are the cell's catalogue: each maps to the object kind the camera identifies. A SKU
the cell knows but that is not in stock (not on the table, never comes down the belt)
ends as a "short" line -- the order is not filled, and that is not a robot fault.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# SKU -> object kind the detector names. The cell can identify these (detector
# references are enrolled for them); the registry also has kinds that are not stocked
# in the cell (a quantity above the stock): orderable, but they end short.
CATALOG = {
    "can": {"kind": "can", "name": "Lon cà chua (YCB 005)"},
    "apple": {"kind": "apple", "name": "Táo (YCB 013)"},
    "orange": {"kind": "orange", "name": "Cam (YCB 017)"},
    "peach": {"kind": "peach", "name": "Đào (YCB 015)"},
}
ORDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_QTY = 10


@dataclass
class Order:
    order_id: str
    items: dict[str, int]

    def __post_init__(self) -> None:
        if not ORDER_ID.match(self.order_id or ""):
            raise ValueError("order_id: 1-40 letters, digits, '-' or '_'")
        if not self.items:
            raise ValueError("an order needs at least one item")
        clean = {}
        for sku, qty in self.items.items():
            if sku not in CATALOG:
                raise ValueError(f"unknown SKU {sku!r}; catalogue: {sorted(CATALOG)}")
            if isinstance(qty, bool) or not isinstance(qty, int) or not 1 <= qty <= MAX_QTY:
                raise ValueError(f"{sku}: quantity must be a whole number 1..{MAX_QTY}")
            clean[sku] = qty
        self.items = clean

    @classmethod
    def from_dict(cls, raw: dict) -> "Order":
        if not isinstance(raw, dict) or not isinstance(raw.get("items"), dict):
            raise ValueError('expected {"order_id": "...", "items": {"sku": qty, ...}}')
        items = {str(k): v for k, v in raw["items"].items() if v not in (0, None)}
        return cls(str(raw.get("order_id", "")), items)

    @classmethod
    def load(cls, path: Path) -> "Order":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def kinds(self) -> dict[str, int]:
        """Quantity per detector kind."""
        out: dict[str, int] = {}
        for sku, qty in self.items.items():
            kind = CATALOG[sku]["kind"]
            out[kind] = out.get(kind, 0) + qty
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class OrderLine:
    sku: str
    requested: int
    picked: int                 # in the tote, by the audit
    missing: int


@dataclass
class OrderResult:
    order_id: str
    status: str                 # complete | short | failed
    lines: list[OrderLine]
    mispicks: list[str]         # objects in the tote that the order did not ask for (audit)
    robot_dropped: list[str]    # what the robot believes it put in the tote
    failed_attempts: list[str]  # picks that did not end in the tote
    sources: dict = field(default_factory=dict)


class OrderBook:
    """The cell's running tally of an order: what is still wanted, what was done."""

    def __init__(self, order: Order) -> None:
        self.order = order
        self.need = order.kinds()
        self.dropped: list[tuple[str, str]] = []     # (object name, kind) the robot let go over the tote
        self.failed: list[str] = []

    def remaining(self, kind: str) -> int:
        return self.need.get(kind, 0) - sum(1 for _, k in self.dropped if k == kind)

    def wanted(self, kind: str | None) -> bool:
        return kind is not None and self.remaining(kind) > 0

    def complete(self) -> bool:
        return all(self.remaining(kind) <= 0 for kind in self.need)

    def record(self, name: str, kind: str, ok: bool) -> None:
        if ok:
            self.dropped.append((name, kind))
        else:
            self.failed.append(name)

    def result(self, audit_in_tote: dict[str, bool], object_kinds: dict[str, str]) -> OrderResult:
        """The order's outcome. `audit_in_tote`: object name -> in the tote, from the
        simulator after the run (the ground truth an inspector would check);
        `object_kinds`: object name -> kind."""
        inside = [name for name, ok in audit_in_tote.items() if ok]
        count: dict[str, int] = {}
        for name in inside:
            count[object_kinds[name]] = count.get(object_kinds[name], 0) + 1
        lines, mispicks = [], []
        budget = dict(self.need)
        for name in inside:
            kind = object_kinds[name]
            if budget.get(kind, 0) > 0:
                budget[kind] -= 1
            else:
                mispicks.append(name)
        for sku, qty in self.order.items.items():
            kind = CATALOG[sku]["kind"]
            picked = min(qty, count.get(kind, 0))
            count[kind] = count.get(kind, 0) - picked
            lines.append(OrderLine(sku, qty, picked, qty - picked))
        filled = all(line.missing == 0 for line in lines)
        # "short": items missing and the robot never failed a pick (the stock ran out);
        # "failed": a wrong item in the tote, or a pick that went wrong left a gap.
        if filled and not mispicks:
            status = "complete"
        elif mispicks or self.failed:
            status = "failed"
        else:
            status = "short"
        return OrderResult(self.order.order_id, status, lines, mispicks, [n for n, _ in self.dropped], list(self.failed),
                           sources={"picked, mispicks": "simulator audit after the run (object_in_basket)",
                                    "robot_dropped": "objects the robot let go over the tote and saw land"})
