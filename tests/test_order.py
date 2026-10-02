import unittest

from product.order import CATALOG, Order, OrderBook


class OrderTest(unittest.TestCase):
    def test_validation(self):
        Order("demo_1", {"apple": 2})
        for bad in ({}, {"banana": 1}, {"apple": 0}, {"apple": 1.5}, {"apple": True}):
            with self.assertRaises(ValueError):
                Order("demo_1", bad)
        with self.assertRaises(ValueError):
            Order("bad id!", {"apple": 1})
        self.assertEqual(Order.from_dict({"order_id": "x", "items": {"can": 1, "apple": 0}}).items, {"can": 1})

    def test_wanted_follows_what_was_dropped(self):
        book = OrderBook(Order("o", {"can": 2, "apple": 1}))
        self.assertTrue(book.wanted("can") and book.wanted("apple"))
        self.assertFalse(book.wanted("orange") or book.wanted(None))
        book.record("can", "can", ok=True)
        book.record("apple", "apple", ok=False)
        self.assertEqual(book.remaining("can"), 1)
        self.assertTrue(book.wanted("apple"))
        book.record("can_belt", "can", ok=True)
        book.record("apple", "apple", ok=True)
        self.assertTrue(book.complete())
        self.assertFalse(book.wanted("can"))

    def test_results(self):
        kinds = {"can": "can", "can_belt": "can", "apple": "apple", "orange": "orange"}
        book = OrderBook(Order("o", {"can": 1, "apple": 1}))
        book.record("can", "can", True)
        book.record("apple", "apple", True)
        done = book.result({"can": True, "apple": True, "can_belt": False, "orange": False}, kinds)
        self.assertEqual(done.status, "complete")
        self.assertEqual(done.mispicks, [])
        # An orange in the tote that the order did not ask for: a mispick.
        wrong = book.result({"can": True, "apple": True, "orange": True}, kinds)
        self.assertEqual((wrong.status, wrong.mispicks), ("failed", ["orange"]))
        # Orange not picked, nothing failed: short, not a robot fault.
        book = OrderBook(Order("o", {"apple": 1, "orange": 1}))
        book.record("apple", "apple", True)
        short = book.result({"apple": True}, kinds)
        self.assertEqual(short.status, "short")
        self.assertEqual([(l.sku, l.missing) for l in short.lines], [("apple", 0), ("orange", 1)])
        # A pick that went wrong and left a gap: failed.
        book = OrderBook(Order("o", {"apple": 1}))
        book.record("apple", "apple", False)
        self.assertEqual(book.result({"apple": False}, kinds).status, "failed")

    def test_catalog_kinds_are_registry_objects(self):
        from simulation.objects import OBJECTS

        self.assertTrue(all(entry["kind"] in OBJECTS for entry in CATALOG.values()))


if __name__ == "__main__":
    unittest.main()
