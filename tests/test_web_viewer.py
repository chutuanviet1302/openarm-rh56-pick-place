import unittest

from scripts.serve_viewer import parse_layout


class LayoutValidationTest(unittest.TestCase):
    def test_accepts_table_coordinates_and_rejects_bad_input(self):
        self.assertEqual(parse_layout({"object": [0.08, -0.38], "basket": [0.25, -0.25]}), ((0.08, -0.38), (0.25, -0.25)))
        with self.assertRaisesRegex(ValueError, "ngoài mặt bàn"):
            parse_layout({"object": [9, 9], "basket": [0.25, -0.25]})
        with self.assertRaises(ValueError):
            parse_layout({"object": [0], "basket": [0.25, -0.25]})
        with self.assertRaisesRegex(ValueError, "15 cm"):
            parse_layout({"object": [0.08, -0.38], "basket": [0.10, -0.36]})
        with self.assertRaisesRegex(ValueError, "đế robot"):
            parse_layout({"object": [0.0, 0.0], "basket": [0.25, -0.25]})


if __name__ == "__main__":
    unittest.main()
