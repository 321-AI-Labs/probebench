"""Thin test wrapper: runs the full probebench selftest suite.

Run with:  python -m unittest tests.test_selftest  (from the repo root)
or simply: probebench selftest
"""

import unittest

from probebench import cli


class Selftest(unittest.TestCase):
    def test_full_selftest(self):
        rc = cli.main(["selftest"])
        self.assertEqual(rc, 0, "probebench selftest reported FAIL")


if __name__ == "__main__":
    unittest.main()
