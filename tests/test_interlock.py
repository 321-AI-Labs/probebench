"""FT-8 hard-interlock tests (REVIEW-P2 D8 remediation / mission D7).

The interlock: `probebench run` and `probebench judges` refuse to contact the
P8-owned live judge defaults (127.0.0.1:809x) unless --live-judges-acknowledged
is explicitly passed. Offline/mock runs pinned via --judge-url need no flag.

ZERO live-judge contact by construction: no test here ever issues a network
request to a 809x port. The refusal paths return before any network call or
file write; the proceed paths talk only to the bundled mock judge on loopback.

Hermeticity note: `cmd_run` applies --judge-url overrides by mutating the
module-global judges.JUDGES (by design: it configures the process-wide judge
registry). These tests snapshot and restore that registry around every test so
results are order-independent, and the refusal tests explicitly reset the
registry to the shipped live defaults to simulate a fresh process.

Run with:  python -m unittest tests.test_interlock   (from the repo root)
"""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from probebench import cli, judges
from probebench.mock_judge import serve_in_thread, stop

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
MOCK_PORT = 18399  # P2-local; never a 809x port

# The shipped defaults (probebench/judges.py) - the interlock's protected targets.
LIVE_DEFAULTS = {"glm": "http://127.0.0.1:8091",
                 "llama": "http://127.0.0.1:8092",
                 "qwen": "http://127.0.0.1:8093"}


def _mock_run_argv(out_dir, extra=()):
    base = [
        "run", "http://127.0.0.1:18999/v1",
        "--suite", str(FIXTURES / "fixture-prompts.jsonl"),
        "--sources", str(REPO_ROOT / "suite" / "SOURCES.md"),
        "--fixture-answers", str(FIXTURES / "fixture-answers.jsonl"),
        "--max-items", "4",
        "--run-name", "interlock-selftest",
        "--out-dir", str(out_dir),
    ]
    for j in ("glm", "llama", "qwen"):
        base += ["--judge-url", f"{j}=http://127.0.0.1:{MOCK_PORT}"]
    return base + list(extra)


class JudgesRegistryIsolated(unittest.TestCase):
    """Base class: snapshot judges.JUDGES and restore it after every test, so
    the cmd_run override mutation never leaks between tests (test pollution
    found and fixed during D7 verification: a mock-pinned run left JUDGES
    pointing at the mock, silently disarming a later refusal test)."""

    def setUp(self):
        self._saved_judges = copy.deepcopy(judges.JUDGES)

    def tearDown(self):
        judges.JUDGES.clear()
        judges.JUDGES.update(self._saved_judges)

    @staticmethod
    def reset_registry_to_live_defaults():
        """Simulate a fresh process: registry = shipped live defaults."""
        judges.JUDGES.clear()
        judges.JUDGES.update({j: {"base_url": u} for j, u in LIVE_DEFAULTS.items()})


class InterlockClassifier(JudgesRegistryIsolated):
    def test_live_ports_classified_live(self):
        urls = dict(LIVE_DEFAULTS)
        self.assertEqual(cli._live_judge_urls(urls), urls)

    def test_mock_ports_not_live(self):
        urls = {j: f"http://127.0.0.1:{MOCK_PORT}" for j in ("glm", "llama", "qwen")}
        self.assertEqual(cli._live_judge_urls(urls), {})

    def test_localhost_and_bracket_ipv6(self):
        self.assertTrue(cli._live_judge_urls({"glm": "http://localhost:8091"}))
        self.assertTrue(cli._live_judge_urls({"glm": "http://[::1]:8093/v1"}))
        self.assertFalse(cli._live_judge_urls({"glm": "http://127.0.0.1:18099"}))
        self.assertFalse(cli._live_judge_urls({"glm": "http://127.0.0.1:8080"}))

    def test_interlock_block_flag_flips_decision(self):
        urls = {"glm": "http://127.0.0.1:8091"}
        self.assertEqual(cli._interlock_block(urls, acknowledged=False), urls)
        self.assertEqual(cli._interlock_block(urls, acknowledged=True), {})
        self.assertEqual(cli._interlock_block({"glm": "http://127.0.0.1:18399"},
                                              acknowledged=False), {})


class InterlockRunRefusal(JudgesRegistryIsolated):
    def setUp(self):
        super().setUp()
        self.reset_registry_to_live_defaults()

    def test_run_refuses_live_defaults_without_flag(self):
        """Default judges point at 8091-8093 -> refused before ANY network call
        or file write (FT-8 failure mode)."""
        buf = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["run", "http://127.0.0.1:18999/v1",
                               "--out-dir", tmp, "--run-name", "must-not-exist"])
            self.assertEqual(rc, 2)
            out = buf.getvalue()
            self.assertIn("LIVE-JUDGE INTERLOCK", out)
            for port in ("8091", "8092", "8093"):
                self.assertIn(port, out)
            self.assertIn("--live-judges-acknowledged", out)
            self.assertFalse((Path(tmp) / "must-not-exist").exists(),
                             "refusal must happen before any file write")

    def test_judges_health_refuses_without_flag(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["judges"])
        self.assertEqual(rc, 2)
        self.assertIn("LIVE-JUDGE INTERLOCK", buf.getvalue())
        self.assertIn("--live-judges-acknowledged", buf.getvalue())


class InterlockProceedsVsMock(JudgesRegistryIsolated):
    """The proceed paths: mock-pinned runs (with and without the flag)."""

    @classmethod
    def setUpClass(cls):
        cls.server, cls.thread = serve_in_thread(MOCK_PORT)

    @classmethod
    def tearDownClass(cls):
        stop(cls.server, cls.thread)

    def _run(self, extra):
        buf = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="interlock-test-") as tmp:
            with contextlib.redirect_stdout(buf):
                rc = cli.main(_mock_run_argv(tmp, extra))
            self.assertEqual(rc, 0, buf.getvalue())
            run_dir = Path(tmp) / "interlock-selftest"
            rows = [json.loads(l) for l in
                    (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            return buf.getvalue(), rows

    def test_run_proceeds_mock_pinned_without_flag(self):
        out, rows = self._run([])
        self.assertNotIn("LIVE-JUDGE INTERLOCK", out)
        manifest, verdicts = rows[0], [r for r in rows if r.get("record") == "verdict"]
        self.assertEqual(len(verdicts), 12)  # 4 items x 3 judges
        for j, ep in manifest["inputs"]["endpoints"]["judges"].items():
            self.assertEqual(ep["base_url"], f"http://127.0.0.1:{MOCK_PORT}")

    def test_run_proceeds_with_flag_vs_mock(self):
        """--live-judges-acknowledged present: mock-pinned run still proceeds
        (the flag gates live targets only; it does not disturb offline runs)."""
        out, _ = self._run(["--live-judges-acknowledged"])
        self.assertNotIn("LIVE-JUDGE INTERLOCK", out)

    def test_manifest_carries_section3_env_fields(self):
        """REVIEW-P2 D3: build_envelope outputs carry driver/ram_speed_mts/
        engine_sha and inputs.repos."""
        _, rows = self._run([])
        manifest = rows[0]
        env = manifest["environment"]
        self.assertEqual(env["driver"], "591.86")
        self.assertEqual(env["ram_speed_mts"], 3600)
        self.assertIn("engine_sha", env)
        self.assertIn("repos", manifest["inputs"])
        self.assertEqual(manifest["inputs"]["repos"], [])


if __name__ == "__main__":
    unittest.main()
