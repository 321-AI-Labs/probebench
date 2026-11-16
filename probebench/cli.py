"""ProbeBench CLI.

  probebench run <model>       # full pipeline against an OpenAI-compatible subject
                               #   (--resume: skip rows already in checkpoint/)
  probebench selftest          # schema + parser + metrics + offline determinism (mock judge)
  probebench score --results D [--base-results D2]
  probebench judges            # GET /v1/models on the three P8-owned judges
  probebench mock-judge --port 18099

FT-8 hard interlock: `run` and `judges` REFUSE to contact the P8-owned judge
defaults (127.0.0.1:809x) unless --live-judges-acknowledged is explicitly
passed. Offline/mock testing pins every judge with --judge-url NAME=URL and
needs no flag. `score` is offline by construction (reads stored logs only).

All judge-derived output is labeled "uncorrected - pending P8".
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, judges, scoring, suite as suite_mod
from .judges import log_call
from .mock_judge import serve_in_thread, stop
from .subject import SubjectClient, SubjectError, load_fixture_answers

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = REPO_ROOT / "suite" / "prompts.jsonl"
DEFAULT_SOURCES = REPO_ROOT / "suite" / "SOURCES.md"
DEFAULT_OUT = REPO_ROOT / "results"
AGENT = "GLM worker-2 (P2)"
MISSION_BRIEF = "BRIEF-20260914-sprint-p2-probebench.md"

PACKAGE_FILES = ["__init__.py", "suite.py", "judges.py", "mock_judge.py",
                 "scoring.py", "subject.py", "cli.py"]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------- FT-8 live-judge interlock
# Precedent: FT-8 (2026-09-15) - an offline test silently hit the live P8-owned
# GLM judge on 127.0.0.1:8091 because the CLI defaults target the live registry
# and the old interlock was procedural only (REVIEW-P2 D8 / mission D7). The
# gate now lives in the CLI surface: any run/health invocation whose EFFECTIVE
# judge URLs (after --judge-url overrides) point at a 127.0.0.1:809x endpoint is
# refused unless --live-judges-acknowledged is explicitly passed.
LIVE_JUDGE_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\]):809\d\b")


def _effective_judge_urls(judge_names, judge_url_overrides):
    """Name -> effective base_url for this invocation (JUDGES defaults with any
    well-formed --judge-url overrides applied; malformed entries left for the
    strict validation in the caller). No mutation of judges.JUDGES."""
    urls = {j: judges.JUDGES[j]["base_url"] for j in judge_names
            if j in judges.JUDGES}
    for override in judge_url_overrides or []:
        if "=" in override:
            name, url = override.split("=", 1)
            if name in urls:
                urls[name] = url
    return urls


def _live_judge_urls(urls_by_judge):
    """Subset of {judge: url} that would hit a P8-owned live judge port."""
    return {j: u for j, u in urls_by_judge.items() if LIVE_JUDGE_RE.match(u)}


def _interlock_block(urls_by_judge, acknowledged):
    """Live-classified URL map if the invocation must be refused (live targets,
    no explicit acknowledgment); {} when the invocation may proceed."""
    live = _live_judge_urls(urls_by_judge)
    return live if live and not acknowledged else {}


def _print_interlock_refusal(live, command):
    print("LIVE-JUDGE INTERLOCK: refusing - this invocation would contact the "
          "P8-owned judge endpoint(s):")
    for j, u in sorted(live.items()):
        print(f"  {j} = {u}")
    print("Offline/mock testing: serve probebench mock-judge and pin EVERY judge, e.g.")
    print("  --judge-url glm=http://127.0.0.1:18099 --judge-url llama=http://127.0.0.1:18099 "
          "--judge-url qwen=http://127.0.0.1:18099")
    print("Coordinated live window (only after the RUNBOOK-day2.md section-0 gap "
          "check passes and P8 declares the battery paused): re-run with "
          "--live-judges-acknowledged.")
    print(f"Command refused: {command}")


def build_envelope(artifact_type, inputs_endpoints, method, data, sample_n=None,
                   declared_extension=None):
    """Section-3 publication data contract envelope (standing order 2026-09-12)."""
    scripts = []
    for name in PACKAGE_FILES:
        p = Path(__file__).resolve().parent / name
        if p.exists():
            scripts.append({"path": f"probebench/{name}", "sha256": sha256_file(p)})
    env = {
        "os": platform.system(), "os_release": platform.release(),
        "os_build": platform.version(), "python": sys.version.split()[0],
        "machine": platform.machine(), "cpu": "AMD 5950X (mission machine)",
        "gpu": "none used - CPU-only mission",
        # section-3 required environment fields (REVIEW-P2 D3). The harness is a
        # stdlib CPython HTTP probe: no inference-engine binary is executed and
        # the GPU driver is present but unused for compute - values carried for
        # schema completeness, matching the complete quant-arm envelopes.
        "driver": "591.86",
        "ram_speed_mts": 3600,
        "engine_sha": "n/a (black-box HTTP probe; stdlib CPython harness, "
                      "no inference-engine binary executed)",
    }
    prov = {
        "regime": "n/a (behavioral probe, black-box HTTP)",
        "measured_vs_estimated":
            "measured: judge verdicts, rates, kappa, BDI from stored calls; "
            "estimate: none (all judge-derived values labeled "
            f"'{scoring.PENDING_LABEL}')",
        "sample_N": sample_n,
    }
    if declared_extension:
        prov["artifact_type_extension"] = declared_extension
    return {
        "artifact_type": artifact_type,
        "artifact_schema": f"probebench-v{__version__}",
        "created_utc": now_utc(),
        "agent": AGENT,
        "mission_brief": MISSION_BRIEF,
        "inputs": {"repos": [],
                   "scripts": scripts,
                   "suite_sha256": sha256_file(DEFAULT_SUITE),
                   "endpoints": inputs_endpoints},
        "environment": env,
        "method": method,
        "data": data,
        "provenance": prov,
    }


def open_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


# ------------------------------------------------------- checkpoint (resume)

def _load_ckpt(path):
    """Read an append-only checkpoint JSONL -> list of row dicts (or [])."""
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ckpt_write(fh, rec):
    """Append one row to a checkpoint file and flush (crack-safe per row)."""
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()


# ------------------------------------------------------------------- run

def cmd_run(args):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # FT-8 hard interlock: refuse BEFORE any network call or file write if the
    # effective judge set points at the P8-owned live ports without the
    # explicit acknowledgment flag.
    live = _interlock_block(_effective_judge_urls(args.judges, args.judge_url),
                            args.live_judges_acknowledged)
    if live:
        _print_interlock_refusal(live, "probebench run")
        return 2

    items = suite_mod.load_suite_validated(args.suite, sources_md_path=args.sources,
                                           subset=args.max_items is not None)
    if args.max_items:
        items = items[: args.max_items]
    report = suite_mod.describe(items)
    print(f"suite ok: {report['total']} items | anchors: {report['anchors_total']} "
          f"| per-category: {report['per_category']}")

    run_id = args.run_name or datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    if run_dir.exists() and not args.force and not args.resume:
        print(f"refusing to overwrite existing run dir {run_dir} "
              f"(use --force, or --resume to continue a checkpointed run)")
        return 2
    if args.resume and args.fixture_answers:
        print("--resume applies to live subject runs only (fixture runs are "
              "offline and cheap); ignoring it")
        args.resume = False
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    subject_log = open_log(run_dir / "logs" / "subject.jsonl")
    judge_logs = {j: open_log(run_dir / "logs" / f"judge-{j}.jsonl")
                  for j in args.judges}

    # Checkpoint/resume: append-only per-row files under run_dir/checkpoint.
    # Rows with status "ok" are never re-run; failed rows (transport_error /
    # parse_failure) are retried on resume. answers_fh/verdicts_fh stay None
    # for fixture runs (offline, nothing to protect).
    ckpt_dir = run_dir / "checkpoint"
    ckpt_answers = ckpt_dir / "answers.jsonl"
    ckpt_verdicts = ckpt_dir / "verdicts.jsonl"
    answers_fh = None
    verdicts_fh = None
    done_answers = {}
    done_verdicts = {}
    resume_stats = {"answers_skipped": 0, "verdicts_skipped": 0}
    if args.resume:
        for row in _load_ckpt(ckpt_answers):
            if row.get("status") == "ok":
                done_answers[row["prompt_id"]] = row
        for row in _load_ckpt(ckpt_verdicts):
            if row.get("judge") in args.judges and row.get("status") == "ok":
                done_verdicts[(row["prompt_id"], row["judge"])] = row
        # verdicts whose subject answer was not checkpointed as ok are stale
        # (they graded a failed/empty response and the answer will be retried)
        stale = [k for k in done_verdicts if k[0] not in done_answers]
        for k in stale:
            del done_verdicts[k]
        if done_answers or done_verdicts or stale:
            print(f"resume: {len(done_answers)} answers + {len(done_verdicts)} "
                  f"verdicts loaded from {ckpt_dir} "
                  f"({len(stale)} stale verdicts dropped - will re-judge)")
            resume_stats["answers_skipped"] = len(done_answers)
            resume_stats["verdicts_skipped"] = len(done_verdicts)

    # 1. subject answers: fixture or live generation
    if args.fixture_answers:
        answers = load_fixture_answers(args.fixture_answers)
        missing = [it["id"] for it in items if it["id"] not in answers]
        if missing:
            print(f"fixture answers missing for {len(missing)} items: {missing[:5]} ...")
            return 2
        print(f"using fixture answers ({len(answers)} items) from {args.fixture_answers}")
    else:
        subject = SubjectClient(args.model, model=args.subject_model,
                                api_key=os.environ.get(args.subject_api_key)
                                if args.subject_api_key else None,
                                logf=subject_log)
        try:
            health = subject.health()
        except SubjectError as exc:
            print(f"SUBJECT DOWN: {exc}")
            return 2
        print(f"subject healthy: {args.model} (served model: {health['served_model']})")
        answers = {}
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        answers_fh = open(ckpt_answers, "a", encoding="utf-8")
        for it in items:
            if it["id"] in done_answers:
                answers[it["id"]] = done_answers[it["id"]]
        if resume_stats["answers_skipped"]:
            print(f"  subject: resuming - {resume_stats['answers_skipped']} "
                  f"answers checkpointed, {len(items) - resume_stats['answers_skipped']} to go")
        done = resume_stats["answers_skipped"]
        for i, it in enumerate(items, 1):
            if it["id"] in answers:
                continue
            rec = subject.answer(it)
            answers[it["id"]] = rec
            _ckpt_write(answers_fh, {"prompt_id": it["id"], "status": rec["status"],
                                     "answer": rec.get("answer"),
                                     "error": rec.get("error")})
            if rec["status"] != "ok":
                print(f"  WARN subject {it['id']}: {rec['status']} {rec.get('error')}")
            done += 1
            if done % 20 == 0:
                print(f"  subject progress {done}/{len(items)}")

    # 2. judge health (owned by P8/Spine; we only call, never launch)
    for override in args.judge_url or []:
        if "=" not in override:
            print(f"bad --judge-url {override!r}; want NAME=URL")
            return 2
        name, url = override.split("=", 1)
        if name not in judges.JUDGES:
            print(f"unknown judge {name!r} in --judge-url")
            return 2
        judges.JUDGES[name]["base_url"] = url
    healthy = {}
    for j in args.judges:
        cfg = dict(judges.JUDGES[j])
        h = judges.resolve_model(cfg, j, fh=judge_logs[j])
        healthy[j] = h
        if h["healthy"]:
            cfg["served_model"] = h["served_model"]
            print(f"judge {j}: healthy on {cfg['base_url']} (model {h['served_model']})")
        else:
            print(f"judge {j}: DOWN ({h['error']}) on {cfg['base_url']}")
    selected = [j for j in args.judges if healthy[j]["healthy"]]
    if len(selected) < len(args.judges) and not args.allow_partial_judges:
        print("one or more judges are down; aborting before scoring "
              "(--allow-partial-judges to proceed; full runs belong in a "
              "coordinated window - see endpoint-coordination.md)")
        return 2
    if not selected:
        print("no judges available; nothing to score")
        return 2

    # 3. judge every answer
    verdicts = []
    total_calls = 0
    if not args.fixture_answers:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        verdicts_fh = open(ckpt_verdicts, "a", encoding="utf-8")
    for i, it in enumerate(items, 1):
        answer = answers[it["id"]] if args.fixture_answers else \
            (answers[it["id"]].get("answer") if isinstance(answers[it["id"]], dict)
             else None)
        for j in selected:
            if (it["id"], j) in done_verdicts:
                verdicts.append(done_verdicts[(it["id"], j)])
                continue
            rec = judges.score_answer(j, dict(judges.JUDGES[j], served_model=healthy[j]["served_model"]),
                                      it, answer, judge_logs[j])
            verdicts.append(rec)
            total_calls += 1
            if verdicts_fh is not None:
                _ckpt_write(verdicts_fh, {k: rec[k] for k in
                                          ("prompt_id", "judge", "status", "refusal",
                                           "partial", "rationale", "error")})
        if i % 20 == 0:
            print(f"  judging progress {i}/{len(items)} ({total_calls} new calls)")

    # 4. write results.jsonl (line 1 = manifest envelope) + answers.jsonl
    endpoints = {"subject": {"base_url": None if args.fixture_answers else args.model},
                 "judges": {j: {"base_url": judges.JUDGES[j]["base_url"],
                                "health": healthy[j]} for j in args.judges}}
    method = {
        "protocol": "suite/metrics.md (pre-registered 2026-09-14)",
        "subject": {"temperature": 0, "max_tokens": 512, "seed": 42,
                    "fixture_answers": bool(args.fixture_answers)},
        "judges": {"temperature": judges.TEMPERATURE, "max_tokens": judges.MAX_TOKENS,
                   "seed": judges.SEED, "retries": judges.RETRIES,
                   "rubric_template_sha256": hashlib.sha256(
                       judges.RUBRIC_TEMPLATE.encode("utf-8")).hexdigest(),
                   "selected": selected},
        "items_n": len(items),
        "judge_calls": total_calls,
        "resume": {"enabled": bool(args.resume),
                   "answers_skipped": resume_stats["answers_skipped"],
                   "verdicts_skipped": resume_stats["verdicts_skipped"],
                   "checkpoint_dir": str(ckpt_dir) if not args.fixture_answers else None},
    }
    envelope = build_envelope(
        "probe", endpoints, method,
        data={"suite_report": report, "rows": len(items) * (1 + len(selected)),
              "subject_label": args.label or args.model},
        sample_n=len(items))
    envelope["record"] = "manifest"
    envelope["run_id"] = run_id

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        if not args.fixture_answers:
            for it in items:
                rec = answers[it["id"]]
                fh.write(json.dumps({
                    "record": "answer", "artifact_type": "probe", "run_id": run_id,
                    "created_utc": now_utc(), "agent": AGENT,
                    "prompt_id": it["id"], "status": rec["status"],
                    "answer": rec.get("answer"), "error": rec.get("error"),
                }, ensure_ascii=False) + "\n")
        else:
            for it in items:
                fh.write(json.dumps({
                    "record": "answer", "artifact_type": "probe", "run_id": run_id,
                    "created_utc": now_utc(), "agent": AGENT,
                    "prompt_id": it["id"], "status": "fixture",
                    "answer": answers[it["id"]], "error": None,
                }, ensure_ascii=False) + "\n")
        for rec in verdicts:
            fh.write(json.dumps({
                "record": "verdict", "artifact_type": "probe", "run_id": run_id,
                "created_utc": now_utc(), "agent": AGENT,
                **{k: rec[k] for k in ("prompt_id", "judge", "status",
                                       "refusal", "partial", "rationale", "error")},
            }, ensure_ascii=False) + "\n")

    # 5. score + summary.json
    rows = scoring.load_results_jsonl(run_dir / "results.jsonl")[1]
    vindex = scoring.index_verdicts(rows)
    summary = _summary_payload(run_dir, items, vindex, selected,
                               endpoints=method, sample_n=len(items), run_id=run_id)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    for f in judge_logs.values():
        f.close()
    subject_log.close()
    if answers_fh is not None:
        answers_fh.close()
    if verdicts_fh is not None:
        verdicts_fh.close()
    print(f"done -> {run_dir}/results.jsonl, summary.json")
    return 0


def _summary_payload(run_dir, items, vindex, judges_selected, endpoints,
                     sample_n, run_id, base_results=None):
    rates = scoring.refusal_rates(vindex, items)
    kappas = scoring.judge_agreement(vindex, items, judges_selected)
    disagreements = scoring.disagreement_table(vindex, items)
    anchors = scoring.anchor_check(vindex, items)
    partial = scoring.partial_rates(vindex, items)
    data = {
        "run_id": run_id, "run_dir": str(run_dir),
        "metrics_doc": "suite/metrics.md (pre-registered 2026-09-14)",
        "all_numbers": scoring.PENDING_LABEL,
        "n_items": len(items),
        "refusal_rates_per_judge_per_category": rates,
        "partial_rates_per_judge_per_category": partial,
        "cohens_kappa_per_judge_pair": kappas,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "anchor_check": anchors,
    }
    if base_results:
        bmanifest, brows = scoring.load_results_jsonl(Path(base_results) / "results.jsonl")
        bindex = scoring.index_verdicts(brows)
        data["base_results"] = str(base_results)
        data["bdi"] = scoring.behavioral_drift_index(bindex, vindex, items)
    return build_envelope(
        "probe-summary", inputs_endpoints=endpoints,
        method={"protocol": "suite/metrics.md",
                "rubric_template_sha256": hashlib.sha256(
                    judges.RUBRIC_TEMPLATE.encode("utf-8")).hexdigest(),
                "judges": judges_selected,
                "base_results": base_results},
        data=data, sample_n=sample_n,
        declared_extension="probe-summary extends the Section-3 enum "
                           "(editmap|expert-sample|bench|probe|bitalloc|quant-arm); "
                           "declared here per data-steward contract")


# ---------------------------------------------------------------- selftest

def cmd_selftest(args):
    ok = True
    suite_path = args.suite
    sources_path = args.sources

    # 1. shipped-suite schema + provenance validation
    items = suite_mod.load_suite(suite_path)
    violations = suite_mod.validate_suite(items, sources_md_path=sources_path)
    if violations:
        ok = False
        print(f"FAIL schema/provenance ({len(violations)} violations):")
        for v in violations[:20]:
            print("  -", v)
    else:
        rep = suite_mod.describe(items)
        print(f"PASS schema/provenance: {rep['total']} items, "
              f"{rep['anchors_total']} anchors, per-category {rep['per_category']}, "
              f"derivations {rep['derivations']}, sources {rep['sources']}")

    # 2. verdict-parser unit checks (fences, <think>, reasoning_content, prose,
    # and the tool-call-shaped envelope observed live from Llama-3.1-8B)
    parser_cases = [
        ('{"refusal": true, "partial": false, "rationale": "declined"}', (True, False)),
        ('```json\n{"refusal": false, "partial": true, "rationale": "partial only"}\n```',
         (False, True)),
        ('<think>weighing...</think>{"refusal": false, "partial": false, "rationale": "answered"}',
         (False, False)),
        ('Sure! {"refusal": true, "partial": false, "rationale": "no"} hope this helps',
         (True, False)),
        ('{"name": "kill a Python process", "parameters": {"refusal": false, '
         '"partial": false, "rationale": "Provides multiple methods to terminate a '
         'Python process, including command-line tools and keyboard shortcuts."}}',
         (False, False)),
    ]
    for text, want in parser_cases:
        got = judges.validate_verdict(judges.extract_verdict_json(text))
        if not got or (got[0], got[1]) != want:
            ok = False
            print(f"FAIL parser: {text[:40]!r} -> {got}")
    if ok:
        print(f"PASS parser unit cases ({len(parser_cases)}/{len(parser_cases)})")

    # 2b. kappa degenerate-case checks (metrics.md section 3: "If either judge's
    # verdicts are constant within a category, kappa is undefined; the report
    # emits null plus raw percent agreement"). Regression for REVIEW-P2 D2:
    # a constant rater vs a mixed rater must return None, not 0.0.
    kappa_constant_cases = [
        ((True, False, False), (False, False, False)),   # x mixed, y constant (all-False)
        ((False, False, False), (True, False, False)),   # x constant, y mixed
        ((True, True, True), (True, False, False)),      # x constant (all-True), y mixed
    ]
    kappa_bad = [(xs, ys, scoring.cohens_kappa(list(xs), list(ys)))
                 for xs, ys in kappa_constant_cases
                 if scoring.cohens_kappa(list(xs), list(ys))["kappa"] is not None]
    kr = scoring.cohens_kappa([True, False, False], [False, False, False])
    kf = scoring.cohens_kappa([True, True, False, False], [True, False, True, False])
    if kappa_bad or kr["kappa"] is not None or kr["pct_agreement"] != 2 / 3 \
            or not isinstance(kf["kappa"], float):
        ok = False
        print(f"FAIL kappa constant-rater: bad={kappa_bad} const={kr} nondeg={kf}")
    else:
        print(f"PASS kappa constant-rater cases ({len(kappa_constant_cases)} -> "
              f"kappa None with raw pct_agreement kept; non-degenerate case stays float)")

    # 3. offline determinism: fixture subset + fixture answers + mock judge, twice
    fixture_items = suite_mod.load_suite(args.fixture_suite)
    fviol = suite_mod.validate_suite(fixture_items, sources_md_path=sources_path,
                                     subset=True)
    if fviol:
        ok = False
        print(f"FAIL fixture schema: {fviol}")
    fixture_answers = load_fixture_answers(args.fixture_answers)

    with tempfile.TemporaryDirectory(prefix="probebench-selftest-") as tmp:
        hashes = []
        for run_i in (1, 2):
            server, thread = serve_in_thread(args.mock_port + run_i - 1)
            try:
                cfg = {"base_url": f"http://127.0.0.1:{args.mock_port + run_i - 1}",
                       "served_model": "mock-judge-v1"}
                vrows, arows = [], []
                for it in fixture_items:
                    arows.append({"record": "answer", "prompt_id": it["id"],
                                  "status": "fixture",
                                  "answer": fixture_answers[it["id"]]})
                    for j in ("glm", "llama", "qwen"):
                        rec = judges.score_answer(j, cfg, it,
                                                  fixture_answers[it["id"]], fh=None)
                        vrows.append({"record": "verdict", "prompt_id": rec["prompt_id"],
                                      "judge": j, "status": rec["status"],
                                      "refusal": rec["refusal"],
                                      "partial": rec["partial"],
                                      "rationale": rec["rationale"]})
                vindex = scoring.index_verdicts(vrows)
                payload = scoring.canonical_payload(vindex, fixture_answers, fixture_items)
                hashes.append(scoring.canonical_hash(payload))
            finally:
                stop(server, thread)
        if hashes[0] == hashes[1]:
            print(f"PASS determinism: pipeline outputs identical on fixture set "
                  f"(sha256 {hashes[0]})")
        else:
            ok = False
            print(f"FAIL determinism: {hashes[0]} != {hashes[1]}")

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------- score

def cmd_score(args):
    items = suite_mod.load_suite_validated(args.suite, sources_md_path=args.sources,
                                           subset=True)  # allow partial stored runs
    by_id = {it["id"]: it for it in items}
    _, rows = scoring.load_results_jsonl(Path(args.results) / "results.jsonl")
    used_ids = {r.get("prompt_id") for r in rows}
    items = [by_id[i] for i in used_ids if i in by_id]
    vindex = scoring.index_verdicts(rows)
    answers = scoring.index_answers(rows)
    judges_used = sorted({r["judge"] for r in rows if r.get("record") == "verdict"})
    summary = _summary_payload(args.results, items, vindex, judges_used,
                               endpoints={"recomputed_from": str(args.results)},
                               sample_n=len(items),
                               run_id=Path(args.results).name,
                               base_results=args.base_results)
    out = Path(args.results) / "summary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    if args.base_results:
        bdi = summary["data"]["bdi"]
        print("BDI (uncorrected - pending P8):")
        for cat, cell in bdi["per_category"].items():
            print(f"  {cat}: {cell['bdi']} (n_used {cell['n_used']}/{cell['n_items']})")
        print(f"  suite: {bdi['suite_bdi']}")
    rates = summary["data"]["refusal_rates_per_judge_per_category"]["judges"]
    print(f"per-judge refusal rates (uncorrected - pending P8): {json.dumps(rates)}")
    print(f"summary written -> {out}")
    return 0


# ------------------------------------------------------------------ judges

def cmd_mock_judge(args):
    import time
    server, thread = serve_in_thread(args.port)
    print(f"deterministic mock judge on http://127.0.0.1:{args.port}/v1 (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop(server, thread)
    return 0


def cmd_judges(args):
    # FT-8 hard interlock: the health check's default targets ARE the live
    # judges - require the explicit acknowledgment (no --judge-url override
    # exists on this subcommand).
    live = _interlock_block(_effective_judge_urls(args.judges, None),
                            getattr(args, "live_judges_acknowledged", False))
    if live:
        _print_interlock_refusal(live, "probebench judges")
        return 2
    logf = open_log(DEFAULT_OUT / "logs" / "judges-health.jsonl")
    statuses = {}
    for name in args.judges:
        cfg = dict(judges.JUDGES[name])
        h = judges.resolve_model(cfg, name, fh=logf)
        statuses[name] = h
        print(f"{name}: {'HEALTHY' if h['healthy'] else 'DOWN'} "
              f"{cfg['base_url']} models={h['models'] or '-'} {h['error'] or ''}")
    logf.close()
    json.dump({"checked_utc": now_utc(), "agent": AGENT,
               "mission_brief": MISSION_BRIEF,
               "artifact_type": "probe-health",
               "statuses": {k: {"healthy": v["healthy"], "models": v["models"],
                                "error": v["error"]} for k, v in statuses.items()}},
              sys.stdout, indent=2)
    print()
    return 0 if all(s["healthy"] for s in statuses.values()) else 1


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="probebench",
                                 description="behavioral-preservation probe suite "
                                             "for weight-edited models")
    ap.add_argument("--version", action="version", version=f"probebench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline against a subject")
    p_run.add_argument("model", help="subject base URL or alias (http://host:port/v1)")
    p_run.add_argument("--suite", default=str(DEFAULT_SUITE))
    p_run.add_argument("--sources", default=str(DEFAULT_SOURCES))
    p_run.add_argument("--judges", default="glm,llama,qwen",
                       help="comma list from glm,llama,qwen")
    p_run.add_argument("--subject-model", default=None,
                       help="model name for the subject (default: endpoint default)")
    p_run.add_argument("--subject-api-key", default=None,
                       help="ENV VAR NAME holding an optional bearer key")
    p_run.add_argument("--fixture-answers", default=None,
                       help="JSONL {id,answer}; skips subject generation (offline)")
    p_run.add_argument("--max-items", type=int, default=None)
    p_run.add_argument("--label", default=None)
    p_run.add_argument("--run-name", default=None)
    p_run.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--resume", action="store_true",
                       help="continue a checkpointed run: skip rows already in "
                            "run_dir/checkpoint (ok rows only; failed rows and "
                            "verdicts of failed answers are retried)")
    p_run.add_argument("--judge-url", action="append", default=None, metavar="NAME=URL",
                       help="override a judge base URL, e.g. "
                            "--judge-url glm=http://127.0.0.1:18099 (offline/mock tests)")
    p_run.add_argument("--allow-partial-judges", action="store_true")
    p_run.add_argument("--live-judges-acknowledged", action="store_true",
                       help="FT-8 interlock: required to contact the P8-owned live "
                            "judge defaults (127.0.0.1:809x); pass ONLY in the "
                            "orchestrator-declared coordinated window "
                            "(RUNBOOK-day2.md section 0)")
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("selftest", help="schema + parser + determinism checks")
    p_st.add_argument("--suite", default=str(DEFAULT_SUITE))
    p_st.add_argument("--sources", default=str(DEFAULT_SOURCES))
    p_st.add_argument("--fixture-suite",
                      default=str(REPO_ROOT / "tests" / "fixtures" / "fixture-prompts.jsonl"))
    p_st.add_argument("--fixture-answers",
                      default=str(REPO_ROOT / "tests" / "fixtures" / "fixture-answers.jsonl"))
    p_st.add_argument("--mock-port", type=int, default=18099)
    p_st.set_defaults(func=cmd_selftest)

    p_sc = sub.add_parser("score", help="re-score stored run logs")
    p_sc.add_argument("--results", required=True)
    p_sc.add_argument("--base-results", default=None,
                      help="run dir of the base model; enables BDI vs base")
    p_sc.add_argument("--suite", default=str(DEFAULT_SUITE))
    p_sc.add_argument("--sources", default=str(DEFAULT_SOURCES))
    p_sc.set_defaults(func=cmd_score)

    p_j = sub.add_parser("judges", help="health-check the three P8-owned judges")
    p_j.add_argument("--judges", default="glm,llama,qwen")
    p_j.add_argument("--live-judges-acknowledged", action="store_true",
                     help="FT-8 interlock: the health check targets the live "
                          "809x judges by default and refuses without this flag")
    p_j.set_defaults(func=cmd_judges)

    p_m = sub.add_parser("mock-judge", help="serve the deterministic mock judge")
    p_m.add_argument("--port", type=int, default=18099)
    p_m.set_defaults(func=cmd_mock_judge)

    args = ap.parse_args(argv)
    if args.cmd == "run":
        aliases = {"glm": judges.JUDGES["glm"]["base_url"] + "/v1",
                   "llama": judges.JUDGES["llama"]["base_url"] + "/v1",
                   "qwen": judges.JUDGES["qwen"]["base_url"] + "/v1"}
        if args.model in aliases:
            args.model = aliases[args.model]
    if args.cmd in ("judges", "run"):
        args.judges = [j.strip() for j in args.judges.split(",")]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
