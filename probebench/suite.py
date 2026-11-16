"""Suite loader + validator.

Schema for suite/prompts.jsonl rows:
  {"id": "P2B-001", "category": <cat>, "text": str,
   "rubric": str,
   "provenance": {"source_id": str, "derivation": str, "note": str},
   "oracle": bool, "oracle_note": str (required iff oracle true)}

Validation is strict by design: no citation = prompt rejected; balance and
anchor minimums enforced for the shipped suite (relaxable for fixture
subsets via subset=True).
"""

import hashlib
import json
import re
from pathlib import Path

CATEGORIES = (
    "borderline-compliance",
    "benign-adjacent",
    "dual-use-framing",
    "jailbreak-adjacent",
)
PER_CATEGORY = 30
TOTAL = 120
MIN_ANCHORS = 10
ANCHOR_CATEGORIES = ("benign-adjacent", "dual-use-framing")
SOURCE_IDS = {"advbench", "xstest", "orbench", "jailbreakbench", "self"}
DERIVATIONS = {"verbatim", "paraphrase", "self-authored"}
ID_RE = re.compile(r"^P2B-(?:0\d\d|1[01]\d|120)$")

# SOURCES.md must contain a "## S-<id>" heading for every source id in use.
SOURCES_HEADER_RE = re.compile(r"^##\s+S-([a-z]+)\b", re.MULTILINE)


def suite_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_suite(path):
    """Load prompts.jsonl -> list[dict]. Raises ValueError on malformed JSON."""
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return items


def parse_source_ids(sources_md_path):
    text = Path(sources_md_path).read_text(encoding="utf-8")
    return set(SOURCES_HEADER_RE.findall(text))


def validate_suite(items, sources_md_path=None, subset=False):
    """Return a list of violation strings (empty list = valid).

    subset=True relaxes the 120-total / 30-per-category / 10-anchor balance
    checks (used by selftest fixtures) while keeping per-row schema,
    provenance-resolution and anchor-placement checks strict.
    """
    violations = []
    seen = set()
    cat_counts = {c: 0 for c in CATEGORIES}
    anchor_count = 0

    available_sources = None
    if sources_md_path is not None:
        available_sources = parse_source_ids(sources_md_path)

    for idx, it in enumerate(items):
        where = f"item[{idx}] {it.get('id', '<no-id>')}"

        pid = it.get("id")
        if not isinstance(pid, str) or not ID_RE.match(pid or ""):
            violations.append(f"{where}: bad id (want P2B-001..P2B-120)")
        if pid in seen:
            violations.append(f"{where}: duplicate id")
        seen.add(pid)

        cat = it.get("category")
        if cat not in CATEGORIES:
            violations.append(f"{where}: unknown category {cat!r}")
        else:
            cat_counts[cat] += 1

        text = it.get("text")
        if not isinstance(text, str) or not (10 <= len(text) <= 1200):
            violations.append(f"{where}: text missing or out of length bounds")

        rubric = it.get("rubric")
        if not isinstance(rubric, str) or len(rubric) < 60:
            violations.append(f"{where}: rubric missing or too short (<60 chars); "
                              "a judge-agnostic rubric is mandatory per prompt")

        prov = it.get("provenance")
        if not isinstance(prov, dict):
            violations.append(f"{where}: provenance missing (no citation = prompt rejected)")
        else:
            sid = prov.get("source_id")
            deriv = prov.get("derivation")
            note = prov.get("note")
            if sid not in SOURCE_IDS:
                violations.append(f"{where}: provenance.source_id {sid!r} not a known source")
            elif available_sources is not None and sid not in available_sources:
                violations.append(f"{where}: source_id {sid!r} has no '## S-{sid}' "
                                  f"section in {sources_md_path}")
            if deriv not in DERIVATIONS:
                violations.append(f"{where}: derivation {deriv!r} not in {sorted(DERIVATIONS)}")
            if not isinstance(note, str) or len(note) < 20:
                violations.append(f"{where}: provenance.note missing or too short")

        oracle = it.get("oracle")
        if not isinstance(oracle, bool):
            violations.append(f"{where}: oracle must be bool")
            continue
        note_ = it.get("oracle_note")
        if oracle:
            if cat not in ANCHOR_CATEGORIES:
                violations.append(f"{where}: anchor outside {ANCHOR_CATEGORIES}")
            if not isinstance(note_, str) or len(note_) < 40:
                violations.append(f"{where}: anchor requires oracle_note (>=40 chars)")
            anchor_count += 1
        elif note_ is not None:
            violations.append(f"{where}: oracle_note present without oracle=true")

    if not subset:
        if len(items) != TOTAL:
            violations.append(f"suite has {len(items)} items, want {TOTAL}")
        for c in CATEGORIES:
            if cat_counts[c] != PER_CATEGORY:
                violations.append(f"category {c}: {cat_counts[c]} items, want {PER_CATEGORY}")
        if anchor_count < MIN_ANCHORS:
            violations.append(f"{anchor_count} oracle anchors, want >= {MIN_ANCHORS}")

    return violations


def load_suite_validated(path, sources_md_path=None, subset=False):
    items = load_suite(path)
    violations = validate_suite(items, sources_md_path=sources_md_path, subset=subset)
    if violations:
        raise ValueError("suite validation failed:\n  - " + "\n  - ".join(violations))
    return items


def describe(items):
    """Count report: per-category sizes, anchor counts, derivations."""
    from collections import Counter
    cats = Counter(it["category"] for it in items)
    anchors = Counter(it["category"] for it in items if it.get("oracle"))
    derivs = Counter(it["provenance"]["derivation"] for it in items)
    srcs = Counter(it["provenance"]["source_id"] for it in items)
    return {
        "total": len(items),
        "per_category": dict(cats),
        "anchors_per_category": dict(anchors),
        "anchors_total": sum(anchors.values()),
        "derivations": dict(derivs),
        "sources": dict(srcs),
    }
