"""Run the question set through the real pipeline and write a results report.

Usage:
    python -m eval.run_eval                 # run everything
    python -m eval.run_eval --set test      # only the 6 untouched test questions
    python -m eval.run_eval --set dev       # dev questions: use these to tune thresholds
    python -m eval.run_eval --ids T1,T2,D8  # run selected cases only

Outputs eval/results.md (human readable, failures included) and eval/results.json.
Failures are written to the report on purpose: the brief asks for honest results.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from voicerag.bootstrap import build_app
from voicerag.config import load_settings
from voicerag.models import FinalAnswer

HERE = Path(__file__).resolve().parent


@dataclass
class TurnResult:
    """Outcome of one question of one case."""

    case_id: str
    set_name: str
    question: str
    answer: str
    found: bool
    pages: list[int]
    seconds: float
    cost_inr: float
    best_score: float | None
    llm_calls: int
    retrieved_context_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)       # empty = passed

    @property
    def passed(self) -> bool:
        return not self.problems


def check_turn(spec: dict, result: FinalAnswer) -> list[str]:
    """Compare one answer with its expectations; return a list of problems."""
    problems: list[str] = []
    text = result.answer.lower()
    if "expect_found" in spec and result.found != spec["expect_found"]:
        problems.append(f"expected found={spec['expect_found']}, got found={result.found}")
    if result.found:
        for group in spec.get("must_contain", []):
            if not any(option.lower() in text for option in group):
                problems.append(f"answer lacks any of {group}")
        for banned in spec.get("must_not_contain", []):
            if banned.lower() in text:
                problems.append(f"answer contains {banned!r}")
        missing_pages = [p for p in spec.get("expect_pages_include", []) if p not in result.pages]
        if missing_pages:
            problems.append(f"expected citations to include pages {missing_pages}, got {result.pages}")
    return problems


def run(selected: str, case_ids: set[str] | None = None) -> list[TurnResult]:
    """Run every selected case in its own fresh conversation."""
    cases = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["cases"]
    if case_ids is not None:
        known_ids = {case["id"] for case in cases}
        unknown_ids = case_ids - known_ids
        if unknown_ids:
            raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown_ids))}")
    app = build_app(load_settings())
    results: list[TurnResult] = []
    for case in cases:
        if selected != "all" and case["set"] != selected:
            continue
        if case_ids is not None and case["id"] not in case_ids:
            continue
        session = app.pipeline.new_session()                 # follow-ups stay inside one case
        for spec in case["turns"]:
            answer = app.pipeline.ask(spec["q"], session)
            retrieved_context_ids = list(
                dict.fromkeys(
                    f"page:{sc.chunk.page}"
                    for sc in sorted(answer.sources, key=lambda source: source.score, reverse=True)
                )
            )
            results.append(
                TurnResult(
                    case_id=case["id"], set_name=case["set"], question=spec["q"], answer=answer.answer,
                    found=answer.found, pages=answer.pages, seconds=answer.metrics.total_seconds,
                    cost_inr=answer.metrics.cost_inr, best_score=answer.best_score,
                    llm_calls=len(answer.metrics.calls), retrieved_context_ids=retrieved_context_ids,
                    problems=check_turn(spec, answer),
                )
            )
            print(f"{'PASS' if results[-1].passed else 'FAIL'}  {case['id']:4} {spec['q']}")
    return results


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (small samples; no interpolation needed)."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))] if ordered else 0.0


def write_report(results: list[TurnResult]) -> None:
    """Write results.md and results.json next to this script."""
    (HERE / "results.json").write_text(
        json.dumps([r.__dict__ | {"passed": r.passed} for r in results], indent=1), encoding="utf-8"
    )
    lines = ["# Evaluation results", ""]
    for set_name in ("test", "dev"):
        rows = [r for r in results if r.set_name == set_name]
        if not rows:
            continue
        passed = sum(r.passed for r in rows)
        times = [r.seconds for r in rows]
        lines += [
            f"## {set_name} set: {passed}/{len(rows)} passed", "",
            f"- latency: median {statistics.median(times):.2f} s, p90 {percentile(times, 90):.2f} s, max {max(times):.2f} s",
            f"- cost per question: mean ₹{statistics.mean(r.cost_inr for r in rows):.4f}, max ₹{max(r.cost_inr for r in rows):.4f}",
            "", "| id | question | found | pages | s | ₹ | best score | result |", "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            score = "-" if r.best_score is None else f"{r.best_score:.2f}"
            verdict = "PASS" if r.passed else "FAIL: " + "; ".join(r.problems)
            lines.append(
                f"| {r.case_id} | {r.question} | {r.found} | {r.pages} | {r.seconds:.2f} | {r.cost_inr:.4f} | {score} | {verdict} |"
            )
        lines += ["", "Answers:", ""]
        lines += [f"- **{r.case_id}**: {r.answer}" for r in rows] + [""]

    # Threshold help: how well does the best rerank score separate found from not-found?
    scored = [r for r in results if r.best_score is not None]
    if scored:
        yes = [r.best_score for r in scored if r.found]
        no = [r.best_score for r in scored if not r.found]
        lines += ["## Rerank-score separation (for choosing RERANK_MIN_SCORE)", ""]
        if yes:
            lines.append(f"- answered questions: min {min(yes):.3f}, median {statistics.median(yes):.3f}")
        if no:
            lines.append(f"- not-found questions: max {max(no):.3f}, median {statistics.median(no):.3f}")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {HERE / 'results.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--ids", help="comma-separated case IDs to run, e.g. T1,T2,D8")
    args = parser.parse_args()
    case_ids = None if args.ids is None else {case_id.strip() for case_id in args.ids.split(",") if case_id.strip()}
    if args.ids is not None and not case_ids:
        parser.error("--ids must contain at least one case ID")
    results = run(args.set, case_ids)
    write_report(results)


if __name__ == "__main__":
    main()
