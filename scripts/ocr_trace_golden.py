#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> int:
    manifest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/golden/ocr/manifest.tsv")
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    failures = []
    checked = 0
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_no, raw in enumerate(manifest, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            cols = raw.split("\t")
            if len(cols) != 3:
                failures.append(f"{manifest_path}:{line_no}: expected 3 columns: id, trace_json, expected_json")
                continue
            case_id, trace_path, expected_path = cols
            checked += 1
            failures.extend(check_case(case_id, Path(trace_path), Path(expected_path)))

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        print(f"ocr trace golden failed: {len(failures)} issue(s), {checked} case(s)", file=sys.stderr)
        return 1

    print(f"ocr trace golden passed: {checked} case(s)")
    return 0


def check_case(case_id: str, trace_path: Path, expected_path: Path) -> list[str]:
    failures: list[str] = []
    if not trace_path.is_file():
        return [f"{case_id}: trace json not found: {trace_path}"]
    if not expected_path.is_file():
        return [f"{case_id}: expected json not found: {expected_path}"]

    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{case_id}: invalid trace json {trace_path}: {exc}"]
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{case_id}: invalid expected json {expected_path}: {exc}"]

    lines = trace_lines(trace)
    candidates = trace_candidates(trace)
    full_text = trace_text(trace, lines)
    summary = trace.get("summary", {})
    failures.extend(check_count(case_id, "line_count", len(lines), expected.get("line_count")))
    failures.extend(check_count(case_id, "region_count", len(trace.get("regions", [])), expected.get("region_count")))
    failures.extend(check_count(case_id, "candidate_count", len(candidates), expected.get("candidate_count")))
    failures.extend(check_float(case_id, "confidence", float(summary.get("confidence", 0.0)), expected.get("confidence")))
    failures.extend(check_float(case_id, "avg_line_confidence", average_field(lines, "confidence"), expected.get("avg_line_confidence")))
    failures.extend(check_float(case_id, "avg_line_margin", average_field(lines, "avg_margin"), expected.get("avg_line_margin")))
    failures.extend(check_float(case_id, "min_line_margin", min_field(lines, "min_margin"), expected.get("min_line_margin")))
    failures.extend(check_count(case_id, "low_margin_line_count", low_margin_line_count(lines, expected), expected.get("low_margin_line_count")))
    if "selected_source" in expected and summary.get("selected_source") != expected["selected_source"]:
        failures.append(
            f"{case_id}: selected_source expected {expected['selected_source']!r}, got {summary.get('selected_source')!r}"
        )
    if "full_text" in expected and full_text != expected["full_text"]:
        failures.append(f"{case_id}: full_text expected {expected['full_text']!r}, got {full_text!r}")

    for idx, text in enumerate(expected.get("text_contains", []), start=1):
        if text not in full_text:
            failures.append(f"{case_id}: text_contains[{idx}] not found: {text!r}")

    for idx, text in enumerate(expected.get("text_not_contains", []), start=1):
        if text in full_text:
            failures.append(f"{case_id}: text_not_contains[{idx}] matched: {text!r}")

    for idx, rule in enumerate(expected.get("must_have", []), start=1):
        if not any(line_matches(line, rule) for line in lines):
            failures.append(f"{case_id}: must_have[{idx}] not found: {json.dumps(rule, ensure_ascii=False)}")

    for idx, rule in enumerate(expected.get("must_not_have", []), start=1):
        matched = [line for line in lines if line_matches(line, rule)]
        if matched:
            failures.append(
                f"{case_id}: must_not_have[{idx}] matched {len(matched)} line(s): {json.dumps(rule, ensure_ascii=False)}"
            )

    for idx, rule in enumerate(expected.get("must_have_candidate", []), start=1):
        if not any(candidate_matches(candidate, rule) for candidate in candidates):
            failures.append(f"{case_id}: must_have_candidate[{idx}] not found: {json.dumps(rule, ensure_ascii=False)}")

    for idx, rule in enumerate(expected.get("must_not_have_candidate", []), start=1):
        matched = [candidate for candidate in candidates if candidate_matches(candidate, rule)]
        if matched:
            failures.append(
                f"{case_id}: must_not_have_candidate[{idx}] matched {len(matched)} candidate(s): {json.dumps(rule, ensure_ascii=False)}"
            )

    return failures


def trace_lines(trace: dict) -> list[dict]:
    lines = trace.get("lines")
    if isinstance(lines, list):
        return [line for line in lines if isinstance(line, dict)]

    out = []
    for region_idx, region in enumerate(trace.get("regions", [])):
        for line_idx, line in enumerate(region.get("lines", [])):
            if not isinstance(line, dict):
                continue
            item = dict(line)
            item.setdefault("region_index", region_idx)
            item.setdefault("line_index", line_idx)
            out.append(item)
    return out


def trace_candidates(trace: dict) -> list[dict]:
    candidates = trace.get("candidates")
    if isinstance(candidates, list):
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    return []


def trace_text(trace: dict, lines: list[dict]) -> str:
    regions = trace.get("regions", [])
    if isinstance(regions, list) and regions:
        parts = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            text = str(region.get("text", "")).strip()
            if text:
                parts.append(text)
        if parts:
            return "\n\n".join(parts)
    return "\n".join(str(line.get("text", "")).strip() for line in lines if str(line.get("text", "")).strip())


def check_count(case_id: str, name: str, actual: int, rule) -> list[str]:
    if rule is None:
        return []
    if isinstance(rule, int):
        return [] if actual == rule else [f"{case_id}: {name} expected {rule}, got {actual}"]
    if not isinstance(rule, dict):
        return [f"{case_id}: {name} rule must be integer or object"]

    failures = []
    if "eq" in rule and actual != rule["eq"]:
        failures.append(f"{case_id}: {name} expected {rule['eq']}, got {actual}")
    if "min" in rule and actual < rule["min"]:
        failures.append(f"{case_id}: {name} expected >= {rule['min']}, got {actual}")
    if "max" in rule and actual > rule["max"]:
        failures.append(f"{case_id}: {name} expected <= {rule['max']}, got {actual}")
    return failures


def check_float(case_id: str, name: str, actual: float, rule) -> list[str]:
    if rule is None:
        return []
    if isinstance(rule, (int, float)):
        return [] if actual == float(rule) else [f"{case_id}: {name} expected {float(rule)}, got {actual:.4f}"]
    if not isinstance(rule, dict):
        return [f"{case_id}: {name} rule must be number or object"]

    failures = []
    if "eq" in rule and actual != float(rule["eq"]):
        failures.append(f"{case_id}: {name} expected {float(rule['eq'])}, got {actual:.4f}")
    if "min" in rule and actual < float(rule["min"]):
        failures.append(f"{case_id}: {name} expected >= {float(rule['min'])}, got {actual:.4f}")
    if "max" in rule and actual > float(rule["max"]):
        failures.append(f"{case_id}: {name} expected <= {float(rule['max'])}, got {actual:.4f}")
    return failures


def average_field(lines: list[dict], name: str) -> float:
    values = [float(line.get(name, 0.0)) for line in lines if isinstance(line.get(name, 0.0), (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def min_field(lines: list[dict], name: str) -> float:
    values = [float(line.get(name, 0.0)) for line in lines if isinstance(line.get(name, 0.0), (int, float))]
    if not values:
        return 0.0
    return min(values)


def low_margin_line_count(lines: list[dict], expected: dict) -> int:
    threshold = float(expected.get("low_margin_threshold", 0.02))
    return sum(1 for line in lines if float(line.get("min_margin", 0.0)) < threshold)


def line_matches(line: dict, rule: dict) -> bool:
    text = str(line.get("text", ""))
    if "text" in rule and text != rule["text"]:
        return False
    if "contains" in rule and rule["contains"] not in text:
        return False
    if "source" in rule and line.get("source") != rule["source"]:
        return False
    if "bbox" in rule and line.get("bbox") != rule["bbox"]:
        return False
    if "crop_size" in rule and line.get("crop_size") != rule["crop_size"]:
        return False
    if "min_confidence" in rule and float(line.get("confidence", 0.0)) < float(rule["min_confidence"]):
        return False
    if "max_confidence" in rule and float(line.get("confidence", 0.0)) > float(rule["max_confidence"]):
        return False
    if "min_avg_margin" in rule and float(line.get("avg_margin", 0.0)) < float(rule["min_avg_margin"]):
        return False
    if "max_avg_margin" in rule and float(line.get("avg_margin", 0.0)) > float(rule["max_avg_margin"]):
        return False
    if "min_min_margin" in rule and float(line.get("min_margin", 0.0)) < float(rule["min_min_margin"]):
        return False
    if "max_min_margin" in rule and float(line.get("min_margin", 0.0)) > float(rule["max_min_margin"]):
        return False
    return True


def candidate_matches(candidate: dict, rule: dict) -> bool:
    for key in ["label", "mode", "action", "reason"]:
        if key in rule and candidate.get(key) != rule[key]:
            return False
    if "label_contains" in rule and rule["label_contains"] not in str(candidate.get("label", "")):
        return False
    if "min_score" in rule and float(candidate.get("score", 0.0)) < float(rule["min_score"]):
        return False
    if "max_score" in rule and float(candidate.get("score", 0.0)) > float(rule["max_score"]):
        return False
    if "min_confidence" in rule and float(candidate.get("confidence", 0.0)) < float(rule["min_confidence"]):
        return False
    if "max_confidence" in rule and float(candidate.get("confidence", 0.0)) > float(rule["max_confidence"]):
        return False
    for key in ["char_count", "line_count", "region_count", "source_family_count"]:
        if f"min_{key}" in rule and int(candidate.get(key, 0)) < int(rule[f"min_{key}"]):
            return False
        if f"max_{key}" in rule and int(candidate.get(key, 0)) > int(rule[f"max_{key}"]):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
