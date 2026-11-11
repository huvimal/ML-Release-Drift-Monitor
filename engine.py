"""Release and monitoring policies with delayed labels and fallback evidence."""
import math
import random

from drift import compare, visible_labels
from model import instant, metrics, predict, sha, validate_events


def validate_policy(policy):
    unit = ["release_min_accuracy", "release_max_accuracy_drop", "release_min_positive_recall",
            "min_label_coverage", "monitor_accuracy_floor", "fallback_min_gain", "js_threshold"]
    positive = ["psi_threshold"]
    integer = ["min_label_count", "consecutive_bad_windows"]
    if set(policy) != set(unit+positive+integer):
        raise ValueError("Policy fields do not match schema.")
    for key in unit+positive:
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key in unit and value > 1):
            raise ValueError(f"Invalid policy threshold: {key}")
    if any(type(policy[k]) is not int or policy[k] < 1 for k in integer):
        raise ValueError("Count thresholds must be positive integers.")


def wilson(correct, count):
    if count == 0:
        return None
    z, p = 1.96, correct/count
    denominator = 1+z*z/count
    centre = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0, centre-radius), min(1, centre+radius)]


def with_ci(model, rows):
    result = metrics(model, rows)
    return {**result, "accuracy_wilson95": wilson(result["correct"], result["count"])}


def paired_gain(active, fallback, rows):
    if not rows:
        return None
    deltas = [int(predict(fallback, r) == r["label"])-int(predict(active, r) == r["label"]) for r in rows]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(500))
    return {"fallback_minus_active_accuracy": sum(deltas)/len(deltas), "bootstrap95": [samples[12], samples[487]],
            "samples": 500, "assumption": "paired IID event bootstrap; correlated traffic needs group/time-aware resampling"}


def release_gate(baseline, candidate, history, policy):
    rows = [r for r in history if r["partition"] == "test"]
    base, cand = with_ci(baseline, rows), with_ci(candidate, rows)
    reasons = []
    if len(rows) < policy["min_label_count"]:
        reasons.append("insufficient_release_test_data")
    if cand["accuracy"] is None or cand["accuracy"] < policy["release_min_accuracy"]:
        reasons.append("accuracy_below_floor")
    if cand["accuracy"] is None or base["accuracy"] is None or base["accuracy"]-cand["accuracy"] > policy["release_max_accuracy_drop"]:
        reasons.append("accuracy_regression")
    if cand["positive_recall"] is None or cand["positive_recall"] < policy["release_min_positive_recall"]:
        reasons.append("positive_recall_below_floor")
    return {"decision": "blocked" if reasons else "passed", "baseline": baseline["version"], "candidate": candidate["version"],
            "baseline_metrics": base, "candidate_metrics": cand, "reasons": reasons,
            "policy_sha256": sha(policy), "test_sha256": sha(rows)}


def monitor_window(registry, window, labels, reference, policy):
    rows, as_of = window["events"], window["as_of"]
    validate_events(rows)
    labeled = visible_labels(rows, labels, as_of)
    event_end = max(rows, key=lambda r: instant(r["event_time"]))["event_time"]
    event_start = min(rows, key=lambda r: instant(r["event_time"]))["event_time"]
    input_sha = sha({"events": rows, "visible_labels": labeled, "as_of": as_of,
                     "reference": reference, "policy": policy})
    def build(state, previous):
        active = registry.model(state["active"])
        fallback = registry.model(state["fallback"]) if state["fallback"] else None
        drift = compare(reference, rows, policy)
        measured = with_ci(active, labeled)
        fallback_metrics = with_ci(fallback, labeled) if fallback else None
        gain = paired_gain(active, fallback, labeled) if fallback else None
        coverage = len(labeled)/len(rows)
        sufficient = len(labeled) >= policy["min_label_count"] and coverage >= policy["min_label_coverage"]
        quality_bad = sufficient and measured["accuracy"] < policy["monitor_accuracy_floor"]
        evidence_bad = bool(quality_bad and gain and gain["fallback_minus_active_accuracy"] >= policy["fallback_min_gain"]
                            and gain["bootstrap95"][0] > 0)
        streak = 1 if evidence_bad else 0
        if evidence_bad:
            for prior in previous:
                if not prior["bad_with_fallback_evidence"]:
                    break
                streak += 1
        if streak >= policy["consecutive_bad_windows"]:
            decision = "rollback_recommended"
        elif evidence_bad:
            decision = "watch"
        elif drift["drifted"] or quality_bad or not sufficient:
            decision = "investigate"
        else:
            decision = "healthy"
        return {"window_id": window["window_id"], "as_of": as_of, "events": len(rows),
                "active": state["active"], "fallback": state["fallback"], "epoch": state["epoch"],
                "label_count": len(labeled), "label_coverage": coverage, "sufficient_labels": sufficient,
                "active_metrics": measured, "fallback_metrics": fallback_metrics, "paired_gain": gain,
                "drift": drift, "quality_bad": bool(quality_bad), "bad_with_fallback_evidence": evidence_bad,
                "consecutive_bad_windows": streak, "decision": decision,
                "input_sha256": input_sha, "active_model_sha256": sha(active),
                "fallback_model_sha256": sha(fallback) if fallback else None,
                "policy_sha256": sha(policy), "reference_sha256": sha(reference),
                "action": "recommendation only; active registry state is unchanged"}
    return registry.record_window(window["window_id"], input_sha, event_start, event_end, as_of,
                                  [r["event_id"] for r in rows], build)
