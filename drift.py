"""Fixed-reference descriptive drift scores. Thresholds are policy, not p-values."""
import bisect
import math
from collections import Counter

from model import instant, number


def psi(reference, current):
    if not reference or not current:
        raise ValueError("PSI needs nonempty reference and current distributions.")
    eps = 1e-6
    p = [(x+eps)/(sum(reference)+eps*len(reference)) for x in reference]
    q = [(x+eps)/(sum(current)+eps*len(current)) for x in current]
    if len(p) != len(q):
        raise ValueError("Histogram bins must match.")
    return sum((b-a)*math.log(b/a) for a, b in zip(p, q))


def js_divergence(reference, current):
    keys = sorted(set(reference) | set(current))
    n, m = sum(reference.values()), sum(current.values())
    if n <= 0 or m <= 0:
        raise ValueError("Jensen-Shannon divergence requires nonempty distributions.")
    result = 0
    for key in keys:
        p, q = reference.get(key, 0)/n, current.get(key, 0)/m
        middle = (p+q)/2
        result += .5*((p*math.log2(p/middle) if p else 0)+(q*math.log2(q/middle) if q else 0))
    return result


def build_reference(rows):
    result = {"numeric": {}, "channel_counts": dict(Counter(r.get("channel") or "__missing__" for r in rows))}
    for name in ["urgency_score", "words"]:
        values = sorted(r[name] for r in rows if r.get(name) is not None)
        if not values:
            raise ValueError("Reference numeric features cannot be entirely missing.")
        edges = sorted({values[min(len(values)-1, int(len(values)*q))] for q in [.2, .4, .6, .8]})
        counts = histogram(rows, name, edges)
        result["numeric"][name] = {"edges": edges, "counts": counts}
    return result


def histogram(rows, name, edges):
    counts = [0]*(len(edges)+2)
    for row in rows:
        value = row.get(name)
        counts[-1 if value is None else bisect.bisect_right(edges, value)] += 1
    return counts


def compare(reference, rows, policy):
    numeric = {}
    for name, spec in reference["numeric"].items():
        counts = histogram(rows, name, spec["edges"])
        score = psi(spec["counts"], counts)
        numeric[name] = {"psi": score, "drifted": score > policy["psi_threshold"],
                         "missing_fraction": counts[-1]/len(rows), "edges": spec["edges"]}
    current = dict(Counter(r.get("channel") or "__missing__" for r in rows))
    js = js_divergence(reference["channel_counts"], current)
    return {"numeric": numeric, "channel": {"js_divergence_base2": js,
             "drifted": js > policy["js_threshold"], "current_counts": current},
            "drifted": any(r["drifted"] for r in numeric.values()) or js > policy["js_threshold"]}


def visible_labels(events, labels, as_of):
    cutoff = instant(as_of)
    event_by_id = {r["event_id"]: r for r in events}
    if any(instant(r["event_time"]) > cutoff for r in events):
        raise ValueError("Cannot score future events.")
    seen, joined = set(), []
    for label in labels:
        if not isinstance(label.get("event_id"), str) or label["event_id"] in seen:
            raise ValueError("Label IDs must be unique.")
        seen.add(label["event_id"])
        if type(label.get("label")) is not int or label["label"] not in {0, 1}:
            raise ValueError("Labels must be binary integers.")
        available = instant(label["available_at"])
        if label["event_id"] in event_by_id:
            row = event_by_id[label["event_id"]]
            if available < instant(row["event_time"]):
                raise ValueError("Label availability cannot precede its event.")
            if available <= cutoff:
                joined.append({**row, "label": label["label"]})
    return joined
