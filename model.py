"""Train and use small logistic classifiers with train-only scaling."""
import copy
import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone.")
    return result


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_events(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Events must be a nonempty list.")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("event_id"), str) or not row["event_id"] or row["event_id"] in ids:
            raise ValueError("Event IDs must be nonempty and unique.")
        ids.add(row["event_id"])
        instant(row["event_time"])
        for field in ["urgency_score", "words"]:
            value = row.get(field)
            if value is not None and (not number(value) or value < 0 or (field == "urgency_score" and value > 1)):
                raise ValueError(f"Invalid {field} in {row['event_id']}.")
        if row.get("channel") is not None and not isinstance(row["channel"], str):
            raise ValueError("Channel must be string or null.")


def vector(model, row):
    return [((row.get(name) if row.get(name) is not None else model["means"][i])-model["means"][i])/model["scales"][i]
            for i, name in enumerate(model["features"])]


def probability(model, row):
    score = model["bias"] + sum(w*x for w, x in zip(model["weights"], vector(model, row)))
    return 1/(1+math.exp(-max(-40, min(40, score))))


def predict(model, row):
    return int(probability(model, row) >= model["threshold"])


def metrics(model, rows):
    if not rows:
        return {"count": 0, "accuracy": None, "positive_recall": None, "correct": 0}
    predictions = [predict(model, r) for r in rows]
    correct = sum(p == r["label"] for p, r in zip(predictions, rows))
    positive = sum(r["label"] == 1 for r in rows)
    tp = sum(p == r["label"] == 1 for p, r in zip(predictions, rows))
    return {"count": len(rows), "correct": correct, "accuracy": correct/len(rows),
            "positive_recall": tp/positive if positive else None}


def fit(history, name, features):
    train = [r for r in history if r["partition"] == "train"]
    dev = [r for r in history if r["partition"] == "dev"]
    means = [sum(r[f] for r in train)/len(train) for f in features]
    scales = [max(1e-6, math.sqrt(sum((r[f]-means[i])**2 for r in train)/len(train))) for i, f in enumerate(features)]
    model = {"version": name, "model_type": "logistic_regression", "features": features,
             "means": means, "scales": scales, "weights": [0.0]*len(features), "bias": 0.0,
             "threshold": .5, "training_sha256": sha(train), "seed": 42}
    rng = random.Random(42)
    for _ in range(80):
        order = list(train)
        rng.shuffle(order)
        for row in order:
            error = probability(model, row)-row["label"]
            values = vector(model, row)
            model["weights"] = [w-.03*(error*x+.001*w) for w, x in zip(model["weights"], values)]
            model["bias"] -= .03*error
    choices = []
    for threshold in [.4, .5, .6]:
        model["threshold"] = threshold
        choices.append((metrics(model, dev)["accuracy"], threshold))
    model["threshold"] = max(choices, key=lambda x: x[0])[1]
    model["dev_selection"] = "highest dev accuracy among fixed thresholds 0.4/0.5/0.6; first wins ties"
    return model


def train_models(history):
    validate_events(history)
    if {r.get("partition") for r in history} != {"train", "dev", "test"}:
        raise ValueError("History requires train/dev/test partitions.")
    for row in history:
        if type(row.get("label")) is not int or row["label"] not in {0, 1} or any(row.get(f) is None for f in ["urgency_score", "words"]):
            raise ValueError("Historical rows require binary labels and complete numeric features.")
    by_split = {s: [instant(r["event_time"]) for r in history if r["partition"] == s] for s in ["train", "dev", "test"]}
    if max(by_split["train"]) >= min(by_split["dev"]) or max(by_split["dev"]) >= min(by_split["test"]):
        raise ValueError("Historical partitions must be strictly chronological.")
    stable = fit(history, "stable-v1", ["urgency_score"])
    candidate = fit(history, "candidate-v2", ["urgency_score", "words"])
    faulty = copy.deepcopy(candidate)
    faulty.update(version="faulty-v0", weights=[-w for w in candidate["weights"]], bias=-candidate["bias"],
                  fixture="deliberately inverted weights; not another trained model")
    return [stable, candidate, faulty]


def save_model(root, model):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root/f"{model['version']}.json"
    content = json.dumps(model, ensure_ascii=False, indent=2)+"\n"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != model:
        raise ValueError("Model version exists with different content; use a new version.")
    path.write_text(content, encoding="utf-8")
    return path
