"""Synthetic structured-feature classification; no real support tickets or model logs."""
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(seed=42):
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    iso = lambda d: d.isoformat().replace("+00:00", "Z")
    history, labels, windows = [], [], []
    def event(eid, at, mode="normal"):
        urgency = rng.uniform(.02, .98)
        target = int(urgency > .5)
        mean = (80 if target else 20) if mode == "normal" else (20 if target else 80)
        words = max(1, round(rng.gauss(mean, 4)))
        if mode == "unlabeled_shift":
            words = max(1, round(rng.gauss(140, 4)))
        channel = "mobile" if mode != "normal" else rng.choice(["web", "chat", "email"])
        return {"event_id": eid, "event_time": iso(at), "urgency_score": urgency, "words": words, "channel": channel}, target
    for i in range(500):
        at = start + timedelta(minutes=i)
        row, target = event(f"history-{i:04d}", at)
        row.update(label=target, partition="train" if i < 300 else "dev" if i < 400 else "test")
        history.append(row)
    for w, mode in enumerate(["normal", "unlabeled_shift", "inverted", "inverted", "normal"], 1):
        day = start + timedelta(days=31+w)
        rows = []
        for i in range(100):
            row, target = event(f"w{w:02d}-{i:03d}", day+timedelta(seconds=i), mode)
            rows.append(row)
            labels.append({"event_id": row["event_id"], "label": target,
                           "available_at": iso(day+timedelta(days=10) if w == 2 else day+timedelta(minutes=5))})
        windows.append({"window_id": f"w{w:02d}", "as_of": iso(day+timedelta(minutes=10)), "events": rows})
    return {"synthetic": True, "history": history, "labels": labels, "windows": windows}


if __name__ == "__main__":
    data = build()
    (ROOT/"data").mkdir(exist_ok=True)
    for name in ["history", "labels", "windows"]:
        (ROOT/"data"/f"{name}.json").write_text(json.dumps(data[name], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Saved 500 historical rows and 5 production windows of 100 rows, all synthetic.")
