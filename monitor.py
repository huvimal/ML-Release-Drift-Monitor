"""Offline ML release gates, drift monitoring and guarded local rollback."""
import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path

from drift import build_reference
from engine import monitor_window, release_gate, validate_policy
from model import metrics, save_model, sha, train_models
from registry import Registry

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def display(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def dashboard(registry):
    history = registry.history()
    esc = lambda x: html.escape(str(x))
    rows = "".join(f"<tr><td>{esc(r['window_id'])}</td><td>{esc(r['active'])}</td><td>{r['label_count']}/{r['events']}</td>"
                   f"<td>{esc(r['active_metrics']['accuracy'])}</td><td>{esc(r['drift']['drifted'])}</td>"
                   f"<td>{r['consecutive_bad_windows']}</td><td>{esc(r['decision'])}</td></tr>" for r in history)
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>ML Release &amp; Drift Monitor</title><style>body{{font:16px system-ui;margin:32px;color:#183247;line-height:1.6}}
    table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border:1px solid #bdcbd6;text-align:left}}
    pre{{white-space:pre-wrap;background:#edf4f8;padding:16px}}</style><h1>ML Release &amp; Drift Monitor</h1>
    <h2>Current local registry</h2><pre>{esc(json.dumps(registry.state(), indent=2))}</pre>
    <table><tr><th>Window</th><th>Model</th><th>Visible labels</th><th>Accuracy</th><th>Data drift</th><th>Bad streak</th><th>Decision</th></tr>{rows}</table>
    <p>Historical rows describe immutable as-of snapshots. Drift alone does not prove accuracy loss.</p>
    <p>All supplied traffic is synthetic. Registry changes affect this local experiment only.</p></html>'''
    target = ROOT/"reports"/"dashboard.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT/"private"/"registry.db")
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--policy", type=Path, default=ROOT/"policy.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    p = sub.add_parser("initialize")
    p.add_argument("--version", default="stable-v1")
    p = sub.add_parser("promote")
    p.add_argument("--candidate", required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--request", required=True)
    p = sub.add_parser("watch")
    p.add_argument("--window", required=True)
    p = sub.add_parser("rollback")
    p.add_argument("--expected", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--request", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    registry = None
    try:
        history = read(args.data/"history.json")
        policy = read(args.policy)
        validate_policy(policy)
        if args.command == "train":
            models = train_models(history)
            report = {"synthetic_demo": True, "history_sha256": sha(history), "models": {}}
            for model in models:
                save_model(ROOT/"models", model)
                report["models"][model["version"]] = {"features": model["features"], "model_sha256": sha(model),
                    "dev": metrics(model, [r for r in history if r["partition"] == "dev"])}
            save(ROOT/"reports"/"training.json", report)
            display(report)
            return 0
        reference = build_reference([r for r in history if r["partition"] == "train"])
        code_hash = sha({name: (ROOT/name).read_text(encoding="utf-8") for name in ["model.py", "drift.py", "engine.py", "registry.py"]})
        registry = Registry(args.db, {"policy": policy, "history_sha256": sha(history), "reference": reference, "implementation_sha256": code_hash})
        if args.command == "initialize":
            for name in ["stable-v1", "candidate-v2", "faulty-v0"]:
                model = read(ROOT/"models"/f"{name}.json")
                if model["version"] != name or model["training_sha256"] != sha([r for r in history if r["partition"] == "train"]):
                    raise ValueError("Model artifact does not match the selected training data.")
                registry.register(model)
            display(registry.transition("initialize", "initialize", None, args.version))
        elif args.command == "promote":
            report = release_gate(registry.model(args.expected), registry.model(args.candidate), history, policy)
            save(ROOT/"reports"/f"release-{args.candidate}.json", report)
            display(report)
            if report["decision"] == "blocked":
                return 1
            display(registry.transition(args.request, "promote", args.expected, args.candidate, report))
        elif args.command == "watch":
            windows = read(args.data/"windows.json")
            if len({w["window_id"] for w in windows}) != len(windows):
                raise ValueError("Window IDs must be unique.")
            matches = [w for w in windows if w["window_id"] == args.window]
            if not matches:
                raise ValueError("Unknown window ID.")
            report = monitor_window(registry, matches[0], read(args.data/"labels.json"), reference, policy)
            save(ROOT/"reports"/f"{args.window}.json", report)
            display({key: report[key] for key in ["window_id", "active", "fallback", "label_count", "label_coverage", "active_metrics",
                                                 "consecutive_bad_windows", "decision", "replayed"]})
            print(f"Drift: {report['drift']['drifted']}")
        elif args.command == "rollback":
            display(registry.transition(args.request, "rollback", args.expected, args.target, args.evidence))
        elif args.command == "status":
            display({"state": registry.state(), "recorded_windows": len(registry.history())})
        print(f"Dashboard: {dashboard(registry)}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        if registry:
            registry.close()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
