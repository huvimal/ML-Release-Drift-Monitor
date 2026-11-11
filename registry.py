"""Transactional local deployment registry. No external service is deployed."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from model import instant, sha


class Registry:
    def __init__(self, path, config):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=15, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY CHECK(id=1), digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS models(version TEXT PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY CHECK(id=1), active TEXT NOT NULL, fallback TEXT, epoch INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS transitions(request TEXT PRIMARY KEY, spec TEXT NOT NULL, result TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(seq INTEGER PRIMARY KEY AUTOINCREMENT, window_id TEXT UNIQUE NOT NULL,
          input_sha TEXT NOT NULL, epoch INTEGER NOT NULL, event_end TEXT NOT NULL, as_of TEXT NOT NULL, report TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS window_events(event_id TEXT PRIMARY KEY, window_id TEXT NOT NULL);
        ''')
        for table in ["models", "config", "transitions", "observations", "window_events"]:
            for operation in ["UPDATE", "DELETE"]:
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation} BEFORE {operation} ON {table} "
                                "BEGIN SELECT RAISE(ABORT, 'immutable record'); END")
        with self.transaction():
            existing = self.db.execute("SELECT digest FROM config WHERE id=1").fetchone()
            if existing and existing[0] != sha(config):
                raise ValueError("Registry configuration changed. Use a new registry DB for a new experiment.")
            if not existing:
                self.db.execute("INSERT INTO config VALUES(1,?)", (sha(config),))

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def register(self, model):
        with self.transaction():
            old = self.db.execute("SELECT digest FROM models WHERE version=?", (model["version"],)).fetchone()
            if old and old[0] != sha(model):
                raise ValueError("Model version is immutable; different content requires another version.")
            if not old:
                self.db.execute("INSERT INTO models VALUES(?,?,?)", (model["version"], json.dumps(model), sha(model)))

    def model(self, version):
        row = self.db.execute("SELECT payload,digest FROM models WHERE version=?", (version,)).fetchone()
        if not row or sha(json.loads(row[0])) != row[1]:
            raise ValueError("Model missing or registry digest mismatch.")
        return json.loads(row[0])

    def state(self):
        row = self.db.execute("SELECT active,fallback,epoch FROM state WHERE id=1").fetchone()
        return dict(row) if row else None

    def transition(self, request, kind, expected, target, proof=None):
        if not request:
            raise ValueError("A nonempty transition request ID is required.")
        spec = {"kind": kind, "expected": expected, "target": target, "proof": proof}
        with self.transaction():
            previous = self.db.execute("SELECT spec,result FROM transitions WHERE request=?", (request,)).fetchone()
            if previous:
                if json.loads(previous[0]) != spec:
                    raise ValueError("Request ID already belongs to another operation.")
                return {**json.loads(previous[1]), "replayed": True, "current_state": self.state()}
            current = self.state()
            self.model(target)
            if kind == "initialize":
                if current:
                    raise ValueError("Registry is already initialized.")
                self.db.execute("INSERT INTO state VALUES(1,?,NULL,1)", (target,))
            elif kind == "promote":
                if not current or current["active"] != expected or target == expected:
                    raise ValueError("Active version changed or target is already active.")
                if not proof or proof.get("decision") != "passed" or proof.get("candidate") != target or proof.get("baseline") != expected:
                    raise ValueError("Promotion requires a matching passing release gate.")
                self.db.execute("UPDATE state SET active=?,fallback=?,epoch=epoch+1 WHERE id=1", (target, expected))
            elif kind == "rollback":
                if not current or current["active"] != expected or current["fallback"] != target:
                    raise ValueError("Rollback target or expected version does not match the active registry state.")
                latest = self.db.execute("SELECT window_id,epoch,report FROM observations ORDER BY seq DESC LIMIT 1").fetchone()
                if not latest or latest[0] != proof or latest[1] != current["epoch"]:
                    raise ValueError("Rollback evidence must be the latest observation of this deployment epoch.")
                evidence = json.loads(latest[2])
                if evidence["decision"] != "rollback_recommended" or evidence["fallback"] != target:
                    raise ValueError("Latest evidence does not recommend this rollback.")
                self.db.execute("UPDATE state SET active=?,fallback=NULL,epoch=epoch+1 WHERE id=1", (target,))
            else:
                raise ValueError("Unknown transition kind.")
            result = {"kind": kind, "request": request, "state": self.state(), "replayed": False}
            self.db.execute("INSERT INTO transitions VALUES(?,?,?)", (request, json.dumps(spec, sort_keys=True), json.dumps(result)))
            return result

    def record_window(self, window_id, input_sha, event_start, event_end, as_of, event_ids, build_report):
        with self.transaction():
            previous = self.db.execute("SELECT input_sha,report FROM observations WHERE window_id=?", (window_id,)).fetchone()
            if previous:
                if previous[0] != input_sha:
                    raise ValueError("Window snapshot is immutable; changed input/as-of requires a separate experiment DB.")
                return {**json.loads(previous[1]), "replayed": True}
            current = self.state()
            if not current:
                raise ValueError("Initialize and promote a model first.")
            latest = self.db.execute("SELECT event_end,as_of FROM observations ORDER BY seq DESC LIMIT 1").fetchone()
            if latest and (instant(event_start) <= instant(latest[0]) or instant(as_of) < instant(latest[1])):
                raise ValueError("New monitoring windows must advance event time and not move as-of time backward.")
            for event_id in event_ids:
                if self.db.execute("SELECT 1 FROM window_events WHERE event_id=?", (event_id,)).fetchone():
                    raise ValueError("Event already belongs to another monitored window.")
            previous_reports = [json.loads(r[0]) for r in self.db.execute(
                "SELECT report FROM observations WHERE epoch=? ORDER BY seq DESC", (current["epoch"],))]
            report = build_report(current, previous_reports)
            self.db.execute("INSERT INTO observations(window_id,input_sha,epoch,event_end,as_of,report) VALUES(?,?,?,?,?,?)",
                            (window_id, input_sha, current["epoch"], event_end, as_of, json.dumps(report, allow_nan=False)))
            self.db.executemany("INSERT INTO window_events VALUES(?,?)", [(event_id, window_id) for event_id in event_ids])
            return {**report, "replayed": False}

    def history(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT report FROM observations ORDER BY seq")]
