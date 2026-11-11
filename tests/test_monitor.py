import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from drift import build_reference, compare, js_divergence, psi, visible_labels
from engine import monitor_window, release_gate, validate_policy, wilson
from make_data import build
from model import instant, metrics, sha, train_models, validate_events
from registry import Registry

ROOT = Path(__file__).resolve().parents[1]


class MonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build()
        cls.models = train_models(cls.data["history"])
        cls.policy = json.loads((ROOT/"policy.json").read_text(encoding="utf-8"))
        cls.reference = build_reference([r for r in cls.data["history"] if r["partition"] == "train"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.registry = Registry(Path(self.temp.name)/"registry.db", {"test": "v1"})
        for model in self.models:
            self.registry.register(model)
        self.registry.transition("init", "initialize", None, "stable-v1")

    def tearDown(self):
        self.registry.close()
        self.temp.cleanup()

    def promote(self):
        gate = release_gate(self.models[0], self.models[1], self.data["history"], self.policy)
        return self.registry.transition("promote", "promote", "stable-v1", "candidate-v2", gate)

    def watch(self, index, labels=None):
        return monitor_window(self.registry, self.data["windows"][index], self.data["labels"] if labels is None else labels,
                              self.reference, self.policy)

    def recommendation(self):
        self.promote()
        for i in range(4):
            report = self.watch(i)
        return report

    def test_data_reproducibility(self):
        self.assertEqual(self.data, build())
        self.assertEqual(len(self.data["history"]), 500)

    def test_no_test_label_or_scaler_leakage(self):
        changed = copy.deepcopy(self.data["history"])
        for row in changed:
            if row["partition"] == "test":
                row["label"] = 1-row["label"]
                row["words"] = 100000
        self.assertEqual(self.models, train_models(changed))

    def test_chronological_partitions_required(self):
        history = copy.deepcopy(self.data["history"])
        history[0]["partition"] = "test"
        with self.assertRaisesRegex(ValueError, "chronological"):
            train_models(history)

    def test_nan_and_duplicate_events_rejected(self):
        rows = copy.deepcopy(self.data["windows"][0]["events"])
        rows[0]["words"] = math.nan
        with self.assertRaises(ValueError):
            validate_events(rows)
        with self.assertRaises(ValueError):
            validate_events([self.data["windows"][0]["events"][0]]*2)

    def test_psi_and_js_known_values(self):
        self.assertAlmostEqual(psi([10, 10], [20, 20]), 0)
        self.assertGreater(psi([90, 10], [10, 90]), 0)
        self.assertEqual(js_divergence({"a": 10}, {"b": 10}), 1)
        self.assertEqual(js_divergence({"a": 10}, {"a": 10}), 0)

    def test_missing_bucket_and_unknown_category(self):
        rows = copy.deepcopy(self.data["windows"][0]["events"])
        for row in rows:
            row["words"] = None
            row["channel"] = "new-channel"
        report = compare(self.reference, rows, self.policy)
        self.assertEqual(report["numeric"]["words"]["missing_fraction"], 1)
        self.assertTrue(report["channel"]["drifted"])

    def test_future_labels_hidden(self):
        window = self.data["windows"][1]
        self.assertEqual(visible_labels(window["events"], self.data["labels"], window["as_of"]), [])

    def test_invalid_label_availability_and_duplicate(self):
        window = self.data["windows"][0]
        label = {"event_id": window["events"][0]["event_id"], "label": 1, "available_at": "2000-01-01T00:00:00Z"}
        with self.assertRaises(ValueError):
            visible_labels(window["events"], [label], window["as_of"])
        with self.assertRaises(ValueError):
            visible_labels(window["events"], [self.data["labels"][0]]*2, window["as_of"])

    def test_wilson_bounds(self):
        self.assertIsNone(wilson(0, 0))
        lo, hi = wilson(10, 10)
        self.assertTrue(0 < lo <= hi <= 1)

    def test_faulty_candidate_gate_blocks(self):
        gate = release_gate(self.models[0], self.models[2], self.data["history"], self.policy)
        self.assertEqual(gate["decision"], "blocked")
        with self.assertRaises(ValueError):
            self.registry.transition("bad", "promote", "stable-v1", "faulty-v0", gate)
        self.assertEqual(self.registry.state()["active"], "stable-v1")

    def test_release_requires_enough_heldout_labels(self):
        gate = release_gate(self.models[0], self.models[1], [r for r in self.data["history"] if r["partition"] != "test"]
                            + [r for r in self.data["history"] if r["partition"] == "test"][:1], self.policy)
        self.assertIn("insufficient_release_test_data", gate["reasons"])

    def test_good_candidate_can_promote(self):
        self.promote()
        self.assertEqual(self.registry.state(), {"active": "candidate-v2", "fallback": "stable-v1", "epoch": 2})

    def test_transition_idempotency_and_conflict(self):
        self.promote()
        result = self.promote()
        self.assertTrue(result["replayed"])
        self.assertEqual(self.registry.state()["epoch"], 2)
        with self.assertRaises(ValueError):
            self.registry.transition("promote", "initialize", None, "candidate-v2")

    def test_model_version_immutable(self):
        changed = {**self.models[0], "bias": 99}
        with self.assertRaises(ValueError):
            self.registry.register(changed)

    def test_drift_without_labels_not_rollback(self):
        self.promote()
        report = self.watch(1)
        self.assertTrue(report["drift"]["drifted"])
        self.assertEqual(report["decision"], "investigate")
        self.assertEqual(report["consecutive_bad_windows"], 0)

    def test_repeated_window_does_not_inflate_streak(self):
        self.promote()
        report = self.watch(2)
        again = self.watch(2)
        self.assertEqual(report["consecutive_bad_windows"], 1)
        self.assertEqual(again["consecutive_bad_windows"], 1)
        self.assertTrue(again["replayed"])
        self.assertEqual(len(self.registry.history()), 1)

    def test_asof_snapshot_cannot_change(self):
        self.promote()
        self.watch(0)
        window = copy.deepcopy(self.data["windows"][0])
        window["as_of"] = "2026-03-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "immutable"):
            monitor_window(self.registry, window, self.data["labels"], self.reference, self.policy)

    def test_backdated_and_reused_events_rejected(self):
        self.promote()
        self.watch(1)
        with self.assertRaises(ValueError):
            self.watch(0)
        window = copy.deepcopy(self.data["windows"][2])
        window["events"][0]["event_id"] = self.data["windows"][1]["events"][0]["event_id"]
        with self.assertRaisesRegex(ValueError, "another monitored"):
            monitor_window(self.registry, window, [], self.reference, self.policy)

    def test_persistent_bad_windows_recommend_without_switch(self):
        report = self.recommendation()
        self.assertEqual(report["decision"], "rollback_recommended")
        self.assertEqual(self.registry.state()["active"], "candidate-v2")
        self.assertGreater(report["paired_gain"]["bootstrap95"][0], 0)

    def test_guarded_rollback_and_replay(self):
        self.recommendation()
        result = self.registry.transition("rb", "rollback", "candidate-v2", "stable-v1", "w04")
        self.assertEqual(result["state"]["active"], "stable-v1")
        self.assertTrue(self.registry.transition("rb", "rollback", "candidate-v2", "stable-v1", "w04")["replayed"])
        self.assertEqual(self.registry.state()["epoch"], 3)
        self.assertEqual(self.watch(4)["decision"], "healthy")

    def test_rollback_with_stale_evidence_rejected(self):
        self.recommendation()
        self.watch(4)
        with self.assertRaisesRegex(ValueError, "latest"):
            self.registry.transition("stale", "rollback", "candidate-v2", "stable-v1", "w04")

    def test_rollback_wrong_target_rejected(self):
        self.recommendation()
        with self.assertRaises(ValueError):
            self.registry.transition("wrong", "rollback", "candidate-v2", "faulty-v0", "w04")

    def test_low_coverage_does_not_trigger(self):
        self.promote()
        ids = {r["event_id"] for r in self.data["windows"][2]["events"][:10]}
        labels = [r for r in self.data["labels"] if r["event_id"] in ids]
        report = self.watch(2, labels)
        self.assertFalse(report["sufficient_labels"])
        self.assertEqual(report["decision"], "investigate")

    def test_invalid_policy_rejected(self):
        for value in [math.nan, -1, True]:
            with self.assertRaises(ValueError):
                validate_policy({**self.policy, "min_label_coverage": value})


if __name__ == "__main__":
    unittest.main()
