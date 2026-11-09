#!/usr/bin/env python3
"""Offline unit tests + optional live smoke for the cloud System One adapter.

    python test_jev_cloud.py                 # offline, no network, no key
    JEV_CLOUD_LIVE=1 python test_jev_cloud.py   # also hit http://127.0.0.1:8090
"""
from __future__ import annotations

import json
import math
import os
import unittest
import urllib.request

import jev_cloud_adapter as jca


class FakeEngine:
    """Deterministic engine: assigns descending logprobs to the labels it sees."""

    def __init__(self, bias: float = 0.0):
        self.model = "fake-systemone"
        self.base = "http://fake"
        self.concurrency = 4
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.bias = bias
        self.seen = []

    def describe(self):
        return {"engine": "fake", "model": self.model}

    def chat(self, messages, **kw):
        self.calls += 1
        text = messages[-2]["content"] if len(messages) >= 2 else ""
        self.seen.append(text)
        rows = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if len(line) >= 2 and line[0].isalpha() and line[1] == ".":
                rows.append({"token": line[0], "logprob": -0.7 * i + self.bias})
        return {"text": rows[0]["token"] if rows else "", "reasoning": "",
                "rows": [{"top_logprobs": rows}], "usage": {}}


class TestProbabilityRead(unittest.TestCase):
    def test_strict_label_tokens(self):
        rows = [{"top_logprobs": [
            {"token": "A", "logprob": -0.2},
            {"token": " A", "logprob": -1.2},
            {"token": "A.", "logprob": -0.9},
            {"token": "B", "logprob": -1.5},
            {"token": "Option", "logprob": -3.0},   # must NOT count as O
            {"token": "The", "logprob": -3.1},      # must NOT count as T
            {"token": "**", "logprob": -3.2},
        ]}]
        got = jca.rows_to_label_logprobs(rows, ["A", "B", "O", "T"])
        self.assertEqual(set(got), {"A", "B"})
        self.assertAlmostEqual(got["A"], -0.2)   # max over whitespace variants

    def test_normalize_missing_labels_are_explicit_zero(self):
        got = jca.normalize({"A": -0.2, "B": -1.5}, ["k1", "k2", "k3"], ["A", "B", "C"])
        self.assertAlmostEqual(sum(got.values()), 1.0)
        self.assertEqual(got["k3"], 0.0)
        self.assertGreater(got["k1"], got["k2"])

    def test_normalize_empty_raises(self):
        with self.assertRaises(jca.CloudHTTPError):
            jca.normalize({}, ["a", "b"], ["A", "B"])

    def test_answer_shapes(self):
        noul = jca.answer({"type": "noul"}, ["true", "false"], [0.8, 0.2])
        self.assertAlmostEqual(noul["noul"], 0.8)
        choice = jca.answer({"type": "choice"}, ["x", "y"], [0.7, 0.3])
        self.assertEqual(choice["choice"], "x")
        score = jca.answer({"type": "score", "criteria": ["lo", "mid", "hi"]},
                           ["0", "1", "2"], [0.1, 0.2, 0.7])
        self.assertAlmostEqual(score["score"], 1.6)
        self.assertIn("legend", score)

    def test_all_labels(self):
        self.assertEqual(jca.all_labels(8), list("ABCDEFGH")[:8])
        self.assertEqual(len(jca.all_labels(30)), 30)
        self.assertEqual(jca.all_labels(30)[26], "AA")

    def test_parse_generated_options(self):
        raw = "attack: push the front line\n- build: raise a mine\n3) wait: hold position"
        opts = jca.parse_generated_options(raw, 4)
        self.assertEqual(len(opts), 3)
        self.assertEqual(opts[0][0], "attack")


class TestService(unittest.TestCase):
    def _svc(self, bias=0.0):
        eng = FakeEngine(bias)
        svc = jca.JevCloudService(eng, temperature=1.0, workers=4)
        svc.capabilities = {**eng.describe(), "label_distribution": True}
        return eng, svc

    def test_noul_and_choice_wire_format(self):
        eng, svc = self._svc()
        answers, usage = svc.run(
            [{"role": "user", "content": "gold=500 mana=120"}],
            {
                "act": {"type": "noul", "instructions": "Act now?",
                        "criteria": {"true": "yes", "false": "no"}},
                "action": {"type": "choice", "instructions": "Which action?",
                           "criteria": {"recruit": "train infantry", "build": "raise a mine",
                                        "wait": "hold"}},
            })
        self.assertIn("act", answers)
        self.assertIn("action", answers)
        self.assertIn("noul", answers["act"])
        self.assertIn(answers["action"]["choice"], {"recruit", "build", "wait"})
        for a in answers.values():
            self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=6)
        self.assertEqual(usage["questions"], 2)
        self.assertEqual(usage["engine"], "cloud-systemone")

    def test_tournament_over_many_criteria(self):
        eng, svc = self._svc()
        criteria = {f"opt_{i}": f"action number {i}" for i in range(19)}
        answers, usage = svc.run(
            "state", {"pick": {"type": "choice", "instructions": "Pick one", "criteria": criteria}})
        probs = answers["pick"]["probabilities"]
        self.assertEqual(len(probs), 19)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertGreater(usage["tournaments"], 0)
        self.assertGreater(usage["reads"], 1)
        # engine concurrency cap is never exceeded by construction
        self.assertLessEqual(eng.concurrency, jca.MAX_CONCURRENCY)

    def test_invalid_question_types_rejected(self):
        eng, svc = self._svc()
        with self.assertRaises(ValueError):
            svc.run("state", {"x": {"type": "bogus"}})

    def test_auto_requires_auto_endpoint(self):
        eng, svc = self._svc()
        with self.assertRaises(ValueError):
            svc.run("state", {"x": {"type": "auto", "instructions": "What now?"}})


class TestLive(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("JEV_CLOUD_LIVE"), "set JEV_CLOUD_LIVE=1")
    def test_live_systemone(self):
        url = os.environ.get("JEV_CLOUD_URL", "http://127.0.0.1:8090")
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(url + "/health", timeout=5) as r:
            health = json.loads(r.read().decode())
        self.assertTrue(health["capabilities"].get("label_distribution"), health)
        body = {
            "state": [{"role": "system", "content": "Four-way RTS. You have 500 gold."},
                      {"role": "user", "content": "Your army is weaker than the neighbor's."}],
            "questions": {
                "act": {"type": "noul", "instructions": "Should this faction act instead of waiting?",
                        "criteria": {"true": "act", "false": "wait"}},
                "action": {"type": "choice", "instructions": "Which action is best?",
                           "criteria": {"recruit": "train infantry now",
                                        "build": "raise a mine for income",
                                        "move": "dispatch the strongest army"}},
            },
        }
        req = urllib.request.Request(url + "/v1/systemone",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with op.open(req, timeout=30) as r:
            out = json.loads(r.read().decode())
        self.assertIn("answers", out)
        for a in out["answers"].values():
            self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
