#!/usr/bin/env python3
"""Smoke test for the local Jev-like service: the example from the jpt-4b model card."""
import json
import sys

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"

payload = {
    "state": [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "I was charged twice. Please refund the duplicate."},
    ],
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent is the request?",
                    "criteria": ["Routine", "Urgent", "Emergency"]},
    },
}

r = requests.post(URL + "/v1/systemone", json=payload, timeout=1800)
r.raise_for_status()
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
