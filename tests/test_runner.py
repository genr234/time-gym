import io
import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from time_gym import request_cap
from time_gym.report_baseline import matrix, openrouter_sample


def call(name, args=None):
    return {"tool_calls": [{"id": name, "type": "function",
                           "function": {"name": name, "arguments": json.dumps(args or {})}}]}


class RunnerTests(unittest.TestCase):
    @patch("time_gym.report_baseline.urlopen")
    def test_mock_http_tool_loop_reasoning_and_seed(self, urlopen):
        for reasoning in (False, True):
            requests = []

            def respond(request, timeout):
                payload = json.loads(request.data)
                requests.append(payload)
                topic = json.loads(payload["messages"][1]["content"])["topics"][0]
                source_id = next(r["id"] for r in topic["sources"] if r["stage"] == 1)
                if len(requests) % 2:
                    reply = {**call("read_source", {"source_id": source_id}), "reasoning_details": [{"id": "mock"}]}
                else:
                    self.assertEqual(payload["messages"][-2]["reasoning_details"], [{"id": "mock"}])
                    result = json.loads(payload["messages"][-1]["content"])
                    self.assertEqual((payload["messages"][-1]["tool_call_id"], result["source"]["id"]),
                                     ("read_source", source_id))
                    reply = call("finish")
                return io.StringIO(json.dumps({"choices": [{"message": reply}]}))

            urlopen.side_effect = respond
            rows = list(matrix(seeds=[1], budgets=[60], task="budget", reasoning=reasoning, samples=2,
                               sample_seed=99, live=lambda messages, **kw: openrouter_sample(messages, "fake", **kw)))
            self.assertEqual(len(rows), 6)
            self.assertEqual([p["seed"] for p in requests[:4]], [99, 99, 100, 100])
            self.assertTrue(all(p["reasoning"] == {"enabled": reasoning} for p in requests))
            self.assertTrue(all(r["requests"] == 2 and r["outcome"] == "finished" for r in rows))

    def test_one_error_does_not_stop_the_matrix(self):
        replies = iter([URLError("offline"), {"tool_calls": [{"id": "a", "function": {
            "name": "read_source", "arguments": "{"}}]}, {"content": "Plain text"},
            call("read_source", {"source_id": "bad"}), call("finish"), call("finish")])

        def live(_messages, **_settings):
            result = next(replies)
            if isinstance(result, Exception):
                raise result
            return result

        rows = list(matrix(seeds=[0], budgets=[15], task="budget", samples=2, live=live, max_requests=1))
        self.assertEqual([r["outcome"] for r in rows],
                         ["api_error", "unparseable_output", "text_only", "request_limit", "finished", "finished"])
        self.assertEqual([r["elapsed"] for r in rows], [0, 0, 0, 1, 0, 0])
        self.assertTrue(all(r["requests"] == 1 for r in rows))
        self.assertGreater(request_cap(60), request_cap(30))
        self.assertGreater(request_cap(30), request_cap(15))
