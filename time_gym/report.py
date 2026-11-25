"""Run both tasks on one simulated clock and score the saved report."""

from copy import deepcopy
from datetime import datetime

INSTRUCTIONS = (
    "Use one tool per turn. In the budget task, maximize verified topic weight before the deadline. "
    "In the ETA task, give an 80% interval before reads. Forecast the time for this fixed procedure: "
    "for each topic, read stages in order; at each stage use the accessible record with the lowest "
    "upper cost (tie: lowest ID) until a current record is read. Add one minute for the final save. "
    "Then verify all listed topics. Your own save time does not set the ETA target. "
    "Each finding must cite one current record you read from each listed stage. "
    "Use the last stage's value to settle disagreement. A link in a read result enables a gated record. "
    "Read costs are hidden within the listed intervals. Failed reads still use time. "
    "Unsupported findings subtract value. Only your last saved draft counts."
)


def normalize(text):
    text = " ".join(text.casefold().split())
    for fmt in ("%d %B", "%B %d", "%m-%d"):
        try:
            return datetime.strptime(f"2000 {text}", f"%Y {fmt}").strftime("%m-%d")
        except ValueError:
            pass
    return text


def next_record(topic, seen):
    """Choose the lowest listed upper cost at the next unresolved stage."""
    opened = {s.get("opens") for s in seen.values()}
    for stage in sorted({s["stage"] for s in topic["sources"]}):
        records = [s for s in topic["sources"] if s["stage"] == stage]
        if any(seen.get(s["id"], {}).get("status") == "current" for s in records):
            continue
        return min((s for s in records if s["id"] not in seen
                    and (not s["link_required"] or s["id"] in opened)),
                   key=lambda s: (s["read_minutes"][1], s["id"]), default=None)
    return None


def useful_time(case, catalog):
    """Replay the public lookup rule on the hidden case, independent of agent actions."""
    minutes = 1  # One final save.
    for name in case.targets:
        topic = next(t for t in case.topics if t.name == name)
        sources = {s.id: s for s in topic.sources}
        seen = {}
        while record := next_record(catalog[name], seen):
            source = sources[record["id"]]
            minutes += source.minutes
            seen[source.id] = {"status": source.status, "opens": source.opens}
    return minutes


class ReportEpisode:
    def __init__(self, case, budget=None):
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("Use a positive integer budget, or None for the ETA task.")
        self.case, self.budget = case, budget
        self.topics = {t.name: t for t in case.topics if budget is not None or t.name in case.targets}
        self.catalog = {name: {"topic": name, "weight": t.weight, "sources": [s.catalog() for s in t.sources]}
                        for name, t in self.topics.items()}
        self.sources = {s.id: (t, s) for t in self.topics.values() for s in t.sources}
        self.eta_actual = useful_time(case, self.catalog) if budget is None else None
        self.elapsed = self.reread_minutes = self.retry_minutes = self.invalid_actions = 0
        self.seen = {}
        self.draft = []
        self.saved = {"verified": 0, "unsupported": 0, "format_errors": 0, "value": 0}
        self.verified_topics = set()
        self.eta = self.outcome = None
        self.actions = []
        self.draft_scores = []

    def brief(self):
        return deepcopy({"task": "budget" if self.budget is not None else "eta",
                         "budget_minutes": self.budget, "topics": list(self.catalog.values())})

    def step(self, action, args):
        if self.outcome:
            raise RuntimeError("The episode has ended.")
        start = self.elapsed
        result = self._apply(action, args)
        if self.budget == self.elapsed and self.outcome is None:
            self.outcome = "deadline"
        observation = {**result, "elapsed_minutes": self.elapsed,
                       "remaining_minutes": None if self.budget is None else self.budget - self.elapsed,
                       "outcome": self.outcome}
        self.actions.append(deepcopy({"action": action, "args": args, "start_minutes": start, **observation}))
        return observation

    def _apply(self, action, args):
        if not isinstance(args, dict):
            return self._invalid("Use an arguments object.")
        if action == "estimate":
            low, high = args.get("low"), args.get("high")
            if (self.budget is not None or self.eta or type(low) is not int or type(high) is not int
                    or not 1 <= low <= high <= 10_000):
                return self._invalid("Give one ETA interval with 1 <= low <= high <= 10000.")
            self.eta = [low, high]
            return {"estimated": True}
        if action == "finish":
            self.outcome = "finished"
            return {"delivered": True}
        if self.budget is None and self.eta is None:
            return self._invalid("Give an ETA before reading or saving.")
        if action == "read_source":
            return self._read(args.get("source_id"))
        if action == "save_draft":
            findings = args.get("findings")
            if not isinstance(findings, list) or not all(self._valid_finding(f) for f in findings):
                return self._invalid("Give findings with a topic, value, and distinct source IDs.")
            if len({normalize(f["topic"]) for f in findings}) != len(findings):
                return self._invalid("Give at most one finding per topic.")
            self._spend(1)
            self.draft = deepcopy(findings)
            self._score()
            self.draft_scores.append({"minute": self.elapsed, **self.saved})
            return {"saved_findings": len(findings)}
        return self._invalid("Use a listed tool.")

    def _read(self, source_id):
        if not isinstance(source_id, str) or source_id not in self.sources:
            return self._invalid("Unknown source ID.")
        topic, source = self.sources[source_id]
        if source.gated and not any(s.get("opens") == source_id for s in self.seen.values()):
            return self._invalid("Read the source that links to this record first.")
        start = self.elapsed
        fits = self._spend(source.minutes)
        if source_id in self.seen:
            if source.status == "failed":
                self.retry_minutes += self.elapsed - start
            else:
                self.reread_minutes += self.elapsed - start
        if not fits:
            return {"error": "Read exceeded the budget. No evidence obtained."}
        result = {"id": source.id, "stage": source.stage, "topic": topic.name,
                  "status": source.status, "read_minutes": source.minutes}
        if source.status != "failed":
            result.update(value=source.value, opens=source.opens)
        self.seen[source_id] = result.copy()
        return {"source": result}

    def _spend(self, minutes):
        self.elapsed += minutes
        if self.budget is not None and self.elapsed > self.budget:
            self.elapsed = self.budget
            self.outcome = "overrun"
            return False
        return True

    def _invalid(self, error):
        self.invalid_actions += 1
        self._spend(1)
        return {"error": error}

    @staticmethod
    def _valid_finding(finding):
        if not isinstance(finding, dict) or not all(isinstance(finding.get(k), str) for k in ("topic", "value")):
            return False
        ids = finding.get("source_ids")
        return (isinstance(ids, list) and 2 <= len(ids) <= 3 and
                all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids))

    def _score(self):
        self.verified_topics.clear()
        value = unsupported = format_errors = 0
        topics = {normalize(name): t for name, t in self.topics.items()}
        for finding in self.draft:
            topic = topics.get(normalize(finding["topic"]))
            records = [self.seen.get(i, {}) for i in finding["source_ids"]]
            if (topic and all(s.get("status") == "current" and s["topic"] == topic.name for s in records)
                    and sorted(s["stage"] for s in records) == sorted({s.stage for s in topic.sources})
                    and normalize(finding["value"]) == normalize(topic.value)):
                value += topic.weight
                self.verified_topics.add(topic.name)
                format_errors += finding["value"] != topic.value
            else:
                unsupported += 1
                value -= topic.weight if topic else 1
        self.saved = {"verified": len(self.verified_topics), "unsupported": unsupported,
                      "format_errors": format_errors, "value": value}

    def metrics(self):
        remaining = None if self.budget is None else self.budget - self.elapsed
        unused = 0
        if remaining and any(1 + sum(s.minutes for s in t.route(self.seen) if s.id not in self.seen) <= remaining
                             for t in self.topics.values() if t.name not in self.verified_topics):
            unused = remaining
        complete = self.saved["verified"] == len(self.topics) and self.saved["unsupported"] == 0
        forecast = {"eta_width": None, "eta_covered": None, "eta_bias": None, "eta_interval_score": None}
        if self.eta:
            low, high = self.eta
            actual = self.eta_actual
            loss = high - low + 10 * max(low - actual, actual - high, 0)
            forecast = {"eta_width": high - low, "eta_covered": low <= actual <= high,
                        "eta_bias": (low + high) / 2 - actual, "eta_interval_score": loss}
        reward = self.saved["value"] / sum(t.weight for t in self.topics.values())
        if self.budget is None:
            reward = 20 / (20 + forecast["eta_interval_score"]) if complete and self.eta else 0.0
        return {"split": self.case.split, "family": self.case.family, "seed": self.case.seed,
                "budget": self.budget, "task": "eta" if self.budget is None else "budget",
                "elapsed": self.elapsed, "outcome": self.outcome, **self.saved, "reward": reward,
                "invalid_actions": self.invalid_actions, "reread_minutes": self.reread_minutes,
                "retry_minutes": self.retry_minutes, "unused_useful_minutes": unused,
                "scope_complete": complete, "eta": self.eta, "eta_actual": self.eta_actual,
                **forecast, "draft_scores": self.draft_scores}
