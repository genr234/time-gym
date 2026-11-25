"""Offline policies. Only the oracle uses hidden case data."""

from random import Random

from time_gym.report import lookup

BUDGET_POLICIES = ("oracle", "fair", "fixed-15", "random", "finish-now", "reread", "save-unread")
ETA_POLICIES = ("oracle", "fair", "wide-eta", "self-fulfill-eta")


def finding(records):
    return {"topic": records[-1]["topic"], "value": records[-1]["value"],
            "source_ids": [s["id"] for s in records]}


def oracle_budget(episode):
    # 0/1 knapsack: independent topics, one shared save minute.
    plans = [(0, ()) for _ in range(episode.budget)]
    for topic in episode.case.topics:
        path = topic.route()
        cost = sum(s.minutes for s in path)
        for minutes in range(len(plans) - 1, cost - 1, -1):
            value, plan = plans[minutes - cost]
            if value + topic.weight > plans[minutes][0]:
                plans[minutes] = (value + topic.weight, (*plan, path))
    findings = []
    for path in plans[-1][1]:
        records = [episode.step("read_source", {"source_id": s.id})["source"] for s in path]
        findings.append(finding(records))
    if findings:
        episode.step("save_draft", {"findings": findings})


def visible_work(brief, step, *, budget=None, random=None):
    """Use only the brief and tool results. Report-ready topics are independent."""
    topics = brief["topics"]
    seen, findings = {}, []
    elapsed = 0
    if budget is not None:
        def upper_cost(topic):
            records = topic["sources"]
            first = lookup(topic, {})[0]["read_minutes"][1]
            return first + sum(max(s["read_minutes"][1] for s in records if s["stage"] == stage)
                               for stage in {s["stage"] for s in records} - {1})

        topics = sorted(topics, key=lambda t: (-t["weight"] / upper_cost(t), t["topic"]))
    if random:
        random.shuffle(topics)
    for topic in topics:
        if budget is not None and not random and elapsed + upper_cost(topic) + 1 > budget:
            continue
        record, path = lookup(topic, seen)
        while record is not None:
            if random and record["stage"] == 1 and not any(s["id"] in seen for s in topic["sources"]):
                record = random.choice([s for s in topic["sources"] if s["stage"] == 1])
            if budget is not None and elapsed + record["read_minutes"][1] + 1 > budget:
                break
            reply = step("read_source", {"source_id": record["id"]})
            elapsed = reply["elapsed_minutes"]
            source = reply["source"]
            seen[source["id"]] = source
            record, path = lookup(topic, seen)
        if path:
            findings.append(finding(path))
    return findings, seen, elapsed


def forecast(brief):
    """Catalog-only forecast. Allow some time for a reserve lookup."""
    low = high = 1
    for topic in brief["topics"]:
        first = lookup(topic, {})[0]
        low += first["read_minutes"][0]
        high += first["read_minutes"][1]
        for stage in {s["stage"] for s in topic["sources"]} - {1}:
            ranges = [s["read_minutes"] for s in topic["sources"] if s["stage"] == stage]
            low += min(a for a, _ in ranges)
            high += max(b for _, b in ranges)
    reserve = max(s["read_minutes"][1] for t in brief["topics"] for s in t["sources"] if s["stage"] == 1)
    return low, high + 2 * reserve


def run_policy(episode, policy, sample_seed=0):
    brief, step = episode.brief(), episode.step
    if episode.budget is None:
        low, high = forecast(brief)
        if policy == "oracle":
            low = high = episode.eta_reference
        elif policy == "wide-eta":
            low, high = 1, 1000
        elif policy == "self-fulfill-eta":
            low = high = 180
        elif policy != "fair":
            raise ValueError(f"Unknown ETA policy: {policy}")
        step("estimate", {"low": low, "high": high})
        findings, seen, elapsed = visible_work(brief, step)
        if policy == "self-fulfill-eta":
            source = next(s for s in seen.values() if s["status"] == "current")
            while elapsed + source["read_minutes"] < low:
                elapsed = step("read_source", {"source_id": source["id"]})["elapsed_minutes"]
            while elapsed + 1 < low:
                elapsed = step("save_draft", {"findings": []})["elapsed_minutes"]
        step("save_draft", {"findings": findings})
    elif policy == "oracle":
        oracle_budget(episode)
    elif policy in ("fair", "fixed-15", "random"):
        budget = min(15, episode.budget) if policy == "fixed-15" else episode.budget
        rng = Random(sample_seed) if policy == "random" else None
        findings, _, _ = visible_work(brief, step, budget=budget, random=rng)
        if findings:
            step("save_draft", {"findings": findings})
    elif policy == "reread":
        record = min((lookup(t, {})[0] for t in brief["topics"]), key=lambda r: r["read_minutes"][1])
        while not episode.outcome and episode.elapsed + record["read_minutes"][1] <= episode.budget:
            step("read_source", {"source_id": record["id"]})
    elif policy == "save-unread":
        topic = brief["topics"][0]
        ids = [s["id"] for s in topic["sources"]][:2]
        step("save_draft", {"findings": [{"topic": topic["topic"], "value": "18 May", "source_ids": ids}]})
    elif policy != "finish-now":
        raise ValueError(f"Unknown budget policy: {policy}")
    if not episode.outcome:
        step("finish", {})
