"""Generate fictional cases. Only the simulator and oracle see the hidden fields."""

from dataclasses import dataclass, replace
from itertools import product
from random import Random

DEV_FAMILIES = ("corroborate", "conflict", "chain")
HELD_OUT_FAMILIES = ("branch",)
DEV_SEEDS = tuple(range(12))
HELD_OUT_SEEDS = tuple(range(100, 112))
THINKING_TRAIN_SEEDS = range(20, 80)
BUDGETS = (15, 30, 60)


@dataclass(frozen=True)
class Source:
    id: str
    stage: int
    bounds: tuple[int, int]
    minutes: int
    value: str
    status: str = "current"
    gated: bool = False
    opens: str | None = None

    def catalog(self):
        return {"id": self.id, "stage": self.stage, "read_minutes": self.bounds,
                "link_required": self.gated}


@dataclass(frozen=True)
class Topic:
    name: str
    weight: int
    value: str
    sources: tuple[Source, ...]

    def route(self, read=()):
        """Cheapest complete path, in access order."""
        stages = sorted({s.stage for s in self.sources})
        choices = ([s for s in self.sources if s.stage == stage and s.status == "current"] for stage in stages)
        paths = (p for p in product(*choices) if all(not s.gated or (i > 0 and p[i - 1].opens == s.id)
                                                   for i, s in enumerate(p)))
        return min(paths, key=lambda p: (sum(s.minutes for s in p if s.id not in read), tuple(s.id for s in p)))


@dataclass(frozen=True)
class Case:
    family: str
    seed: int
    topics: tuple[Topic, ...]
    targets: tuple[str, ...]

    @property
    def split(self):
        return "held-out" if self.family in HELD_OUT_FAMILIES else "dev"


def generate(family, seed):
    if family not in (*DEV_FAMILIES, *HELD_OUT_FAMILIES, "detour"):
        raise ValueError(f"Unknown family: {family}")
    rng = Random(f"time-gym-v2/{family}/{seed}")
    topics = []
    for index in range(rng.randint(12, 20)):
        name = f"Site {index + 1:02d} opening date"
        month = rng.choice(("April", "May", "June", "July", "August"))
        day = rng.randint(1, 27)
        value, other = f"{day} {month}", f"{day + 1} {month}"
        quick = index == 0
        a, reserve, b, c, d = rng.sample(range(10, 99), 5)
        a, reserve, b, c, d = (f"{index + 1:02d}-{n}" for n in (a, reserve, b, c, d))
        branch = rng.choice((b, c))
        link = b if family in ("chain", "branch", "detour") else None

        def record(id, stage, low, width, text, *, status="current", gated=False, opens=None):
            return Source(id, stage, (low, low + width), rng.randint(low, low + width),
                          text, status, gated, opens)

        first = 3 if quick else rng.randint(5, 8)
        second = 5 if quick else rng.randint(7, 10)
        status = "current" if quick or family in ("branch", "detour") else rng.choices(("current", "failed", "archived"), (7, 2, 1))[0]
        sources = [record(a, 1, first, 2, other if status == "archived" else value,
                          status=status, opens=link),
                   record(reserve, 1, first + 2, 2, value, opens=c if family in ("branch", "detour") else link)]
        if family == "branch":
            live = b if quick else branch
            for id in (b, c):
                sources.append(record(id, 2, second, 2, value if id == live else other,
                                      status="current" if id == live else "archived",
                                      gated=True, opens=d if id == live else None))
            sources.append(record(d, 3, 1 if quick else rng.randint(3, 6), 1, value, gated=True))
        elif family == "detour":
            status = "current" if quick else rng.choice(("current", "archived", "failed"))
            sources.extend((record(b, 2, second, 2, other if status == "archived" else value,
                                   status=status, gated=True), record(c, 2, second, 2, value, gated=True)))
        else:
            sources.append(record(b, 2, second, 2, other if family == "conflict" else value,
                                  gated=family == "chain"))
            if family == "conflict":
                sources.append(record(c, 3, 1 if quick else rng.randint(3, 6), 1, value))
        # Alternate which record is correct in a conflict. Never disclose that in the catalog.
        if family == "conflict" and rng.randrange(2):
            value = other
            sources[-1] = replace(sources[-1], value=value)
        rng.shuffle(sources)
        topics.append(Topic(name, rng.randint(1, 5), value, tuple(sources)))
    targets = tuple(t.name for t in rng.sample(topics[1:], 3))
    rng.shuffle(topics)
    return Case(family, seed, tuple(topics), targets)


def matrix_cases(split, seeds=None):
    """Use only development families unless evaluation is explicitly selected."""
    if split == "dev":
        families, defaults = DEV_FAMILIES, DEV_SEEDS
    elif split == "held-out":
        families, defaults = HELD_OUT_FAMILIES, HELD_OUT_SEEDS
    else:
        raise ValueError(f"Unknown split: {split}")
    for family in families:
        for seed in defaults if seeds is None else seeds:
            yield generate(family, seed)


def thinking_training_cases():
    """Expanded development cases, outside either frozen evaluation matrix."""
    for family in (*DEV_FAMILIES, "detour"):
        for seed in THINKING_TRAIN_SEEDS:
            yield generate(family, seed)
