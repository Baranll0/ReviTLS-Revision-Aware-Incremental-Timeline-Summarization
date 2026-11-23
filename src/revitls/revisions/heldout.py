"""Seeded held-out revision suite with fictional entities and negative controls.

Unlike the hand-written scenario templates in ``benchmark.py`` (which the cue lexicon and
prompts were developed against), this suite is generated from slot grammars with several
surface forms per case type. It is intended to be run once, after the system is frozen.

Case types
----------
numeric / date / cause : a fact is corrected after a delay (``revision``).
retraction             : an unconfirmed event is shown not to have occurred (``retraction``).
neg_denial             : a real event followed by an article that denies a side allegation or
                         retracts a *remark*. The event must stay active.
neg_progress           : a real event followed by a follow-up that contains a revision cue
                         word ("clarified") but does not contradict anything. The original
                         fact must be retained.

Cues: ``explicit`` articles use correction vocabulary; ``implicit`` articles state the new
fact with no lexicon cue and a longer delay.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta

from revitls.evaluation.revision_metrics import RevisionTestCase
from revitls.schemas import Article

PLACES = [
    "Brandvik", "Oskarsholm", "Talvera", "Merrow Bay", "Kestrel Point", "Ildara", "Norwick",
    "Havenmoor", "Selkirk Reach", "Dunmara", "Veldt Harbour", "Pellam", "Ostrava Heights",
    "Corvane", "Lysmoor", "Tarnowe", "Quillon", "Harrowgate Falls", "Zerrin", "Ambersea",
    "Wexmoor", "Folkran", "Garrowby", "Ithmere", "Jorvik Sands", "Kellam", "Lorrimar",
    "Maldane", "Nethervale", "Orsolo",
]
SITES = [
    "water treatment plant", "rail bridge", "ferry terminal", "grain silo complex", "power substation",
    "hospital annex", "dam spillway", "container depot", "cable car station", "sewage works",
    "cold storage warehouse", "textile mill", "fuel depot", "pumping station", "bus garage",
    "observatory", "tunnel ventilation hub", "fish market", "radio tower", "customs hall",
]
ORGS = [
    "the regional civil protection agency", "the city council", "the provincial health authority",
    "the transport ministry", "the port authority", "the national statistics office",
    "the county sheriff's office", "the utilities regulator", "the fire service",
    "the environmental agency",
]
UNITS = [
    ("residents", "evacuated"), ("workers", "injured"), ("passengers", "stranded"),
    ("households", "left without power"), ("patients", "transferred"), ("vessels", "delayed"),
    ("students", "sent home"), ("livestock", "lost"),
]
CAUSES = [
    "a cracked coolant line", "a firmware update", "an unsecured loading hatch", "a lightning strike",
    "a transformer fault", "a contaminated fuel batch", "a mislabelled valve", "a failed weld seam",
    "a software scheduling error", "a worn drive belt", "a flooded cable duct", "a blocked drainage channel",
    "an overloaded circuit breaker", "a corroded support bracket", "a sensor calibration fault",
    "a misrouted shipment",
]
RUMORS = [
    "a chemical leak", "a second explosion", "a gas pipeline rupture", "a prison escape",
    "a bridge collapse", "a bank run", "a fuel shortage declaration", "a dam breach",
    "a hospital fire", "a train derailment",
]
SPOKESPEOPLE = ["Marta Voss", "Idris Kalem", "Joanna Pryce", "Tomas Arden", "Lena Okafor", "Ravi Dolan"]
KINDS = ("numeric", "date", "cause", "retraction")


@dataclass(frozen=True)
class HeldoutSpec:
    n_per_cell: int = 10
    n_neg_denial: int = 15
    n_neg_progress: int = 15
    seed: int = 20261123
    start: date = date(2015, 1, 5)
    spacing_days: int = 120


def _fmt_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def _num(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randrange(lo, hi)


def _number_pair(rng: random.Random) -> tuple[int, int]:
    """(old, new) counts where neither digit string contains the other nor looks like a year."""
    while True:
        n1 = _num(rng, 1100, 8900)
        n2 = n1 + _num(rng, 1500, 6000)
        if 2000 <= n1 <= 2100 or 2000 <= n2 <= 2100:
            continue
        if str(n1) in str(n2) or str(n2) in str(n1):
            continue
        return n1, n2


def _fmt_n(n: int) -> str:
    return f"{n:,}"


def _background(case_id: str, place: str, base: date, rng: random.Random) -> list[Article]:
    topics = [
        ("Council approves new bus lanes", "The council in {p} approved a package of new bus lanes on Tuesday."),
        ("Local bakery wins regional award", "A family bakery in {p} won a regional food award this week."),
        ("School board debates timetable change", "The school board in {p} debated moving start times by thirty minutes."),
    ]
    out = []
    for k, (title, body) in enumerate(rng.sample(topics, 2)):
        out.append(
            Article(
                article_id=f"bg_{case_id}_{k}",
                topic="heldout",
                title=title,
                content=body.format(p=place),
                publish_date=base + timedelta(days=1 + k),
                source="local",
            )
        )
    return out


def _repeats(init_title: str, init_body: str, n: int, d0: date, rng: random.Random) -> list[tuple[str, str, date]]:
    leads = ["Regional outlets repeated that ", "Wire services relayed that ", "Local radio reported that "]
    out = [(init_title, init_body, d0)]
    for k in range(1, n):
        lead = leads[(k - 1) % len(leads)]
        out.append((init_title, lead + init_body[0].lower() + init_body[1:], d0 + timedelta(days=rng.choice([0, 0, 1]))))
    return out


def _build_numeric(rng: random.Random, cue: str, place: str, site: str, org: str, d0: date):
    n1, n2 = _number_pair(rng)
    noun, verb = rng.choice(UNITS)
    form = rng.randrange(2)
    init_title = f"{org.capitalize()} reports {_fmt_n(n1)} {noun} {verb} in {place}"
    init_body = (
        f"{org.capitalize()} said that {_fmt_n(n1)} {noun} were {verb} after the incident at the {site} in {place}."
        if form == 0
        else f"According to {org}, a total of {_fmt_n(n1)} {noun} have been {verb} following trouble at the {site} in {place}."
    )
    if cue == "explicit":
        delay = rng.randint(2, 8)
        title = f"{org.capitalize()} revises {noun} figure for {place}"
        body = (
            f"Correction: {org} revised its earlier report and now says {_fmt_n(n2)} {noun} were {verb} after the incident at the {site} in {place}."
            if rng.randrange(2) == 0
            else f"An updated tally from {org} puts the number of {noun} {verb} at the {site} in {place} at {_fmt_n(n2)}; the earlier figure was revised after a recount."
        )
    else:
        delay = rng.randint(15, 35)
        title = f"{place} {site} incident: new count of {noun}"
        body = (
            f"A full count at the {site} in {place} found {_fmt_n(n2)} {noun} {verb}, {org} said."
            if rng.randrange(2) == 0
            else f"{org.capitalize()} now says {_fmt_n(n2)} {noun} were {verb} after the incident at the {site} in {place}."
        )
    return init_title, init_body, title, body, delay, _fmt_n(n2), _fmt_n(n1), d0, "NUMERIC_FACT"


def _build_date(rng: random.Random, cue: str, place: str, site: str, org: str, d0: date):
    wrong = d0
    true = wrong + timedelta(days=rng.choice([-3, -2, -1, 1, 2, 3]))
    pub1 = max(wrong, true) if true > wrong else wrong
    event = rng.choice(["inspection", "board hearing", "public vote", "safety review"])
    init_title = f"{org.capitalize()} {event} at {place} {site}"
    init_body = f"{org.capitalize()} held its {event} at the {site} in {place} on {_fmt_date(wrong)}."
    if cue == "explicit":
        delay = rng.randint(2, 8)
        title = f"Clarification on date of {place} {event}"
        body = (
            f"Correction: the {event} at the {site} in {place} took place on {_fmt_date(true)}, not {_fmt_date(wrong)} as previously reported."
            if rng.randrange(2) == 0
            else f"{org.capitalize()} corrected its earlier report: the {event} at the {site} in {place} was held on {_fmt_date(true)}."
        )
    else:
        delay = rng.randint(15, 35)
        title = f"Records show date of {place} {event}"
        body = (
            f"Official minutes show the {event} at the {site} in {place} was held on {_fmt_date(true)}."
            if rng.randrange(2) == 0
            else f"The {event} at the {site} in {place} happened on {_fmt_date(true)}, {org} confirmed."
        )
    return init_title, init_body, title, body, delay, true.isoformat(), wrong.isoformat(), pub1, "DATE_CORRECTION"


def _build_cause(rng: random.Random, cue: str, place: str, site: str, org: str, d0: date):
    c1, c2 = rng.sample(CAUSES, 2)
    init_title = f"Outage at {place} {site} blamed on {c1}"
    init_body = f"Early statements blamed {c1} for the outage at the {site} in {place}."
    if cue == "explicit":
        delay = rng.randint(2, 8)
        title = f"Investigators rule out {c1} in {place} outage"
        body = (
            f"Investigators ruled out {c1} and concluded that {c2} caused the outage at the {site} in {place}."
            if rng.randrange(2) == 0
            else f"Official update: {org} retracted the earlier explanation and said {c2} caused the outage at the {site} in {place}."
        )
    else:
        delay = rng.randint(15, 35)
        title = f"Final report on {place} {site} outage"
        body = (
            f"The final report found that {c2} caused the outage at the {site} in {place}."
            if rng.randrange(2) == 0
            else f"{org.capitalize()} said the outage at the {site} in {place} was traced to {c2}."
        )
    return init_title, init_body, title, body, delay, c2, c1, d0, "ENTITY_ATTRIBUTION"


def _build_retraction(rng: random.Random, cue: str, place: str, site: str, org: str, d0: date):
    r = rng.choice(RUMORS)
    init_title = f"Unconfirmed reports of {r} at {place} {site}"
    init_body = f"Social media users claimed {r} had occurred at the {site} in {place}."
    if cue == "explicit":
        delay = rng.randint(2, 8)
        title = f"{org.capitalize()} debunks {r} rumor in {place}"
        body = (
            f"{org.capitalize()} confirmed that no {r[2:]} occurred at the {site} in {place}; the earlier reports were false."
            if rng.randrange(2) == 0
            else f"Officials debunked the rumor: there was no {r[2:]} at the {site} in {place}, {org} said."
        )
    else:
        delay = rng.randint(10, 30)
        title = f"Inspectors visit {place} {site}"
        body = (
            f"Inspectors who visited the {site} in {place} found no sign of {r}, {org} said."
            if rng.randrange(2) == 0
            else f"{org.capitalize()} said nothing resembling {r} took place at the {site} in {place}."
        )
    return init_title, init_body, title, body, delay, "", r, d0, "RETRACTION"


_BUILDERS = {"numeric": _build_numeric, "date": _build_date, "cause": _build_cause, "retraction": _build_retraction}


def generate_heldout_suite(spec: HeldoutSpec | None = None) -> tuple[list[Article], list[RevisionTestCase]]:
    """Return (chronologically sorted articles, cases) for the held-out suite."""
    spec = spec or HeldoutSpec()
    rng = random.Random(spec.seed)
    plan: list[tuple[str, str]] = []
    for kind in KINDS:
        for cue in ("explicit", "implicit"):
            plan += [(kind, cue)] * spec.n_per_cell
    plan += [("neg_denial", "explicit")] * spec.n_neg_denial
    plan += [("neg_progress", "explicit")] * spec.n_neg_progress
    rng.shuffle(plan)

    places = PLACES * ((len(plan) // len(PLACES)) + 1)
    rng.shuffle(places)

    articles: list[Article] = []
    cases: list[RevisionTestCase] = []
    for idx, (kind, cue) in enumerate(plan):
        place = places[idx]
        site = rng.choice(SITES)
        org = rng.choice(ORGS)
        d0 = spec.start + timedelta(days=idx * spec.spacing_days)
        case_id = f"h{idx:03d}_{kind}_{cue}"
        n_rep = rng.choice([2, 3])

        if kind in _BUILDERS:
            it, ib, ct, cb, delay, expected, stale, pub1, rtype = _BUILDERS[kind](rng, cue, place, site, org, d0)
            for k, (t, b, d) in enumerate(_repeats(it, ib, n_rep, pub1, rng)):
                articles.append(Article(article_id=f"{case_id}_init_{k}", topic="heldout", title=t, content=b, publish_date=d, source="wire"))
            rev_pub = max(pub1, d0) + timedelta(days=delay)
            articles.append(Article(article_id=f"{case_id}_rev", topic="heldout", title=ct, content=cb, publish_date=rev_pub, source="official"))
            cases.append(
                RevisionTestCase(
                    case_id=case_id, expected_final_fact=expected, stale_fact_substring=stale,
                    should_be_retracted=(kind == "retraction"), case_type="retraction" if kind == "retraction" else "revision",
                    cue=cue, revision_type=rtype,
                )
            )
        elif kind == "neg_denial":
            name = rng.choice(SPOKESPEOPLE)
            facility = f"{place} {site}"
            it = f"{org.capitalize()} opens {facility}"
            ib = f"{org.capitalize()} opened the {facility} to the public on {_fmt_date(d0)} after two years of construction."
            if rng.randrange(2) == 0:
                ct = f"{name} denies cost-cutting claims over {facility}"
                cb = f"{name}, a spokesperson for {org}, denied claims that corners were cut during construction of the {facility}, which opened as planned."
            else:
                ct = f"{name} retracts remark about {facility}"
                cb = f"{name} retracted a remark criticising the {facility} and apologised; the {facility} remains open to the public."
            for k, (t, b, d) in enumerate(_repeats(it, ib, n_rep, d0, rng)):
                articles.append(Article(article_id=f"{case_id}_init_{k}", topic="heldout", title=t, content=b, publish_date=d, source="wire"))
            articles.append(Article(article_id=f"{case_id}_rev", topic="heldout", title=ct, content=cb, publish_date=d0 + timedelta(days=rng.randint(2, 8)), source="press"))
            cases.append(RevisionTestCase(case_id=case_id, expected_final_fact=facility.lower(), stale_fact_substring="", case_type="neg_denial", cue=cue))
        else:  # neg_progress
            n1, _ = _number_pair(rng)
            noun, verb = rng.choice(UNITS)
            k_sh = rng.randint(3, 9)
            it = f"{org.capitalize()} reports {_fmt_n(n1)} {noun} {verb} in {place}"
            ib = f"{org.capitalize()} said that {_fmt_n(n1)} {noun} were {verb} after the incident at the {site} in {place}."
            ct = f"{org.capitalize()} clarifies shelter arrangements in {place}"
            cb = f"{org.capitalize()} clarified that the {_fmt_n(n1)} {noun} {verb} after the incident at the {site} in {place} are being housed in {k_sh} shelters."
            for k, (t, b, d) in enumerate(_repeats(it, ib, n_rep, d0, rng)):
                articles.append(Article(article_id=f"{case_id}_init_{k}", topic="heldout", title=t, content=b, publish_date=d, source="wire"))
            articles.append(Article(article_id=f"{case_id}_rev", topic="heldout", title=ct, content=cb, publish_date=d0 + timedelta(days=rng.randint(1, 4)), source="official"))
            cases.append(RevisionTestCase(case_id=case_id, expected_final_fact=_fmt_n(n1), stale_fact_substring="", case_type="neg_progress", cue=cue))

        articles.extend(_background(case_id, place, d0, rng))

    articles.sort(key=lambda a: (a.publish_date, a.article_id))
    return articles, cases
