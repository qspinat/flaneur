"""Read a photo's title and caption with plain rules: what kind of picture it is, which
streets it names, which arrondissement, and where the camera stood ("vue prise de …").

What the rules cannot settle is left for a fine reading (see `needs_reading`).
"""

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field

from flaneur import db

STREET_TYPES = (
    "rue|ruelle|boulevard|avenue|place|quai|impasse|passage|pont|carrefour|faubourg|"
    "cour|cité|square|villa|allée|chaussée|port|porte|barrière|parvis|rond-point|"
    "galerie|marché|cul-de-sac|sentier|chemin(?! de fer)|route|butte|montée|esplanade|jardin"
)
# Abbreviated types, as written in captions ("bd Saint-Michel", "av. de l'Opéra").
ABBREVIATIONS = r"bd\b\.?|av\."
# A street mention: optional house number(s), a street type, then its name up to a
# delimiter. "rue de la Poterie", "11 rue de Vouillé", "54-56 boulevard Saint-Michel".
# The name starts with a capital or a digit, unless an article comes first ("rue de la
# montagne Sainte-Geneviève"): without that, "rue marquée par…" would read as a street.
STREET = re.compile(
    r"(?:(?P<number>\d+(?:\s?(?:bis|ter))?(?:\s?[-–]\s?\d+)?)\s+)?"
    rf"\b(?P<type>(?i:{STREET_TYPES}|{ABBREVIATIONS}))\s+"
    r"(?P<name>(?:(?:du|des|de\sla|de\sl'|d'|de|aux?)\s?[^,.;:()\"]*?|[A-ZÀ-Ý0-9][^,.;:()\"]*?))"
    r"(?=\s*(?:,|\.|;|:|\(|\"|$|\svue\b|\set\b|\sà\s|\sau\s|\sangle\b|\sdepuis\b|\svers\b|\sParis\b))"
)
ORDINAL = r"\d{1,2}\s?(?:er|e|ème|eme)"
# "5ème arrondissement", "5ème et 7ème arrondissements", "1er, 4ème, 5ème et 6ème arr…".
ARRONDISSEMENT = re.compile(rf"\b(?:{ORDINAL}(?:\s*,\s*|\s+et\s+))*{ORDINAL}\s+arrondissement")
ROMAN_ARRONDISSEMENT = re.compile(r"\b([IVXL]+)(?:er|e|ème)?\s?arr\.")
_PLACE = r"(?P<place>[^\d,.;()][^,.;()]*?)(?=\s+vers\b|[,.;()]|$)"
# "vue prise de la rue X", "vue prise coin rue X", "vue prise au bas du pont Neuf".
VIEWPOINT = re.compile(
    rf"vue prise (?:de|du|des|depuis|au|à|coin)\s+(?:la\s+|l'|le\s+|les\s+)?{_PLACE}",
    re.IGNORECASE,
)
# "vers le Sud-Est", but not the date wording "vers 1900".
TOWARDS = re.compile(rf"\bvers (?:la |le |l'|les )?{_PLACE}", re.IGNORECASE)

# What a photo shows. A rule reads either the whole title, or only its subject (the title
# before its first comma): "Le déambulatoire, église Saint-Séverin, rue …" is an interior,
# "La rue X, vue prise de la cour…" is not a courtyard. The vocabulary of churches and of
# facade details also names streets and monuments ("rue de la Chapelle", "passage des
# Charniers", "porte des Lilas", "la colonne de Juillet"), so those rules apply only when the
# subject does not start with a street, and the church words only when the title names a
# building. First matching rule wins.
TITLE, SUBJECT, SUBJECT_IN_BUILDING = "title", "subject", "subject in a building"
KIND_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("portrait", SUBJECT, re.compile(r"^\s*(portrait|carte de visite)\b", re.IGNORECASE)),
    ("aerial", TITLE, re.compile(r"vue a[ée]rienne", re.IGNORECASE)),
    # "Intérieur de la cour du Dragon" is a passage seen from inside: still a street view.
    (
        "interior",
        TITLE,
        re.compile(
            r"\bint[ée]rieurs?\b(?!\s+(?:de\s+la\s+cour|du\s+passage|de\s+la\s+cité))",
            re.IGNORECASE,
        ),
    ),
    (
        "courtyard",
        TITLE,
        re.compile(
            r"\b(?:sur cour|cour int[ée]rieure|vue sur cour|dans la cour|vieille cour)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "interior",
        SUBJECT_IN_BUILDING,
        re.compile(
            r"\b(?:nef|ch[œo]e?ur|d[ée]ambulatoire|bas-c[ôo]t[ée]s?|vitra(?:il|ux)|piliers?|"
            r"chapiteaux?|stalles?|b[ée]nitier|retable|baptist[èe]re|base de colonne|"
            r"vo[ûu]tes?|chapelle|autel|orgues?|tribune|jub[ée]|crypte|sacristie|chaire|"
            r"colonnes?|tombeau|toits?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "courtyard",
        SUBJECT,
        re.compile(
            r"\b(?:jardin du presbyt[èe]re|charniers?|clo[îi]tres?|cimeti[èe]re)\b"
            # a garden as the subject itself (a presbytery is often seen from the street)
            r"|^\s*(?:le )?jardin\b",
            re.IGNORECASE,
        ),
    ),
    # A close-up of a facade: reached from the street photo that shows it.
    (
        "detail",
        SUBJECT,
        re.compile(
            r"\b(?:portail|niche|fronton|d[ée]cor sculpt[ée]|bas-relief|statue|mascaron|"
            r"heurtoir|cariatides?|gargouilles?|grilles?)\b|^\s*(?:la )?porte\b",
            re.IGNORECASE,
        ),
    ),
]
# Kinds shown once you enter a place (see places.py).
PLACE_KINDS = {"interior", "courtyard"}
# Kinds that never belong in the walk. A "detail" (close-up of a facade) is reached from the
# street photo that shows that facade.
NOT_A_STREET_VIEW = PLACE_KINDS | {"detail", "portrait", "aerial"}

# The place a title names, most specific building first: "…, église Saint-Séverin, rue …",
# "…, hôtel des abbés de Cluny, …". A courtyard or garden with no name is named by the
# address that follows it: "… dans la cour, 3 rue des Prêtres-Saint-Séverin".
_NAME_END = r"(?=\s*[,()]|\s+(?:et|rue|quai|place|boulevard|avenue)\b|$)"
BUILDINGS = [
    re.compile(rf"\b(?:église|basilique|temple)\s+.+?{_NAME_END}", re.IGNORECASE),
    re.compile(
        rf"\b(?:h[ôo]tel|couvent|collège|abbaye)\s+(?:de |du |des |d')\S.*?{_NAME_END}",
        re.IGNORECASE,
    ),
]
COURTYARD = re.compile(
    r"\b(?:cour|jardin)\b[^,]*,\s*(?P<address>\d+[^,]*?\b(?:rue|quai|place|boulevard)\b[^,]+)",
    re.IGNORECASE,
)
# Applied before reading: typographic apostrophes, and "St." / "Ste." (whose "." would end
# the street name).
NORMALISE = [
    (re.compile("[’ʼ]"), "'"),
    (re.compile(r"\b(S)(t|te)\.\s*(?=[A-ZÀ-Ý])"), r"\1ain\2-"),
]

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


@dataclass(frozen=True)
class StreetMention:
    type: str
    name: str
    number: str | None = None

    @property
    def label(self) -> str:
        return f"{self.type} {self.name}"


@dataclass(frozen=True)
class Reading:
    kind: str  # one of KIND_RULES' kinds, or "unknown" (maybe a street view)
    streets: list[StreetMention] = field(default_factory=list)
    arrondissements: list[int] = field(default_factory=list)
    viewpoint: str | None = None  # where the camera stood, as written ("rue de la Lingerie")
    towards: str | None = None  # what it looked at, when written ("vers le Sud-Est")

    @property
    def needs_reading(self) -> bool:
        """True when rules could not tell where the photo is: a fine reading is needed."""
        return self.kind not in NOT_A_STREET_VIEW and not self.streets


def _roman_to_int(numeral: str) -> int:
    values = [ROMAN[c] for c in numeral.upper()]
    return sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values))


def _arrondissements(text: str) -> list[int]:
    found = {int(n) for m in ARRONDISSEMENT.finditer(text) for n in re.findall(r"\d{1,2}", m[0])}
    found |= {_roman_to_int(m[1]) for m in ROMAN_ARRONDISSEMENT.finditer(text)}
    return sorted(n for n in found if 1 <= n <= 20)


def find_streets(text: str) -> list[StreetMention]:
    mentions: list[StreetMention] = []
    seen: set[str] = set()
    for m in STREET.finditer(text):
        mention = StreetMention(
            type=m["type"].lower(), name=m["name"].strip(" '"), number=m["number"]
        )
        if mention.label.lower() not in seen:
            seen.add(mention.label.lower())
            mentions.append(mention)
    return mentions


def _starts_with_street(subject: str) -> bool:
    """True when the subject is a street ("Rue de la Chapelle", "La porte des Lilas"): a
    street type, then a name with a capital, after at most an article."""
    match = STREET.search(subject)
    return bool(
        match
        and subject[: match.start()].strip().lower() in {"", "la", "le", "les", "l'"}
        and any(c.isupper() for c in match["name"])
    )


def _kind(title: str) -> str:
    subject = title.split(",")[0]
    street_first = _starts_with_street(subject)
    names_a_building = any(pattern.search(title) for pattern in BUILDINGS)
    for kind, scope, rule in KIND_RULES:
        if scope != TITLE and street_first and kind != "portrait":
            continue
        if scope == SUBJECT_IN_BUILDING and not names_a_building:
            continue
        if rule.search(title if scope == TITLE else subject):
            return kind
    return "unknown"


def _normalise(text: str) -> str:
    for pattern, replacement in NORMALISE:
        text = pattern.sub(replacement, text)
    return text


def _place(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match["place"].strip() if match else None


def read(title: str | None, caption: str | None) -> Reading:
    """Read one photo. The title is the short, primary text (it decides the kind); the
    caption adds details, and its streets are used only when the title names none."""
    title, caption = _normalise(title or ""), _normalise(caption or "")
    text = f"{title}. {caption}"
    kind = _kind(title)
    arrondissements = _arrondissements(text)
    return Reading(
        kind=kind,
        streets=find_streets(title) or find_streets(caption),
        arrondissements=arrondissements,
        viewpoint=_place(VIEWPOINT, text),
        towards=_place(TOWARDS, text),
    )


def read_all(conn: sqlite3.Connection) -> int:
    """Read every photo with the rules, replacing earlier rule readings (a fine reading
    is never overwritten). Returns how many photos were read."""
    photos = conn.execute(
        "SELECT id, title, caption FROM photos WHERE id NOT IN "
        "(SELECT photo_id FROM readings WHERE method = 'fine')"
    ).fetchall()
    return db.save_readings(
        conn,
        (
            {
                "photo_id": photo["id"],
                "method": "rules",
                "kind": r.kind,
                "streets": json.dumps([asdict(s) for s in r.streets], ensure_ascii=False),
                "arrondissements": json.dumps(r.arrondissements),
                "viewpoint": r.viewpoint,
                "towards": r.towards,
            }
            for photo in photos
            if (r := read(photo["title"], photo["caption"]))
        ),
    )


def place_of(title: str) -> str | None:
    """The place a title names (a building, or a courtyard by its address), or None."""
    title = _normalise(title)
    for pattern in BUILDINGS:
        if match := pattern.search(title):
            name = match.group(0).strip()
            return name[0].upper() + name[1:]
    if match := COURTYARD.search(title):
        return f"Cour, {match['address'].strip()}"
    return None
