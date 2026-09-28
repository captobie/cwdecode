"""Text for the labels: callsigns, QSO exchanges, Q-codes and abbreviations, plain English,
random characters, and empty (noise-only) samples, mixed by weight.

Vocabulary lists are ported from HamLexicon (smart-cleanup branch). Callsigns are assigned
to splits by a stable hash, so no callsign appears in more than one of train, val and test.
"""
import re
import zlib
from importlib import resources
from pathlib import Path

import numpy as np

from cwsynth.alphabet import ALIASES, KEYED, PATTERNS, normalize

DEFAULT_WEIGHTS: dict[str, float] = {
    "callsign": 0.20, "qso": 0.30, "abbreviation": 0.15,
    "english": 0.20, "random": 0.10, "noise": 0.05,
}

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ITU prefixes (ported from HamLexicon). Ones ending in a digit already include the call area.
PREFIXES = [
    "K", "W", "N", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AI", "AJ", "AK", "KA", "KB",
    "KC", "KD", "KE", "KF", "KG", "KI", "KJ", "KK", "WA", "WB", "WD", "NA", "NB",
    "KH6", "KL7", "KP4", "KP2", "VE", "VA", "VO", "VY", "XE", "G", "M", "2E", "GW", "GM",
    "GI", "EI", "F", "DL", "DK", "DJ", "DF", "DG", "DH", "DB", "DC", "DD", "DM", "DO",
    "ON", "PA", "PD", "PE", "OZ", "SM", "SA", "LA", "OH", "OE", "HB9", "I", "IK", "IZ",
    "EA", "EB", "EC", "EA8", "CT", "CT3", "SP", "SQ", "OK", "OM", "HA", "YO", "LZ", "SV",
    "9A", "S5", "YU", "UA", "RA", "R", "UR", "UT", "LY", "YL", "ES", "TA", "4X", "4Z",
    "JA", "JH", "JR", "JE", "JF", "JG", "JI", "HL", "BY", "BV", "VU", "VK", "ZL", "ZS",
    "PY", "PU", "LU", "CE", "CX", "HK", "YV", "OA", "VP9", "ZF", "TF", "OX",
]
Q_CODES = [
    "QRA", "QRG", "QRK", "QRL", "QRM", "QRN", "QRO", "QRP", "QRQ", "QRS", "QRT", "QRU",
    "QRV", "QRX", "QRZ", "QSB", "QSK", "QSL", "QSO", "QSP", "QST", "QSX", "QSY", "QTC",
    "QTH", "QTR",
]
ABBREVIATIONS = [
    "CQ", "DE", "TU", "73", "88", "UR", "RST", "ES", "FB", "OM", "YL", "XYL", "OP", "NAME",
    "HR", "WX", "RIG", "ANT", "PWR", "W", "TNX", "TKS", "PSE", "AGN", "CPY", "SRI", "HW",
    "CUL", "CUAGN", "GM", "GA", "GE", "GN", "GL", "DX", "TEST", "NR", "ABT", "BURO", "DR",
    "FER", "HPE", "INFO", "MNI", "OB", "RPT", "SIG", "SOLID", "VY", "WID", "WL", "WKD",
    "ENUF", "HI", "BCNU", "AA", "AB", "EU", "NA", "SA", "AF", "OC", "POTA", "SOTA",
    "K", "KN", "BK", "R", "CL", "AS", "<AR>", "<BT>", "<KN>", "<SK>", "<AS>",
]
NAMES = ["JOHN", "BOB", "JIM", "TOM", "MIKE", "DAVE", "BILL", "STEVE", "DAN", "KEN", "ED",
         "PAUL", "JACK", "CARL", "AL", "PETE", "RAY", "DON", "JOE", "ANN", "SUE", "MARY",
         "HANS", "PIERRE", "YUKI", "IVAN", "LARS", "JAN", "MARCO", "LUIS", "KATE", "LIZ"]
PLACES = ["BOSTON", "DENVER", "SEATTLE", "AUSTIN", "OMAHA", "TULSA", "RENO", "FARGO",
          "BOISE", "TAMPA", "MUNICH", "BERLIN", "PARIS", "LYON", "MADRID", "ROME", "OSLO",
          "TOKYO", "SYDNEY", "TORONTO", "LONDON", "DUBLIN", "PRAGUE", "VIENNA", "WARSAW"]
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
          "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "ON", "QC", "BC"]
RIGS = ["IC7300", "K3", "K4", "KX2", "KX3", "TS590", "FT991", "FT817", "IC705", "FTDX10",
        "HOMEBREW", "HB", "FLEX", "QCX", "TS520"]
ANTENNAS = ["DIPOLE", "VERTICAL", "YAGI", "EFHW", "LOOP", "G5RV", "BEAM", "INV V",
            "HEXBEAM", "WINDOM", "LONG WIRE", "GP"]
WX = ["SUNNY", "RAIN", "CLOUDY", "SNOW", "CLEAR", "WINDY", "COLD", "HOT", "FOG", "WARM"]
SPLITS = ("train", "val", "test")


def _pick(rng: np.random.Generator, items: list[str]) -> str:
    return items[int(rng.integers(len(items)))]


def _letters(rng: np.random.Generator, n: int) -> str:
    return "".join(_pick(rng, list(_LETTERS)) for _ in range(n))


def callsign_split(call: str) -> str:
    """Stable assignment of a callsign to a split: 80 % train, 10 % val, 10 % test."""
    bucket = zlib.crc32(call.encode()) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def _raw_callsign(rng: np.random.Generator) -> str:
    if rng.random() < 0.7:
        prefix = _pick(rng, PREFIXES)
        base = prefix if prefix[-1].isdigit() else prefix + str(rng.integers(10))
    else:
        # Unlisted but well-formed prefixes, so the model doesn't memorize the table.
        form = rng.integers(4)
        prefix = (_letters(rng, 1) if form == 0 else _letters(rng, 2) if form == 1
                  else str(rng.integers(2, 10)) + _letters(rng, 1) if form == 2
                  else _letters(rng, 1) + str(rng.integers(2, 10)))
        base = prefix + str(rng.integers(10))
    suffix_len = int(rng.choice([1, 2, 3, 4], p=[0.10, 0.35, 0.50, 0.05]))
    call = base + _letters(rng, suffix_len)
    r = rng.random()
    if r < 0.06:
        call += _pick(rng, ["/P", "/M", "/MM", "/QRP", "/AM", "/" + str(rng.integers(10))])
    elif r < 0.10:
        call = _pick(rng, ["EA8", "VP2E", "DL", "F", "KH6", "VE3", "PJ4", "HB0"]) + "/" + call
    return call


def callsign(rng: np.random.Generator, split: str = "train") -> str:
    while True:
        call = _raw_callsign(rng)
        if callsign_split(call) == split:
            return call


def _cut(number: str, rng: np.random.Generator) -> str:
    """Contest cut numbers: 0 → T, 9 → N (sometimes)."""
    if rng.random() < 0.5:
        return number
    return number.replace("0", "T").replace("9", "N")


def _rst(rng: np.random.Generator) -> str:
    if rng.random() < 0.5:
        return _pick(rng, ["5NN", "599"])
    return f"{rng.integers(3, 6)}{rng.integers(3, 10)}9"


def _exchange(rng: np.random.Generator, split: str) -> str:
    kind = rng.integers(7)
    if kind == 0:   # CQ WW: zone
        return f"{_rst(rng)} {_cut(f'{rng.integers(1, 41):02d}', rng)}"
    if kind == 1:   # WPX: serial
        return f"{_rst(rng)} {_cut(str(rng.integers(1, 2500)).zfill(3), rng)}"
    if kind == 2:   # ARRL DX: state or power
        return f"{_rst(rng)} {_pick(rng, STATES + ['KW', 'K', '100', '1TT', '5TT'])}"
    if kind == 3:   # Sweepstakes
        return (f"{rng.integers(1, 1500)} {_pick(rng, list('QABUMS'))} {callsign(rng, split)} "
                f"{rng.integers(0, 100):02d} {_pick(rng, STATES)}")
    if kind == 4:   # Field Day
        return f"{rng.integers(1, 20)}{_pick(rng, list('ABCDEF'))} {_pick(rng, STATES)}"
    if kind == 5:   # NAQP / CWT
        return f"{_pick(rng, NAMES)} {_pick(rng, STATES + [str(rng.integers(1, 40000))])}"
    return f"{_rst(rng)} {_pick(rng, STATES)} {_pick(rng, STATES)}"  # POTA / QSO party


def qso(rng: np.random.Generator, split: str = "train") -> str:
    """One over from a realistic exchange: calling, contest, POTA, rag-chew or sign-off."""
    me, you = callsign(rng, split), callsign(rng, split)
    name, place = _pick(rng, NAMES), _pick(rng, PLACES)
    kind = rng.integers(8)
    if kind == 0:
        return _pick(rng, [f"CQ CQ CQ DE {me} {me} K", f"CQ CQ DE {me} {me} <AR> K",
                           f"CQ TEST {me} {me}", f"CQ POTA DE {me} {me} K", f"QRZ? DE {me} K",
                           f"CQ DX CQ DX DE {me} {me} K", f"CQ SOTA DE {me}/P K"])
    if kind == 1:
        return _pick(rng, [f"{you} DE {me} {me} K", f"{you}", f"{you} {you}",
                           f"{you} DE {me} <KN>", f"DE {me}"])
    if kind == 2:
        return _pick(rng, [f"{you} {_exchange(rng, split)}", f"TU {_exchange(rng, split)}",
                           f"{_exchange(rng, split)} TU", f"R {_exchange(rng, split)}"])
    if kind == 3:
        return _pick(rng, [f"TU {me}", "TU", f"R TU {me} TEST", f"{you} TU", "NR?", "AGN?",
                           f"{you}?", "? AGN", "QRL?", "UP", "5NN TU"])
    if kind in (4, 5):
        parts = [_pick(rng, ["GM", "GA", "GE", "GM OM", "GE DR OM", "TNX FER CALL"]),
                 f"UR RST {_rst(rng)} {_rst(rng)}", f"QTH {place}", f"NAME {name} {name}"]
        if rng.random() < 0.5:
            parts.append(f"RIG {_pick(rng, RIGS)} ES ANT {_pick(rng, ANTENNAS)}")
        if rng.random() < 0.4:
            parts.append(f"WX {_pick(rng, WX)} ES {rng.integers(-10, 40)} C")
        if rng.random() < 0.3:
            parts.append(f"PWR {rng.integers(1, 100)} W")
        body = " <BT> ".join(parts[:int(rng.integers(2, len(parts) + 1))])
        return f"{you} DE {me} <BT> {body} <BT> HW? {you} DE {me} <KN>"
    if kind == 6:
        return _pick(rng, [f"{you} DE {me} <BT> TNX FER QSO {name} <BT> 73 ES GL <SK> {you} DE {me}",
                           f"R R TNX {name} 73 <SK>", f"73 CUL <SK> E E", f"{you} 73 TU <SK>",
                           f"FB {name} HPE CUAGN 73 <SK>"])
    return _pick(rng, [f"SRI QRM PSE RPT UR NAME", "QRS PSE", "QSB <BT> PSE AGN",
                       f"QSY UP 2 {me}", f"{you} QRX 5", f"<AS> {you}", "QRN HR <BT> PSE RPT", "BK", "R R <BT> OK"])


def abbreviations(rng: np.random.Generator, split: str = "train") -> str:
    count = int(rng.integers(1, 9))
    words = []
    for _ in range(count):
        r = rng.random()
        if r < 0.35:
            words.append(_pick(rng, Q_CODES) + ("?" if rng.random() < 0.25 else ""))
        elif r < 0.9:
            words.append(_pick(rng, ABBREVIATIONS))
        else:
            words.append(_rst(rng))
    return " ".join(words)


def random_characters(rng: np.random.Generator, split: str = "train") -> str:
    """Code groups drawn uniformly from every keyed token, prosigns included, so rare
    characters are seen and the model can't lean on spelling."""
    groups = int(rng.integers(1, 9))
    return " ".join(
        "".join(_pick(rng, KEYED) for _ in range(int(rng.integers(1, 8)))) for _ in range(groups)
    )


def clean_text(text: str) -> str:
    """Canonical label text from arbitrary prose: a few punctuation marks are mapped to their
    CW equivalents, aliases become prosigns, and everything unsupported becomes a space."""
    text = text.upper().replace(";", ",").replace(":", ",").replace("!", ".")
    text = text.replace('"', " ").replace("(", " ").replace(")", " ")
    allowed = {k for k in PATTERNS if len(k) == 1} | set(ALIASES)
    return normalize("".join(c if c in allowed else " " for c in text))


class CorpusMix:
    """Weighted mix of text sources. `text_file` replaces the bundled word list as the
    source of plain English (random spans of its words, punctuation kept where sendable)."""

    def __init__(self, weights: dict[str, float] | None = None, text_file: str | Path | None = None):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        unknown = set(self.weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown corpus sources: {sorted(unknown)}")
        self.sources = [s for s, w in self.weights.items() if w > 0]
        p = np.array([self.weights[s] for s in self.sources], dtype=float)
        self.p = p / p.sum()
        self.text_file = str(text_file) if text_file else None
        if text_file:
            self.words = clean_text(Path(text_file).read_text(encoding="utf-8", errors="ignore")).split()
            self.from_file = True
        else:
            data = resources.files("cwsynth").joinpath("data/english_words.txt").read_text()
            self.words = data.split()
            self.from_file = False
        if not self.words:
            raise ValueError("the English text source is empty")

    def english(self, rng: np.random.Generator, split: str = "train") -> str:
        count = int(rng.integers(2, 13))
        if self.from_file:
            start = int(rng.integers(0, max(1, len(self.words) - count)))
            return " ".join(self.words[start:start + count])
        words = [_pick(rng, self.words) for _ in range(count)]
        if rng.random() < 0.3:
            words[-1] += _pick(rng, [".", "?", ","])
        return " ".join(words)

    def _generate(self, source: str, rng: np.random.Generator, split: str) -> str:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if source == "noise":
            return ""
        return {"callsign": callsign, "qso": qso, "abbreviation": abbreviations,
                "random": random_characters, "english": self.english}[source](rng, split)

    def sample(self, rng: np.random.Generator, split: str = "train") -> tuple[str, str]:
        """(source name, text). Noise-only samples have empty text."""
        source = self.sources[int(rng.choice(len(self.sources), p=self.p))]
        return source, self._generate(source, rng, split)

    def sample_text(self, rng: np.random.Generator, split: str = "train") -> str:
        """Non-empty text, e.g. for an interfering station."""
        while True:
            source, text = self.sample(rng, split)
            if text:
                return text
