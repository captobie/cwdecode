"""The label vocabulary: one token per distinct Morse pattern.

A CTC label has to be a function of the audio, so sounds that are identical get a single
canonical token: `+` is sent exactly like AR, `=` like BT and `(` like KN, and all three are
normalized to the prosign. Prosigns are written in angle brackets (`<SK>`); the letters
`SK` without brackets are two characters with a letter gap between them.

The token order below *is* the model's output layer. Append new tokens at the end, never
reorder, or trained models stop matching.
"""
import re

BLANK = "<blank>"
SPACE = " "

PATTERNS: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "-": "-....-",
    "@": ".--.-.", "'": ".----.",
    "<AR>": ".-.-.", "<BT>": "-...-", "<KN>": "-.--.", "<SK>": "...-.-",
}

ALIASES: dict[str, str] = {"+": "<AR>", "=": "<BT>", "(": "<KN>"}

TOKENS: list[str] = [BLANK, SPACE, *PATTERNS]
TOKEN_TO_ID: dict[str, int] = {token: i for i, token in enumerate(TOKENS)}
PROSIGNS: list[str] = [t for t in PATTERNS if t.startswith("<")]
# Every token that is keyed (everything except blank and space).
KEYED: list[str] = list(PATTERNS)

_PIECE = re.compile(r"<[A-Za-z]+>|\s+|.")

assert len(set(PATTERNS.values())) == len(PATTERNS), "each pattern must map to one token"


def tokenize(text: str) -> list[str]:
    """Canonical tokens for `text`. Runs of whitespace become one space and leading or
    trailing whitespace is dropped. Raises ValueError for anything not in the vocabulary."""
    tokens: list[str] = []
    for match in _PIECE.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            if tokens and tokens[-1] != SPACE:
                tokens.append(SPACE)
            continue
        token = piece.upper()
        token = ALIASES.get(token, token)
        if token not in PATTERNS:
            raise ValueError(f"not in the CW vocabulary: {piece!r}")
        tokens.append(token)
    if tokens and tokens[-1] == SPACE:
        tokens.pop()
    return tokens


def normalize(text: str) -> str:
    """The canonical label string for `text`."""
    return "".join(tokenize(text))


def encode(text: str) -> list[int]:
    """Token IDs for CTC targets (blank is 0 and never appears in a target)."""
    return [TOKEN_TO_ID[t] for t in tokenize(text)]


def decode(ids: list[int]) -> str:
    """Label string for already-collapsed token IDs; blanks are skipped."""
    return "".join(TOKENS[i] for i in ids if i != 0)


def is_supported(text: str) -> bool:
    try:
        tokenize(text)
    except ValueError:
        return False
    return True
