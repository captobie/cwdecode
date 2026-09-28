"""Greedy CTC decoding and character error rate, counted in tokens (a prosign is one)."""
import torch

from cwsynth.alphabet import TOKENS, tokenize


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor | None = None) -> list[str]:
    """[frames, batch, classes] → one string per batch item: best token per frame, repeats
    collapsed, blanks removed."""
    best = log_probs.argmax(dim=-1).T.cpu()   # [batch, frames]
    out = []
    for b, row in enumerate(best):
        if lengths is not None:
            row = row[:int(lengths[b])]
        ids = torch.unique_consecutive(row)
        out.append("".join(TOKENS[i] for i in ids.tolist() if i != 0).strip())
    return out


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def token_errors(prediction: str, reference: str) -> tuple[int, int]:
    """(edits, reference length) in tokens, with runs of spaces normalized."""
    pred, ref = tokenize(prediction), tokenize(reference)
    return edit_distance(pred, ref), len(ref)
