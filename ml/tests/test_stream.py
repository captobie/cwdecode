"""The streaming reference decoder (the Swift port's tests mirror these cases)."""
import numpy as np

from cwmodel.stream import OUTPUT_FRAME_S, StreamingDecoder, drops_repeat, emissions, handoff, reconcile
from cwsynth.alphabet import TOKEN_TO_ID, TOKENS


def test_reconcile_adds_missed_tokens_and_skips_repeats():
    assert reconcile(["N", "O", "W"], ["O", "W", " ", "5"], [" "]) == ([" ", "5"], 0)
    assert reconcile(["O", "W", " "], ["O", "W"], [" ", "5"]) == ([], 1)
    assert reconcile(["A"], ["B"], ["C"]) == ([], 0)


def test_handoff_lands_in_the_widest_gap():
    assert abs(handoff([1.0, 1.2], 0.9, 1.5) - (0.9 + 37 * OUTPUT_FRAME_S)) < 1e-9
    assert handoff([], 2.0, 2.0) == 2.0


def test_emissions_are_stamped_at_their_first_frame():
    ids = [0, TOKEN_TO_ID["C"], TOKEN_TO_ID["C"], 0, TOKEN_TO_ID["Q"]]
    log_probs = np.full((len(ids), len(TOKENS)), -10.0)
    log_probs[np.arange(len(ids)), ids] = 0.0
    assert emissions(log_probs, 10.0) == [("C", 10.0 + OUTPUT_FRAME_S), ("Q", 10.0 + 4 * OUTPUT_FRAME_S)]


def test_repeated_spaces_and_commas_are_dropped():
    assert drops_repeat(",", ",") and drops_repeat(" ", " ")
    assert not drops_repeat(",", " ") and not drops_repeat("E", "E") and not drops_repeat("<BT>", "<BT>")


def test_streaming_commits_one_comma_for_a_double_firing():
    ids = [TOKEN_TO_ID[t] for t in "A"] + [0, TOKEN_TO_ID[","], 0, TOKEN_TO_ID[","], 0, TOKEN_TO_ID[" "], TOKEN_TO_ID["B"]]
    def fixed(spec):
        log_probs = np.full((len(ids), len(TOKENS)), -10.0)
        log_probs[np.arange(len(ids)), ids] = 0.0
        return log_probs
    decoder = StreamingDecoder(fixed)
    decoder.process(np.zeros(800, np.float32))
    decoder.finish()
    assert decoder.text == "A, B"
