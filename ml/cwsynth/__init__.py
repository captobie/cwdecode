"""Synthetic CW audio with ground-truth labels, for training a CTC decoder."""

from cwsynth.alphabet import TOKENS, decode, encode, normalize, tokenize
from cwsynth.corpus import CorpusMix
from cwsynth.dataset import write_dataset
from cwsynth.render import Sample, generate, render
from cwsynth.spec import ConditionRanges, SampleSpec, sample_spec

__version__ = "0.1.0"

__all__ = [
    "TOKENS", "decode", "encode", "normalize", "tokenize",
    "CorpusMix", "write_dataset", "Sample", "generate", "render",
    "ConditionRanges", "SampleSpec", "sample_spec",
]
