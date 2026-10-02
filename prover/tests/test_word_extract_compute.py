"""The word extraction witness (compute_fns.word_extract_compute), on the CPU.

A one-word extraction (N = 1, which the top-k builders use whenever one word
covers the width) has no stride to infer the word width from; it was taken as
0 and every word came out 0, so an honest proof failed its own relation. With
one word, x = 1·w0 forces w0 = x. Two and more words keep the bit slicing."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from claims import WordExtractionClaim       # noqa: E402
from compute_fns import word_extract_compute  # noqa: E402
from core import Variable                    # noqa: E402


def _extract(vals, n_words, word_bits):
    x = Variable("x", length=len(vals))
    words = [Variable(f"w{n}", length=len(vals)) for n in range(n_words)]
    claim = WordExtractionClaim(x=x, words=words,
                                coeffs=[1 << (n * word_bits) for n in range(n_words)],
                                length=len(vals))
    live = {x: torch.tensor(vals, dtype=torch.int64).to(torch.uint64)}
    out = word_extract_compute(claim, live)
    return [out[w].view(torch.int64).tolist() for w in words]


def test_one_word_is_the_value():
    assert _extract([5, 0, 4095, 1234], 1, 12) == [[5, 0, 4095, 1234]]


def test_two_words_are_the_low_and_high_slices():
    assert _extract([5, 0x1234, 0xFFFF], 2, 8) == [[5, 0x34, 0xFF], [0, 0x12, 0xFF]]


def test_one_word_keeps_an_out_of_range_value_for_the_range_check():
    # the range LogUp on w0, not the extraction, is what rejects x >= 2^B
    assert _extract([1 << 20], 1, 12) == [[1 << 20]]
