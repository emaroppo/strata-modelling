"""A document as the windows an encoder can take, with offsets into the original."""

import sys
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ...model import Example

DEFAULT_ENCODER = "distilbert-base-uncased"


def read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"warning: unreadable document {path} ({e}), using empty text", file=sys.stderr)
        return ""


def tokens_touching(span, windows) -> list[tuple[int, int]]:
    """Every token a span overlaps, across each window the document became.

    Bisected rather than scanned: a dense document runs to hundreds of
    spans over five hundred tokens, and this is asked once per span on
    every dataset built.
    """
    touched: list[tuple[int, int]] = []
    for window in windows:
        tokens = [(int(s), int(e)) for s, e in window if not (s == 0 and e == 0)]
        if not tokens:
            continue
        starts = [s for s, _ in tokens]
        ends = [e for _, e in tokens]
        # First token reaching past the span's start, up to the first one
        # starting at or after its end
        touched.extend(tokens[bisect_right(ends, span.start) : bisect_left(starts, span.end)])
    return touched


class WindowDataset(Dataset):
    """One item per *window*, not per document.

    A document longer than the encoder's input becomes several training
    items rather than a truncated one (``docs/adr/0014``). The index is built once by counting
    each document's windows; ``__getitem__`` re-encodes its document and
    returns the window asked for, which costs a tokenisation per access and
    saves holding every padded window in memory at once.
    """

    def __init__(self, samples: list[Example], encode, scan):
        self.samples = samples
        self.encode = encode
        self.index: list[tuple[int, int]] = []
        #: What was noticed while counting windows: supervision a target
        #: loses on the way in. docs/adr/0036
        self.diagnostics: Counter = Counter()
        for i, sample in enumerate(samples):
            windows, noticed = scan(read_text(sample.path), sample.target)
            self.index.extend((i, w) for w in range(windows))
            self.diagnostics.update(noticed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sample_idx, window = self.index[idx]
        sample = self.samples[sample_idx]
        return self.encode(read_text(sample.path), sample.target)[window]


class Windowed:
    """Tokenisation into windows, shared by every text head.

    ``window`` off means truncation at ``MAX_LENGTH``. ``offset_mapping`` is
    character offsets into the *original* string for every window, so
    nothing is rebased. See ``docs/adr/0014``.
    """

    MAX_LENGTH = 512

    def __init__(self, encoder: str, window: int | None, window_overlap: int):
        self.encoder = encoder
        self.window = window
        self.window_overlap = window_overlap
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.encoder)
        return self._tokenizer

    # -- what a head adds ----------------------------------------------------

    def _item(self, fields: dict, text: str, offsets, target) -> dict:
        """Add this head's supervision to one already-tokenised window."""
        raise NotImplementedError

    def _trainable(self, target) -> bool:
        """Whether this head can learn anything from a target. See the base."""
        return True

    def _alignment(self, target, windows) -> dict[str, int]:
        """What this head loses aligning a target to tokens. Nothing, by default."""
        return {}

    # -- shared --------------------------------------------------------------

    def _windowing(self) -> dict:
        if self.window is None:
            return {"max_length": self.MAX_LENGTH}
        # The tokenizer calls the overlap "stride"; the step is window - overlap.
        return {
            "max_length": self.window,
            "stride": self.window_overlap,
            "return_overflowing_tokens": True,
        }

    def _encode(self, text: str, target) -> list[dict]:
        """One encoded item per window, and a list even when there is one."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
            **self._windowing(),
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        return [
            self._item({k: v[i] for k, v in encoded.items()}, text, offsets[i], target)
            for i in range(len(offsets))
        ]

    def _scan(self, text: str, target) -> tuple[int, dict[str, int]]:
        """How many windows this document needs, and what it loses on the way in.

        One tokenisation answering both. See ``docs/adr/0036``.
        """
        encoded = self.tokenizer(
            text, truncation=True, return_offsets_mapping=True, **self._windowing()
        )
        windows = encoded["offset_mapping"]
        if self.window is None:
            windows = [windows]
        if target is not None and not self._trainable(target):
            # No windows, so no training items, and counted. docs/adr/0036
            return 0, {"empty_targets": 1}
        return len(windows), self._alignment(target, windows)
