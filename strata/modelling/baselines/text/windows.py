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
    """A document exactly as stored, which is what span offsets count.

    ``newline=""`` keeps ``\r\n`` as two characters, as the offsets do.
    """
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read()
    except OSError as e:
        print(f"warning: unreadable document {path} ({e}), using empty text", file=sys.stderr)
        return ""


def tokens_touching(span, windows) -> list[tuple[int, int]]:
    """Every token a span overlaps, across each window the document became."""
    touched: list[tuple[int, int]] = []
    for window in windows:
        tokens = [(int(s), int(e)) for s, e in window if not (s == 0 and e == 0)]
        if not tokens:
            continue
        starts = [s for s, _ in tokens]
        ends = [e for _, e in tokens]
        # From the first token past the span's start to the first starting at or after its end.
        touched.extend(tokens[bisect_right(ends, span.start) : bisect_left(starts, span.end)])
    return touched


class WindowDataset(Dataset):
    """One item per window, not per document; re-encoded per access rather than held in memory.

    See ``docs/adr/0014``.
    """

    def __init__(self, samples: list[Example], encode, scan):
        self.samples = samples
        self.encode = encode
        self.index: list[tuple[int, int]] = []
        #: Supervision a target loses on the way in (docs/adr/0036).
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
    """Tokenisation into windows, shared by every text head; offsets are into the original string.
    See ``docs/adr/0014``.
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
        """Whether this head can learn anything from a target."""
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
        """How many windows this document needs, and what it loses on the way in. See
        ``docs/adr/0036``.
        """
        encoded = self.tokenizer(
            text, truncation=True, return_offsets_mapping=True, **self._windowing()
        )
        windows = encoded["offset_mapping"]
        if self.window is None:
            windows = [windows]
        if target is not None and not self._trainable(target):
            # docs/adr/0036
            return 0, {"empty_targets": 1}
        return len(windows), self._alignment(target, windows)
