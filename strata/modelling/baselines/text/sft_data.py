"""What the text-sft baseline does to its examples before and after the model: prompts, encoding,
padding, near copies, seeded sampling.

Loads no model, so a stand-in tokenizer tests it. See ``docs/adr/0054``.
"""

import string
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: What the loss ignores: the prompt's positions and the padding.
IGNORE = -100

VARIANTS = ("base", "instruct")

# -- the prompt --------------------------------------------------------------


def template_fields(template: str) -> tuple[str, ...]:
    """The input fields a prompt template names, in order of first use.

    Positional, attribute and index fields are refused, so what the template reads is what
    ``requires_features`` declares.
    """
    found: list[str] = []
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        raise ValueError(f"prompt_template is not a format string: {e}") from None
    for _, name, spec, _ in parsed:
        if name is None:
            continue
        if not name.isidentifier():
            raise ValueError(
                f"prompt_template names {{{name}}}; a template names input fields "
                f"by name only, such as {{subject}}"
            )
        if spec and ("{" in spec):
            raise ValueError(f"prompt_template nests a field in {{{name}:{spec}}}")
        if name not in found:
            found.append(name)
    return tuple(found)


class Prompts:
    """An input as the text the model is given, and a target as the text it learns, tokenized apart.

    ``base`` is completion after ``response_marker``, ending in the end token;
    ``instruct`` is a chat through the tokenizer's template, ending in its end of turn.
    """

    def __init__(
        self,
        variant: str,
        template: str,
        system_prompt: str | None = None,
        response_marker: str = "",
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant is {variant!r}; it takes {' or '.join(VARIANTS)}")
        self.variant = variant
        self.template = template
        self.fields = template_fields(template)
        self.system_prompt = system_prompt
        self.response_marker = response_marker

    def values(self, features: dict) -> str:
        """The template filled from an input. A field it lacks is refused."""
        missing = [f for f in self.fields if f not in features]
        if missing:
            raise ValueError(
                f"the input has no {', '.join(missing)}, which prompt_template "
                f"names (it has: {', '.join(sorted(features)) or 'nothing'})"
            )
        return self.template.format(**{f: features[f] for f in self.fields})

    def _messages(self, features: dict, body: str | None = None) -> list[dict]:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.values(features)})
        if body is not None:
            messages.append({"role": "assistant", "content": body})
        return messages

    def prompt(self, tokenizer, features: dict) -> str:
        """The prompt as a string, special tokens included, ready to tokenize."""
        if self.variant == "base":
            return (tokenizer.bos_token or "") + self.values(features) + self.response_marker
        return tokenizer.apply_chat_template(
            self._messages(features), add_generation_prompt=True, tokenize=False
        )

    def completion(self, tokenizer, features: dict, body: str) -> str:
        """The target and the end token the model must learn to emit."""
        if self.variant == "base":
            return body + tokenizer.eos_token
        # Whatever the chat template appends after the prompt, end of turn included.
        full = tokenizer.apply_chat_template(
            self._messages(features, body), add_generation_prompt=False, tokenize=False
        )
        prompt = self.prompt(tokenizer, features)
        if not full.startswith(prompt):
            raise ValueError(
                "the chat template does not extend its own generation prompt, so "
                "the prompt trained on would not be the one generated from"
            )
        return full[len(prompt) :]

    def end(self, tokenizer) -> str:
        """What follows a target: the end token, or the chat template's end of turn."""
        if self.variant == "base":
            return tokenizer.eos_token
        sentinel = "x"
        after = self.completion(tokenizer, dict.fromkeys(self.fields, ""), sentinel)
        if not after.startswith(sentinel):
            raise ValueError("the chat template changes the assistant's text it is given")
        return after[len(sentinel) :]


def ids(tokenizer, text: str) -> list[int]:
    """``text``'s tokens alone: the prompt carries its own special tokens."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def encode(prompts: Prompts, tokenizer, features: dict, body: str) -> dict:
    """``input_ids`` and ``labels``, the prompt's labels masked out of the loss."""
    prompt_ids = ids(tokenizer, prompts.prompt(tokenizer, features))
    completion_ids = ids(tokenizer, prompts.completion(tokenizer, features, body))
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [IGNORE] * len(prompt_ids) + completion_ids,
    }


class Collator:
    """Right-pads a batch, the padding masked out of attention and loss by position, not by id."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch  # the sft extra; this module loads without it

        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(list(b["input_ids"]) + [self.pad_id] * pad)
            labels.append(list(b["labels"]) + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


# -- the targets -------------------------------------------------------------


def read_target(path: str | Path) -> str:
    """A target exactly as stored. Unreadable is an error: an empty target teaches stopping."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def checksum_of(path: str | Path) -> str:
    """A materialised sample is named for its checksum. See ``docs/adr/0001``."""
    return Path(path).stem


def folded(text: str) -> str:
    """Whitespace collapsed and case folded."""
    return " ".join(text.split()).lower()


def shingles(text: str, n: int = 5) -> frozenset:
    """The text's runs of ``n`` words; a shorter text is one run of all of it."""
    words = text.split()
    if len(words) <= n:
        return frozenset([tuple(words)])
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    shared = len(a & b)
    return shared / (len(a) + len(b) - shared)


def collapse(texts: Sequence[str], keys: Sequence[str], threshold: float) -> list[int]:
    """Which of ``texts`` to keep: one per chained group of near copies, by index, in the order
    given.

    Near copies are identical folded or their five-word shingles reach ``threshold`` Jaccard; a
    group keeps its smallest key.
    """
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    norm = [folded(t) for t in texts]
    first: dict[str, int] = {}
    for i, text in enumerate(norm):
        if text in first:
            union(first[text], i)
        else:
            first[text] = i

    sets = [shingles(t) for t in norm]
    order = sorted(range(len(texts)), key=lambda i: len(sets[i]))
    for a, i in enumerate(order):
        for j in order[a + 1 :]:
            # Sorted by size: once j is too big to reach the threshold, every later one is.
            if len(sets[i]) < threshold * len(sets[j]):
                break
            if jaccard(sets[i], sets[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    return sorted(min(members, key=lambda i: keys[i]) for members in groups.values())


def copies_of(texts: Sequence[str], reference: Sequence[str], threshold: float) -> list[bool]:
    """For each of ``texts``, whether it nearly copies any of ``reference``, by :func:`collapse`'s
    test.
    """
    folded_reference = [folded(t) for t in reference]
    exact = set(folded_reference)
    sets = [shingles(t) for t in folded_reference]
    ranked = sorted((len(s), n) for n, s in enumerate(sets))
    sizes = [size for size, _ in ranked]
    found = []
    for text in texts:
        norm = folded(text)
        if norm in exact:
            found.append(True)
            continue
        own = shingles(norm)
        low = bisect_left(sizes, threshold * len(own))
        high = bisect_right(sizes, len(own) / threshold)
        found.append(any(jaccard(own, sets[n]) >= threshold for _, n in ranked[low:high]))
    return found


# -- generation --------------------------------------------------------------


class SeededSampling:
    """Samples each row of a batch from its own generator, so an output is its seed's whatever
    shared its batch.

    A logits processor for ``generate`` with sampling off: every token but the draw is set to minus
    infinity.
    """

    def __init__(self, seeds: Sequence[int], temperature: float, top_p: float):
        self.seeds = list(seeds)
        self.temperature = temperature
        self.top_p = top_p
        self._generators: list[Any] | None = None

    def __call__(self, input_ids, scores):
        import torch  # the sft extra; this module loads without it

        if self._generators is None:
            self._generators = [
                torch.Generator(device=scores.device).manual_seed(seed) for seed in self.seeds
            ]
        probs = torch.softmax(scores.float() / self.temperature, dim=-1)
        if self.top_p < 1.0:
            ordered, order = probs.sort(dim=-1, descending=True)
            # Kept while the likelier tokens hold less than top_p between them; the likeliest always
            # is.
            before = ordered.cumsum(dim=-1) - ordered
            ordered = ordered.masked_fill(before >= self.top_p, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, order, ordered)
        picked = torch.stack(
            [
                torch.multinomial(row, 1, generator=generator)[0]
                for row, generator in zip(probs, self._generators, strict=True)
            ]
        )
        out = torch.full_like(scores, float("-inf"))
        return out.scatter(-1, picked.unsqueeze(-1), 0.0)
