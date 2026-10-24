"""What the text-sft baseline does to its examples before and after the model.

The prompt each input becomes, the tokens and labels an example is trained
on, the batches they are padded into, the near copies left out, and the
sampling that gives every output its own seed. None of it loads a model, so
all of it is tested with a stand-in tokenizer. torch is imported only where
a tensor is made. See ``docs/adr/0054``.
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

    A field is named, ``{subject}``, never positional: the input is the
    sample's fields, which have names and no order. Attribute and index
    access are refused too, so what the template reads is exactly what
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
    """An input as the text the model is given, and a target as the text it learns.

    The two are tokenized apart, at training as at generation, so the
    prompt a model learned after is the one it is given. ``base`` is text
    completion: the begin token, the template, then ``response_marker``,
    and the target ends with the end token. ``instruct`` is a chat: the
    system prompt, the template as the user's turn, and the target as the
    assistant's, wrapped by the tokenizer's chat template, whose end-of-turn
    token is what the model learns to stop on.
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
        # Whatever the chat template appends after the prompt: the body as
        # it renders it, and its own end of turn
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
    """Right-pads a batch; the padding is masked out of the attention and the loss.

    Masked by position, not by token id, so a tokenizer whose padding token
    is its end token still learns to stop.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

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
    """Which of ``texts`` to keep: one per group of near copies, by index.

    Two texts are near copies when, folded, they are identical or their
    five-word runs overlap by a Jaccard of at least ``threshold``. Matches
    chain: A like B and B like C make one group. Each group keeps the
    member with the smallest key, its checksum, so a rerun keeps the same
    one. Returned in the order given.
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
            # Sorted by size: once j is too big for the Jaccard to reach the
            # threshold, every later one is too
            if len(sets[i]) < threshold * len(sets[j]):
                break
            if jaccard(sets[i], sets[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(texts)):
        groups[find(i)].append(i)
    return sorted(min(members, key=lambda i: keys[i]) for members in groups.values())


def copies_of(texts: Sequence[str], reference: Sequence[str], threshold: float) -> list[bool]:
    """For each of ``texts``, whether it nearly copies any of ``reference``.

    The same test as :func:`collapse`. Only references of a size that
    could reach the threshold are compared.
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
    """Samples each row of a batch with its own seed, so an output is its seed's.

    Torch samples a batch from one generator, so an output would depend on
    which outputs shared its batch. This draws each row from a generator
    of its own, with the temperature and top-p applied here, and leaves
    every token but the drawn one at minus infinity: given to ``generate``
    as a logits processor with sampling off, the greedy pick is the draw.
    """

    def __init__(self, seeds: Sequence[int], temperature: float, top_p: float):
        self.seeds = list(seeds)
        self.temperature = temperature
        self.top_p = top_p
        self._generators: list[Any] | None = None

    def __call__(self, input_ids, scores):
        import torch

        if self._generators is None:
            self._generators = [
                torch.Generator(device=scores.device).manual_seed(seed) for seed in self.seeds
            ]
        probs = torch.softmax(scores.float() / self.temperature, dim=-1)
        if self.top_p < 1.0:
            ordered, order = probs.sort(dim=-1, descending=True)
            # A token is kept while the ones more likely than it hold less
            # than top_p between them; the likeliest is always kept
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
