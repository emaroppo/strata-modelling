"""GLiNER as a strata span model, against a stand-in for the library.

No download and no network: the stand-in finds two fixed words, so what is
under test is everything around the model — words and windows, offsets
back into the document, zero-shot and training, checkpoints — not
GLiNER's own judgement.
"""

import sys
import types
from typing import ClassVar

import pytest
import torch

pytest.importorskip("transformers", reason="needs the text extra")

from strata.contracts import Span, Spans
from strata.modelling import Example
from strata.modelling.baselines.text import gliner as module
from strata.modelling.baselines.text.gliner import GlinerSpanTagger, to_training, windows, words
from strata.modelling.plugins.conformance import ModelContract

TEXT = "alpha beta gamma delta"


class StandIn(torch.nn.Module):
    """Labels "alpha" with the first prompt asked for and "gamma" with the second."""

    loaded: ClassVar[list] = []
    trained_on: ClassVar[list] = []

    def __init__(self, max_len=384):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.config = types.SimpleNamespace(max_len=max_len, max_width=12)

    @classmethod
    def from_pretrained(cls, checkpoint, revision=None, **overrides):
        cls.loaded.append((checkpoint, revision, overrides))
        return cls(max_len=StandIn.max_len)

    def inference(self, texts, labels, flat_ner=True, threshold=0.5, batch_size=8):
        found = []
        for text in texts:
            # As GLiNER does: words past max_len are dropped, with no error
            bounds = words(text)
            if len(bounds) > self.config.max_len:
                text = text[: bounds[self.config.max_len - 1][1]]
            entities = []
            for word, label in zip(("alpha", "gamma"), labels, strict=False):
                start = text.find(word)
                while start != -1:
                    entities.append(
                        {"start": start, "end": start + len(word), "label": label, "score": 0.9}
                    )
                    start = text.find(word, start + 1)
            found.append(entities)
        return found

    def create_training_args(self, **kwargs):
        return kwargs

    def train_model(self, records, eval_dataset, training_args=None):
        StandIn.trained_on.append((records, training_args))
        with torch.no_grad():
            self.weight += 1
        return types.SimpleNamespace(state=types.SimpleNamespace(log_history=[{"train_loss": 0.5}]))


StandIn.max_len = 384


@pytest.fixture(autouse=True)
def stand_in(monkeypatch):
    StandIn.loaded, StandIn.trained_on, StandIn.max_len = [], [], 384
    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=StandIn))
    monkeypatch.setattr(module, "_resolved", lambda checkpoint: "c0ffee")


@pytest.fixture
def documents(tmp_path):
    paths = []
    for i in range(4):
        path = tmp_path / f"doc{i}.txt"
        path.write_text(TEXT)
        paths.append(path)
    return paths


def target():
    return Spans(
        values=[
            Span(labels=["WORD"], start=0, end=5, text="alpha"),
            Span(labels=["OTHER"], start=11, end=16, text="gamma"),
        ]
    )


class TestGlinerConforms(ModelContract):
    @pytest.fixture
    def model(self):
        return GlinerSpanTagger(device="cpu", prompts={"WORD": "word", "OTHER": "other"})

    @pytest.fixture
    def fresh(self):
        return GlinerSpanTagger(device="cpu")

    @pytest.fixture
    def examples(self, documents):
        return [Example(path=path, target=target()) for path in documents]


class TestFinetunedGlinerConforms(TestGlinerConforms):
    @pytest.fixture
    def model(self):
        return GlinerSpanTagger(
            device="cpu", num_epochs=1, prompts={"WORD": "word", "OTHER": "other"}
        )


# ----------------------------------------------------------------------
# Words and windows
# ----------------------------------------------------------------------


def test_words_are_split_the_way_gliner_splits_them():
    text = "Mail ana.novak@x.org now-ish"
    assert [text[a:b] for a, b in words(text)] == [
        "Mail",
        "ana",
        ".",
        "novak",
        "@",
        "x",
        ".",
        "org",
        "now-ish",
    ]


def test_windows_cover_everything_and_overlap():
    assert windows(5, 10, 2) == [(0, 5)]
    assert windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]


def test_a_training_span_off_word_edges_is_dropped_and_counted():
    spans = [Span(labels=["WORD"], start=0, end=5), Span(labels=["WORD"], start=1, end=5)]
    [record], dropped = to_training(TEXT, spans, {"WORD": "word"}, 384)
    assert record["ner"] == [[0, 0, "word"]]
    assert dropped == 1


def test_a_training_span_across_a_window_edge_is_dropped_and_counted():
    # Two-word windows: "beta gamma" straddles the first edge
    spans = [Span(labels=["X"], start=6, end=16), Span(labels=["X"], start=17, end=22)]
    records, dropped = to_training(TEXT, spans, {"X": "x"}, 2)
    assert [r["ner"] for r in records] == [[], [[1, 1, "x"]]]
    assert dropped == 1


# ----------------------------------------------------------------------
# Behaviour
# ----------------------------------------------------------------------


def test_zero_shot_trains_nothing_and_still_scores_val(documents):
    model = GlinerSpanTagger(device="cpu", prompts={"WORD": "word", "OTHER": "other"})
    examples = [Example(path=p, target=target()) for p in documents]
    metrics = model.finetune(examples[:2], ["WORD", "OTHER"], val=examples[2:])
    assert StandIn.trained_on == []
    assert metrics["val_span_f1"] == 1.0


def test_fine_tuning_trains_on_word_indexed_records(documents):
    model = GlinerSpanTagger(device="cpu", num_epochs=2, batch_size=2, seed=3)
    examples = [Example(path=p, target=target()) for p in documents]
    metrics = model.finetune(examples, ["WORD", "OTHER"])
    [(records, args)] = StandIn.trained_on
    assert records[0] == {"tokenized_text": TEXT.split(), "ner": [[0, 0, "WORD"], [2, 2, "OTHER"]]}
    assert (args["max_steps"], args["seed"]) == (4, 3)
    assert metrics["train_spans_dropped"] == 0.0


def test_a_long_document_is_read_in_windows_and_answered_whole(tmp_path):
    # Past the model's limit, the second "alpha" would be silently lost
    StandIn.max_len = 6
    path = tmp_path / "long.txt"
    text = "alpha one two three four five six seven alpha"
    path.write_text(text)
    model = GlinerSpanTagger(device="cpu", chunk_overlap=2)
    model.finetune([Example(path=path, target=Spans())], ["PER"])
    [prediction] = model.predict([path])
    assert [(s.start, s.end) for s in prediction.values] == [(0, 5), (40, 45)]
    assert all(s.text == "alpha" for s in prediction.values)


def test_an_entity_seen_twice_in_an_overlap_is_kept_once(tmp_path):
    StandIn.max_len = 4
    path = tmp_path / "doc.txt"
    path.write_text("one two alpha three four")
    model = GlinerSpanTagger(device="cpu", chunk_overlap=2)
    model.finetune([Example(path=path, target=Spans())], ["PER"])
    [prediction] = model.predict([path])
    assert [(s.start, s.end) for s in prediction.values] == [(8, 13)]


def test_offsets_count_carriage_returns(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"Hi\r\nthere alpha")
    model = GlinerSpanTagger(device="cpu")
    model.finetune([Example(path=path, target=Spans())], ["PER"])
    [prediction] = model.predict([path])
    assert [(s.start, s.text) for s in prediction.values] == [(10, "alpha")]


def test_a_zero_shot_checkpoint_pins_the_revision_and_keeps_no_weights(documents, tmp_path):
    model = GlinerSpanTagger(device="cpu")
    model.finetune([Example(path=documents[0], target=target())], ["WORD", "OTHER"])
    model.save(tmp_path / "run.pt")
    saved = torch.load(tmp_path / "run.pt", weights_only=True)
    assert (saved["revision"], saved["state_dict"]) == ("c0ffee", None)

    restored = GlinerSpanTagger(device="cpu")
    restored.load(tmp_path / "run.pt")
    assert StandIn.loaded[-1][:2] == (module.DEFAULT_CHECKPOINT, "c0ffee")


def test_a_fine_tuned_checkpoint_carries_its_weights(documents, tmp_path):
    model = GlinerSpanTagger(device="cpu", num_epochs=1)
    model.finetune([Example(path=documents[0], target=target())], ["WORD", "OTHER"])
    model.save(tmp_path / "run.pt")
    restored = GlinerSpanTagger(device="cpu")
    restored.load(tmp_path / "run.pt")
    assert torch.equal(restored._model.weight, model._model.weight)
    assert restored._model.weight.sum().item() == 2.0


def test_overlapping_or_multi_label_schemas_are_refused():
    from strata.contracts import SpanSchema

    with pytest.raises(ValueError, match="overlapping"):
        GlinerSpanTagger(device="cpu").requires_schema(SpanSchema(classes=["A"], overlapping=True))
