"""The text-sft baseline, everything around the model.

No GPU, no download and no network: a stand-in tokenizer, one token per
character, is enough to test what the baseline does to its examples — the
prompt each input becomes, the labels the loss reads, what is dropped and
collapsed, how outputs are batched, seeded and cut. Whether a LoRA learns
is the last test's question, and it runs only where
``STRATA_TEST_SFT_MODEL`` names a model to try it on.

Needs the text extra, which the baselines' package imports; not the sft
extra, which is the point of one test here.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Skipped, not failed, on a base install: the text package imports both
pytest.importorskip("torch", reason="needs the text extra")
pytest.importorskip("transformers", reason="needs the text extra")

import torch

from strata.contracts import SFTTextExample
from strata.modelling import Example
from strata.modelling.baselines.text.sft import (
    DEFAULT_BASE_MODELS,
    TextSFT,
    batches,
    configure_tokenizer,
    decoded,
    epoch_callback,
    stop_ids,
)
from strata.modelling.baselines.text.sft_data import (
    IGNORE,
    Collator,
    Prompts,
    SeededSampling,
    collapse,
    copies_of,
    encode,
    template_fields,
)
from strata.modelling.model import GenerationInput, MediaModel

TEMPLATE = "subject: {subject}\nurls: {urls}\nattachments: {attachments}\n"
INPUT = {"subject": "Hi", "urls": True, "attachments": False}


class Tokenizer:
    """One token per character; each special token is one token.

    Its chat template wraps every turn in ``<|role|>...<|end|>`` and trims
    the content, as Llama 3's does.
    """

    SPECIAL = ("<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
    SPLIT = re.compile("|".join(re.escape(s) for s in SPECIAL) + "|.", re.DOTALL)

    def __init__(self, chat_template: str | None = "stub", pad_token: str | None = None):
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = pad_token
        self.unk_token_id = None
        self.chat_template = chat_template
        self.padding_side = "right"

    def convert_tokens_to_ids(self, token: str):
        if token in self.SPECIAL:
            return self.SPECIAL.index(token)
        return 100 + ord(token) if len(token) == 1 else None

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL.index(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        return None if self.pad_token is None else self.SPECIAL.index(self.pad_token)

    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False):
        assert not add_special_tokens, "the baseline tokenizes with its own special tokens"
        if isinstance(text, str):
            return {"input_ids": [self.convert_tokens_to_ids(t) for t in self.SPLIT.findall(text)]}
        # A batch, as generation asks for one: of one length, so never padded
        rows = [self(t)["input_ids"] for t in text]
        assert len({len(r) for r in rows}) == 1, "a batch of prompts of different lengths"
        from transformers import BatchEncoding

        ids = torch.tensor(rows)
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def save_pretrained(self, folder: str) -> None:
        (Path(folder) / "tokenizer.json").write_text(str(self.chat_template))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(
            ("" if skip_special_tokens else self.SPECIAL[i]) if i < 100 else chr(i - 100)
            for i in ids
        )

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        assert not tokenize
        text = "<s>" + "".join(f"<|{m['role']}|>{m['content'].strip()}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")


def write(tmp_path: Path, checksum: str, body: str, **features) -> Example:
    """An SFT sample as materialised: its file named for its checksum."""
    path = tmp_path / f"{checksum}.txt"
    path.write_text(body, encoding="utf-8", newline="")
    return Example(path=path, target=None, features={**INPUT, **features})


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


# -- the template ------------------------------------------------------------


def test_the_template_names_the_fields_the_model_requires():
    assert template_fields(TEMPLATE) == ("subject", "urls", "attachments")
    assert template_fields("{b} {a!r} {b:>10} {{literal}}") == ("b", "a")
    model = TextSFT(prompt_template=TEMPLATE)
    assert model.requires_features == ("subject", "urls", "attachments")
    assert isinstance(model, MediaModel)
    assert TextSFT.trains_on is SFTTextExample


@pytest.mark.parametrize("template", ["{}", "{0}", "{a.b}", "{a[0]}", "{a", "{a:{b}}"])
def test_a_template_naming_anything_but_a_field_is_refused(template):
    with pytest.raises(ValueError, match="prompt_template"):
        TextSFT(prompt_template=template)


def test_a_field_the_input_lacks_is_refused_before_anything_loads(tmp_path, monkeypatch):
    # unsloth unimportable: reaching it would fail differently
    monkeypatch.setitem(sys.modules, "unsloth", None)
    model = TextSFT(prompt_template=TEMPLATE + "corpus: {corpus}\n")
    with pytest.raises(ValueError, match="no corpus"):
        model.finetune([write(tmp_path, "aa", "body")])


def test_without_the_sft_extra_training_says_which_extra(tmp_path, monkeypatch):
    # Set here so it is unset again after: the Turing workaround sets it
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    monkeypatch.setitem(sys.modules, "unsloth", None)
    with pytest.raises(ImportError, match=r"strata-modelling\[sft\]"):
        TextSFT(prompt_template=TEMPLATE).finetune([write(tmp_path, "aa", "body")])


# -- the parameters ----------------------------------------------------------


def test_the_resolved_parameters_fill_in_every_default():
    params = TextSFT(prompt_template=TEMPLATE).resolved_params()
    assert params["base_model"] == DEFAULT_BASE_MODELS["base"]
    assert params["response_marker"] == "->\n"
    assert params["max_body_tokens"] == 450
    assert params["duplicate_threshold"] == 0.8
    assert params["generation"] == {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 460}
    assert len(params["lora_targets"]) == 7

    instruct = TextSFT(prompt_template=TEMPLATE, variant="instruct", system_prompt="Write.")
    resolved = instruct.resolved_params()
    assert resolved["base_model"] == DEFAULT_BASE_MODELS["instruct"]
    assert resolved["response_marker"] is None
    # What a run records rebuilds the same model
    assert TextSFT(**resolved).resolved_params() == resolved


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"variant": "chat"}, "variant"),
        ({"system_prompt": "Write."}, "system_prompt"),
        ({"chat_template": "llama-3"}, "chat_template"),
        ({"variant": "instruct", "response_marker": ">"}, "response_marker"),
        ({"duplicate_threshold": 0}, "duplicate_threshold"),
        ({"generation": {"temprature": 0.5}}, "temprature"),
        ({"generation": {"top_p": 0}}, "top_p"),
    ],
)
def test_a_parameter_that_would_go_unused_or_wrong_is_refused(params, match):
    with pytest.raises(ValueError, match=match):
        TextSFT(prompt_template=TEMPLATE, **params)


def test_settings_are_given_over_the_defaults():
    model = TextSFT(prompt_template=TEMPLATE, generation={"top_p": 0.9})
    assert model.settings_for({"temperature": 0.0}) == {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_new_tokens": 460,
    }
    with pytest.raises(ValueError, match="seed"):
        model.settings_for({"seed": 1})


# -- prompts and labels ------------------------------------------------------


def test_a_base_prompt_is_the_template_after_the_begin_token_and_before_the_marker():
    prompts = Prompts("base", TEMPLATE, response_marker="->\n")
    tokenizer = Tokenizer()
    assert prompts.prompt(tokenizer, INPUT) == (
        "<s>subject: Hi\nurls: True\nattachments: False\n->\n"
    )
    assert prompts.completion(tokenizer, INPUT, "Dear all") == "Dear all</s>"
    assert prompts.end(tokenizer) == "</s>"


def test_an_instruct_prompt_is_a_chat_ending_where_the_assistant_begins():
    prompts = Prompts("instruct", TEMPLATE, system_prompt="Write an email.")
    tokenizer = Tokenizer()
    prompt = prompts.prompt(tokenizer, INPUT)
    assert prompt == (
        "<s><|system|>Write an email.<|end|>"
        "<|user|>subject: Hi\nurls: True\nattachments: False<|end|><|assistant|>"
    )
    # The target ends with the chat template's own end of turn
    assert prompts.completion(tokenizer, INPUT, "Dear all") == "Dear all<|end|>"
    assert prompts.end(tokenizer) == "<|end|>"


def test_a_chat_template_that_does_not_extend_its_prompt_is_refused():
    class Reordering(Tokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
            return super().apply_chat_template(messages[::-1], add_generation_prompt)

    with pytest.raises(ValueError, match="does not extend"):
        Prompts("instruct", TEMPLATE).completion(Reordering(), INPUT, "body")


def test_the_loss_reads_the_completion_and_its_end_token_only():
    tokenizer = Tokenizer()
    prompts = Prompts("base", TEMPLATE, response_marker="->\n")
    row = encode(prompts, tokenizer, INPUT, "Dear all")
    prompt_length = len(prompts.prompt(tokenizer, INPUT)) - len("<s>") + 1
    assert row["labels"][:prompt_length] == [IGNORE] * prompt_length
    learned = row["labels"][prompt_length:]
    assert learned == row["input_ids"][prompt_length:]
    assert tokenizer.decode(learned) == "Dear all</s>"
    assert learned[-1] == tokenizer.eos_token_id


def test_the_collator_pads_on_the_right_and_masks_the_padding():
    batch = Collator(pad_id=2)(
        [
            {"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
            {"input_ids": [5], "labels": [5]},
        ]
    )
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 2, 2]]
    assert batch["labels"].tolist() == [[IGNORE, 6, 7], [5, IGNORE, IGNORE]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


# -- what is trained on ------------------------------------------------------


def test_a_body_over_the_budget_is_dropped_and_counted_never_cut(tmp_path):
    model = TextSFT(prompt_template="{subject}", max_body_tokens=10)
    tokenizer = Tokenizer()
    train = [write(tmp_path, "aa", "x" * 10), write(tmp_path, "bb", "y" * 11)]
    rows, _, counts = model.prepare(train, None, tokenizer)
    assert counts["train_dropped_long_body"] == 1
    assert counts["train_kept"] == 1
    # The one kept is whole, end token and all
    learned = [t for t in rows[0]["labels"] if t != IGNORE]
    assert tokenizer.decode(learned) == "x" * 10 + "</s>"


def test_a_prompt_and_body_over_the_sequence_length_are_dropped_and_counted(tmp_path):
    model = TextSFT(prompt_template="{subject}", max_seq_length=20)
    train = [write(tmp_path, "aa", "short"), write(tmp_path, "bb", "short", subject="s" * 20)]
    _, _, counts = model.prepare(train, None, Tokenizer())
    assert counts["train_dropped_long_sequence"] == 1
    assert counts["train_kept"] == 1


def test_bodies_identical_but_for_case_and_spacing_are_one():
    kept = collapse(["Dear  John,\nhello", "dear john, HELLO", "else"], ["b", "a", "c"], 0.8)
    # The smaller checksum of the two, and the unrelated one
    assert kept == [1, 2]


def test_bodies_sharing_most_five_word_runs_are_one():
    base = words(40)
    near = base + " extra"
    far = words(40, start=100)
    assert collapse([base, near, far], ["c", "a", "b"], 0.8) == [1, 2]
    # Under the threshold they stay apart
    assert collapse([base, words(20)], ["a", "b"], 0.8) == [0, 1]


def test_near_copies_chain_and_the_group_keeps_its_smallest_checksum():
    a = words(30)
    b = words(30) + " " + words(4, start=200)
    c = b + " " + words(4, start=300)
    sets = collapse([a, c], ["x", "y"], 0.8)
    assert sets == [0, 1], "a and c alone are too far apart to be one"
    assert collapse([a, b, c], ["m", "z", "k"], 0.8) == [2]


def test_a_validation_body_copying_a_training_one_is_dropped():
    train = [words(40), "Totally different words here"]
    val = [words(40) + " tail", "  TOTALLY different words HERE ", words(40, start=500)]
    assert copies_of(val, train, 0.8) == [True, True, False]


def test_preparing_counts_each_way_an_example_was_left_out(tmp_path):
    model = TextSFT(prompt_template="{subject}")
    train = [
        write(tmp_path, "b1", words(40)),
        write(tmp_path, "a1", words(40) + " again"),
        write(tmp_path, "c1", words(40, start=100)),
    ]
    val = [
        write(tmp_path, "d1", words(40, start=100) + " too"),
        write(tmp_path, "e1", words(40, start=900)),
        write(tmp_path, "f1", words(40, start=900).upper()),
    ]
    train_rows, val_rows, counts = model.prepare(train, val, Tokenizer())
    assert counts["train_dropped_near_copy"] == 1
    assert counts["val_dropped_near_copy"] == 1
    assert counts["val_dropped_near_train"] == 1
    assert counts["train_kept"] == len(train_rows) == 2
    assert counts["val_kept"] == len(val_rows) == 1


def test_each_epoch_is_kept_then_reported_and_a_stop_is_honoured():
    from transformers import TrainerControl

    happened: list = []

    def on_epoch(done, total, metrics):
        happened.append(("report", done, total, metrics))
        return done == 2

    callback = epoch_callback(3, True, on_epoch, lambda name: happened.append(("keep", name)))
    callback.trainer = type("Trainer", (), {"evaluate": lambda self: {"eval_loss": 0.5}})()
    control = TrainerControl()
    for loss in (1.0, 3.0):
        callback.on_log(None, None, control, logs={"loss": loss})
    callback.on_log(None, None, control, logs={"eval_loss": 9.0})
    callback.on_epoch_end(None, None, control)
    assert not control.should_training_stop
    callback.on_log(None, None, control, logs={"loss": 1.0})
    callback.on_epoch_end(None, None, control)

    assert happened == [
        ("keep", "epoch-1"),
        ("report", 1, 3, {"loss": 2.0, "val_loss": 0.5}),
        ("keep", "epoch-2"),
        ("report", 2, 3, {"loss": 1.0, "val_loss": 0.5}),
    ]
    assert control.should_training_stop
    assert callback.last == {"loss": 1.0, "val_loss": 0.5, "epochs": 2.0}


# -- the tokenizer -----------------------------------------------------------


def test_the_instruct_variant_refuses_a_tokenizer_without_a_chat_template():
    with pytest.raises(ValueError, match="chat template"):
        configure_tokenizer(Tokenizer(chat_template=None), "instruct", None, None, "m")
    # The base variant needs none
    configure_tokenizer(Tokenizer(chat_template=None), "base", None, None, "m")


def test_a_jinja_chat_template_replaces_the_tokenizers_own():
    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer = configure_tokenizer(Tokenizer(), "instruct", template, None, "m")
    assert tokenizer.chat_template == template


def test_padding_is_an_existing_token_or_the_end_token():
    assert configure_tokenizer(Tokenizer(), "base", None, None, "m").pad_token == "</s>"
    assert configure_tokenizer(Tokenizer(), "base", None, "<pad>", "m").pad_token == "<pad>"
    with pytest.raises(ValueError, match="no embedding"):
        configure_tokenizer(Tokenizer(), "base", None, "<|new|>", "m")


# -- generation --------------------------------------------------------------


def test_generating_from_samples_is_refused():
    model = TextSFT(prompt_template=TEMPLATE)
    given = [GenerationInput(values=INPUT, samples={"text": [Path("x.txt")]})]
    with pytest.raises(ValueError, match="text"):
        model.generate(given, [{}], [0])
    with pytest.raises(ValueError, match="one of each"):
        model.generate([GenerationInput(values=INPUT)], [{}, {}], [0])


def test_outputs_are_batched_by_settings_and_prompt_length():
    keys = [("t1", 5), ("t1", 6), ("t1", 5), ("t2", 5), ("t1", 5)]
    assert batches(keys, 2) == [[0, 2], [4], [1], [3]]


def test_an_output_ends_at_its_first_stop_token():
    tokenizer = Tokenizer()
    end = tokenizer.convert_tokens_to_ids("<|end|>")
    row = tokenizer("ok")["input_ids"] + [end] + tokenizer("junk")["input_ids"]
    assert decoded(row, [end], tokenizer).data == b"ok"
    assert decoded(row, [end], tokenizer).ended is True
    cut = decoded(tokenizer("no end")["input_ids"], [end], tokenizer)
    assert (cut.data, cut.ended) == (b"no end", False)


def test_the_stop_tokens_include_the_end_of_turn_trained_on():
    tokenizer = Tokenizer()
    model = type("M", (), {"generation_config": type("C", (), {"eos_token_id": [42, 43]})()})()
    stop = stop_ids(tokenizer, model, Prompts("instruct", TEMPLATE))
    assert tokenizer.convert_tokens_to_ids("<|end|>") in stop
    assert {tokenizer.eos_token_id, 42, 43} <= set(stop)


def test_each_row_is_drawn_with_its_own_seed_whatever_shares_its_batch():
    torch.manual_seed(0)
    scores = torch.randn(3, 50)

    def drawn(seeds, rows):
        out = SeededSampling(seeds, temperature=1.0, top_p=1.0)(None, scores[rows])
        assert torch.isfinite(out).sum(dim=-1).tolist() == [1] * len(rows)
        return out.argmax(dim=-1).tolist()

    alone = drawn([7], [1])
    assert drawn([3, 7, 9], [0, 1, 2])[1] == alone[0]
    assert drawn([7, 11], [1, 0])[0] == alone[0]


def test_a_tight_top_p_leaves_only_the_likeliest_token():
    scores = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
    for seed in range(20):
        out = SeededSampling([seed], temperature=1.0, top_p=0.01)(None, scores)
        assert out.argmax(dim=-1).item() == 1


# -- the model's side, with a stand-in for it --------------------------------


class Net:
    """A causal model in the shape generate and save use. Draws a letter a step.

    Each step's scores are the same for every row, so what a row writes is
    its seed's alone. The row whose prompt holds "$" ends after two
    letters; every other runs to the limit.
    """

    LETTERS = torch.arange(100 + ord("a"), 100 + ord("z") + 1)

    def __init__(self):
        self.calls: list[dict] = []
        self.device = "cpu"
        self.config = type("Config", (), {"pad_token_id": None})()
        self.generation_config = type("Config", (), {"eos_token_id": None})()

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append({"rows": input_ids.shape[0], **kwargs})
        processors = kwargs.get("logits_processor") or []
        stopping = [i for i, row in enumerate(input_ids.tolist()) if 100 + ord("$") in row]
        out = input_ids
        for step in range(kwargs["max_new_tokens"]):
            scores = torch.full((input_ids.shape[0], 300), float("-inf"))
            scores[:, self.LETTERS] = torch.linspace(0, 1, len(self.LETTERS))
            for processor in processors:
                scores = processor(out, scores)
            picked = scores.argmax(dim=-1)
            if step == 2:
                picked[stopping] = kwargs["eos_token_id"][0]
            out = torch.cat([out, picked.unsqueeze(-1)], dim=-1)
        return out

    def save_pretrained(self, folder: str) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "adapter_config.json").write_text("{}")


@pytest.fixture
def stand_in(monkeypatch):
    """unsloth, as far as this baseline calls it, and a model and tokenizer for it."""
    import types

    net, tokenizer = Net(), Tokenizer()
    unsloth = types.ModuleType("unsloth")
    loaded: list[str] = []

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            loaded.append(model_name)
            return net, tokenizer

        @staticmethod
        def for_inference(model):
            return model

    unsloth.FastLanguageModel = FastLanguageModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    import transformers

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(folder):
            assert (Path(folder) / "tokenizer.json").exists()
            return tokenizer

    monkeypatch.setattr(transformers, "AutoTokenizer", AutoTokenizer)
    return types.SimpleNamespace(net=net, tokenizer=tokenizer, loaded=loaded)


def ready(stand_in) -> TextSFT:
    model = TextSFT(prompt_template="{subject}")
    model._model, model._tokenizer = (
        stand_in.net,
        configure_tokenizer(stand_in.tokenizer, "base", None, None, "m"),
    )
    return model


def test_generating_returns_each_output_in_order_seeded_and_ended(stand_in):
    model = ready(stand_in)
    inputs = [GenerationInput(values={"subject": s}) for s in ("ab", "$", "ab", "abc", "ab")]
    settings = [{"max_new_tokens": 6}] * 4 + [{"max_new_tokens": 6, "temperature": 0}]
    seeds = [1, 1, 1, 2, 1]
    done: list[tuple[int, int]] = []
    out = model.generate(inputs, settings, seeds, lambda d, t: done.append((d, t)))

    assert len(out) == 5 and done[-1] == (5, 5)
    # Same input, same seed, same text, whatever shared its batch
    assert out[0] == out[2]
    assert [o.ended for o in out] == [False, True, False, False, False]
    assert len(out[1].data) == 2 and len(out[0].data) == 6
    # Greedy draws nothing: the likeliest letter every step
    assert out[4].data == b"z" * 6
    # Batched by settings and prompt length: (ab, ab), ($), (abc), (ab greedy)
    assert sorted(c["rows"] for c in stand_in.net.calls) == [1, 1, 1, 2]
    for call in stand_in.net.calls:
        assert call["do_sample"] is False
    assert sum("logits_processor" in c for c in stand_in.net.calls) == 3


def test_a_checkpoint_is_one_file_that_rebuilds_the_model_as_trained(stand_in, tmp_path):
    model = ready(stand_in)
    model.revision = "abc123"
    model.save(tmp_path / "run.pt")
    assert (tmp_path / "run.pt").is_file()

    again = TextSFT(prompt_template="something else {entirely}", num_epochs=9)
    again.load(tmp_path / "run.pt")
    assert again.resolved_params() == model.resolved_params()
    assert again.requires_features == ("subject",)
    # Loaded from the checkpoint's own adapter, not the base model by name
    assert stand_in.loaded[-1] != model.base_model
    assert again.generate([GenerationInput(values={"subject": "q"})], [{}], [0])[0].data


# -- the extra ---------------------------------------------------------------


def test_the_module_imports_without_the_sft_extra():
    absent = ["unsloth", "unsloth_zoo", "peft", "datasets", "bitsandbytes", "trl"]
    script = textwrap.dedent(
        f"""
        import sys
        for name in {absent!r}:
            sys.modules[name] = None
        from strata.modelling.baselines.text.sft import TextSFT
        model = TextSFT(prompt_template="{{subject}}")
        assert model.requires_features == ("subject",)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


# -- a real model ------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("STRATA_TEST_SFT_MODEL"),
    reason="trains a real model: set STRATA_TEST_SFT_MODEL to a base model id, on a GPU host",
)
def test_a_real_model_trains_keeps_each_epoch_and_generates(tmp_path):
    pytest.importorskip("unsloth", reason="needs the sft extra")
    train = [
        write(tmp_path, f"{n:02x}", f"Hello {n}, the meeting is at {n} o'clock.", subject=f"m{n}")
        for n in range(16)
    ]
    val = [write(tmp_path, "ff", "See you all tomorrow at noon.", subject="noon")]
    model = TextSFT(
        prompt_template=TEMPLATE,
        base_model=os.environ["STRATA_TEST_SFT_MODEL"],
        num_epochs=2,
        batch_size=2,
        gradient_accumulation=1,
        max_body_tokens=64,
        max_seq_length=256,
    )
    kept: list[str] = []
    reports: list[dict] = []

    def save_checkpoint(name: str) -> None:
        model.save(tmp_path / f"{name}.pt")
        kept.append(name)

    metrics = model.finetune(train, val, lambda done, total, m: reports.append(m), save_checkpoint)
    assert kept == ["epoch-1", "epoch-2"]
    assert all("val_loss" in r for r in reports)
    assert metrics["train_kept"] == 16

    again = TextSFT(prompt_template="unused {x}")
    again.load(tmp_path / "epoch-2.pt")
    assert again.resolved_params() == model.resolved_params()
    inputs = [GenerationInput(values={**INPUT, "subject": "m3"})] * 3
    first = again.generate(inputs, [{"max_new_tokens": 32}] * 3, [1, 2, 1])
    assert len(first) == 3 and all(isinstance(o.data, bytes) for o in first)
    # The same seed, the same output, whatever shared its batch
    assert first[0].data == first[2].data
