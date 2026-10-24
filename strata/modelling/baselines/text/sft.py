"""A causal language model LoRA-tuned to write a text from its input: ``text-sft``.

Trained on ``SFTTextExample`` samples, whose file is the text to write and
whose features are the input it is written from (record 0053). Nothing in
the loop belongs to one project: a 4-bit LoRA on every linear layer through
unsloth, a loss on the completion only, and a base or instruct wrapping. A
project brings its prompt template and system prompt as parameters, and
its own checks as a metrics plugin.

Two things are done to the examples on the way in, because only this model
has the base model's tokens. A target over ``max_body_tokens`` is dropped,
never cut: a cut example has no end token and teaches the model not to
stop. And near-identical targets are collapsed within each side, with a
validation target that nearly copies a training one dropped. Each is
counted on the run (record 0036).

A checkpoint is kept after every epoch, since a LoRA's best epoch is found
by evaluating each. Behind the ``sft`` extra: unsloth, peft, datasets and
bitsandbytes are imported only when a model is loaded. See
``docs/adr/0054``.
"""

# unsloth and datasets resolve only with the sft extra, which every CI job
# but the sft one leaves out; that one type-checks them as real imports.
# pyright: reportMissingImports=false

import inspect
import json
import os
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from strata.contracts import SFTTextExample

from ...model import (
    BatchReport,
    EpochReport,
    Example,
    Generated,
    GenerationInput,
    MediaModel,
    SaveCheckpoint,
)
from .sft_data import (
    Collator,
    Prompts,
    SeededSampling,
    checksum_of,
    collapse,
    copies_of,
    encode,
    ids,
    read_target,
)

#: The base model each variant starts from when none is named: unsloth's
#: 4-bit Llama 3.1 8B, text completion or chat.
DEFAULT_BASE_MODELS = {
    "base": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "instruct": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
}

#: Every linear layer of a Llama-shaped block, which is what a LoRA on
#: "every linear layer" means for the models unsloth serves.
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: What follows the input in a base prompt, so the model knows where the
#: text it writes begins.
DEFAULT_RESPONSE_MARKER = "->\n"

#: The settings a generate request may give, per output.
SETTINGS = ("temperature", "top_p", "max_new_tokens")

#: The checkpoint file's layout, so a later one can still read this one.
FORMAT = 1


class TextSFT(MediaModel):
    """A LoRA over a causal language model, writing a text from an input's fields."""

    trains_on = SFTTextExample
    version = "1"

    def __init__(
        self,
        *,
        prompt_template: str,
        variant: str = "base",
        base_model: str | None = None,
        revision: str | None = None,
        system_prompt: str | None = None,
        response_marker: str | None = None,
        chat_template: str | None = None,
        pad_token: str | None = None,
        load_in_4bit: bool = True,
        max_body_tokens: int = 450,
        max_seq_length: int = 1024,
        duplicate_threshold: float = 0.8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_targets: list[str] | None = None,
        lr: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation: int = 8,
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.01,
        seed: int = 3407,
        generation: dict | None = None,
        generate_batch_size: int = 8,
    ):
        """``prompt_template`` is a format string over the input fields, such
        as ``"subject: {subject}\\n"``; the fields it names are this model's
        ``requires_features``.

        ``variant`` is ``base``, text completion, or ``instruct``, a chat.
        ``system_prompt`` and ``chat_template`` are the instruct variant's
        only, and ``response_marker`` the base variant's only: given to the
        other, each is refused rather than quietly unused. A base model has
        no system turn, so an instruction for one goes in the template.

        ``chat_template`` replaces the tokenizer's own: a Jinja template, or
        the name of one of unsloth's (``llama-3``). Left out, the base
        model's own is used, which comes with the model the run records.
        ``pad_token`` names an existing token to pad with; left out, the
        tokenizer's own, else its end token. ``revision`` pins the base
        model; left out, the one loaded is resolved and recorded.

        ``generation`` is the default for each generate setting:
        ``temperature`` (0 is greedy), ``top_p`` and ``max_new_tokens``,
        which defaults to ``max_body_tokens`` and ten more.
        """
        self.prompts = Prompts(
            variant,
            prompt_template,
            system_prompt=system_prompt,
            response_marker=_marker(variant, response_marker),
        )
        if variant == "base" and system_prompt is not None:
            raise ValueError(
                "system_prompt is the instruct variant's: a base model has no system "
                "turn. Put the instruction in prompt_template instead."
            )
        if variant == "base" and chat_template is not None:
            raise ValueError("chat_template is the instruct variant's; a base prompt is plain text")
        if not 0 < duplicate_threshold <= 1:
            raise ValueError(f"duplicate_threshold is {duplicate_threshold}; it is in (0, 1]")
        if max_body_tokens < 1:
            raise ValueError(f"max_body_tokens is {max_body_tokens}; it is at least 1")
        self.variant = variant
        self.base_model = base_model or DEFAULT_BASE_MODELS[variant]
        self.revision = revision
        self.chat_template = chat_template
        self.pad_token = pad_token
        self.load_in_4bit = load_in_4bit
        self.max_body_tokens = max_body_tokens
        self.max_seq_length = max_seq_length
        self.duplicate_threshold = duplicate_threshold
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = list(lora_targets or DEFAULT_LORA_TARGETS)
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.seed = seed
        self.generation = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": max_body_tokens + 10,
        }
        self.generation = self.settings_for(generation or {})
        self.generate_batch_size = generate_batch_size
        # Read from the built model, as the handler reads it. docs/adr/0011
        self.requires_features = self.prompts.fields
        self._model = None
        self._tokenizer = None

    def resolved_params(self) -> dict:
        """Every parameter as this model uses it, its defaults filled in.

        What a run records, so a changed default never changes what a
        recorded run means; ``save`` keeps it, and ``load`` rebuilds from
        it. ``revision`` is resolved when the base model is first loaded,
        so ask after ``finetune``. See ``docs/adr/0054``.
        """
        return {
            "prompt_template": self.prompts.template,
            "variant": self.variant,
            "base_model": self.base_model,
            "revision": self.revision,
            "system_prompt": self.prompts.system_prompt,
            "response_marker": self.prompts.response_marker if self.variant == "base" else None,
            "chat_template": self.chat_template,
            "pad_token": self.pad_token,
            "load_in_4bit": self.load_in_4bit,
            "max_body_tokens": self.max_body_tokens,
            "max_seq_length": self.max_seq_length,
            "duplicate_threshold": self.duplicate_threshold,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_targets": list(self.lora_targets),
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "generation": dict(self.generation),
            "generate_batch_size": self.generate_batch_size,
        }

    def settings_for(self, given: dict) -> dict:
        """The settings an output is generated with: ``given`` over the defaults.

        An unknown setting is refused, so a misspelt one is never silently
        the default. What a generation record digests is this, not what
        was asked for. See ``docs/adr/0053``.
        """
        unknown = sorted(set(given) - set(SETTINGS))
        if unknown:
            raise ValueError(
                f"text-sft has no generation setting {', '.join(unknown)}; it takes "
                f"{', '.join(SETTINGS)}"
            )
        settings = {**self.generation, **given}
        if settings["temperature"] < 0:
            raise ValueError(f"temperature is {settings['temperature']}; it is 0 or more")
        if not 0 < settings["top_p"] <= 1:
            raise ValueError(f"top_p is {settings['top_p']}; it is in (0, 1]")
        if int(settings["max_new_tokens"]) < 1:
            raise ValueError(f"max_new_tokens is {settings['max_new_tokens']}; it is at least 1")
        settings["max_new_tokens"] = int(settings["max_new_tokens"])
        return settings

    # -- the model -----------------------------------------------------------

    def _net(self):
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model has no weights. Call finetune() or load() first.")
        return self._model, self._tokenizer

    def _load(self, saved: Path | None = None) -> None:
        """The base model, or a saved adapter on it, with its tokenizer set up."""
        _unsloth()
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel

        if saved is None and self.revision is None:
            self.revision = _resolved(self.base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(saved) if saved is not None else self.base_model,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            dtype=None,
            **({"revision": self.revision} if self.revision else {}),
        )
        if saved is None:
            tokenizer = configure_tokenizer(
                tokenizer, self.variant, self.chat_template, self.pad_token, self.base_model
            )
        else:
            # The tokenizer as trained, chat template and padding included
            tokenizer = AutoTokenizer.from_pretrained(str(saved))
        model.config.pad_token_id = tokenizer.pad_token_id
        self._model, self._tokenizer = model, tokenizer

    def _trainable(self):
        """The LoRA to train: new on the base model, or the one loaded, warm."""
        from unsloth import FastLanguageModel

        if self._model is None:
            self._load()
            self._model = FastLanguageModel.get_peft_model(
                self._model,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=list(self.lora_targets),
                use_gradient_checkpointing="unsloth",
                random_state=self.seed,
            )
        else:
            # Loaded from a checkpoint, which loads for inference. Declared
            # through a wrapper of a method unsloth attaches at import, so
            # pyright cannot see it
            FastLanguageModel.for_training(self._model)  # pyright: ignore[reportAttributeAccessIssue]
            for name, parameter in self._model.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
        return self._net()

    # -- training ------------------------------------------------------------

    def prepare(
        self, train: list[Example], val: list[Example] | None, tokenizer
    ) -> tuple[list[dict], list[dict], dict[str, float]]:
        """What is trained and validated on, encoded, and what was left out.

        A target over the token budget, or whose prompt and target together
        exceed ``max_seq_length``, is dropped, never cut. Near copies are
        collapsed within each side, and a validation target that nearly
        copies a training one is dropped. Every count is returned.
        """
        counts: dict[str, float] = {}
        sides = {}
        for side, examples in (("train", train), ("val", val or [])):
            fitting = []
            long_body = long_sequence = 0
            for example in examples:
                body = read_target(example.path)
                if len(ids(tokenizer, body)) > self.max_body_tokens:
                    long_body += 1
                    continue
                row = encode(self.prompts, tokenizer, example.features, body)
                if len(row["input_ids"]) > self.max_seq_length:
                    long_sequence += 1
                    continue
                fitting.append((checksum_of(example.path), body, row))
            kept = collapse(
                [body for _, body, _ in fitting],
                [checksum for checksum, _, _ in fitting],
                self.duplicate_threshold,
            )
            counts[f"{side}_dropped_long_body"] = float(long_body)
            counts[f"{side}_dropped_long_sequence"] = float(long_sequence)
            counts[f"{side}_dropped_near_copy"] = float(len(fitting) - len(kept))
            sides[side] = [fitting[i] for i in kept]

        # Collapsing after the split is a stopgap: a freeze draws first, so
        # this guards against the leak rather than preventing it
        near_train = copies_of(
            [body for _, body, _ in sides["val"]],
            [body for _, body, _ in sides["train"]],
            self.duplicate_threshold,
        )
        counts["val_dropped_near_train"] = float(sum(near_train))
        sides["val"] = [
            item for item, near in zip(sides["val"], near_train, strict=True) if not near
        ]
        counts["train_kept"] = float(len(sides["train"]))
        counts["val_kept"] = float(len(sides["val"]))
        return [row for *_, row in sides["train"]], [row for *_, row in sides["val"]], counts

    def finetune(
        self,
        train: list[Example],
        val: list[Example] | None = None,
        on_epoch: EpochReport | None = None,
        save_checkpoint: SaveCheckpoint | None = None,
    ) -> dict[str, float]:
        # Refused before anything is loaded. docs/adr/0054
        for example in [*train, *(val or [])]:
            self.prompts.values(example.features)

        # First, so a missing extra is named as one
        _unsloth()
        from datasets import Dataset
        from transformers import Trainer, TrainingArguments
        from unsloth import is_bfloat16_supported

        from .._shared import seed_everything

        seed_everything(self.seed)
        model, tokenizer = self._trainable()
        train_rows, val_rows, counts = self.prepare(train, val, tokenizer)
        if not train_rows:
            raise ValueError(f"nothing is left to train on: {counts}")

        bf16 = is_bfloat16_supported()
        callback = epoch_callback(self.num_epochs, bool(val_rows), on_epoch, save_checkpoint)
        with tempfile.TemporaryDirectory() as scratch:
            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=Dataset.from_list(train_rows),
                eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
                data_collator=Collator(tokenizer.pad_token_id),
                callbacks=[callback],
                args=TrainingArguments(
                    output_dir=scratch,
                    num_train_epochs=self.num_epochs,
                    per_device_train_batch_size=self.batch_size,
                    per_device_eval_batch_size=self.batch_size,
                    gradient_accumulation_steps=self.gradient_accumulation,
                    learning_rate=self.lr,
                    lr_scheduler_type="cosine",
                    weight_decay=self.weight_decay,
                    optim="adamw_8bit",
                    fp16=not bf16,
                    bf16=bf16,
                    # Every step, so an epoch's loss is the mean of its steps
                    logging_steps=1,
                    # Evaluated and kept by the callback, once an epoch
                    eval_strategy="no",
                    save_strategy="no",
                    # Only the loss: the logits of a whole validation side
                    # over a large vocabulary do not fit in memory
                    prediction_loss_only=True,
                    seed=self.seed,
                    report_to="none",
                    **_across_versions(TrainingArguments, self.warmup_ratio),
                ),
            )
            callback.trainer = trainer
            trainer.train()
        return {**callback.last, **counts}

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        inputs: list[GenerationInput],
        settings: list[dict],
        seeds: list[int],
        on_batch: BatchReport | None = None,
    ) -> list[Generated]:
        """One text per input, prompted exactly as at training.

        Outputs with the same settings and the same prompt length share a
        batch, so nothing is padded: unsloth's fast generation corrupts
        left-padded rows. Each row is sampled with its own seed, so an
        output does not depend on what shared its batch.
        """
        if not len(inputs) == len(settings) == len(seeds):
            raise ValueError(
                f"{len(inputs)} inputs, {len(settings)} settings and {len(seeds)} seeds: "
                f"one of each per output"
            )
        given = [sorted(i.samples) for i in inputs if i.samples]
        if given:
            raise ValueError(
                f"text-sft writes from an input's values alone, and was given samples "
                f"under {', '.join(given[0])}"
            )
        if not inputs:
            return []

        import torch
        from transformers import LogitsProcessorList
        from unsloth import FastLanguageModel

        model, tokenizer = self._net()
        FastLanguageModel.for_inference(model)
        tokenizer.padding_side = "left"
        resolved = [self.settings_for(s) for s in settings]
        prompts = [self.prompts.prompt(tokenizer, i.values) for i in inputs]
        stop = stop_ids(tokenizer, model, self.prompts)
        keys = [
            (json.dumps(s, sort_keys=True), len(ids(tokenizer, p)))
            for s, p in zip(resolved, prompts, strict=True)
        ]
        outputs: list[Generated] = [Generated(b"") for _ in inputs]
        done = 0
        for batch in batches(keys, self.generate_batch_size):
            chosen = resolved[batch[0]]
            encoded = tokenizer(
                [prompts[n] for n in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            # A temperature of 0 is greedy, and draws nothing
            sampling = {}
            if chosen["temperature"] > 0:
                draw = SeededSampling(
                    [seeds[n] for n in batch], chosen["temperature"], chosen["top_p"]
                )
                sampling["logits_processor"] = LogitsProcessorList([draw])
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    **sampling,
                    max_new_tokens=chosen["max_new_tokens"],
                    # The draw is SeededSampling's; generate picks it greedily
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=stop,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for n, row in zip(batch, generated[:, width:].tolist(), strict=True):
                outputs[n] = decoded(row, stop, tokenizer)
            done += len(batch)
            if on_batch is not None:
                on_batch(done, len(inputs))
        return outputs

    # -- checkpoints ---------------------------------------------------------

    def save(self, path: Path) -> None:
        """One file, which is what a run's checkpoint is: the adapter, the
        tokenizer as trained, and the parameters as used, in a tar.
        """
        model, tokenizer = self._net()
        path = Path(path)
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch) / "model"
            model.save_pretrained(str(folder))
            tokenizer.save_pretrained(str(folder))
            params = Path(scratch) / "params.json"
            params.write_text(
                json.dumps({"format": FORMAT, "params": self.resolved_params()}, indent=2) + "\n"
            )
            # Written beside and moved into place, so a checkpoint read while
            # it is being saved is the old one or the new one, never half
            partial = path.with_name(path.name + ".partial")
            with tarfile.open(partial, "w") as tar:
                tar.add(params, arcname="params.json")
                tar.add(folder, arcname="model")
            partial.replace(path)

    def load(self, path: Path) -> None:
        """Rebuild from :meth:`save`, with the parameters it was saved with."""
        with tempfile.TemporaryDirectory() as scratch:
            with tarfile.open(path) as tar:
                tar.extractall(scratch, filter="data")
            saved = json.loads((Path(scratch) / "params.json").read_text())
            if saved.get("format") != FORMAT:
                raise ValueError(
                    f"{path} is a text-sft checkpoint of format {saved.get('format')}, "
                    f"and this one reads format {FORMAT}"
                )
            # The saved parameters win over the ones constructed with: they
            # are what the adapter was trained under
            self.__init__(**saved["params"])
            self._load(Path(scratch) / "model")


# ----------------------------------------------------------------------


def _marker(variant: str, marker: str | None) -> str:
    if variant == "base":
        return DEFAULT_RESPONSE_MARKER if marker is None else marker
    if marker is not None:
        raise ValueError(
            "response_marker is the base variant's; an instruct prompt ends where "
            "the chat template opens the assistant's turn"
        )
    return ""


def configure_tokenizer(
    tokenizer, variant: str, chat_template: str | None, pad_token: str | None, base_model: str
):
    """The tokenizer as this model trains and generates with it.

    For the instruct variant, its chat template: the one asked for, or its
    own, and a tokenizer with neither is refused. Then its padding token,
    which must be one it already has: a new one would have no embedding.
    """
    if variant == "instruct":
        if chat_template is not None and ("{%" in chat_template or "{{" in chat_template):
            tokenizer.chat_template = chat_template
        elif chat_template is not None:
            from unsloth.chat_templates import get_chat_template

            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        elif not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                f"the instruct variant needs a chat template, and {base_model} has "
                f"none. Name one with chat_template, or train the base variant."
            )
    if pad_token is not None:
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        if not isinstance(pad_id, int) or pad_id == tokenizer.unk_token_id:
            raise ValueError(
                f"pad_token {pad_token!r} is not a token of {base_model}'s tokenizer, "
                f"and a new one would have no embedding"
            )
        tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None:
        # Safe because padding is masked by position, never by id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def epoch_callback(
    epochs: int,
    validates: bool,
    on_epoch: EpochReport | None,
    save_checkpoint: SaveCheckpoint | None,
):
    """What the trainer calls after each epoch: report it, keep it, and stop when asked.

    The epoch's loss is the mean of its steps', and its validation loss is
    evaluated here, once an epoch. The checkpoint is kept before the epoch
    is reported, so a caller told of epoch two can use ``epoch-2``. What
    the last epoch reported stays on ``last``, the metrics ``finetune``
    returns.
    """
    from transformers import TrainerCallback

    class Epochs(TrainerCallback):
        def __init__(self):
            # The trainer this is given to, set once it exists
            self.trainer: Any = None
            self.done = 0
            self.losses: list[float] = []
            self.last: dict[str, float] = {}

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.losses.append(float(logs["loss"]))

        def on_epoch_end(self, args, state, control, **kwargs):
            self.done += 1
            reported: dict[str, float] = {}
            if self.losses:
                reported["loss"] = sum(self.losses) / len(self.losses)
                self.losses.clear()
            if validates and self.trainer is not None:
                reported["val_loss"] = float(self.trainer.evaluate()["eval_loss"])
            self.last = {**reported, "epochs": float(self.done)}
            if save_checkpoint is not None:
                save_checkpoint(f"epoch-{self.done}")
            # An answer of True asks to stop after this epoch. docs/adr/0031
            if on_epoch is not None and on_epoch(self.done, epochs, dict(reported)):
                control.should_training_stop = True

    return Epochs()


def stop_ids(tokenizer, model, prompts: Prompts) -> list[int]:
    """The tokens an output ends on: the end token, the model's own, and
    the first token of what follows a target as trained."""
    stop: set[int] = set()
    config = getattr(model, "generation_config", None)
    for source in (tokenizer.eos_token_id, getattr(config, "eos_token_id", None)):
        if isinstance(source, int):
            stop.add(source)
        elif isinstance(source, list | tuple):
            stop.update(int(s) for s in source)
    end = ids(tokenizer, prompts.end(tokenizer))
    if end:
        stop.add(end[0])
    return sorted(stop)


def decoded(row: list[int], stop: list[int], tokenizer) -> Generated:
    """An output's tokens as text, up to the first stop, and whether there was one."""
    ending = next((n for n, token in enumerate(row) if token in stop), None)
    if ending is not None:
        row = row[:ending]
    text = tokenizer.decode(row, skip_special_tokens=True)
    return Generated(data=text.encode("utf-8"), ended=ending is not None)


def batches(keys: list, size: int) -> list[list[int]]:
    """Indices grouped by key, each group cut into batches of at most ``size``."""
    groups: dict = {}
    for n, key in enumerate(keys):
        groups.setdefault(key, []).append(n)
    return [
        members[i : i + size]
        for _, members in sorted(groups.items())
        for i in range(0, len(members), max(1, size))
    ]


def _unsloth():
    """unsloth, imported after the workaround it needs on older GPUs."""
    import torch

    # unsloth's fused loss (cut_cross_entropy) fails to compile its Triton
    # kernel on Turing (compute 7.5, the 2080 Ti); returning logits takes
    # the plain loss instead. Read when unsloth is imported, so set first.
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    try:
        import unsloth
    except ImportError as e:
        raise ImportError(
            f"text-sft needs the sft extra: pip install 'strata-modelling[sft]' ({e})"
        ) from e
    return unsloth


def _across_versions(arguments: Callable, warmup_ratio: float) -> dict:
    """The training arguments transformers 4 and 5 spell differently.

    5 groups by length through ``train_sampling_strategy``, where 4 had
    ``group_by_length``, and takes a warmup ratio as a ``warmup_steps``
    below one, where 4 had ``warmup_ratio``.
    """
    if "train_sampling_strategy" in inspect.signature(arguments).parameters:
        return {"train_sampling_strategy": "group_by_length", "warmup_steps": warmup_ratio}
    return {"group_by_length": True, "warmup_ratio": warmup_ratio}


def _resolved(base_model: str) -> str | None:
    """The commit a model name points at now, so a run can be repeated.

    None when it can't be asked (offline, or a local directory): the load
    then uses whatever the cache holds.
    """
    try:
        from huggingface_hub import model_info

        return model_info(base_model).sha
    # Any failure to ask means "not pinned", never a failed run
    except Exception:
        return None
