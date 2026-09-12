"""Every label type through what modelling reads and returns.

Checked against the examples ``strata.labels`` ships, so a type added there
is covered here on the next upgrade, and fails until modelling handles it.
"""

import pytest

from strata.labels import MANIFEST_FORMAT, MANIFEST_NAME, Manifest, ManifestSample
from strata.labels.examples import EXAMPLES
from strata.modelling import PredictionCache, ScoredPath, examples

each_type = pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.name)


@each_type
def test_a_model_is_handed_the_type_it_was_annotated_with(example, tmp_path):
    """The last layer between a reviewer and a model, and once the one nothing checked.

    It read every value as a classification, so a span or box dataset raised
    on the first sample and no model ever saw one. The manifest is written
    and read back from disk, as a dataset version travels.
    """
    (tmp_path / "files").mkdir()
    samples = []
    for i, split in enumerate(["train", "val"]):
        (tmp_path / "files" / f"{i}.bin").write_bytes(f"sample {i}".encode())
        samples.append(
            ManifestSample(
                checksum=f"{i:064x}", path=f"files/{i}.bin", split=split, value=example.value
            )
        )
    written = Manifest(
        format=MANIFEST_FORMAT,
        dataset="d",
        label_set="x",
        label_schema=example.schema,
        samples=samples,
    )
    (tmp_path / MANIFEST_NAME).write_text(written.model_dump_json())
    manifest = Manifest.model_validate_json((tmp_path / MANIFEST_NAME).read_text())

    train, val = examples(tmp_path, manifest)
    assert [e.target for e in [*train, *val]] == [example.value, example.value]


@each_type
def test_a_prediction_survives_the_cache(example, tmp_path):
    cache = PredictionCache.local(tmp_path / "runs")
    cache.put(1, {"a" * 64: example.prediction})
    assert cache.get(1, ["a" * 64])["a" * 64] == example.prediction


@each_type
def test_a_prediction_survives_being_tied_to_its_sample(example, tmp_path):
    """The wrapper a scoring pass returns, on the way to the ranking.

    Typed to one concrete prediction it refused every other kind outright,
    so a scoring pass over a span or box project failed after minutes of
    work rather than at the contract.
    """
    scored = ScoredPath(path=tmp_path / "a.bin", value=example.prediction)
    back = ScoredPath.model_validate_json(scored.model_dump_json())
    assert back.value == example.prediction


@each_type
def test_a_prediction_survives_the_wire(example):
    from strata.modelling.remote.service import PredictionResponse

    response = PredictionResponse(predictions={"a" * 64: example.prediction})
    back = PredictionResponse.model_validate_json(response.model_dump_json())
    assert back.predictions["a" * 64] == example.prediction
