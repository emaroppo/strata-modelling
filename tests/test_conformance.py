"""The conformance suite, run against a model, which is the only way to
know the suite itself works.

A plugin's tests look exactly like this: subclass, supply two fixtures.
"""

import pytest

from strata.labels import Choices
from strata.modelling import Example
from strata.modelling.conformance import ModelContract


@pytest.fixture
def counting_model_class(tmp_path):
    from counting_model import COUNTING_MODEL

    from strata.modelling import resolve

    (tmp_path / "counter.py").write_text(COUNTING_MODEL)
    return resolve("counter.py:CountingModel", root=tmp_path)


class TestCountingModelConformance(ModelContract):
    """What a plugin author writes."""

    @pytest.fixture
    def model(self, counting_model_class):
        return counting_model_class()

    @pytest.fixture
    def examples(self, tmp_path):
        paths = []
        for i in range(4):
            path = tmp_path / f"sample{i}.jpg"
            path.write_bytes(f"sample {i}".encode())
            paths.append(path)
        return [
            Example(path=paths[0], target=Choices(values=["cat"])),
            Example(path=paths[1], target=Choices(values=["cat", "dog"])),
            Example(path=paths[2], target=Choices(values=["dog"])),
            # A human looked and found nothing: a real answer, and a model
            # has to cope with it
            Example(path=paths[3], target=Choices()),
        ]


def test_the_contract_refuses_to_run_without_a_model():
    # The fixtures are not optional, and the failure has to say which one is
    # missing rather than erroring somewhere inside the suite
    with pytest.raises(NotImplementedError, match="`model` fixture"):
        ModelContract.model.__wrapped__(None)


def test_the_contract_refuses_to_run_without_examples(tmp_path):
    with pytest.raises(NotImplementedError, match="`examples` fixture"):
        ModelContract.examples.__wrapped__(None, tmp_path)
