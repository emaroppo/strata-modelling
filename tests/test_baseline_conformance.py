"""The shipped baselines, held to the same contract as any plugin.

Nothing checked before this that a baseline's checkpoint round-trips, or
that it never predicts a class outside the list it was given. The contract
existed only as prose in an ABC. These are real training runs against a
stub backbone, so they are slow by this suite's standards and skip entirely
without the image extra.
"""

import pytest

torch = pytest.importorskip("torch", reason="needs the image extra")
pytest.importorskip("timm", reason="needs the image extra")

from PIL import Image  # noqa: E402

from strata.labels import Choices  # noqa: E402
from strata.modelling import Example  # noqa: E402
from strata.modelling.baselines.classifier import (  # noqa: E402
    MulticlassClassifier,
    MultiLabelClassifier,
    PresenceClassifier,
)
from strata.modelling.conformance import ModelContract  # noqa: E402


class _TinyBackbone(torch.nn.Module):
    """Stands in for ConvNeXt, which would fetch pretrained weights.

    Carries the two hooks the classifiers reach for when growing a head, so
    warm-start behaviour is exercised rather than stubbed out.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, x):
        return self.head(self.pool(x).flatten(1))

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, *args, **kwargs):
        self.head = torch.nn.Linear(3, num_classes)


class _BaselineContract(ModelContract):
    """Shared setup; each baseline supplies only its class."""

    MODEL: type = MultiLabelClassifier

    @pytest.fixture
    def model(self, monkeypatch):
        instance = self.MODEL(num_epochs=1, batch_size=2, device="cpu")
        monkeypatch.setattr(
            type(instance),
            "_build_backbone",
            lambda self, n: _TinyBackbone(n),
            raising=False,
        )
        return instance

    @pytest.fixture
    def fresh(self, model, monkeypatch):
        # Pinned to CPU like the model under test: the stub backbone is built
        # wherever the instance says, so a default-constructed one would land
        # on the GPU and its weights on the CPU
        return self.MODEL(num_epochs=1, batch_size=2, device="cpu")

    @pytest.fixture
    def examples(self, tmp_path):
        made = []
        for i, labels in enumerate([["cat"], ["dog"], ["cat", "dog"], []]):
            path = tmp_path / f"img{i}.png"
            Image.new("RGB", (32, 32), color=(i * 40, 90, 120)).save(path)
            made.append(Example(path=path, target=Choices(values=labels)))
        return made

    @pytest.fixture
    def classes(self):
        # Fixed rather than derived, so the empty target above does not
        # shrink the list the model is held to
        return ["cat", "dog"]


class TestMultiLabelClassifier(_BaselineContract):
    MODEL = MultiLabelClassifier


class TestMulticlassClassifier(_BaselineContract):
    MODEL = MulticlassClassifier


class TestPresenceClassifier(_BaselineContract):
    MODEL = PresenceClassifier

    @pytest.fixture
    def classes(self):
        # The negative class is appended by the model itself, and a
        # prediction of it must still count as inside the class list
        return ["cat", "dog", PresenceClassifier.NEGATIVE_LABEL]
