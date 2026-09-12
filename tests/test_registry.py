"""Model resolution: short names through the registry, refs for everything else."""

import pytest
from counting_model import COUNTING_MODEL

from strata.modelling import ENTRY_POINT_GROUP, Model, ModelError, available, resolve


def test_a_ref_loads_a_class_from_a_file(tmp_path):
    (tmp_path / "mine.py").write_text(COUNTING_MODEL)
    loaded = resolve("mine.py:CountingModel", root=tmp_path)
    assert issubclass(loaded, Model)


def test_a_file_ref_is_anchored_to_the_root(tmp_path):
    # Found relative to the work rather than the current directory, so a
    # project-local model does not depend on where the command was run
    (tmp_path / "mine.py").write_text(COUNTING_MODEL)
    assert resolve("mine.py:CountingModel", root=tmp_path).__name__ == "CountingModel"


def test_a_module_ref_imports_an_installed_class():
    assert resolve("strata.modelling.model:Model") is Model


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(ModelError, match="missing file"):
        resolve("absent.py:Anything", root=tmp_path)


def test_a_missing_class_names_what_it_looked_in(tmp_path):
    (tmp_path / "mine.py").write_text(COUNTING_MODEL)
    with pytest.raises(ModelError, match="No class 'Absent' in mine.py"):
        resolve("mine.py:Absent", root=tmp_path)


def test_an_unimportable_module_is_an_error():
    with pytest.raises(ModelError, match="will not import"):
        resolve("no_such_package.at_all:Thing")


def test_a_model_bringing_a_missing_dependency_says_so(tmp_path):
    (tmp_path / "needy.py").write_text("import a_framework_nobody_installed\n")
    with pytest.raises(ModelError, match="brings its own dependencies"):
        resolve("needy.py:Whatever", root=tmp_path)


def test_a_bare_name_goes_to_the_registry():
    # No ':' means a short name, which is what a request carries once it
    # crosses a wire
    pytest.importorskip("timm", reason="multilabel is an image baseline")
    assert resolve("multilabel").__name__ == "MultiLabelClassifier"


def test_a_name_whose_extra_is_missing_says_which_extra(monkeypatch):
    """Asked for by name it once raised a bare ModuleNotFoundError from inside importlib."""
    import strata.modelling.plugins.registry as registry

    class Unimportable:
        name = "multilabel"
        value = "strata.modelling.baselines.image:MultiLabelClassifier"
        module = "strata.modelling.baselines.image"

        def load(self):
            raise ModuleNotFoundError("No module named 'timm'")

    monkeypatch.setattr(registry, "entry_points", lambda group: [Unimportable()])
    with pytest.raises(ModelError, match="needs the 'image' extra"):
        resolve("multilabel")


def test_every_advertised_name_actually_resolves():
    # An entry point can name a class that was renamed or removed, and
    # nothing notices until someone asks for it. A baseline whose extra is
    # not installed here cannot resolve, and has to say which extra instead.
    for name in available():
        try:
            model = resolve(name)
        except ModelError as e:
            assert "extra" in str(e), f"{name}: {e}"
            continue
        assert issubclass(model, Model)


def test_an_unknown_short_name_explains_the_other_path():
    with pytest.raises(ModelError, match="direct reference"):
        resolve("whatever")


def test_available_reads_what_is_installed():
    # The only honest answer to "what can this backend serve" — a
    # maintained list would drift from reality
    assert isinstance(available(), dict)


def test_the_entry_point_group_is_stable():
    # Changing it silently unregisters every plugin already published
    assert ENTRY_POINT_GROUP == "strata.models"


def test_resolving_the_same_file_twice_gives_the_same_class(tmp_path):
    # Without this a model cannot be compared to itself, which is what
    # checking a warm start against its parent needs
    (tmp_path / "mine.py").write_text(COUNTING_MODEL)
    assert resolve("mine.py:CountingModel", root=tmp_path) is resolve(
        "mine.py:CountingModel", root=tmp_path
    )


def test_two_projects_with_the_same_filename_do_not_collide(tmp_path):
    # Both carry a model.py; naming the modules after the stem made one
    # silently shadow the other
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.py").write_text(
            COUNTING_MODEL.replace("class CountingModel", f"class Model{name.upper()}")
        )
    first = resolve("model.py:ModelA", root=tmp_path / "a")
    second = resolve("model.py:ModelB", root=tmp_path / "b")
    assert first is not second
    assert (first.__name__, second.__name__) == ("ModelA", "ModelB")
