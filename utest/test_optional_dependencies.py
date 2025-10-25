import builtins
import importlib
import sys

import pytest


def _purge_modules(monkeypatch, *prefixes):
    for prefix in prefixes:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("DocTest.VisualTest", "VisualTest"),
        ("DocTest.PdfTest", "PdfTest"),
    ],
)
def test_import_without_pydantic_dependencies(monkeypatch, module_name, class_name):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pydantic"):
            raise ModuleNotFoundError("No module named 'pydantic'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _purge_modules(
        monkeypatch,
        module_name,
        "DocTest.llm.client",
        "DocTest.llm.types",
    )

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls()
    assert instance is not None
    if class_name == "VisualTest":
        import numpy as np

        ref = np.zeros((4, 4, 3), dtype=np.uint8)
        cand = np.zeros((4, 4, 3), dtype=np.uint8)
        combined = instance.concatenate_images_safely(ref, cand, axis=1)
        assert combined.shape == (4, 8, 3)
    else:
        assert hasattr(instance, "compare_pdf_documents")
