"""Load the compiled OCRS extension bundled with DocTest."""
from importlib import import_module as _import_module
import importlib.util as _importlib_util
# Prefer the bundled shared library, fallback to globally installed module.
_spec = _importlib_util.find_spec("DocTest._ocrs._ocrs")
if _spec is not None:
    module = _import_module("DocTest._ocrs._ocrs")
else:
    module = _import_module("_ocrs")
globals().update(module.__dict__)
