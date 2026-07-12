"""doctest_dashboard: visual review dashboard for robotframework-doctestlibrary.

Ships as part of the robotframework-doctestlibrary distribution (install the
``[dashboard]`` extra) and therefore carries the same version.
"""

from importlib import metadata

try:
    __version__ = metadata.version("robotframework-doctestlibrary")
except metadata.PackageNotFoundError:  # running from a plain checkout
    __version__ = "0.0.0.dev0"
