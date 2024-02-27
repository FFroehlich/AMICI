"""Test support functions"""
import os
import sys
from tempfile import TemporaryDirectory

import pytest

# Indicates whether we are currently running under valgrind
#  see also https://stackoverflow.com/a/62364698
ON_VALGRIND = any(
    needle in haystack
    for needle in ("valgrind", "vgpreload")
    for haystack in (
        os.getenv("LD_PRELOAD", ""),
        os.getenv("DYLD_INSERT_LIBRARIES", ""),
    )
)

# Decorator to skip certain tests when we are under valgrind
#  (those that are independent of the AMICI C++ parts, or that take too long,
#  or that test performance)
skip_on_valgrind = pytest.mark.skipif(
    ON_VALGRIND, reason="Takes too long or is meaningless under valgrind"
)


class TemporaryDirectoryWinSafe(TemporaryDirectory):
    """TemporaryDirectory that will not raise if cleanup fails.

    If any extension was loaded from the temporary directory, cleanup would
    otherwise fail on Windows with a ``PermissionError``. This class ignores
    such failures.
    """

    def __init__(self, *args, delete=True, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO Python3.12 TemporaryDirectory already has a delete argument
        #  remove this once we drop support for Python3.11
        self.delete = delete

        if not self.delete:
            self._finalizer.detach()

    def cleanup(self):
        if not self.delete:
            return

        try:
            super().cleanup()
        except PermissionError as e:
            if sys.platform not in {"win32", "cygwin"}:
                raise e
        except NotADirectoryError:
            # Ignore exception on Windows for pyd files:
            #  NotADirectoryError: [WinError 267] The directory name is
            #  invalid: '....pyd'
            pass
