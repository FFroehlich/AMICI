"""Tests related to the sympy_utils module."""

from amici.sympy_utils import _custom_pow_eval_derivative, _monkeypatched
import sympy as sp
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_monkeypatch():
    t = sp.Symbol("t")
    n = sp.Symbol("n")
    vals = [(t, 0), (n, 1)]

    # check that the removable singularity still exists
    assert (t**n).diff(t).subs(vals) is sp.nan

    # check that we can monkeypatch it out
    with _monkeypatched(
        sp.Pow, "_eval_derivative", _custom_pow_eval_derivative
    ):
        assert (t**n).diff(t).subs(vals) is not sp.nan

    # check that the monkeypatch is transient
    assert (t**n).diff(t).subs(vals) is sp.nan
