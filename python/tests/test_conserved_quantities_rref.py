import numpy as np
import pytest
import sympy as sp
from amici.conserved_quantities_rref import nullspace_by_rref, pivots, rref
from amici.testing import skip_on_valgrind


def random_matrix_generator(min_dim, max_dim, count):
    """Generate random 2D square matrix"""
    rng = np.random.default_rng(12345)
    for rows, cols in rng.integers(min_dim, max_dim, [count, 2]):
        yield np.random.rand(rows, cols)


@skip_on_valgrind
@pytest.mark.parametrize("mat", random_matrix_generator(0, 10, 200))
def test_rref(mat):
    """Create some random matrices and compare output of ``rref`` and
    ``pivots`` to that of sympy"""
    actual_rref = rref(mat)
    expected_rref, expected_pivots = sp.Matrix(mat).rref()
    expected_rref = np.asarray(expected_rref, dtype=float)

    assert list(expected_pivots) == pivots(actual_rref)
    assert np.allclose(expected_rref, actual_rref)


@skip_on_valgrind
@pytest.mark.parametrize("mat", random_matrix_generator(0, 50, 50))
def test_nullspace_by_rref(mat):
    """Test ``nullspace_by_rref`` on a number of random matrices and compare
    to sympy results"""
    actual = nullspace_by_rref(mat)

    if len(actual):
        assert np.allclose(mat.dot(actual.T), 0)

    expected = sp.Matrix(mat).nullspace()
    expected = (
        np.hstack(np.asarray(expected, dtype=float)).T
        if len(expected)
        else np.array([])
    )

    assert np.allclose(actual, expected, rtol=1e-8)
