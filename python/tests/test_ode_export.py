"""Miscellaneous AMICI Python interface tests"""

import sympy as sp
from amici.cxxcodeprinter import csc_matrix
from amici.testing import skip_on_valgrind


@skip_on_valgrind
def test_csc_matrix():
    """Test sparse CSC matrix creation"""
    matrix = sp.Matrix([[1, 0], [2, 3]])
    (
        symbol_col_ptrs,
        symbol_row_vals,
        sparse_list,
        symbol_list,
        sparse_matrix,
    ) = csc_matrix(
        matrix,
        rownames=[sp.Symbol("a1"), sp.Symbol("a2")],
        colnames=[sp.Symbol("b1"), sp.Symbol("b2")],
    )

    assert symbol_col_ptrs == [0, 2, 3]
    assert symbol_row_vals == [0, 1, 1]
    assert sparse_list == sp.Matrix([[1], [2], [3]])
    assert symbol_list == ["da1_db1", "da2_db1", "da2_db2"]
    assert str(sparse_matrix) == "Matrix([[da1_db1, 0], [da2_db1, da2_db2]])"


@skip_on_valgrind
def test_csc_matrix_empty():
    """Test sparse CSC matrix creation for empty matrix"""
    matrix = sp.Matrix()
    (
        symbol_col_ptrs,
        symbol_row_vals,
        sparse_list,
        symbol_list,
        sparse_matrix,
    ) = csc_matrix(matrix, rownames=[], colnames=[])

    assert symbol_col_ptrs == []
    assert symbol_row_vals == []
    assert sparse_list == sp.Matrix(0, 0, [])
    assert symbol_list == []
    assert str(sparse_matrix) == "Matrix(0, 0, [])"


@skip_on_valgrind
def test_csc_matrix_vector():
    """Test sparse CSC matrix creation from matrix slice"""
    matrix = sp.Matrix([[1, 0], [2, 3]])
    (
        symbol_col_ptrs,
        symbol_row_vals,
        sparse_list,
        symbol_list,
        sparse_matrix,
    ) = csc_matrix(
        matrix[:, 0],
        colnames=[sp.Symbol("b")],
        rownames=[sp.Symbol("a1"), sp.Symbol("a2")],
    )

    assert symbol_col_ptrs == [0, 2]
    assert symbol_row_vals == [0, 1]
    assert sparse_list == sp.Matrix([[1], [2]])
    assert symbol_list == ["da1_db", "da2_db"]
    assert str(sparse_matrix) == "Matrix([[da1_db], [da2_db]])"

    # Test continuation of numbering of symbols
    (
        symbol_col_ptrs,
        symbol_row_vals,
        sparse_list,
        symbol_list,
        sparse_matrix,
    ) = csc_matrix(
        matrix[:, 1],
        colnames=[sp.Symbol("b")],
        rownames=[sp.Symbol("a1"), sp.Symbol("a2")],
        identifier=1,
    )

    assert symbol_col_ptrs == [0, 1]
    assert symbol_row_vals == [1]
    assert sparse_list == sp.Matrix([[3]])
    assert symbol_list == ["da2_db_1"]
    assert str(sparse_matrix) == "Matrix([[0], [da2_db_1]])"


@skip_on_valgrind
def test_match_deriv():
    from amici.de_export import DERIVATIVE_PATTERN as pat

    def check(str, out1, out2):
        match = pat.match(str)
        assert match[1] == out1, (str, match[1], match[2])
        assert match[2] == out2, (str, match[1], match[2])

    check("dwdx", "w", "x")
    check("dx_rdatadtotal_cl", "x_rdata", "total_cl")
    check("dtotal_cldx_rdata", "total_cl", "x_rdata")
    check("dxdotdw", "xdot", "w")
    check("dxdotdx_explicit", "xdot", "x")
