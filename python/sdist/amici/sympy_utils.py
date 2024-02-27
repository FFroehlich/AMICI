"""Functionality for working with sympy objects."""
import os
from itertools import starmap
from typing import Union, Any, Callable
import contextlib
import sympy as sp
import logging
from amici.de_export import get_logger
from amici.logging import log_execution_time


logger = get_logger(__name__, logging.ERROR)


def _custom_pow_eval_derivative(self, s):
    """
    Custom Pow derivative that removes a removable singularity for
    ``self.base == 0`` and ``self.base.diff(s) == 0``. This function is
    intended to be monkeypatched into :py:method:`sympy.Pow._eval_derivative`.

    :param self:
        sp.Pow class

    :param s:
        variable with respect to which the derivative will be computed
    """
    dbase = self.base.diff(s)
    dexp = self.exp.diff(s)
    part1 = sp.Pow(self.base, self.exp - 1) * self.exp * dbase
    part2 = self * dexp * sp.log(self.base)
    if self.base.is_nonzero or dbase.is_nonzero or part2.is_zero:
        # first piece never applies or is zero anyway
        return part1 + part2

    return part1 + sp.Piecewise(
        (self.base, sp.And(sp.Eq(self.base, 0), sp.Eq(dbase, 0))),
        (part2, True),
    )


@contextlib.contextmanager
def _monkeypatched(obj: object, name: str, patch: Any):
    """
    Temporarily monkeypatches an object.

    :param obj:
        object to be patched

    :param name:
        name of the attribute to be patched

    :param patch:
        patched value
    """
    pre_patched_value = getattr(obj, name)
    setattr(obj, name, patch)
    try:
        yield object
    finally:
        setattr(obj, name, pre_patched_value)


@log_execution_time("running smart_jacobian", logger)
def smart_jacobian(
    eq: sp.MutableDenseMatrix, sym_var: sp.MutableDenseMatrix
) -> sp.MutableSparseMatrix:
    """
    Wrapper around symbolic jacobian with some additional checks that reduce
    computation time for large matrices

    :param eq:
        equation
    :param sym_var:
        differentiation variable
    :return:
        jacobian of eq wrt sym_var
    """
    nrow = eq.shape[0]
    ncol = sym_var.shape[0]
    if (
        not min(eq.shape)
        or not min(sym_var.shape)
        or smart_is_zero_matrix(eq)
        or smart_is_zero_matrix(sym_var)
    ):
        return sp.MutableSparseMatrix(nrow, ncol, dict())

    # preprocess sparsity pattern
    elements = (
        (i, j, a, b)
        for i, a in enumerate(eq)
        for j, b in enumerate(sym_var)
        if a.has(b)
    )

    if (n_procs := int(os.environ.get("AMICI_IMPORT_NPROCS", 1))) == 1:
        # serial
        return sp.MutableSparseMatrix(
            nrow, ncol, dict(starmap(_jacobian_element, elements))
        )

    # parallel
    from multiprocessing import get_context

    # "spawn" should avoid potential deadlocks occurring with fork
    #  see e.g. https://stackoverflow.com/a/66113051
    ctx = get_context("spawn")
    with ctx.Pool(n_procs) as p:
        mapped = p.starmap(_jacobian_element, elements)
    return sp.MutableSparseMatrix(nrow, ncol, dict(mapped))


@log_execution_time("running smart_multiply", logger)
def smart_multiply(
    x: Union[sp.MutableDenseMatrix, sp.MutableSparseMatrix],
    y: sp.MutableDenseMatrix,
) -> Union[sp.MutableDenseMatrix, sp.MutableSparseMatrix]:
    """
    Wrapper around symbolic multiplication with some additional checks that
    reduce computation time for large matrices

    :param x:
        educt 1
    :param y:
        educt 2
    :return:
        product
    """
    if (
        not x.shape[0]
        or not y.shape[1]
        or smart_is_zero_matrix(x)
        or smart_is_zero_matrix(y)
    ):
        return sp.zeros(x.shape[0], y.shape[1])
    return x.multiply(y)


def smart_is_zero_matrix(
    x: Union[sp.MutableDenseMatrix, sp.MutableSparseMatrix],
) -> bool:
    """A faster implementation of sympy's is_zero_matrix

    Avoids repeated indexer type checks and double iteration to distinguish
    False/None. Found to be about 100x faster for large matrices.

    :param x: Matrix to check
    """

    if isinstance(x, sp.MutableDenseMatrix):
        return all(xx.is_zero is True for xx in x.flat())

    if isinstance(x, list):
        return all(smart_is_zero_matrix(xx) for xx in x)

    return x.nnz() == 0


def _jacobian_element(i, j, eq_i, sym_var_j):
    """Compute a single element of a jacobian"""
    return (i, j), eq_i.diff(sym_var_j)


def _parallel_applyfunc(obj: sp.Matrix, func: Callable) -> sp.Matrix:
    """Parallel implementation of sympy's Matrix.applyfunc"""
    if (n_procs := int(os.environ.get("AMICI_IMPORT_NPROCS", 1))) == 1:
        # serial
        return obj.applyfunc(func)

    # parallel
    from multiprocessing import get_context
    from pickle import PicklingError

    from sympy.matrices.dense import DenseMatrix

    # "spawn" should avoid potential deadlocks occurring with fork
    #  see e.g. https://stackoverflow.com/a/66113051
    ctx = get_context("spawn")
    with ctx.Pool(n_procs) as p:
        try:
            if isinstance(obj, DenseMatrix):
                return obj._new(obj.rows, obj.cols, p.map(func, obj))
            elif isinstance(obj, sp.SparseMatrix):
                dok = obj.todok()
                mapped = p.map(func, dok.values())
                dok = {k: v for k, v in zip(dok.keys(), mapped) if v != 0}
                return obj._new(obj.rows, obj.cols, dok)
            else:
                raise ValueError(f"Unsupported matrix type {type(obj)}")
        except PicklingError as e:
            raise ValueError(
                f"Couldn't pickle {func}. This is likely because the argument "
                "was not a module-level function. Either rewrite the argument "
                "to a module-level function or disable parallelization by "
                "setting `AMICI_IMPORT_NPROCS=1`."
            ) from e
