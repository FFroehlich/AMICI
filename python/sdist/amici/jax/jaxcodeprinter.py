"""Jax code generation"""

import re
from collections.abc import Iterable
from logging import warning

import sympy as sp
from sympy.printing.numpy import NumPyPrinter


class AmiciJaxCodePrinter(NumPyPrinter):
    """JAX code printer"""

    def doprint(self, expr: sp.Expr, assign_to: str | None = None) -> str:
        try:
            code = super().doprint(expr, assign_to)
            code = re.sub(r"numpy\.", r"jnp.", code)

            return code
        except TypeError as e:
            raise ValueError(
                f'Encountered unsupported function in expression "{expr}"'
            ) from e

    def _print_AmiciSpline(self, expr: sp.Expr) -> str:
        warning("Spline interpolation is support in JAX is untested")
        # FIXME: untested, where are spline nodes coming from anyways?
        return f'interp1d(time, {self.doprint(expr.args[2:])}, kind="cubic")'

    def _print_log(self, expr: sp.Expr) -> str:
        return f"safe_log({self.doprint(expr.args[0])})"

    def _print_Mul(self, expr: sp.Expr) -> str:
        numer, denom = expr.as_numer_denom()
        if denom == 1:
            return super()._print_Mul(expr)
        return f"safe_div({self.doprint(numer)}, {self.doprint(denom)})"

    def _print_Function(self, expr):
        return f"self.nns['{expr.func.__name__}'].forward(jnp.array([{', '.join(self.doprint(a) for a in expr.args[:-1])}]))[{expr.args[-1]}]"

    def _get_sym_lines(
        self,
        symbols: sp.Matrix | Iterable[str],
        equations: sp.Matrix | Iterable[sp.Expr],
        indent_level: int,
    ) -> list[str]:
        """
        Generate C++ code for assigning symbolic terms in symbols to C++ array
        `variable`.

        :param equations:
            vectors of symbolic expressions

        :param symbols:
            names of the symbols to assign to

        :param indent_level:
            indentation level (number of leading blanks)

        :return:
            C++ code as list of lines
        """
        indent = " " * indent_level
        return [
            f"{indent}{s} = {self.doprint(e)}"
            for s, e in zip(symbols, equations)
        ]
