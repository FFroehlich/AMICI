"""Miscellaneous functions related to model import, independent of any specific
 model format"""
import enum
import itertools as itt
import numbers
import sys
from typing import (
    Any,
    Callable,
    Optional,
    SupportsFloat,
    Union,
)
from collections.abc import Iterable, Sequence

import sympy as sp
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import BooleanAtom
from toposort import toposort

RESERVED_SYMBOLS = ["x", "k", "p", "y", "w", "h", "t", "AMICI_EMPTY_BOLUS"]

try:
    import pysb
except ImportError:
    pysb = None


class SBMLException(Exception):
    pass


SymbolDef = dict[sp.Symbol, Union[dict[str, sp.Expr], sp.Expr]]


# Monkey-patch toposort CircularDependencyError to handle non-sortable objects,
#  such as sympy objects
class CircularDependencyError(ValueError):
    def __init__(self, data):
        # Sort the data just to make the output consistent, for use in
        #  error messages.  That's convenient for doctests.
        s = "Circular dependencies exist among these items: {{{}}}".format(
            ", ".join(
                f"{key!r}:{value!r}"
                for key, value in sorted(
                    {str(k): v for k, v in data.items()}.items()
                )
            )
        )
        super().__init__(s)
        self.data = data


setattr(
    sys.modules["toposort"], "CircularDependencyError", CircularDependencyError
)

annotation_namespace = "https://github.com/AMICI-dev/AMICI"


class ObservableTransformation(str, enum.Enum):
    """
    Different modes of observable transformation.
    """

    LOG10 = "log10"
    LOG = "log"
    LIN = "lin"


def noise_distribution_to_observable_transformation(
    noise_distribution: Union[str, Callable],
) -> ObservableTransformation:
    """
    Parse noise distribution string and extract observable transformation

    :param noise_distribution:
        see :func:`noise_distribution_to_cost_function`

    :return:
        observable transformation
    """
    if isinstance(noise_distribution, str):
        if noise_distribution.startswith("log-"):
            return ObservableTransformation.LOG
        if noise_distribution.startswith("log10-"):
            return ObservableTransformation.LOG10

    return ObservableTransformation.LIN


def noise_distribution_to_cost_function(
    noise_distribution: Union[str, Callable],
) -> Callable[[str], str]:
    """
    Parse noise distribution string to a cost function definition amici can
    work with.

    The noise distributions listed in the following are supported. :math:`m`
    denotes the measurement, :math:`y` the simulation, and :math:`\\sigma` a
    distribution scale parameter
    (currently, AMICI only supports a single distribution parameter).

    - `'normal'`, `'lin-normal'`: A normal distribution:

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{\\sqrt{2\\pi}\\sigma}\\
         exp\\left(-\\frac{(m-y)^2}{2\\sigma^2}\\right)

    - `'log-normal'`: A log-normal distribution (i.e. log(m) is
      normally distributed):

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{\\sqrt{2\\pi}\\sigma m}\\
         exp\\left(-\\frac{(\\log m - \\log y)^2}{2\\sigma^2}\\right)

    - `'log10-normal'`: A log10-normal distribution (i.e. log10(m) is
      normally distributed):

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{\\sqrt{2\\pi}\\sigma m \\log(10)}\\
         exp\\left(-\\frac{(\\log_{10} m - \\log_{10} y)^2}{2\\sigma^2}\\right)

    - `'laplace'`, `'lin-laplace'`: A laplace distribution:

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{2\\sigma}
         \\exp\\left(-\\frac{|m-y|}{\\sigma}\\right)

    - `'log-laplace'`: A log-Laplace distribution (i.e. log(m) is Laplace
      distributed):

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{2\\sigma m}
         \\exp\\left(-\\frac{|\\log m - \\log y|}{\\sigma}\\right)

    - `'log10-laplace'`: A log10-Laplace distribution (i.e. log10(m) is
      Laplace distributed):

      .. math::
         \\pi(m|y,\\sigma) = \\frac{1}{2\\sigma m \\log(10)}
         \\exp\\left(-\\frac{|\\log_{10} m - \\log_{10} y|}{\\sigma}\\right)

    - `'binomial'`, `'lin-binomial'`: A (continuation of a discrete) binomial
      distribution, parameterized via the success probability
      :math:`p=\\sigma`:

      .. math::
         \\pi(m|y,\\sigma) = \\operatorname{Heaviside}(y-m) \\cdot
                \\frac{\\Gamma(y+1)}{\\Gamma(m+1) \\Gamma(y-m+1)}
                \\sigma^m (1-\\sigma)^{(y-m)}

    - `'negative-binomial'`, `'lin-negative-binomial'`: A (continuation of a
      discrete) negative binomial distribution, with with `mean = y`,
      parameterized via success probability `p`:

      .. math::

         \\pi(m|y,\\sigma) = \\frac{\\Gamma(m+r)}{\\Gamma(m+1) \\Gamma(r)}
            (1-\\sigma)^m \\sigma^r

      where

      .. math::
         r = \\frac{1-\\sigma}{\\sigma} y

    The distributions above are for a single data point.
    For a collection :math:`D=\\{m_i\\}_i` of data points and corresponding
    simulations :math:`Y=\\{y_i\\}_i` and noise parameters
    :math:`\\Sigma=\\{\\sigma_i\\}_i`, AMICI assumes independence,
    i.e. the full distributions is

    .. math::
       \\pi(D|Y,\\Sigma) = \\prod_i\\pi(m_i|y_i,\\sigma_i)

    AMICI uses the logarithm :math:`\\log(\\pi(m|y,\\sigma)`.

    In addition to the above mentioned distributions, it is also possible to
    pass a function taking a symbol string and returning a log-distribution
    string with variables '{str_symbol}', 'm{str_symbol}', 'sigma{str_symbol}'
    for y, m, sigma, respectively.

    :param noise_distribution: An identifier specifying a noise model.
        Possible values are

        {`'normal'`, `'lin-normal'`, `'log-normal'`, `'log10-normal'`,
        `'laplace'`, `'lin-laplace'`, `'log-laplace'`, `'log10-laplace'`,
        `'binomial'`, `'lin-binomial'`, `'negative-binomial'`,
        `'lin-negative-binomial'`, `<Callable>`}

        For the meaning of the values see above.

    :return: A function that takes a strSymbol and then creates a cost
        function string (negative log-likelihood) from it, which can be
        sympified.
    """

    if isinstance(noise_distribution, Callable):
        return noise_distribution

    if noise_distribution in ["normal", "lin-normal"]:
        y_string = "0.5*log(2*pi*{sigma}**2) + 0.5*(({y} - {m}) / {sigma})**2"
    elif noise_distribution == "log-normal":
        y_string = (
            "0.5*log(2*pi*{sigma}**2*{m}**2) "
            "+ 0.5*((log({y}) - log({m})) / {sigma})**2"
        )
    elif noise_distribution == "log10-normal":
        y_string = (
            "0.5*log(2*pi*{sigma}**2*{m}**2*log(10)**2) "
            "+ 0.5*((log({y}, 10) - log({m}, 10)) / {sigma})**2"
        )
    elif noise_distribution in ["laplace", "lin-laplace"]:
        y_string = "log(2*{sigma}) + Abs({y} - {m}) / {sigma}"
    elif noise_distribution == "log-laplace":
        y_string = "log(2*{sigma}*{m}) + Abs(log({y}) - log({m})) / {sigma}"
    elif noise_distribution == "log10-laplace":
        y_string = (
            "log(2*{sigma}*{m}*log(10)) "
            "+ Abs(log({y}, 10) - log({m}, 10)) / {sigma}"
        )
    elif noise_distribution in ["binomial", "lin-binomial"]:
        # Binomial noise model parameterized via success probability p
        y_string = (
            "- log(Heaviside({y} - {m})) - loggamma({y}+1) "
            "+ loggamma({m}+1) + loggamma({y}-{m}+1) "
            "- {m} * log({sigma}) - ({y} - {m}) * log(1-{sigma})"
        )
    elif noise_distribution in ["negative-binomial", "lin-negative-binomial"]:
        # Negative binomial noise model of the number of successes m
        # (data) before r=(1-sigma)/sigma * y failures occur,
        # with mean number of successes y (simulation),
        # parameterized via success probability p = sigma.
        r = "{y} * (1-{sigma}) / {sigma}"
        y_string = (
            f"- loggamma({{m}}+{r}) + loggamma({{m}}+1) "
            f"+ loggamma({r}) - {r} * log(1-{{sigma}}) "
            f"- {{m}} * log({{sigma}})"
        )
    else:
        raise ValueError(
            f"Cost identifier {noise_distribution} not recognized."
        )

    def nllh_y_string(str_symbol):
        y, m, sigma = _get_str_symbol_identifiers(str_symbol)
        return y_string.format(y=y, m=m, sigma=sigma)

    return nllh_y_string


def _get_str_symbol_identifiers(str_symbol: str) -> tuple:
    """Get identifiers for simulation, measurement, and sigma."""
    y, m, sigma = f"{str_symbol}", f"m{str_symbol}", f"sigma{str_symbol}"
    return y, m, sigma


def smart_subs_dict(
    sym: sp.Expr,
    subs: SymbolDef,
    field: Optional[str] = None,
    reverse: bool = True,
) -> sp.Expr:
    """
    Substitutes expressions completely flattening them out. Requires
    sorting of expressions with toposort.

    :param sym:
        Symbolic expression in which expressions will be substituted

    :param subs:
        Substitutions

    :param field:
        Field of substitution expressions in subs.values(), if applicable

    :param reverse:
        Whether ordering in subs should be reversed. Note that substitution
        requires the reverse order of what is required for evaluation.

    :return:
        Substituted symbolic expression
    """
    s = [
        (eid, expr[field] if field is not None else expr)
        for eid, expr in subs.items()
    ]
    if reverse:
        s.reverse()
    for substitution in s:
        # note that substitution may change free symbols, so we have to do
        # this recursively
        if sym.has(substitution[0]):
            sym = sym.subs(*substitution)
    return sym


def smart_subs(element: sp.Expr, old: sp.Symbol, new: sp.Expr) -> sp.Expr:
    """
    Optimized substitution that checks whether anything needs to be done first

    :param element:
        substitution target

    :param old:
        to be substituted

    :param new:
        subsitution value

    :return:
        substituted expression
    """
    return element.subs(old, new) if element.has(old) else element


def toposort_symbols(
    symbols: SymbolDef, field: Optional[str] = None
) -> SymbolDef:
    """
    Topologically sort symbol definitions according to their interdependency

    :param symbols:
        symbol definitions

    :param field:
        field of definition.values() that is used to compute interdependency

    :return:
        ordered symbol definitions
    """
    sorted_symbols = toposort(
        {
            identifier: {
                s
                for s in (
                    definition[field] if field is not None else definition
                ).free_symbols
                if s in symbols
            }
            for identifier, definition in symbols.items()
        }
    )
    return {
        s: symbols[s]
        for symbol_group in sorted_symbols
        for s in sorted(symbol_group, key=str)
    }


def _parse_special_functions(sym: sp.Expr, toplevel: bool = True) -> sp.Expr:
    """
    Recursively checks the symbolic expression for functions which have be
    to parsed in a special way, such as piecewise functions

    :param sym:
        symbolic expressions

    :param toplevel:
        as this is called recursively, are we in the top level expression?
    """
    args = tuple(
        arg
        if arg.__class__.__name__ == "piecewise"
        and sym.__class__.__name__ == "piecewise"
        else _parse_special_functions(arg, False)
        for arg in sym.args
    )

    fun_mappings = {
        "times": sp.Mul,
        "xor": sp.Xor,
        "abs": sp.Abs,
        "min": sp.Min,
        "max": sp.Max,
        "ceil": sp.functions.ceiling,
        "floor": sp.functions.floor,
        "factorial": sp.functions.factorial,
        "arcsin": sp.functions.asin,
        "arccos": sp.functions.acos,
        "arctan": sp.functions.atan,
        "arccot": sp.functions.acot,
        "arcsec": sp.functions.asec,
        "arccsc": sp.functions.acsc,
        "arcsinh": sp.functions.asinh,
        "arccosh": sp.functions.acosh,
        "arctanh": sp.functions.atanh,
        "arccoth": sp.functions.acoth,
        "arcsech": sp.functions.asech,
        "arccsch": sp.functions.acsch,
    }

    if sym.__class__.__name__ in fun_mappings:
        return fun_mappings[sym.__class__.__name__](*args)

    elif sym.__class__.__name__ == "piecewise" or isinstance(
        sym, sp.Piecewise
    ):
        if isinstance(sym, sp.Piecewise):
            # this is sympy piecewise, can't be nested
            denested_args = args
        else:
            # this is sbml piecewise, can be nested
            denested_args = _denest_piecewise(args)
        return _parse_piecewise_to_heaviside(denested_args)

    if sym.__class__.__name__ == "plus" and not sym.args:
        return sp.Float(0.0)

    if isinstance(sym, (sp.Function, sp.Mul, sp.Add, sp.Pow)):
        sym._args = args

    elif toplevel and isinstance(sym, BooleanAtom):
        # Replace boolean constants by numbers so they can be differentiated
        #  must not replace in Piecewise function. Therefore, we only replace
        #  it the complete expression consists only of a Boolean value.
        sym = sp.Float(int(bool(sym)))

    return sym


def _denest_piecewise(
    args: Sequence[Union[sp.Expr, sp.logic.boolalg.Boolean, bool]],
) -> tuple[Union[sp.Expr, sp.logic.boolalg.Boolean, bool]]:
    """
    Denest piecewise functions that contain piecewise as condition

    :param args:
        Arguments to the piecewise function

    :return:
        Arguments where conditions no longer contain piecewise functions and
        the conditional dependency is flattened out
    """
    args_out = []
    for coeff, cond in grouper(args, 2, True):
        # handling of this case is explicitely disabled in
        # _parse_special_functions as keeping track of coeff/cond
        # arguments is tricky. Simpler to just parse them out here
        if coeff.__class__.__name__ == "piecewise":
            coeff = _parse_special_functions(coeff, False)

        # we can have conditions that are piecewise function
        # returning True or False
        if cond.__class__.__name__ == "piecewise":
            # this keeps track of conditional that the previous
            # piece was picked
            previous_was_picked = sp.false
            # recursively denest those first
            for sub_coeff, sub_cond in grouper(
                _denest_piecewise(cond.args), 2, True
            ):
                # flatten the individual pieces
                pick_this = sp.And(sp.Not(previous_was_picked), sub_cond)
                if sub_coeff == sp.true:
                    args_out.extend([coeff, pick_this])
                previous_was_picked = pick_this

        else:
            args_out.extend([coeff, cond])
    # cut off last condition as that's the default
    return tuple(args_out[:-1])


def _parse_piecewise_to_heaviside(args: Iterable[sp.Expr]) -> sp.Expr:
    """
    Piecewise functions cannot be transformed into C++ right away, but AMICI
    has a special interface for Heaviside functions, so we transform them.

    :param args:
        symbolic expressions for arguments of the piecewise function
    """
    # how many condition-expression pairs will we have?
    formula = sp.Float(0.0)
    not_condition = sp.Float(1.0)

    if all(isinstance(arg, ExprCondPair) for arg in args):
        # sympy piecewise
        grouped_args = args
    else:
        # smbl piecewise
        grouped_args = grouper(args, 2, True)

    for coeff, trigger in grouped_args:
        if isinstance(coeff, BooleanAtom):
            coeff = sp.Float(int(bool(coeff)))

        if trigger == sp.true:
            return formula + coeff * not_condition

        if trigger == sp.false:
            continue

        tmp = _parse_heaviside_trigger(trigger)
        formula += coeff * sp.simplify(not_condition * tmp)
        not_condition *= 1 - tmp

    return formula


def _parse_heaviside_trigger(trigger: sp.Expr) -> sp.Expr:
    """
    Recursively translates a boolean trigger function into a real valued
    root function

    :param trigger:
    :return: real valued root function expression
    """
    if trigger.is_Relational:
        root = trigger.args[0] - trigger.args[1]
        _check_unsupported_functions(root, "sympy.Expression")

        # normalize such that we always implement <,
        # this ensures that we can correctly evaluate the condition if
        # simulation starts at H(0). This is achieved by translating
        # conditionals into Heaviside functions H that is implemented as unit
        # step with H(0) = 1
        if isinstance(trigger, sp.core.relational.StrictLessThan):
            # x < y => x - y < 0 => r < 0
            return 1 - sp.Heaviside(root)
        if isinstance(trigger, sp.core.relational.LessThan):
            # x <= y => not(y < x) => not(y - x < 0) => not -r < 0
            return sp.Heaviside(-root)
        if isinstance(trigger, sp.core.relational.StrictGreaterThan):
            # y > x => y - x < 0 => -r < 0
            return 1 - sp.Heaviside(-root)
        if isinstance(trigger, sp.core.relational.GreaterThan):
            # y >= x => not(x < y) => not(x - y < 0) => not r < 0
            return sp.Heaviside(root)

    # or(x,y) = not(and(not(x),not(y))
    if isinstance(trigger, sp.Or):
        return 1 - sp.Mul(
            *[1 - _parse_heaviside_trigger(arg) for arg in trigger.args]
        )

    if isinstance(trigger, sp.And):
        return sp.Mul(*[_parse_heaviside_trigger(arg) for arg in trigger.args])

    raise RuntimeError(
        "AMICI can not parse piecewise/event trigger functions with argument "
        f"{trigger}."
    )


def grouper(
    iterable: Iterable, n: int, fillvalue: Any = None
) -> Iterable[tuple[Any]]:
    """
    Collect data into fixed-length chunks or blocks

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"

    :param iterable:
        any iterable

    :param n:
        chunk length

    :param fillvalue:
        padding for last chunk if length < n

    :return: itertools.zip_longest of requested chunks
    """
    args = [iter(iterable)] * n
    return itt.zip_longest(*args, fillvalue=fillvalue)


def _check_unsupported_functions(
    sym: sp.Expr, expression_type: str, full_sym: Optional[sp.Expr] = None
):
    """
    Recursively checks the symbolic expression for unsupported symbolic
    functions

    :param sym:
        symbolic expressions

    :param expression_type:
        type of expression, only used when throwing errors

    :param full sym:
        outermost symbolic expression in recursive checks, only used for errors
    """
    if full_sym is None:
        full_sym = sym

    # note that sp.functions.factorial, sp.functions.ceiling,
    # sp.functions.floor applied to numbers should be simplified out and
    # thus pass this test
    unsupported_functions = (
        sp.functions.factorial,
        sp.functions.ceiling,
        sp.functions.floor,
        sp.functions.sec,
        sp.functions.csc,
        sp.functions.cot,
        sp.functions.asec,
        sp.functions.acsc,
        sp.functions.acot,
        sp.functions.acsch,
        sp.functions.acoth,
        sp.Mod,
        sp.core.function.UndefinedFunction,
    )

    if (
        isinstance(sym.func, unsupported_functions)
        or isinstance(sym, unsupported_functions)
    ) and getattr(sym.func, "name", "") != "rateOf":
        raise RuntimeError(
            f"Encountered unsupported expression "
            f'"{sym.func}" of type '
            f'"{type(sym.func)}" as part of a '
            f'{expression_type}: "{full_sym}"!'
        )
    for arg in list(sym.args):
        _check_unsupported_functions(arg, expression_type)


def cast_to_sym(
    value: Union[SupportsFloat, sp.Expr, BooleanAtom], input_name: str
) -> sp.Expr:
    """
    Typecasts the value to :py:class:`sympy.Float` if possible, and ensures the
    value is a symbolic expression.

    :param value:
        value to be cast

    :param input_name:
        name of input variable

    :return:
        typecast value
    """
    if isinstance(value, (sp.RealNumber, numbers.Number)):
        value = sp.Float(float(value))
    elif isinstance(value, BooleanAtom):
        value = sp.Float(float(bool(value)))

    if not isinstance(value, sp.Expr):
        raise TypeError(
            f"Couldn't cast {input_name} to sympy.Expr, was " f"{type(value)}"
        )

    return value


def generate_measurement_symbol(observable_id: Union[str, sp.Symbol]):
    """
    Generates the appropriate measurement symbol for the provided observable

    :param observable_id:
        symbol (or string representation) of the observable

    :return:
        symbol for the corresponding measurement
    """
    if not isinstance(observable_id, str):
        observable_id = strip_pysb(observable_id)
    return symbol_with_assumptions(f"m{observable_id}")


def generate_regularization_symbol(observable_id: Union[str, sp.Symbol]):
    """
    Generates the appropriate regularization symbol for the provided observable

    :param observable_id:
        symbol (or string representation) of the observable

    :return:
        symbol for the corresponding regularization
    """
    if not isinstance(observable_id, str):
        observable_id = strip_pysb(observable_id)
    return symbol_with_assumptions(f"r{observable_id}")


def generate_flux_symbol(
    reaction_index: int, name: Optional[str] = None
) -> sp.Symbol:
    """
    Generate identifier symbol for a reaction flux.
    This function will always return the same unique python object for a
    given entity.

    :param reaction_index:
        index of the reaction to which the flux corresponds
    :param name:
        an optional identifier of the reaction to which the flux corresponds
    :return:
        identifier symbol
    """
    if name is not None:
        return symbol_with_assumptions(name)

    return symbol_with_assumptions(f"flux_r{reaction_index}")


def symbol_with_assumptions(name: str):
    """
    Central function to create symbols with consistent, canonical assumptions

    :param name:
        name of the symbol

    :return:
        symbol with canonical assumptions
    """
    return sp.Symbol(name, real=True)


def strip_pysb(symbol: sp.Basic) -> sp.Basic:
    """
    Strips pysb info from a :class:`pysb.Component` object

    :param symbol:
        symbolic expression

    :return:
        stripped expression
    """
    # strip pysb type and transform into a flat sympy.Symbol.
    # this ensures that the pysb type specific __repr__ is used when converting
    # to string
    if pysb and isinstance(symbol, pysb.Component):
        return sp.Symbol(symbol.name, real=True)
    else:
        # in this case we will use sympy specific transform anyways
        return symbol


def unique_preserve_order(seq: Sequence) -> list:
    """Return a list of unique elements in Sequence, keeping only the first
    occurrence of each element

    Parameters:
        seq: Sequence to prune

    Returns:
        List of unique elements in ``seq``
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


sbml_time_symbol = symbol_with_assumptions("time")
amici_time_symbol = symbol_with_assumptions("t")


def _default_simplify(x):
    """Default simplification applied in DEModel"""
    # We need this as a free function instead of a lambda to have it picklable
    #  for parallel simplification
    return sp.powsimp(x, deep=True)
