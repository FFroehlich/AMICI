"""Symbolic differential equation model."""

from __future__ import annotations

import contextlib
import copy
import itertools
import re
from itertools import chain
from typing import TYPE_CHECKING
from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
import sympy as sp
from sympy import ImmutableDenseMatrix, MutableDenseMatrix

from ._codegen.cxx_functions import (
    sparse_functions,
    sensi_functions,
    nobody_functions,
    var_in_function_signature,
)
from .cxxcodeprinter import csc_matrix
from .de_model_components import (
    DifferentialState,
    AlgebraicState,
    AlgebraicEquation,
    Observable,
    EventObservable,
    Sigma,
    SigmaY,
    SigmaZ,
    Parameter,
    Constant,
    LogLikelihood,
    LogLikelihoodY,
    LogLikelihoodZ,
    LogLikelihoodRZ,
    Expression,
    ConservationLaw,
    Event,
    State,
    ModelQuantity,
)
from .import_utils import (
    _default_simplify,
    SBMLException,
    toposort_symbols,
    smart_subs_dict,
    ObservableTransformation,
    amici_time_symbol,
    strip_pysb,
    unique_preserve_order,
)
from .sympy_utils import (
    smart_jacobian,
    smart_is_zero_matrix,
    smart_multiply,
    _parallel_applyfunc,
)
from .logging import get_logger, log_execution_time, set_log_level
import logging

if TYPE_CHECKING:
    from .splines import AbstractSpline

logger = get_logger(__name__, logging.ERROR)


DERIVATIVE_PATTERN = re.compile(r"^d(x_rdata|xdot|\w+?)d(\w+?)(?:_explicit)?$")


class DEModel:
    """
    Defines a Differential Equation as set of ModelQuantities.
    This class provides general purpose interfaces to compute arbitrary
    symbolic derivatives that are necessary for model simulation or
    sensitivity computation.

    :ivar _differential_states:
        list of differential state variables

    :ivar _algebraic_states:
        list of algebraic state variables

    :ivar _observables:
        list of observables

    :ivar _event_observables:
        list of event observables

    :ivar _sigma_ys:
        list of sigmas for observables

    :ivar _sigma_zs:
        list of sigmas for event observables

    :ivar _parameters:
        list of parameters

    :ivar _log_likelihood_ys:
        list of loglikelihoods for observables

    :ivar _log_likelihood_zs:
        list of loglikelihoods for event observables

    :ivar _log_likelihood_rzs:
        list of loglikelihoods for event observable regularizations

    :ivar _expressions:
        list of expressions instances

    :ivar _conservation_laws:
        list of conservation laws

    :ivar _symboldim_funs:
        define functions that compute model dimensions, these
        are functions as the underlying symbolic expressions have not been
        populated at compile time

    :ivar _eqs:
        carries symbolic formulas of the symbolic variables of the model

    :ivar _sparseeqs:
        carries linear list of all symbolic formulas for sparsified
        variables

    :ivar _vals:
        carries numeric values of symbolic identifiers of the symbolic
        variables of the model

    :ivar _names:
        carries the names of symbolic identifiers of the symbolic variables
        of the model

    :ivar _syms:
        carries symbolic identifiers of the symbolic variables of the
        model

    :ivar _sparsesyms:
        carries linear list of all symbolic identifiers for sparsified
        variables

    :ivar _colptrs:
        carries column pointers for sparsified variables. See
        SUNMatrixContent_Sparse definition in ``sunmatrix/sunmatrix_sparse.h``

    :ivar _rowvals:
        carries row values for sparsified variables. See
        SUNMatrixContent_Sparse definition in ``sunmatrix/sunmatrix_sparse.h``

    :ivar _equation_prototype:
        defines the attribute from which an equation should be generated via
        list comprehension (see :meth:`OEModel._generate_equation`)

    :ivar _variable_prototype:
        defines the attribute from which a variable should be generated via
        list comprehension (see :meth:`DEModel._generate_symbol`)

    :ivar _value_prototype:
        defines the attribute from which a value should be generated via
        list comprehension (see :meth:`DEModel._generate_value`)

    :ivar _total_derivative_prototypes:
        defines how a total derivative equation is computed for an equation,
        key defines the name and values should be arguments for
        :meth:`DEModel.totalDerivative`

    :ivar _lock_total_derivative:
        add chainvariables to this set when computing total derivative from
        a partial derivative call to enforce a partial derivative in the
        next recursion. prevents infinite recursion

    :ivar _simplify:
        If not None, this function will be used to simplify symbolic
        derivative expressions. Receives sympy expressions as only argument.
        To apply multiple simplifications, wrap them in a lambda expression.

    :ivar _x0_fixedParameters_idx:
        Index list of subset of states for which x0_fixedParameters was
        computed

    :ivar _w_recursion_depth:
        recursion depth in w, quantified as nilpotency of dwdw

    :ivar _has_quadratic_nllh:
        whether all observables have a gaussian noise model, i.e. whether
        res and FIM make sense.

    :ivar _static_indices:
        dict of lists list of indices of static variables for different
        model entities.

    :ivar _z2event:
        list of event indices for each event observable
    """

    def __init__(
        self,
        verbose: bool | int | None = False,
        simplify: Callable | None = _default_simplify,
        cache_simplify: bool = False,
        hybridisation: bool = False,
    ):
        """
        Create a new DEModel instance.

        :param verbose:
            verbosity level for logging, True/False default to
            ``logging.DEBUG``/``logging.ERROR``

        :param simplify:
            see :meth:`DEModel._simplify`

        :param cache_simplify:
            Whether to cache calls to the simplify method. Can e.g. decrease
            import times for models with events.
        """
        self._differential_states: list[DifferentialState] = []
        self._algebraic_states: list[AlgebraicState] = []
        self._algebraic_equations: list[AlgebraicEquation] = []
        self._observables: list[Observable] = []
        self._event_observables: list[EventObservable] = []
        self._sigma_ys: list[SigmaY] = []
        self._sigma_zs: list[SigmaZ] = []
        self._parameters: list[Parameter] = []
        self._constants: list[Constant] = []
        self._log_likelihood_ys: list[LogLikelihoodY] = []
        self._log_likelihood_zs: list[LogLikelihoodZ] = []
        self._log_likelihood_rzs: list[LogLikelihoodRZ] = []
        self._expressions: list[Expression] = []
        self._conservation_laws: list[ConservationLaw] = []
        self._events: list[Event] = []
        self._splines: list[AbstractSpline] = []
        self._symboldim_funs: dict[str, Callable[[], int]] = {
            "sx": self.num_states_solver,
            "v": self.num_states_solver,
            "vB": self.num_states_solver,
            "xB": self.num_states_solver,
            "sigmay": self.num_obs,
            "sigmaz": self.num_eventobs,
        }
        self._eqs: dict[
            str,
            (sp.Matrix | sp.SparseMatrix | list[sp.Matrix | sp.SparseMatrix]),
        ] = dict()
        self._sparseeqs: dict[str, sp.Matrix | list[sp.Matrix]] = dict()
        self._vals: dict[str, list[sp.Expr]] = dict()
        self._names: dict[str, list[str]] = dict()
        self._syms: dict[str, sp.Matrix | list[sp.Matrix]] = dict()
        self._sparsesyms: dict[str, list[str] | list[list[str]]] = dict()
        self._colptrs: dict[str, list[int] | list[list[int]]] = dict()
        self._rowvals: dict[str, list[int] | list[list[int]]] = dict()

        self._equation_prototype: dict[str, Callable] = {
            "total_cl": self.conservation_laws,
            "x0": self.states,
            "y": self.observables,
            "Jy": self.log_likelihood_ys,
            "Jz": self.log_likelihood_zs,
            "Jrz": self.log_likelihood_rzs,
            "w": self.expressions,
            "root": self.events,
            "sigmay": self.sigma_ys,
            "sigmaz": self.sigma_zs,
        }
        self._variable_prototype: dict[str, Callable] = {
            "tcl": self.conservation_laws,
            "x_rdata": self.states,
            "y": self.observables,
            "z": self.event_observables,
            "p": self.parameters,
            "k": self.constants,
            "w": self.expressions,
            "sigmay": self.sigma_ys,
            "sigmaz": self.sigma_zs,
            "h": self.events,
        }
        self._value_prototype: dict[str, Callable] = {
            "p": self.parameters,
            "k": self.constants,
        }
        self._total_derivative_prototypes: dict[
            str, dict[str, str | list[str]]
        ] = {
            "sroot": {
                "eq": "root",
                "chainvars": ["x"],
                "var": "p",
                "dxdz_name": "sx",
            },
        }

        self._lock_total_derivative: list[str] = list()
        self._simplify: Callable = simplify
        if cache_simplify and simplify is not None:

            def cached_simplify(
                expr: sp.Expr,
                _simplified: dict[str, sp.Expr] = {},  # noqa B006
                _simplify: Callable = simplify,
            ) -> sp.Expr:
                """Speed up expression simplification with caching.

                NB: This can decrease model import times for models that have
                    many repeated expressions during C++ file generation.
                    For example, this can be useful for models with events.
                    However, for other models, this may increase model import
                    times.

                :param expr:
                    The SymPy expression.
                :param _simplified:
                    The cache.
                :param _simplify:
                    The simplification method.

                :return:
                    The simplified expression.
                """
                expr_str = repr(expr)
                if expr_str not in _simplified:
                    _simplified[expr_str] = _simplify(expr)
                return _simplified[expr_str]

            self._simplify = cached_simplify
        self._x0_fixedParameters_idx: None | Sequence[int]
        self._w_recursion_depth: int = 0
        self._has_quadratic_nllh: bool = True
        set_log_level(logger, verbose)

        self._static_indices: dict[str, list[int]] = {}

    def differential_states(self) -> list[DifferentialState]:
        """Get all differential states."""
        return self._differential_states

    def algebraic_states(self) -> list[AlgebraicState]:
        """Get all algebraic states."""
        return self._algebraic_states

    def observables(self) -> list[Observable]:
        """Get all observables."""
        return self._observables

    def parameters(self) -> list[Parameter]:
        """Get all parameters."""
        return self._parameters

    def constants(self) -> list[Constant]:
        """Get all constants."""
        return self._constants

    def expressions(self) -> list[Expression]:
        """Get all expressions."""
        return self._expressions

    def events(self) -> list[Event]:
        """Get all events."""
        return self._events

    def event_observables(self) -> list[EventObservable]:
        """Get all event observables."""
        return self._event_observables

    def sigma_ys(self) -> list[SigmaY]:
        """Get all observable sigmas."""
        return self._sigma_ys

    def sigma_zs(self) -> list[SigmaZ]:
        """Get all event observable sigmas."""
        return self._sigma_zs

    def conservation_laws(self) -> list[ConservationLaw]:
        """Get all conservation laws."""
        return self._conservation_laws

    def log_likelihood_ys(self) -> list[LogLikelihoodY]:
        """Get all observable log likelihoodss."""
        return self._log_likelihood_ys

    def log_likelihood_zs(self) -> list[LogLikelihoodZ]:
        """Get all event observable log likelihoods."""
        return self._log_likelihood_zs

    def log_likelihood_rzs(self) -> list[LogLikelihoodRZ]:
        """Get all event observable regularization log likelihoods."""
        return self._log_likelihood_rzs

    def is_ode(self) -> bool:
        """Check if model is ODE model."""
        return len(self._algebraic_equations) == 0

    def states(self) -> list[State]:
        """Get all states."""
        return self._differential_states + self._algebraic_states

    def _process_sbml_rate_of(self) -> None:
        """Substitute any SBML-rateOf constructs in the model equations"""
        rate_of_func = sp.core.function.UndefinedFunction("rateOf")
        species_sym_to_xdot = dict(
            zip(self.sym("x"), self.sym("xdot"), strict=True)
        )
        species_sym_to_idx = {x: i for i, x in enumerate(self.sym("x"))}

        def get_rate(symbol: sp.Symbol):
            """Get rate of change of the given symbol"""
            if symbol.find(rate_of_func):
                raise SBMLException("Nesting rateOf() is not allowed.")

            # Replace all rateOf(some_species) by their respective xdot equation
            with contextlib.suppress(KeyError):
                return self._eqs["xdot"][species_sym_to_idx[symbol]]

            # For anything other than a state, rateOf(.) is 0 or invalid
            return 0

        # replace rateOf-instances in xdot by xdot symbols
        made_substitutions = False
        for i_state in range(len(self.eq("xdot"))):
            if rate_ofs := self._eqs["xdot"][i_state].find(rate_of_func):
                self._eqs["xdot"][i_state] = self._eqs["xdot"][i_state].subs(
                    {
                        # either the rateOf argument is a state, or it's 0
                        rate_of: species_sym_to_xdot.get(rate_of.args[0], 0)
                        for rate_of in rate_ofs
                    }
                )
                made_substitutions = True

        if made_substitutions:
            # substitute in topological order
            subs = toposort_symbols(
                dict(zip(self.sym("xdot"), self.eq("xdot"), strict=True))
            )
            self._eqs["xdot"] = smart_subs_dict(self.eq("xdot"), subs)

        # replace rateOf-instances in x0 by xdot equation
        for i_state in range(len(self.eq("x0"))):
            if rate_ofs := self._eqs["x0"][i_state].find(rate_of_func):
                self._eqs["x0"][i_state] = self._eqs["x0"][i_state].subs(
                    {
                        rate_of: get_rate(rate_of.args[0])
                        for rate_of in rate_ofs
                    }
                )

        # replace rateOf-instances in w by xdot equation
        #  here we may need toposort, as xdot may depend on w
        made_substitutions = False
        for i_expr in range(len(self.eq("w"))):
            if rate_ofs := self._eqs["w"][i_expr].find(rate_of_func):
                self._eqs["w"][i_expr] = self._eqs["w"][i_expr].subs(
                    {
                        rate_of: get_rate(rate_of.args[0])
                        for rate_of in rate_ofs
                    }
                )
                made_substitutions = True

        if made_substitutions:
            # Sort expressions in self._expressions, w symbols, and w equations
            #  in topological order. Ideally, this would already happen before
            #  adding the expressions to the model, but at that point, we don't
            #  have access to xdot yet.
            # NOTE: elsewhere, conservations law expressions are expected to
            #  occur before any other w expressions, so we must maintain their
            #  position
            # toposort everything but conservation law expressions,
            #  then prepend conservation laws
            w_sorted = toposort_symbols(
                dict(
                    zip(
                        self.sym("w")[self.num_cons_law() :, :],
                        self.eq("w")[self.num_cons_law() :, :],
                        strict=True,
                    )
                )
            )
            w_sorted = (
                dict(
                    zip(
                        self.sym("w")[: self.num_cons_law(), :],
                        self.eq("w")[: self.num_cons_law(), :],
                        strict=True,
                    )
                )
                | w_sorted
            )
            old_syms = tuple(self._syms["w"])
            topo_expr_syms = tuple(w_sorted.keys())
            new_order = [old_syms.index(s) for s in topo_expr_syms]
            self._expressions = [self._expressions[i] for i in new_order]
            self._syms["w"] = sp.Matrix(topo_expr_syms)
            self._eqs["w"] = sp.Matrix(list(w_sorted.values()))

        for component in chain(
            self.observables(),
            self.events(),
            self._algebraic_equations,
        ):
            if rate_ofs := component.get_val().find(rate_of_func):
                if isinstance(component, Event):
                    # TODO froot(...) can currently not depend on `w`, so this substitution fails for non-zero rates
                    #  see, e.g., sbml test case 01293
                    raise SBMLException(
                        "AMICI does currently not support rateOf(.) inside event trigger functions."
                    )

                if isinstance(component, AlgebraicEquation):
                    # TODO IDACalcIC fails with
                    #   "The linesearch algorithm failed: step too small or too many backtracks."
                    #  see, e.g., sbml test case 01482
                    raise SBMLException(
                        "AMICI does currently not support rateOf(.) inside AlgebraicRules."
                    )

                component.set_val(
                    component.get_val().subs(
                        {
                            rate_of: get_rate(rate_of.args[0])
                            for rate_of in rate_ofs
                        }
                    )
                )

        for event in self.events():
            if event._state_update is None:
                continue

            for i_state in range(len(event._state_update)):
                if rate_ofs := event._state_update[i_state].find(rate_of_func):
                    raise SBMLException(
                        "AMICI does currently not support rateOf(.) inside event state updates."
                    )
                    # TODO here we need xdot sym, not eqs
                    # event._state_update[i_state] = event._state_update[i_state].subs(
                    #     {rate_of: get_rate(rate_of.args[0]) for rate_of in rate_ofs}
                    # )

    def add_component(
        self, component: ModelQuantity, insert_first: bool | None = False
    ) -> None:
        """
        Adds a new ModelQuantity to the model.

        :param component:
            model quantity to be added

        :param insert_first:
            whether to add quantity first or last, relevant when components
            may refer to other components of the same type.
        """
        if type(component) not in {
            Observable,
            Expression,
            Parameter,
            Constant,
            DifferentialState,
            AlgebraicState,
            AlgebraicEquation,
            LogLikelihoodY,
            LogLikelihoodZ,
            LogLikelihoodRZ,
            SigmaY,
            SigmaZ,
            ConservationLaw,
            Event,
            EventObservable,
        }:
            raise ValueError(f"Invalid component type {type(component)}")

        component_list = getattr(
            self,
            "_"
            + "_".join(
                s.lower()
                for s in re.split(r"([A-Z][^A-Z]+)", type(component).__name__)
                if s
            )
            + "s",
        )
        if insert_first:
            component_list.insert(0, component)
        else:
            component_list.append(component)

    def add_conservation_law(
        self,
        state: sp.Symbol,
        total_abundance: sp.Symbol,
        coefficients: dict[sp.Symbol, sp.Expr],
    ) -> None:
        r"""
        Adds a new conservation law to the model. A conservation law is defined
        by the conserved quantity :math:`T = \sum_i(a_i * x_i)`, where
        :math:`a_i` are coefficients and :math:`x_i` are different state
        variables.

        :param state:
            symbolic identifier of the state that should be replaced by
            the conservation law (:math:`x_j`)

        :param total_abundance:
            symbolic identifier of the total abundance (:math:`T/a_j`)

        :param coefficients:
            Dictionary of coefficients {x_i: a_i}
        """
        try:
            ix = next(
                filter(
                    lambda is_s: is_s[1].get_id() == state,
                    enumerate(self._differential_states),
                )
            )[0]
        except StopIteration:
            raise ValueError(
                f"Specified state {state} was not found in the "
                f"model states."
            )

        state_id = self._differential_states[ix].get_id()

        # \sum_{i≠j}(a_i * x_i)/a_j
        target_expression = (
            sp.Add(
                *(
                    c_i * x_i
                    for x_i, c_i in coefficients.items()
                    if x_i != state
                )
            )
            / coefficients[state]
        )

        # x_j = T/a_j - \sum_{i≠j}(a_i * x_i)/a_j
        state_expr = total_abundance - target_expression

        # T/a_j = \sum_{i≠j}(a_i * x_i)/a_j + x_j
        abundance_expr = target_expression + state_id

        self.add_component(
            Expression(state_id, str(state_id), state_expr), insert_first=True
        )

        cl = ConservationLaw(
            total_abundance,
            f"total_{state_id}",
            abundance_expr,
            coefficients,
            state_id,
        )

        self.add_component(cl)
        self._differential_states[ix].set_conservation_law(cl)

    def add_spline(self, spline: AbstractSpline, spline_expr: sp.Expr) -> None:
        """Add a spline to the model.

        :param spline:
            Spline instance to be added
        :param spline_expr:
            Sympy function representation of `spline` from
            ``spline.ode_model_symbol()``.
        """
        self._splines.append(spline)
        self.add_component(
            Expression(
                identifier=spline.sbml_id,
                name=str(spline.sbml_id),
                value=spline_expr,
            )
        )

    def get_observable_transformations(self) -> list[ObservableTransformation]:
        """
        List of observable transformations

        :return:
            list of transformations
        """
        return [obs.trafo for obs in self._observables]

    def num_states_rdata(self) -> int:
        """
        Number of states.

        :return:
            number of state variable symbols
        """
        return len(self.sym("x_rdata"))

    def num_states_solver(self) -> int:
        """
        Number of states after applying conservation laws.

        :return:
            number of state variable symbols
        """
        return len(self.sym("x"))

    def num_cons_law(self) -> int:
        """
        Number of conservation laws.

        :return:
            number of conservation laws
        """
        return self.num_states_rdata() - self.num_states_solver()

    def num_state_reinits(self) -> int:
        """
        Number of solver states which would be reinitialized after
        preequilibration

        :return:
            number of state variable symbols with reinitialization
        """
        reinit_states = self.eq("x0_fixedParameters")
        solver_states = self.eq("x_solver")
        return sum(ix in solver_states for ix in reinit_states)

    def num_obs(self) -> int:
        """
        Number of Observables.

        :return:
            number of observable symbols
        """
        return len(self.sym("y"))

    def num_eventobs(self) -> int:
        """
        Number of Event Observables.

        :return:
            number of event observable symbols
        """
        return len(self.sym("z"))

    def num_const(self) -> int:
        """
        Number of Constants.

        :return:
            number of constant symbols
        """
        return len(self.sym("k"))

    def num_par(self) -> int:
        """
        Number of Parameters.

        :return:
            number of parameter symbols
        """
        return len(self.sym("p"))

    def num_expr(self) -> int:
        """
        Number of Expressions.

        :return:
            number of expression symbols
        """
        return len(self.sym("w"))

    def num_events(self) -> int:
        """
        Total number of Events (those for which root-functions are added and those without).

        :return:
            number of events
        """
        return len(self.sym("h"))

    def num_events_solver(self) -> int:
        """
        Number of Events.

        :return:
            number of event symbols (length of the root vector in AMICI)
        """
        return sum(
            not event.triggers_at_fixed_timepoint() for event in self.events()
        )

    def sym(self, name: str) -> sp.Matrix:
        """
        Returns (and constructs if necessary) the identifiers for a symbolic
        entity.

        :param name:
            name of the symbolic variable

        :return:
            matrix of symbolic identifiers
        """
        if name not in self._syms:
            self._generate_symbol(name)

        return self._syms[name]

    def sparsesym(self, name: str, force_generate: bool = True) -> list[str]:
        """
        Returns (and constructs if necessary) the sparsified identifiers for
        a sparsified symbolic variable.

        :param name:
            name of the symbolic variable

        :param force_generate:
            whether the symbols should be generated if not available

        :return:
            linearized Matrix containing the symbolic identifiers
        """
        if name not in sparse_functions:
            raise ValueError(f"{name} is not marked as sparse")
        if name not in self._sparsesyms and force_generate:
            self._generate_sparse_symbol(name)
        return self._sparsesyms.get(name, [])

    def eq(self, name: str) -> sp.Matrix:
        """
        Returns (and constructs if necessary) the formulas for a symbolic
        entity.

        :param name:
            name of the symbolic variable

        :return:
            matrix of symbolic formulas
        """

        if name not in self._eqs:
            dec = log_execution_time(f"computing {name}", logger)
            dec(self._compute_equation)(name)
        return self._eqs[name]

    def sparseeq(self, name) -> sp.Matrix:
        """
        Returns (and constructs if necessary) the sparsified formulas for a
        sparsified symbolic variable.

        :param name:
            name of the symbolic variable

        :return:
            linearized matrix containing the symbolic formulas
        """
        if name not in sparse_functions:
            raise ValueError(f"{name} is not marked as sparse")
        if name not in self._sparseeqs:
            self._generate_sparse_symbol(name)
        return self._sparseeqs[name]

    def colptrs(self, name: str) -> list[sp.Number] | list[list[sp.Number]]:
        """
        Returns (and constructs if necessary) the column pointers for
        a sparsified symbolic variable.

        :param name:
            name of the symbolic variable

        :return:
            list containing the column pointers
        """
        if name not in sparse_functions:
            raise ValueError(f"{name} is not marked as sparse")
        if name not in self._sparseeqs:
            self._generate_sparse_symbol(name)
        return self._colptrs[name]

    def rowvals(self, name: str) -> list[sp.Number] | list[list[sp.Number]]:
        """
        Returns (and constructs if necessary) the row values for a
        sparsified symbolic variable.

        :param name:
            name of the symbolic variable

        :return:
            list containing the row values
        """
        if name not in sparse_functions:
            raise ValueError(f"{name} is not marked as sparse")
        if name not in self._sparseeqs:
            self._generate_sparse_symbol(name)
        return self._rowvals[name]

    def val(self, name: str) -> list[sp.Number]:
        """
        Returns (and constructs if necessary) the numeric values of a
        symbolic entity

        :param name:
            name of the symbolic variable

        :return:
            list containing the numeric values
        """
        if name not in self._vals:
            self._generate_value(name)
        return self._vals[name]

    def name(self, name: str) -> list[str]:
        """
        Returns (and constructs if necessary) the names of a symbolic
        variable

        :param name:
            name of the symbolic variable

        :return:
            list of names
        """
        if name not in self._names:
            self._generate_name(name)
        return self._names[name]

    def free_symbols(self) -> set[sp.Basic]:
        """
        Returns list of free symbols that appear in RHS and initial
        conditions.
        """
        return set(
            chain.from_iterable(
                state.get_free_symbols() for state in self.states()
            )
        )

    def static_indices(self, name: str) -> list[int]:
        """
        Returns the indices of static expressions in the given model entity.

        Static expressions are those that do not depend on time,
        neither directly nor indirectly.

        :param name: Name of the model entity.
        :return: List of indices of static expressions.
        """
        # already computed?
        if (res := self._static_indices.get(name)) is not None:
            return res

        if name == "w":
            dwdx = self.sym("dwdx")
            dwdw = self.sym("dwdw")
            w = self.eq("w")

            # Check for direct (via `t`) or indirect (via `x`, `h`, or splines)
            # time dependency.
            # To avoid lengthy symbolic computations, we only check if we have
            # any non-zeros in hierarchy. We currently neglect the case where
            # different hierarchy levels may cancel out. Treating a static
            # expression as dynamic in such rare cases shouldn't be a problem.
            dynamic_dependency = np.asarray(
                dwdx.applyfunc(lambda x: int(not x.is_zero))
            ).astype(np.int64)
            # to check for other time-dependence, we add a column to the dwdx
            #  matrix
            dynamic_syms = [
                # FIXME: see spline comment below
                # *self.sym("spl"),
                *self.sym("h"),
                amici_time_symbol,
            ]
            dynamic_dependency = np.hstack(
                (
                    dynamic_dependency,
                    np.array(
                        [
                            expr.has(*dynamic_syms)
                            # FIXME: the current spline implementation is a giant pita
                            #  currently, the splines occur in the form of sympy functions, e.g.:
                            #   AmiciSpline(y0, time, y0_3, y0_1)
                            #   AmiciSplineSensitivity(y0, time, y0_1, y0_3, y0_1)
                            #  until it uses the proper self.sym("spl") / self.sym("sspl")
                            #  symbols, which will require quite some refactoring,
                            #  we just do dumb string checks :|
                            or (
                                bool(self._splines)
                                and "AmiciSpline" in str(expr)
                            )
                            for expr in w
                        ]
                    )[:, np.newaxis],
                )
            )

            nonzero_dwdw = np.asarray(
                dwdw.applyfunc(lambda x: int(not x.is_zero))
            ).astype(np.int64)

            # `w` is made up an expression hierarchy. Any given entry is only
            # static if all its dependencies are static. Here, we unravel
            # the hierarchical structure of `w`.
            # If for an entry in `w`, the row sum of the intermediate products
            # is 0 across all levels, the expression is static.
            tmp = dynamic_dependency
            res = np.sum(tmp, axis=1)
            while np.any(tmp != 0):
                tmp = nonzero_dwdw.dot(tmp)
                res += np.sum(tmp, axis=1)
            self._static_indices[name] = (
                np.argwhere(res == 0).flatten().tolist()
            )

            return self._static_indices[name]

        if name in ("dwdw", "dwdx", "dwdp"):
            static_indices_w = set(self.static_indices("w"))
            dynamic_syms = [
                *(
                    sym
                    for i, sym in enumerate(self.sym("w"))
                    if i not in static_indices_w
                ),
                amici_time_symbol,
                *self.sym("x"),
                *self.sym("h"),
                # FIXME see spline comment above
                # *(self.sym("spl") if name in ("dwdw", "dwdx") else ()),
                # *(self.sym("sspl") if name == "dwdp" else ()),
            ]
            dynamic_syms = sp.Matrix(dynamic_syms)
            rowvals = self.rowvals(name)
            sparseeq = self.sparseeq(name)

            # collect the indices of static expressions of dwd* from the list
            #  of non-zeros entries of the sparse matrix
            self._static_indices[name] = [
                i
                for i, (expr, row_idx) in enumerate(
                    zip(sparseeq, rowvals, strict=True)
                )
                # derivative of a static expression is static
                if row_idx in static_indices_w
                # constant expressions
                or expr.is_Number
                # check for dependencies on non-static entities
                or (
                    # FIXME see spline comment above
                    #  (check str before diff, as diff will fail on spline functions)
                    (
                        # splines: non-static
                        not self._splines or "AmiciSpline" not in str(expr)
                    )
                    and (
                        # If the expression contains dynamic symbols, it might
                        # still be static. However, checking the derivative
                        # is currently too expensive, and we rather accept
                        # false negatives.
                        not expr.has(*dynamic_syms)
                        # or all(
                        #     expr.diff(dyn_sym).is_zero
                        #     for dyn_sym in dynamic_syms
                        # )
                    )
                )
            ]
            return self._static_indices[name]

        raise NotImplementedError(name)

    def dynamic_indices(self, name: str) -> list[int]:
        """
        Return the indices of dynamic expressions in the given model entity.

        :param name: Name of the model entity.
        :return: List of indices of dynamic expressions.
        """
        static_idxs = set(self.static_indices(name))
        length = len(
            self.sparsesym(name)
            if name in sparse_functions
            else self.sym(name)
        )
        return [i for i in range(length) if i not in static_idxs]

    def _generate_symbol(self, name: str) -> None:
        """
        Generates the symbolic identifiers for a symbolic variable

        :param name:
            name of the symbolic variable
        """
        if name in self._variable_prototype:
            components = self._variable_prototype[name]()
            self._syms[name] = sp.Matrix(
                [comp.get_id() for comp in components]
            )
            if name == "y":
                self._syms["my"] = sp.Matrix(
                    [comp.get_measurement_symbol() for comp in components]
                )
            if name == "z":
                self._syms["mz"] = sp.Matrix(
                    [comp.get_measurement_symbol() for comp in components]
                )
                self._syms["rz"] = sp.Matrix(
                    [comp.get_regularization_symbol() for comp in components]
                )
            return
        elif name == "x":
            self._syms[name] = sp.Matrix(
                [
                    state.get_id()
                    for state in self.states()
                    if not state.has_conservation_law()
                ]
            )
            return
        elif name == "xdot":
            self._syms[name] = sp.Matrix(
                [
                    f"d{x.get_id()}dt" if self.is_ode() else f"de_{ix}"
                    for ix, x in enumerate(self._differential_states)
                    if not x.has_conservation_law()
                ]
                + [f"ae_{ix}" for ix in range(len(self._algebraic_equations))]
            )
            return
        elif name == "dx":
            self._syms[name] = sp.Matrix(
                [
                    f"d{state.get_id()}dt"
                    for state in self.states()
                    if not state.has_conservation_law()
                ]
            )
            return
        elif name == "sx0":
            self._syms[name] = sp.Matrix(
                [
                    f"s{state.get_id()}_0"
                    for state in self.states()
                    if not state.has_conservation_law()
                ]
            )
            return
        elif name == "sx_rdata":
            self._syms[name] = sp.Matrix(
                [f"sx_rdata_{i}" for i in range(len(self.states()))]
            )
            return
        elif name == "dtcldp":
            # check, whether the CL consists of only one state. Then,
            # sensitivities drop out, otherwise generate symbols
            self._syms[name] = sp.Matrix(
                [
                    [
                        sp.Symbol(
                            f"s{strip_pysb(tcl.get_id())}__"
                            f"{strip_pysb(par.get_id())}",
                            real=True,
                        )
                        for par in self._parameters
                    ]
                    if self.conservation_law_has_multispecies(tcl)
                    else [0] * self.num_par()
                    for tcl in self._conservation_laws
                ]
            )
            return
        elif name == "xdot_old":
            length = len(self.eq("xdot"))
        elif name in sparse_functions:
            self._generate_sparse_symbol(name)
            return
        elif name in self._symboldim_funs:
            length = self._symboldim_funs[name]()
        elif name == "stau":
            length = self.eq(name)[0].shape[1]
        elif name in sensi_functions:
            length = self.eq(name).shape[0]
        elif name == "spl":
            # placeholders for the numeric spline values.
            # Need to create symbols
            self._syms[name] = sp.Matrix(
                [[f"spl_{isp}" for isp in range(len(self._splines))]]
            )
            return
        elif name == "sspl":
            # placeholders for spline sensitivities. Need to create symbols
            self._syms[name] = sp.Matrix(
                [
                    [f"sspl_{isp}_{ip}" for ip in range(len(self._syms["p"]))]
                    for isp in range(len(self._splines))
                ]
            )
            return
        else:
            length = len(self.eq(name))
        self._syms[name] = sp.Matrix(
            [
                sp.Symbol(f'{name}{0 if name == "stau" else i}', real=True)
                for i in range(length)
            ]
        )

    def generate_basic_variables(self) -> None:
        """
        Generates the symbolic identifiers for all variables in
        ``DEModel._variable_prototype``
        """
        # We need to process events and Heaviside functions in the ``DEModel`,
        # before adding it to DEExporter
        self.parse_events()

        for var in self._variable_prototype:
            if var not in self._syms:
                self._generate_symbol(var)
        # symbols for spline values need to be created in addition
        for var in ["spl", "sspl"]:
            self._generate_symbol(var)

        self._generate_symbol("x")

    def parse_events(self) -> None:
        """
        This function checks the right-hand side for roots of Heaviside
        functions or events, collects the roots, removes redundant roots,
        and replaces the formulae of the found roots by identifiers of AMICI's
        Heaviside function implementation in the right-hand side
        """
        # Track all roots functions in the right-hand side
        roots = copy.deepcopy(self._events)
        for state in self._differential_states:
            state.set_dt(self._process_heavisides(state.get_dt(), roots))

        for expr in self._expressions:
            expr.set_val(self._process_heavisides(expr.get_val(), roots))

        # remove all possible Heavisides from roots, which may arise from
        # the substitution of `'w'` in `_collect_heaviside_roots`
        for root in roots:
            root.set_val(self._process_heavisides(root.get_val(), roots))

        # Now add the found roots to the model components
        for root in roots:
            # skip roots of SBML events, as these have already been added
            if root in self._events:
                continue
            # add roots of heaviside functions
            self.add_component(root)

        # re-order events - first those that require root tracking, then the others
        self._events = list(
            chain(
                itertools.filterfalse(
                    Event.triggers_at_fixed_timepoint, self._events
                ),
                filter(Event.triggers_at_fixed_timepoint, self._events),
            )
        )

    def get_appearance_counts(self, idxs: list[int]) -> list[int]:
        """
        Counts how often a state appears in the time derivative of
        another state and expressions for a subset of states

        :param idxs:
            list of state indices for which counts are to be computed

        :return:
            list of counts for the states ordered according to the provided
            indices
        """
        free_symbols_dt = list(
            itertools.chain.from_iterable(
                [str(symbol) for symbol in state.get_dt().free_symbols]
                for state in self.states()
            )
        )

        free_symbols_expr = list(
            itertools.chain.from_iterable(
                [str(symbol) for symbol in expr.get_val().free_symbols]
                for expr in self._expressions
            )
        )

        return [
            free_symbols_dt.count(str(self._differential_states[idx].get_id()))
            + free_symbols_expr.count(
                str(self._differential_states[idx].get_id())
            )
            for idx in idxs
        ]

    def _generate_sparse_symbol(self, name: str) -> None:
        """
        Generates the sparse symbolic identifiers, symbolic identifiers,
        sparse equations, column pointers and row values for a symbolic
        variable

        :param name:
            name of the symbolic variable
        """
        matrix = self.eq(name)

        if match_deriv := DERIVATIVE_PATTERN.match(name):
            eq = match_deriv[1]
            var = match_deriv[2]

            rownames = self.sym(eq)
            colnames = self.sym(var)

        if name == "dJydy":
            # One entry per y-slice
            self._colptrs[name] = []
            self._rowvals[name] = []
            self._sparseeqs[name] = []
            self._sparsesyms[name] = []
            self._syms[name] = []

            for iy in range(self.num_obs()):
                (
                    symbol_col_ptrs,
                    symbol_row_vals,
                    sparse_list,
                    symbol_list,
                    sparse_matrix,
                ) = csc_matrix(
                    matrix[iy, :],
                    rownames=rownames,
                    colnames=colnames,
                    identifier=iy,
                )
                self._colptrs[name].append(symbol_col_ptrs)
                self._rowvals[name].append(symbol_row_vals)
                self._sparseeqs[name].append(sparse_list)
                self._sparsesyms[name].append(symbol_list)
                self._syms[name].append(sparse_matrix)
        else:
            (
                symbol_col_ptrs,
                symbol_row_vals,
                sparse_list,
                symbol_list,
                sparse_matrix,
            ) = csc_matrix(
                matrix,
                rownames=rownames,
                colnames=colnames,
                pattern_only=name in nobody_functions,
            )

            self._colptrs[name] = symbol_col_ptrs
            self._rowvals[name] = symbol_row_vals
            self._sparseeqs[name] = sparse_list
            self._sparsesyms[name] = symbol_list
            self._syms[name] = sparse_matrix

    def _compute_equation(self, name: str) -> None:
        """
        Computes the symbolic formula for a symbolic variable

        :param name:
            name of the symbolic variable
        """
        # replacement ensures that we don't have to adapt name in abstract
        # model and keep backwards compatibility with matlab
        match_deriv = DERIVATIVE_PATTERN.match(
            re.sub(r"dJ(y|z|rz)dsigma", r"dJ\1dsigma\1", name)
            .replace("sigmarz", "sigmaz")
            .replace("dJrzdz", "dJrzdrz")
        )
        time_symbol = sp.Matrix([amici_time_symbol])

        if name in self._equation_prototype:
            self._equation_from_components(
                name, self._equation_prototype[name]()
            )

        elif name in self._total_derivative_prototypes:
            args = self._total_derivative_prototypes[name]
            args["name"] = name
            self._lock_total_derivative += args["chainvars"]
            self._total_derivative(**args)
            for cv in args["chainvars"]:
                self._lock_total_derivative.remove(cv)

        elif name == "xdot":
            if self.is_ode():
                self._eqs[name] = sp.Matrix(
                    [
                        state.get_dt()
                        for state in self._differential_states
                        if not state.has_conservation_law()
                    ]
                )
            else:
                self._eqs[name] = sp.Matrix(
                    [
                        x.get_dt() - dx
                        for x, dx in zip(
                            (
                                s
                                for s in self._differential_states
                                if not s.has_conservation_law()
                            ),
                            self.sym("dx"),
                            # dx contains extra elements for algebraic states
                            strict=False,
                        )
                    ]
                    + [eq.get_val() for eq in self._algebraic_equations]
                )

        elif name == "x_rdata":
            self._eqs[name] = sp.Matrix(
                [state.get_x_rdata() for state in self.states()]
            )

        elif name == "x_solver":
            self._eqs[name] = sp.Matrix(
                [
                    state.get_id()
                    for state in self.states()
                    if not state.has_conservation_law()
                ]
            )

        elif name == "sx_solver":
            self._eqs[name] = sp.Matrix(
                [
                    self.sym("sx_rdata")[ix]
                    for ix, state in enumerate(self.states())
                    if not state.has_conservation_law()
                ]
            )

        elif name == "sx0":
            self._derivative(name[1:], "p", name=name)

        elif name == "sx0_fixedParameters":
            # deltax = -x+x0_fixedParameters if x0_fixedParameters>0 else 0
            # deltasx = -sx+dx0_fixed_parametersdx*sx+dx0_fixedParametersdp
            # if x0_fixedParameters>0 else 0
            # sx0_fixedParameters = sx+deltasx =
            # dx0_fixed_parametersdx*sx+dx0_fixedParametersdp
            self._eqs[name] = smart_jacobian(
                self.eq("x0_fixedParameters"), self.sym("p")
            )

            dx0_fixed_parametersdx = smart_jacobian(
                self.eq("x0_fixedParameters"), self.sym("x")
            )

            if not smart_is_zero_matrix(dx0_fixed_parametersdx):
                if isinstance(self._eqs[name], ImmutableDenseMatrix):
                    self._eqs[name] = MutableDenseMatrix(self._eqs[name])
                tmp = smart_multiply(dx0_fixed_parametersdx, self.sym("sx0"))
                for ip in range(self._eqs[name].shape[1]):
                    self._eqs[name][:, ip] += tmp

        elif name == "x0_fixedParameters":
            k = self.sym("k")
            self._x0_fixedParameters_idx = [
                ix
                for ix, eq in enumerate(self.eq("x0"))
                if any(sym in eq.free_symbols for sym in k)
            ]
            eq = self.eq("x0")
            self._eqs[name] = sp.Matrix(
                [eq[ix] for ix in self._x0_fixedParameters_idx]
            )

        elif name == "dtotal_cldx_rdata":
            x_rdata = self.sym("x_rdata")
            self._eqs[name] = sp.Matrix(
                [
                    [cl.get_ncoeff(xr) for xr in x_rdata]
                    for cl in self._conservation_laws
                ]
            )

        elif name == "dtcldx":
            # this is always zero
            self._eqs[name] = sp.zeros(
                self.num_cons_law(), self.num_states_solver()
            )

        elif name == "dtcldp":
            # force symbols
            self._eqs[name] = self.sym(name)

        elif name == "dx_rdatadx_solver":
            if self.num_cons_law():
                x_solver = self.sym("x")
                self._eqs[name] = sp.Matrix(
                    [
                        [state.get_dx_rdata_dx_solver(xs) for xs in x_solver]
                        for state in self.states()
                    ]
                )
            else:
                # so far, dx_rdatadx_solver is only required for sx_rdata
                # in case of no conservation laws, C++ code will directly use
                # sx, we don't need this
                self._eqs[name] = sp.zeros(
                    self.num_states_rdata(), self.num_states_solver()
                )

        elif name == "dx_rdatadp":
            if self.num_cons_law():
                self._eqs[name] = smart_jacobian(
                    self.eq("x_rdata"), self.sym("p")
                )
            else:
                # so far, dx_rdatadp is only required for sx_rdata
                # in case of no conservation laws, C++ code will directly use
                # sx, we don't need this
                self._eqs[name] = sp.zeros(
                    self.num_states_rdata(), self.num_par()
                )

        elif name == "dx_rdatadtcl":
            self._eqs[name] = smart_jacobian(
                self.eq("x_rdata"), self.sym("tcl")
            )

        elif name == "dxdotdx_explicit":
            # force symbols
            self._derivative("xdot", "x", name=name)

        elif name == "dxdotdp_explicit":
            # force symbols
            self._derivative("xdot", "p", name=name)

        elif name == "spl":
            self._eqs[name] = self.sym(name)

        elif name == "sspl":
            # force symbols
            self._eqs[name] = self.sym(name)

        elif name == "spline_values":
            # force symbols
            self._eqs[name] = sp.Matrix(
                [y for spline in self._splines for y in spline.values_at_nodes]
            )

        elif name == "spline_slopes":
            # force symbols
            self._eqs[name] = sp.Matrix(
                [
                    d
                    for spline in self._splines
                    for d in (
                        sp.zeros(len(spline.derivatives_at_nodes), 1)
                        if spline.derivatives_by_fd
                        else spline.derivatives_at_nodes
                    )
                ]
            )

        elif name == "drootdt":
            self._eqs[name] = smart_jacobian(self.eq("root"), time_symbol)

        elif name == "drootdt_total":
            self._eqs[name] = self.eq("drootdt")
            # backsubstitution of optimized right-hand side terms into RHS
            # calling subs() is costly. We can skip it if we don't have any
            # state-dependent roots.
            if self.num_states_solver() and not smart_is_zero_matrix(
                drootdx := self.eq("drootdx")
            ):
                w_sorted = toposort_symbols(
                    dict(zip(self.sym("w"), self.eq("w"), strict=True))
                )
                tmp_xdot = smart_subs_dict(self.eq("xdot"), w_sorted)
                self._eqs[name] += smart_multiply(drootdx, tmp_xdot)

        elif name == "deltax":
            # fill boluses for Heaviside functions, as empty state updates
            # would cause problems when writing the function file later
            event_eqs = []
            for event in self._events:
                if event._state_update is None:
                    event_eqs.append(sp.zeros(self.num_states_solver(), 1))
                else:
                    event_eqs.append(event._state_update)

            self._eqs[name] = event_eqs

        elif name == "z":
            event_observables = [
                sp.zeros(self.num_eventobs(), 1) for _ in self._events
            ]
            event_ids = [e.get_id() for e in self._events]
            # TODO: get rid of this stupid 1-based indexing as soon as we can
            # the matlab interface
            z2event = [
                event_ids.index(event_obs.get_event()) + 1
                for event_obs in self._event_observables
            ]
            for (iz, ie), event_obs in zip(
                enumerate(z2event), self._event_observables, strict=True
            ):
                event_observables[ie - 1][iz] = event_obs.get_val()

            self._eqs[name] = event_observables
            self._z2event = z2event

        elif name in ["ddeltaxdx", "ddeltaxdp", "ddeltaxdt", "dzdp", "dzdx"]:
            if match_deriv[2] == "t":
                var = time_symbol
            else:
                var = self.sym(match_deriv[2])

            self._eqs[name] = [
                smart_jacobian(self.eq(match_deriv[1])[ie], var)
                for ie in range(self.num_events())
            ]
            if name == "dzdx":
                for ie in range(self.num_events()):
                    dtaudx = self.eq("dtaudx")
                    for iz in range(self.num_eventobs()):
                        if ie != self._z2event[iz] - 1:
                            continue
                        dzdt = sp.diff(self.eq("z")[ie][iz], time_symbol)
                        self._eqs[name][ie][iz, :] += dzdt * -dtaudx[ie]

        elif name in ["rz", "drzdx", "drzdp"]:
            eq_events = []
            for ie in range(self.num_events()):
                val = sp.zeros(
                    self.num_eventobs(),
                    1 if name == "rz" else len(self.sym(match_deriv[2])),
                )
                # match event observables to root function
                for iz in range(self.num_eventobs()):
                    if ie == self._z2event[iz] - 1:
                        val[iz, :] = self.eq(name.replace("rz", "root"))[ie, :]
                eq_events.append(val)

            self._eqs[name] = eq_events

        elif name == "stau":
            self._eqs[name] = [
                self.eq("sroot")[ie, :] / self.eq("drootdt_total")[ie]
                if not self.eq("drootdt_total")[ie].is_zero
                else sp.zeros(*self.eq("sroot")[ie, :].shape)
                for ie in range(self.num_events())
            ]

        elif name == "dtaudx":
            self._eqs[name] = [
                self.eq("drootdx")[ie, :] / self.eq("drootdt_total")[ie]
                for ie in range(self.num_events())
            ]

        elif name == "dtaudp":
            self._eqs[name] = [
                self.eq("drootdp")[ie, :] / self.eq("drootdt_total")[ie]
                for ie in range(self.num_events())
            ]

        elif name == "deltasx":
            if self.num_states_solver() * self.num_par() == 0:
                self._eqs[name] = []
                return

            event_eqs = []
            for ie, event in enumerate(self._events):
                tmp_eq = sp.zeros(self.num_states_solver(), self.num_par())

                # need to check if equations are zero since we are using
                # symbols
                if not smart_is_zero_matrix(
                    self.eq("stau")[ie]
                ) and not smart_is_zero_matrix(self.eq("xdot")):
                    tmp_eq += smart_multiply(
                        self.sym("xdot") - self.sym("xdot_old"),
                        self.sym("stau").T,
                    )

                # only add deltax part if there is state update
                if event._state_update is not None:
                    # partial derivative for the parameters
                    tmp_eq += self.eq("ddeltaxdp")[ie]

                    # initial part of chain rule state variables
                    tmp_dxdp = self.sym("sx") * sp.ones(1, self.num_par())

                    # need to check if equations are zero since we are using
                    # symbols
                    if not smart_is_zero_matrix(self.eq("stau")[ie]):
                        # chain rule for the time point
                        tmp_eq += smart_multiply(
                            self.eq("ddeltaxdt")[ie],
                            -self.sym("stau").T,
                        )

                        # additional part of chain rule state variables
                        tmp_dxdp += smart_multiply(
                            self.sym("xdot_old"),
                            -self.sym("stau").T,
                        )

                    # finish chain rule for the state variables
                    tmp_eq += smart_multiply(
                        self.eq("ddeltaxdx")[ie], tmp_dxdp
                    )

                else:
                    tmp_eq = smart_multiply(
                        self.sym("xdot") - self.sym("xdot_old"),
                        self.eq("stau")[ie],
                    )
                event_eqs.append(tmp_eq)

            self._eqs[name] = event_eqs

        elif name == "xdot_old":
            # force symbols
            self._eqs[name] = self.sym(name)

        elif name == "dwdx":
            if (
                expected := list(
                    map(
                        ConservationLaw.get_x_rdata,
                        reversed(self.conservation_laws()),
                    )
                )
            ) != (actual := self.eq("w")[: self.num_cons_law()]):
                raise AssertionError(
                    "Conservation laws are not at the beginning of 'w'. "
                    f"Got {actual}, expected {expected}."
                )
            x = self.sym("x")
            self._eqs[name] = sp.Matrix(
                [
                    [-cl.get_ncoeff(xs) for xs in x]
                    # the insert first in ode_model._add_conservation_law() means
                    # that we need to reverse the order here
                    for cl in reversed(self._conservation_laws)
                ]
            ).col_join(
                smart_jacobian(self.eq("w")[self.num_cons_law() :, :], x)
            )

        elif match_deriv:
            self._derivative(match_deriv[1], match_deriv[2], name)

        else:
            raise ValueError(f"Unknown equation {name}")

        if name in ("sigmay", "sigmaz"):
            # check for states in sigma{y,z}, which is currently not supported
            syms_x = self.sym("x")
            syms_yz = self.sym(name.removeprefix("sigma"))
            xs_in_sigma = {}
            for sym_yz, eq_yz in zip(syms_yz, self._eqs[name], strict=True):
                yz_free_syms = eq_yz.free_symbols
                if tmp := {x for x in syms_x if x in yz_free_syms}:
                    xs_in_sigma[sym_yz] = tmp
            if xs_in_sigma:
                msg = ", ".join(
                    [f"{yz} depends on {xs}" for yz, xs in xs_in_sigma.items()]
                )
                raise NotImplementedError(
                    f"State-dependent observables are not supported, but {msg}."
                )

        if name == "root":
            # Events are processed after the model has been set up.
            # Equations are there, but symbols for roots must be added
            self.sym("h")

        if name in {"Jy", "dydx"}:
            # do not transpose if we compute the partial derivative as part of
            # a total derivative
            if not len(self._lock_total_derivative):
                self._eqs[name] = self._eqs[name].transpose()

        if name in {"dzdx", "drzdx"}:
            self._eqs[name] = [e.T for e in self._eqs[name]]

        if self._simplify:
            dec = log_execution_time(f"simplifying {name}", logger)
            if isinstance(self._eqs[name], list):
                self._eqs[name] = [
                    dec(_parallel_applyfunc)(sub_eq, self._simplify)
                    for sub_eq in self._eqs[name]
                ]
            else:
                self._eqs[name] = dec(_parallel_applyfunc)(
                    self._eqs[name], self._simplify
                )

    def sym_names(self) -> list[str]:
        """
        Returns a list of names of generated symbolic variables

        :return:
            list of names
        """
        return list(self._syms.keys())

    def _derivative(self, eq: str, var: str, name: str = None) -> None:
        """
        Creates a new symbolic variable according to a derivative

        :param eq:
            name of the symbolic variable that defines the formula

        :param var:
            name of the symbolic variable that defines the identifiers
            with respect to which the derivatives are to be computed

        :param name:
            name of resulting symbolic variable, default is ``d{eq}d{var}``
        """
        if not name:
            name = f"d{eq}d{var}"

        ignore_chainrule = {
            ("xdot", "p"): "w",  # has generic implementation in c++ code
            ("xdot", "x"): "w",  # has generic implementation in c++ code
            ("w", "w"): "tcl",  # dtcldw = 0
            ("w", "x"): "tcl",  # dtcldx = 0
        }
        # automatically detect chainrule
        chainvars = [
            cv
            for cv in ["w", "tcl"]
            if var_in_function_signature(eq, cv, self.is_ode())
            and cv not in self._lock_total_derivative
            and var != cv
            and min(self.sym(cv).shape)
            and (
                (eq, var) not in ignore_chainrule
                or ignore_chainrule[(eq, var)] != cv
            )
        ]
        if len(chainvars):
            self._lock_total_derivative += chainvars
            self._total_derivative(name, eq, chainvars, var)
            for cv in chainvars:
                self._lock_total_derivative.remove(cv)
            return

        # partial derivative
        sym_eq = self.eq(eq).transpose() if eq == "Jy" else self.eq(eq)

        sym_var = self.sym(var)

        derivative = smart_jacobian(sym_eq, sym_var)

        self._eqs[name] = derivative

        # compute recursion depth based on nilpotency of jacobian. computing
        # nilpotency can be done more efficiently on numerical sparsity pattern
        if name == "dwdw":
            nonzeros = np.asarray(
                derivative.applyfunc(lambda x: int(not x.is_zero))
            ).astype(np.int64)
            recursion = nonzeros.copy()
            if max(recursion.shape):
                while recursion.max():
                    recursion = recursion.dot(nonzeros)
                    self._w_recursion_depth += 1
                    if self._w_recursion_depth > len(sym_eq):
                        raise RuntimeError(
                            "dwdw is not nilpotent. Something, somewhere went "
                            "terribly wrong. Please file a bug report at "
                            "https://github.com/AMICI-dev/AMICI/issues and "
                            "attach this model."
                        )

        if name == "dydw" and not smart_is_zero_matrix(derivative):
            dwdw = self.eq("dwdw")
            # h(k) = d{eq}dw*dwdw^k* (k=1)
            h = smart_multiply(derivative, dwdw)
            while not smart_is_zero_matrix(h):
                self._eqs[name] += h
                # h(k+1) = d{eq}dw*dwdw^(k+1) = h(k)*dwdw
                h = smart_multiply(h, dwdw)

    def _total_derivative(
        self,
        name: str,
        eq: str,
        chainvars: list[str],
        var: str,
        dydx_name: str = None,
        dxdz_name: str = None,
    ) -> None:
        """
        Creates a new symbolic variable according to a total derivative
        using the chain rule

        :param name:
            name of resulting symbolic variable

        :param eq:
            name of the symbolic variable that defines the formula

        :param chainvars:
            names of the symbolic variable that define the
            identifiers with respect to which the chain rules are applied

        :param var:
            name of the symbolic variable that defines the identifiers
            with respect to which the derivatives are to be computed

        :param dydx_name:
            defines the name of the symbolic variable that
            defines the derivative of the ``eq`` with respect to ``chainvar``,
            default is ``d{eq}d{chainvar}``

        :param dxdz_name:
            defines the name of the symbolic variable that
            defines the derivative of the ``chainvar`` with respect to ``var``,
            default is d{chainvar}d{var}
        """
        # compute total derivative according to chainrule
        # Dydz = dydx*dxdz + dydz

        # initialize with partial derivative dydz without chain rule
        self._eqs[name] = self.sym_or_eq(name, f"d{eq}d{var}")
        if not isinstance(self._eqs[name], sp.Symbol):
            # if not a Symbol, create a copy using sympy API
            # NB deepcopy does not work safely, see sympy issue #7672
            self._eqs[name] = self._eqs[name].copy()

        for chainvar in chainvars:
            if dydx_name is None:
                dydx_name = f"d{eq}d{chainvar}"
            if dxdz_name is None:
                dxdz_name = f"d{chainvar}d{var}"

            dydx = self.sym_or_eq(name, dydx_name)
            dxdz = self.sym_or_eq(name, dxdz_name)
            # Save time for large models if one multiplicand is zero,
            # which is not checked for by sympy
            if not smart_is_zero_matrix(dydx) and not smart_is_zero_matrix(
                dxdz
            ):
                dydx_times_dxdz = smart_multiply(dydx, dxdz)
                if (
                    dxdz.shape[1] == 1
                    and self._eqs[name].shape[1] != dxdz.shape[1]
                ):
                    for iz in range(self._eqs[name].shape[1]):
                        self._eqs[name][:, iz] += dydx_times_dxdz
                else:
                    self._eqs[name] += dydx_times_dxdz

    def sym_or_eq(self, name: str, varname: str) -> sp.Matrix:
        """
        Returns symbols or equations depending on whether a given
        variable appears in the function signature or not.

        :param name:
            name of function for which the signature should be checked

        :param varname:
            name of the variable which should be contained in the
            function signature

        :return:
            the variable symbols if the variable is part of the signature and
            the variable equations otherwise.
        """
        # dwdx and dwdp will be dynamically computed and their ordering
        # within a column may differ from the initialization of symbols here,
        # so those are not safe to use. Not removing them from signature as
        # this would break backwards compatibility.
        if var_in_function_signature(
            name, varname, self.is_ode()
        ) and varname not in [
            "dwdx",
            "dwdp",
        ]:
            return self.sym(varname)
        else:
            return self.eq(varname)

    def _multiplication(
        self,
        name: str,
        x: str,
        y: str,
        transpose_x: bool | None = False,
        sign: int | None = 1,
    ):
        """
        Creates a new symbolic variable according to a multiplication

        :param name:
            name of resulting symbolic variable, default is ``d{eq}d{var}``

        :param x:
            name of the symbolic variable that defines the first factor

        :param y:
            name of the symbolic variable that defines the second factor

        :param transpose_x:
            indicates whether the first factor should be
            transposed before multiplication

        :param sign:
            defines the sign of the product, should be +1 or -1
        """
        if sign not in [-1, 1]:
            raise TypeError(f"sign must be +1 or -1, was {sign}")

        variables = {
            varname: self.sym(varname)
            if var_in_function_signature(name, varname, self.is_ode())
            else self.eq(varname)
            for varname in [x, y]
        }

        xx = variables[x].transpose() if transpose_x else variables[x]
        yy = variables[y]

        self._eqs[name] = sign * smart_multiply(xx, yy)

    def _equation_from_components(
        self, name: str, components: list[ModelQuantity]
    ) -> None:
        """
        Generates the formulas of a symbolic variable from the attributes

        :param name:
            name of resulting symbolic variable

        :param component:
            name of the attribute
        """
        self._eqs[name] = sp.Matrix([comp.get_val() for comp in components])

    def get_conservation_laws(self) -> list[tuple[sp.Symbol, sp.Expr]]:
        """Returns a list of states with conservation law set

        :return:
            list of state identifiers
        """
        return [
            (state.get_id(), state.get_x_rdata())
            for state in self.states()
            if state.has_conservation_law()
        ]

    def _generate_value(self, name: str) -> None:
        """
        Generates the numeric values of a symbolic variable from value
        prototypes

        :param name:
            name of resulting symbolic variable
        """
        if name in self._value_prototype:
            components = self._value_prototype[name]()
        else:
            raise ValueError(f"No values for {name}")

        self._vals[name] = [comp.get_val() for comp in components]

    def _generate_name(self, name: str) -> None:
        """
        Generates the names of a symbolic variable from variable prototypes or
        equation prototypes

        :param name:
            name of resulting symbolic variable
        """
        if name in self._variable_prototype:
            components = self._variable_prototype[name]()
        elif name in self._equation_prototype:
            components = self._equation_prototype[name]()
        else:
            raise ValueError(f"No names for {name}")

        self._names[name] = [comp.get_name() for comp in components]

    def state_has_fixed_parameter_initial_condition(self, ix: int) -> bool:
        """
        Checks whether the state at specified index has a fixed parameter
        initial condition

        :param ix:
            state index

        :return:
            boolean indicating if any of the initial condition free
            variables is contained in the model constants
        """
        ic = self.states()[ix].get_val()
        if not isinstance(ic, sp.Basic):
            return False
        return any(
            fp in (c.get_id() for c in self._constants)
            for fp in ic.free_symbols
        )

    def state_has_conservation_law(self, ix: int) -> bool:
        """
        Checks whether the state at specified index has a conservation
        law set

        :param ix:
            state index

        :return:
            boolean indicating if conservation_law is not None
        """
        return self.states()[ix].has_conservation_law()

    def get_solver_indices(self) -> dict[int, int]:
        """
        Returns a mapping that maps rdata species indices to solver indices

        :return:
            dictionary mapping rdata species indices to solver indices
        """
        solver_index = {}
        ix_solver = 0
        for ix in range(len(self.states())):
            if self.state_has_conservation_law(ix):
                continue
            solver_index[ix] = ix_solver
            ix_solver += 1
        return solver_index

    def state_is_constant(self, ix: int) -> bool:
        """
        Checks whether the temporal derivative of the state is zero

        :param ix:
            state index

        :return:
            boolean indicating if constant over time
        """
        state = self.states()[ix]
        if isinstance(state, AlgebraicState):
            return False

        return state.get_dt().is_zero

    def conservation_law_has_multispecies(self, tcl: ConservationLaw) -> bool:
        """
        Checks whether a conservation law has multiple species or it just
        defines one constant species

        :param tcl:
            conservation law

        :return:
            boolean indicating if conservation_law is not None
        """
        state_set = set(self.sym("x_rdata"))
        n_species = len(state_set.intersection(tcl.get_val().free_symbols))
        return n_species > 1

    def _expr_is_time_dependent(self, expr: sp.Expr) -> bool:
        """Determine whether an expression is time-dependent.

        :param expr:
            The expression.

        :returns:
            Whether the expression is time-dependent.
        """
        # `expr.free_symbols` will be different to `self._states.keys()`, so
        # it's easier to compare as `str`.
        expr_syms = {str(sym) for sym in expr.free_symbols}

        # Check if the time variable is in the expression.
        if "t" in expr_syms:
            return True

        # Check if any time-dependent states are in the expression.
        state_syms = [str(sym) for sym in self.states()]
        return any(
            not self.state_is_constant(state_syms.index(state))
            for state in expr_syms.intersection(state_syms)
        )

    def _get_unique_root(
        self,
        root_found: sp.Expr,
        roots: list[Event],
    ) -> sp.Symbol | None:
        """
        Collects roots of Heaviside functions and events and stores them in
        the roots list. It checks for redundancy to not store symbolically
        equivalent root functions more than once.

        :param root_found:
            equation of the root function
        :param roots:
            list of already known root functions with identifier

        :returns:
            unique identifier for root, or ``None`` if the root is not
            time-dependent
        """
        if not self._expr_is_time_dependent(root_found):
            return None

        for root in roots:
            if sp.simplify(root_found - root.get_val()).is_zero:
                return root.get_id()

        # create an event for a new root function
        root_symstr = f"Heaviside_{len(roots)}"
        roots.append(
            Event(
                identifier=sp.Symbol(root_symstr),
                name=root_symstr,
                value=root_found,
                state_update=None,
            )
        )
        return roots[-1].get_id()

    def _collect_heaviside_roots(
        self,
        args: Sequence[sp.Basic],
    ) -> list[sp.Expr]:
        """
        Recursively checks an expression for the occurrence of Heaviside
        functions and return all roots found

        :param args:
            args attribute of the expanded expression

        :returns:
            root functions that were extracted from Heaviside function
            arguments
        """
        root_funs = []
        for arg in args:
            if arg.func == sp.Heaviside:
                root_funs.append(arg.args[0])
            elif arg.has(sp.Heaviside):
                root_funs.extend(self._collect_heaviside_roots(arg.args))

        if not root_funs:
            return []

        # substitute 'w' expressions into root expressions now, to avoid
        # rewriting 'root.cpp' and 'stau.cpp' headers
        # to include 'w.h'
        w_sorted = toposort_symbols(
            dict(
                zip(
                    [expr.get_id() for expr in self._expressions],
                    [expr.get_val() for expr in self._expressions],
                    strict=True,
                )
            )
        )
        root_funs = [r.subs(w_sorted) for r in root_funs]

        return root_funs

    def _process_heavisides(
        self,
        dxdt: sp.Expr,
        roots: list[Event],
    ) -> sp.Expr:
        """
        Parses the RHS of a state variable, checks for Heaviside functions,
        collects unique roots functions that can be tracked by SUNDIALS and
        replaces Heaviside Functions by amici helper variables that will be
        updated based on SUNDIALS root tracking.

        :param dxdt:
            right-hand side of state variable
        :param roots:
            list of known root functions with identifier

        :returns:
            dxdt with Heaviside functions replaced by amici helper variables
        """
        # track all the old Heaviside expressions in tmp_roots_old
        # replace them later by the new expressions
        heavisides = []
        # run through the expression tree and get the roots
        tmp_roots_old = self._collect_heaviside_roots((dxdt,))
        for tmp_old in unique_preserve_order(tmp_roots_old):
            # we want unique identifiers for the roots
            tmp_new = self._get_unique_root(tmp_old, roots)
            # `tmp_new` is None if the root is not time-dependent.
            if tmp_new is None:
                continue
            # For Heavisides, we need to add the negative function as well
            self._get_unique_root(sp.sympify(-tmp_old), roots)
            heavisides.append((sp.Heaviside(tmp_old), tmp_new))

        if heavisides:
            # only apply subs if necessary
            for heaviside_sympy, heaviside_amici in heavisides:
                dxdt = dxdt.subs(heaviside_sympy, heaviside_amici)

        return dxdt

    @property
    def _components(self) -> list[ModelQuantity]:
        """
        Returns the components of the model

        :return:
            components of the model
        """
        return (
            self._algebraic_states
            + self._algebraic_equations
            + self._conservation_laws
            + self._constants
            + self._differential_states
            + self._event_observables
            + self._events
            + self._expressions
            + self._log_likelihood_ys
            + self._log_likelihood_zs
            + self._log_likelihood_rzs
            + self._observables
            + self._parameters
            + self._sigma_ys
            + self._sigma_zs
            + self._splines
        )

    def _process_hybridization(self, hybridization: dict) -> None:
        """
        Parses the hybridisation information and updates the model accordingly

        :param hybridization:
            hybridization information
        """
        added_expressions = False
        for net_id, net in hybridization.items():
            if not (
                net["hybridization"]["output"] == "ode"
                or net["hybridization"]["input"] == "ode"
            ):
                continue  # do not integrate into ODEs, handle in amici.jax.petab
            inputs = [
                comp
                for comp in self._components
                if str(comp.get_id()) in net["input_vars"]
            ]
            # sort inputs by order in input_vars
            inputs = sorted(
                inputs,
                key=lambda comp: net["input_vars"].index(str(comp.get_id())),
            )
            if len(inputs) != len(net["input_vars"]):
                raise ValueError(
                    f"Could not find all input variables for neural network {net_id}"
                )
            for inp in inputs:
                if isinstance(
                    inp,
                    Sigma
                    | LogLikelihood
                    | Event
                    | ConservationLaw
                    | Observable,
                ):
                    raise NotImplementedError(
                        f"{inp.get_name()} ({type(inp)}) is not supported as neural network input."
                    )

            outputs = {
                out_var: comp
                for comp in self._components
                if (out_var := str(comp.get_id())) in net["output_vars"]
                # TODO: SYNTAX NEEDS to CHANGE
                or (out_var := str(comp.get_id()) + "_dot")
                in net["output_vars"]
            }
            if len(outputs.keys()) != len(net["output_vars"]):
                raise ValueError(
                    f"Could not find all output variables for neural network {net_id}"
                )
            for iout, (out_var, comp) in enumerate(outputs.items()):
                # remove output from model components
                if isinstance(comp, Parameter):
                    self._parameters.remove(comp)
                elif isinstance(comp, Expression):
                    self._expressions.remove(comp)
                elif isinstance(comp, DifferentialState):
                    pass
                else:
                    raise NotImplementedError(
                        f"{comp.get_name()} ({type(comp)}) is not supported as neural network output."
                    )

                # generate dummy Function
                out_val = sp.Function(net_id)(
                    *[input.get_id() for input in inputs], iout
                )

                # add to the model
                if isinstance(comp, DifferentialState):
                    ix = self._differential_states.index(comp)
                    # TODO: SYNTAX NEEDS to CHANGE
                    if out_var.endswith("_dot"):
                        self._differential_states[ix].set_dt(out_val)
                    else:
                        self._differential_states[ix].set_val(out_val)
                else:
                    self.add_component(
                        Expression(
                            identifier=comp.get_id(),
                            name=net_id,
                            value=out_val,
                        )
                    )
                    added_expressions = True

        if added_expressions:
            # toposort expressions
            w_sorted = toposort_symbols(
                dict(
                    zip(
                        self.sym("w"),
                        self.eq("w"),
                        strict=True,
                    )
                )
            )
            old_syms = tuple(self._syms["w"])
            topo_expr_syms = tuple(w_sorted.keys())
            new_order = [old_syms.index(s) for s in topo_expr_syms]
            self._expressions = [self._expressions[i] for i in new_order]
            self._syms["w"] = sp.Matrix(topo_expr_syms)
            self._eqs["w"] = sp.Matrix(list(w_sorted.values()))
