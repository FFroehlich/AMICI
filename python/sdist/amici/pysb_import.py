"""
PySB Import
------------
This module provides all necessary functionality to import a model specified
in the :class:`pysb.core.Model` format.
"""

import itertools
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)
from collections.abc import Iterable

import numpy as np
import pysb
import pysb.bng
import pysb.pattern
import sympy as sp

from .de_export import (
    Constant,
    DEExporter,
    DEModel,
    DifferentialState,
    Expression,
    LogLikelihoodY,
    Observable,
    Parameter,
    SigmaY,
)
from .import_utils import (
    _get_str_symbol_identifiers,
    _parse_special_functions,
    generate_measurement_symbol,
    noise_distribution_to_cost_function,
    noise_distribution_to_observable_transformation,
    _default_simplify,
)
from .logging import get_logger, log_execution_time, set_log_level

CL_Prototype = dict[str, dict[str, Any]]
ConservationLaw = dict[str, Union[dict, str, sp.Basic]]

logger = get_logger(__name__, logging.ERROR)


def pysb2amici(
    model: pysb.Model,
    output_dir: Optional[Union[str, Path]] = None,
    observables: list[str] = None,
    constant_parameters: list[str] = None,
    sigmas: dict[str, str] = None,
    noise_distributions: Optional[dict[str, Union[str, Callable]]] = None,
    verbose: Union[int, bool] = False,
    assume_pow_positivity: bool = False,
    compiler: str = None,
    compute_conservation_laws: bool = True,
    compile: bool = True,
    simplify: Callable = _default_simplify,
    # Do not enable by default without testing.
    # See https://github.com/AMICI-dev/AMICI/pull/1672
    cache_simplify: bool = False,
    generate_sensitivity_code: bool = True,
    model_name: Optional[str] = None,
):
    r"""
    Generate AMICI C++ files for the provided model.

    .. warning::
        **PySB models with Compartments**

        When importing a PySB model with ``pysb.Compartment``\ s, BioNetGen
        scales reaction fluxes with the compartment size. Instead of using the
        respective symbols, the compartment size Parameter or Expression is
        evaluated when generating equations. This may lead to unexpected
        results if the compartment size parameter is changed for AMICI
        simulations.

    :param model:
        pysb model, :attr:`pysb.Model.name` will determine the name of the
        generated module

    :param output_dir:
        see :meth:`amici.de_export.ODEExporter.set_paths`

    :param observables:
        list of :class:`pysb.core.Expression` or :class:`pysb.core.Observable`
        names in the provided model that should be mapped to observables

    :param sigmas:
        dict of :class:`pysb.core.Expression` names that should be mapped to
        sigmas

    :param noise_distributions:
        dict with names of observable Expressions as keys and a noise type
        identifier, or a callable generating a custom noise formula string
        (see :py:func:`amici.import_utils.noise_distribution_to_cost_function`
        ). If nothing is passed for some observable id, a normal model is
        assumed as default.

    :param constant_parameters:
        list of :class:`pysb.core.Parameter` names that should be mapped as
        fixed parameters

    :param verbose: verbosity level for logging, True/False default to
        :attr:`logging.DEBUG`/:attr:`logging.ERROR`

    :param assume_pow_positivity:
        if set to ``True``, a special pow function is used to avoid problems
        with state variables that may become negative due to numerical
        errors

    :param compiler:
        Absolute path to the compiler executable to be used to build the Python
        extension, e.g. ``/usr/bin/clang``.

    :param compute_conservation_laws:
        if set to ``True``, conservation laws are automatically computed and
        applied such that the state-jacobian of the ODE right-hand-side has
        full rank. This option should be set to ``True`` when using the Newton
        algorithm to compute steadystates

    :param compile:
        If ``True``, build the python module for the generated model. If false,
        just generate the source code.

    :param simplify:
        see :attr:`amici.DEModel._simplify`

    :param cache_simplify:
            see :func:`amici.DEModel.__init__`
            Note that there are possible issues with PySB models:
            https://github.com/AMICI-dev/AMICI/pull/1672

    :param generate_sensitivity_code:
        if set to ``False``, code for sensitivity computation will not be
        generated

    :param model_name:
        Name for the generated model module. If None, :attr:`pysb.Model.name`
        will be used.
    """
    if observables is None:
        observables = []
    if constant_parameters is None:
        constant_parameters = []

    if sigmas is None:
        sigmas = {}

    model_name = model_name or model.name

    set_log_level(logger, verbose)
    ode_model = ode_model_from_pysb_importer(
        model,
        constant_parameters=constant_parameters,
        observables=observables,
        sigmas=sigmas,
        noise_distributions=noise_distributions,
        compute_conservation_laws=compute_conservation_laws,
        simplify=simplify,
        cache_simplify=cache_simplify,
        verbose=verbose,
    )
    exporter = DEExporter(
        ode_model,
        outdir=output_dir,
        model_name=model_name,
        verbose=verbose,
        assume_pow_positivity=assume_pow_positivity,
        compiler=compiler,
        generate_sensitivity_code=generate_sensitivity_code,
    )
    # Sympy code optimizations are incompatible with PySB objects, as
    #  `pysb.Observable` comes with its own `.match` which overrides
    #  `sympy.Basic.match()`, breaking `sympy.codegen.rewriting.optimize`.
    exporter._code_printer._fpoptimizer = None
    exporter.generate_model_code()

    if compile:
        exporter.compile_model()


@log_execution_time("creating ODE model", logger)
def ode_model_from_pysb_importer(
    model: pysb.Model,
    constant_parameters: list[str] = None,
    observables: list[str] = None,
    sigmas: dict[str, str] = None,
    noise_distributions: Optional[dict[str, Union[str, Callable]]] = None,
    compute_conservation_laws: bool = True,
    simplify: Callable = sp.powsimp,
    # Do not enable by default without testing.
    # See https://github.com/AMICI-dev/AMICI/pull/1672
    cache_simplify: bool = False,
    verbose: Union[int, bool] = False,
) -> DEModel:
    """
    Creates an :class:`amici.DEModel` instance from a :class:`pysb.Model`
    instance.

    :param model:
        see :func:`amici.pysb_import.pysb2amici`

    :param constant_parameters:
        see :func:`amici.pysb_import.pysb2amici`

    :param observables:
        see :func:`amici.pysb_import.pysb2amici`

    :param sigmas:
        dict with names of observable Expressions as keys and names of sigma
        Expressions as value sigma

    :param noise_distributions:
        see :func:`amici.pysb_import.pysb2amici`

    :param compute_conservation_laws:
        see :func:`amici.pysb_import.pysb2amici`

    :param simplify:
            see :attr:`amici.DEModel._simplify`

    :param cache_simplify:
            see :func:`amici.DEModel.__init__`
            Note that there are possible issues with PySB models:
            https://github.com/AMICI-dev/AMICI/pull/1672

    :param verbose: verbosity level for logging, True/False default to
        :attr:`logging.DEBUG`/:attr:`logging.ERROR`

    :return:
        New DEModel instance according to pysbModel
    """

    ode = DEModel(
        verbose=verbose,
        simplify=simplify,
        cache_simplify=cache_simplify,
    )

    if constant_parameters is None:
        constant_parameters = []

    if observables is None:
        observables = []

    if sigmas is None:
        sigmas = {}

    pysb.bng.generate_equations(model, verbose=verbose)

    _process_pysb_species(model, ode)
    _process_pysb_parameters(model, ode, constant_parameters)
    if compute_conservation_laws:
        _process_pysb_conservation_laws(model, ode)
    _process_pysb_observables(
        model, ode, observables, sigmas, noise_distributions
    )
    _process_pysb_expressions(
        model, ode, observables, sigmas, noise_distributions
    )
    ode._has_quadratic_nllh = not noise_distributions or all(
        noise_distr in ["normal", "lin-normal", "log-normal", "log10-normal"]
        for noise_distr in noise_distributions.values()
    )

    _process_stoichiometric_matrix(model, ode, constant_parameters)

    ode.generate_basic_variables()

    return ode


@log_execution_time("processing PySB stoich. matrix", logger)
def _process_stoichiometric_matrix(
    pysb_model: pysb.Model, ode_model: DEModel, constant_parameters: list[str]
) -> None:
    """
    Exploits the PySB stoichiometric matrix to generate xdot derivatives

    :param pysb_model:
        pysb model instance

    :param ode_model:
        DEModel instance

    :param constant_parameters:
        list of constant parameters
    """

    x = ode_model.sym("x")
    w = list(ode_model.sym("w"))
    p = list(ode_model.sym("p"))
    x_rdata = list(ode_model.sym("x_rdata"))

    n_x = len(x)
    n_w = len(w)
    n_p = len(p)
    n_r = len(pysb_model.reactions)

    solver_index = ode_model.get_solver_indices()
    dflux_dx_dict = {}
    dflux_dw_dict = {}
    dflux_dp_dict = {}

    w_idx = dict()
    p_idx = dict()
    wx_idx = dict()

    def get_cached_index(symbol, sarray, index_cache):
        idx = index_cache.get(symbol, None)
        if idx is not None:
            return idx
        idx = sarray.index(symbol)
        index_cache[symbol] = idx
        return idx

    for ir, rxn in enumerate(pysb_model.reactions):
        for ix in np.unique(rxn["reactants"]):
            idx = solver_index.get(ix, None)
            if idx is not None:
                # species
                values = dflux_dx_dict
            else:
                # conservation law
                idx = get_cached_index(x_rdata[ix], w, wx_idx)
                values = dflux_dw_dict

            values[(ir, idx)] = sp.diff(rxn["rate"], x_rdata[ix])

        # typically <= 3 free symbols in rate, we already account for
        # species above so we only need to account for propensity, which
        # can only be a parameter or expression
        for fs in rxn["rate"].free_symbols:
            # dw
            if isinstance(fs, pysb.Expression):
                var = w
                idx_cache = w_idx
                values = dflux_dw_dict
            # dp
            elif isinstance(fs, pysb.Parameter):
                if fs.name in constant_parameters:
                    continue
                var = p
                idx_cache = p_idx
                values = dflux_dp_dict
            else:
                continue

            idx = get_cached_index(fs, var, idx_cache)
            values[(ir, idx)] = sp.diff(rxn["rate"], fs)

    dflux_dx = sp.ImmutableSparseMatrix(n_r, n_x, dflux_dx_dict)
    dflux_dw = sp.ImmutableSparseMatrix(n_r, n_w, dflux_dw_dict)
    dflux_dp = sp.ImmutableSparseMatrix(n_r, n_p, dflux_dp_dict)

    # use dok format to convert numeric csc to sparse symbolic
    S = sp.ImmutableSparseMatrix(
        n_x,
        n_r,  # don't use shape here as we are eliminating rows
        pysb_model.stoichiometry_matrix[
            np.asarray(list(solver_index.keys())), :
        ].todok(),
    )
    # don't use `.dot` since it's awfully slow
    ode_model._eqs["dxdotdx_explicit"] = S * dflux_dx
    ode_model._eqs["dxdotdw"] = S * dflux_dw
    ode_model._eqs["dxdotdp_explicit"] = S * dflux_dp


@log_execution_time("processing PySB species", logger)
def _process_pysb_species(pysb_model: pysb.Model, ode_model: DEModel) -> None:
    """
    Converts pysb Species into States and adds them to the DEModel instance

    :param pysb_model:
        pysb model instance

    :param ode_model:
        DEModel instance
    """
    xdot = sp.Matrix(pysb_model.odes)

    for ix, specie in enumerate(pysb_model.species):
        init = sp.sympify("0.0")
        for ic in pysb_model.odes.model.initials:
            if pysb.pattern.match_complex_pattern(
                ic.pattern, specie, exact=True
            ):
                # we don't want to allow expressions in initial conditions
                if ic.value in pysb_model.expressions:
                    init = pysb_model.expressions[ic.value.name].expand_expr()
                else:
                    init = ic.value

        ode_model.add_component(
            DifferentialState(
                sp.Symbol(f"__s{ix}"), f"{specie}", init, xdot[ix]
            )
        )
    logger.debug("Finished Processing PySB species ")


@log_execution_time("processing PySB parameters", logger)
def _process_pysb_parameters(
    pysb_model: pysb.Model, ode_model: DEModel, constant_parameters: list[str]
) -> None:
    """
    Converts pysb parameters into Parameters or Constants and adds them to
    the DEModel instance

    :param pysb_model:
        pysb model

    :param constant_parameters:
        list of Parameters that should be constants

    :param ode_model:
        DEModel instance
    """
    for par in pysb_model.parameters:
        if par.name in constant_parameters:
            comp = Constant
        else:
            comp = Parameter

        ode_model.add_component(comp(par, f"{par.name}", par.value))


@log_execution_time("processing PySB expressions", logger)
def _process_pysb_expressions(
    pysb_model: pysb.Model,
    ode_model: DEModel,
    observables: list[str],
    sigmas: dict[str, str],
    noise_distributions: Optional[dict[str, Union[str, Callable]]] = None,
) -> None:
    r"""
    Converts pysb expressions/observables into Observables (with
    corresponding standard deviation SigmaY and LogLikelihoodY) or
    Expressions and adds them to the DEModel instance

    :param pysb_model:
        pysb model

    :param observables:
        list of names of :class`pysb.Expression`\ s or
        :class:`pysb.Observable`\ s that are to be mapped to DEModel
        observables

    :param sigmas:
        dict with names of observable :class:`pysb.Expression` /
        :class:`pysb.Observable` names as keys and names of sigma
        :class:`pysb.Expressions` as values

    :param noise_distributions:
        see :func:`amici.pysb_import.pysb2amici`

    :param ode_model:
        DEModel instance
    """
    # we no longer expand expressions here. pysb/bng guarantees that
    # they are ordered according to their dependency and we can
    # evaluate them sequentially without reordering. Important to make
    # sure that observables are processed first though.

    # we use _constant and _dynamic functions to get access to derived
    # expressions that are otherwise only accessible as private attribute
    for expr in pysb_model.expressions_constant(
        include_derived=True
    ) | pysb_model.expressions_dynamic(include_derived=True):
        if any(
            isinstance(symbol, pysb.Tag)
            for symbol in expr.expand_expr().free_symbols
        ):
            # we only need explicit instantiations of expressions with tags,
            # which are defined in the derived expressions. The abstract
            # expressions are not needed and lead to compilation errors so
            # we skip them.
            continue
        _add_expression(
            expr,
            expr.name,
            expr.expr,
            pysb_model,
            ode_model,
            observables,
            sigmas,
            noise_distributions,
        )


def _add_expression(
    sym: sp.Symbol,
    name: str,
    expr: sp.Basic,
    pysb_model: pysb.Model,
    ode_model: DEModel,
    observables: list[str],
    sigmas: dict[str, str],
    noise_distributions: Optional[dict[str, Union[str, Callable]]] = None,
):
    """
    Adds expressions to the ODE model given and adds observables/sigmas if
    appropriate

    :param sym:
        symbol how the expression is referenced in the model

    :param name:
        name of the expression

    :param expr:
        symbolic expression that the symbol refers to

    :param pysb_model:
        see :py:func:`_process_pysb_expressions`

    :param observables:
        see :py:func:`_process_pysb_expressions`

    :param sigmas:
        see :py:func:`_process_pysb_expressions`

    :param noise_distributions:
        see :py:func:`amici.pysb_import.pysb2amici`

    :param ode_model:
        see :py:func:`_process_pysb_expressions`
    """
    ode_model.add_component(
        Expression(sym, name, _parse_special_functions(expr))
    )

    if name in observables:
        noise_dist = (
            noise_distributions.get(name, "normal")
            if noise_distributions
            else "normal"
        )

        y = sp.Symbol(f"{name}")
        trafo = noise_distribution_to_observable_transformation(noise_dist)
        obs = Observable(y, name, sym, transformation=trafo)
        ode_model.add_component(obs)

        sigma_name, sigma_value = _get_sigma_name_and_value(
            pysb_model, name, sigmas
        )

        sigma = sp.Symbol(sigma_name)
        ode_model.add_component(SigmaY(sigma, f"{sigma_name}", sigma_value))

        cost_fun_str = noise_distribution_to_cost_function(noise_dist)(name)
        my = generate_measurement_symbol(obs.get_id())
        cost_fun_expr = sp.sympify(
            cost_fun_str,
            locals=dict(
                zip(_get_str_symbol_identifiers(name), (y, my, sigma))
            ),
        )
        ode_model.add_component(
            LogLikelihoodY(
                sp.Symbol(f"llh_{name}"), f"llh_{name}", cost_fun_expr
            )
        )


def _get_sigma_name_and_value(
    pysb_model: pysb.Model, obs_name: str, sigmas: dict[str, str]
) -> tuple[str, sp.Basic]:
    """
    Tries to extract standard deviation symbolic identifier and formula
    for a given observable name from the pysb model and if no specification is
    available sets default values

    :param pysb_model:
        pysb model

    :param obs_name:
        name of the observable

    :param sigmas:
        dict of :class:`pysb.core.Expression` names that should be mapped to
        sigmas

    :return:
        tuple containing symbolic identifier and formula for the specified
        observable
    """
    if obs_name in sigmas:
        sigma_name = sigmas[obs_name]
        try:
            # find corresponding Expression instance
            sigma_expr = next(
                x for x in pysb_model.expressions if x.name == sigma_name
            )
        except StopIteration:
            raise ValueError(
                f"value of sigma {obs_name} is not a " f"valid expression."
            )
        sigma_value = sigma_expr.expand_expr()
    else:
        sigma_name = f"sigma_{obs_name}"
        sigma_value = sp.sympify(1.0)

    return sigma_name, sigma_value


@log_execution_time("processing PySB observables", logger)
def _process_pysb_observables(
    pysb_model: pysb.Model,
    ode_model: DEModel,
    observables: list[str],
    sigmas: dict[str, str],
    noise_distributions: Optional[dict[str, Union[str, Callable]]] = None,
) -> None:
    """
    Converts :class:`pysb.core.Observable` into
    :class:`DEModel.Expressions` and adds them to the DEModel instance

    :param pysb_model:
        pysb model

    :param ode_model:
        DEModel instance

    :param observables:
        list of names of pysb.Expressions or pysb.Observables that are to be
        mapped to DEModel observables

    :param sigmas:
        dict with names of observable pysb.Expressions/pysb.Observables
        names as keys and names of sigma pysb.Expressions as values

    :param noise_distributions:
        see :func:`amici.pysb_import.pysb2amici`
    """
    # only add those pysb observables that occur in the added
    # Observables as expressions
    for obs in pysb_model.observables:
        _add_expression(
            obs,
            obs.name,
            obs.expand_obs(),
            pysb_model,
            ode_model,
            observables,
            sigmas,
            noise_distributions,
        )


@log_execution_time("computing PySB conservation laws", logger)
def _process_pysb_conservation_laws(
    pysb_model: pysb.Model, ode_model: DEModel
) -> None:
    """
    Removes species according to conservation laws to ensure that the
    jacobian has full rank

    :param pysb_model:
        pysb model

    :param ode_model:
        DEModel instance
    """

    monomers_without_conservation_law = set()
    for rule in pysb_model.rules:
        monomers_without_conservation_law |= _get_unconserved_monomers(
            rule, pysb_model
        )

    monomers_without_conservation_law |= (
        _compute_monomers_with_fixed_initial_conditions(pysb_model)
    )

    cl_prototypes = _generate_cl_prototypes(
        monomers_without_conservation_law, pysb_model, ode_model
    )
    conservation_laws = _construct_conservation_from_prototypes(
        cl_prototypes, pysb_model
    )
    _add_conservation_for_constant_species(ode_model, conservation_laws)

    _flatten_conservation_laws(conservation_laws)

    for cl in conservation_laws:
        ode_model.add_conservation_law(**cl)


def _compute_monomers_with_fixed_initial_conditions(
    pysb_model: pysb.Model,
) -> set[str]:
    """
    Computes the set of monomers in a model with species that have fixed
    initial conditions

    :param pysb_model: pysb model

    :return:
        set of monomer names with fixed initial conditions
    """
    monomers_with_fixed_initial_conditions = set()

    for monomer in pysb_model.monomers:
        # check if monomer has an initial condition that is fixed (means
        # that corresponding state is constant and all conservation
        # laws are broken)
        if any(
            [
                ic.fixed  # true or false
                for ic in pysb_model.initials
                if monomer.name in extract_monomers(ic.pattern)
            ]
        ):
            monomers_with_fixed_initial_conditions |= {monomer.name}

    return monomers_with_fixed_initial_conditions


def _generate_cl_prototypes(
    excluded_monomers: Iterable[str],
    pysb_model: pysb.Model,
    ode_model: DEModel,
) -> CL_Prototype:
    """
    Constructs a dict that contains preprocessed information for the
    construction of conservation laws

    :param excluded_monomers:
        list of monomer names for which no prototypes
        should be computed

    :param pysb_model:
        pysb model

    :param ode_model:
        DEModel instance

    :return:
        dict('monomer.name':{'possible_indices': ..., 'target_indices': ...}
    """
    cl_prototypes = dict()

    _compute_possible_indices(
        cl_prototypes, pysb_model, ode_model, excluded_monomers
    )
    _compute_dependency_idx(cl_prototypes)
    _compute_target_index(cl_prototypes, ode_model)

    return cl_prototypes


def _compute_possible_indices(
    cl_prototypes: CL_Prototype,
    pysb_model: pysb.Model,
    ode_model: DEModel,
    excluded_monomers: Iterable[str],
) -> None:
    """
    Computes viable choices for target_index, ie species that could be
    removed and replaced by an algebraic expression according to the
    conservation law

    :param cl_prototypes:
        dict in which possible indices will be written

    :param pysb_model:
        pysb model

    :param ode_model:
        DEModel instance

    :param excluded_monomers:
        monomers for which no conservation laws will be
        computed
    """
    for monomer in pysb_model.monomers:
        if monomer.name not in excluded_monomers:
            compartments = [
                str(mp.compartment)  # string based comparison as
                # compartments are not hashable
                for cp in pysb_model.species
                for mp in cp.monomer_patterns
                if mp.monomer.name == monomer.name
            ]

            if len(set(compartments)) > 1:
                raise ValueError(
                    "Conservation laws involving species in "
                    "multiple compartments are currently not "
                    "supported! Please run pysb2amici with "
                    "compute_conservation_laws=False"
                )
                # TODO: implement this, multiply species by the volume of
                # their respective compartment and allow total_cl to depend
                # on parameters + constants and update the respective symbolic
                # derivative accordingly

            prototype = dict()
            prototype["possible_indices"] = [
                ix
                for ix, specie in enumerate(pysb_model.species)
                if monomer.name in extract_monomers(specie)
                and not ode_model.state_is_constant(ix)
            ]

            prototype["species_count"] = len(prototype["possible_indices"])

            if prototype["possible_indices"]:
                cl_prototypes[monomer.name] = prototype


def _compute_dependency_idx(cl_prototypes: CL_Prototype) -> None:
    """
    Compute connecting species, this allows us to efficiently compute
    whether the respective conservation law would induce a cyclic dependency.
    Adds a 'dependency_idx' field to the prototype dict that
    itself is a dict where keys correspond to indexes that, when used as
    target index yield dependencies on conservation laws of monomers in
    the respective values

    :param cl_prototypes:
        dict in which possible indices will be written
    """
    #
    for monomer_i, prototype_i in cl_prototypes.items():
        if "dependency_idx" not in prototype_i:
            prototype_i["dependency_idx"] = dict()

        for monomer_j, prototype_j in cl_prototypes.items():
            if monomer_i == monomer_j:
                continue

            if "dependency_idx" not in prototype_j:
                prototype_j["dependency_idx"] = dict()

            idx_overlap = set(prototype_i["possible_indices"]).intersection(
                set(prototype_j["possible_indices"])
            )
            if len(idx_overlap) == 0:
                continue

            for idx in idx_overlap:
                if idx not in prototype_i["dependency_idx"]:
                    prototype_i["dependency_idx"][idx] = set()

                if idx not in prototype_j["dependency_idx"]:
                    prototype_j["dependency_idx"][idx] = set()

                prototype_i["dependency_idx"][idx] |= {monomer_j}
                prototype_j["dependency_idx"][idx] |= {monomer_i}


def _compute_target_index(
    cl_prototypes: CL_Prototype, ode_model: DEModel
) -> None:
    """
    Computes the target index for every monomer

    :param cl_prototypes:
        dict that contains possible indices for every monomer

    :param ode_model:
        DEModel instance
    """
    possible_indices = list(
        set(
            list(
                itertools.chain(
                    *[
                        cl_prototypes[monomer]["possible_indices"]
                        for monomer in cl_prototypes
                    ]
                )
            )
        )
    )

    # Note: currently this function is supposed to also count appearances in
    # expressions. However, expressions are currently still empty as they
    # are also populated from conservation laws. In case there are many
    # state heavy expressions in the model (should not be the case for mass
    # action kinetics). This may lead to suboptimal results and could improved.
    # As this would require substantial code shuffling, this will only be
    # fixed if this becomes an actual problem
    appearance_counts = ode_model.get_appearance_counts(possible_indices)

    # in this initial guess we ignore the cost of having cyclic dependencies
    # between conservation laws
    for monomer in cl_prototypes:
        prototype = cl_prototypes[monomer]
        # extract monomer specific appearance counts
        prototype["appearance_counts"] = [
            appearance_counts[possible_indices.index(idx)]
            for idx in prototype["possible_indices"]
        ]
        # select target index as possible index with minimal appearance count
        if len(prototype["appearance_counts"]) == 0:
            raise RuntimeError(
                f"Failed to compute conservation law for " f"monomer {monomer}"
            )

        idx = np.argmin(prototype["appearance_counts"])

        # remove entries from possible indices and appearance counts so we
        # do not consider them again in later iterations
        prototype["target_index"] = prototype["possible_indices"].pop(idx)
        prototype["appearance_count"] = prototype["appearance_counts"].pop(idx)

        # this is only an approximation as the effective species count
        # of other conservation laws may also be affected by the chosen
        # target index. As long as the number of unique monomers in
        # multimers has a low upper bound and the species count does not
        # vary too much across conservation laws, this approximation
        # should be fine
        prototype["fillin"] = (
            prototype["appearance_count"] * prototype["species_count"]
        )

    # we might end up with the same index for multiple monomers, so loop until
    # we have a set of unique target indices
    while not _cl_prototypes_are_valid(cl_prototypes):
        _greedy_target_index_update(cl_prototypes)


def _cl_prototypes_are_valid(cl_prototypes: CL_Prototype) -> bool:
    """
    Checks consistency of cl_prototypes by asserting that target indices
    are unique and there are no cyclic dependencies

    :param cl_prototypes:
        dict that contains dependency and target indexes for
        every monomer
    """
    # target indices are unique
    if len(cl_prototypes) != len(set(_get_target_indices(cl_prototypes))):
        return False
    # conservation law dependencies are cycle free
    if any(_cl_has_cycle(monomer, cl_prototypes) for monomer in cl_prototypes):
        return False

    return True


def _cl_has_cycle(monomer: str, cl_prototypes: CL_Prototype) -> bool:
    """
    Checks whether monomer has a conservation law that is part of a
    cyclic dependency

    :param monomer:
        name of monomer for which conservation law is to be checked

    :param cl_prototypes:
        dict that contains dependency and target indexes for every monomer

    :return:
        boolean indicating whether the conservation law is cyclic
    """

    prototype = cl_prototypes[monomer]

    if prototype["target_index"] not in prototype["dependency_idx"]:
        return False

    visited = [monomer]
    root = monomer
    return any(
        _is_in_cycle(connecting_monomer, cl_prototypes, visited, root)
        for connecting_monomer in prototype["dependency_idx"][
            prototype["target_index"]
        ]
    )


def _is_in_cycle(
    monomer: str, cl_prototypes: CL_Prototype, visited: list[str], root: str
) -> bool:
    """
    Recursively checks for cycles in conservation law dependencies via
    Depth First Search

    :param monomer:
        current location in cl dependency graph

    :param cl_prototypes:
        dict that contains dependency and target indexes for
        every monomer

    :param visited:
        history of visited monomers with conservation laws

    :param root:
        monomer at which the cycle search was started

    :return:
        boolean indicating whether the specified monomer is part of a cyclic
        conservation law

    """
    if monomer == root:
        return True  # we found a cycle and root is part of it

    if monomer in visited:
        return False  # we found a cycle but root is not part of it

    visited.append(monomer)

    prototype = cl_prototypes[monomer]

    if prototype["target_index"] not in prototype["dependency_idx"]:
        return False

    return any(
        _is_in_cycle(connecting_monomer, cl_prototypes, visited, root)
        for connecting_monomer in prototype["dependency_idx"][
            prototype["target_index"]
        ]
    )


def _greedy_target_index_update(cl_prototypes: CL_Prototype) -> None:
    """
    Computes unique target indices for conservation laws from possible
    indices  such that expected fill in in symbolic derivatives is minimized

    :param cl_prototypes:
        dict that contains possible indices and non-unique target indices
        for every monomer
    """

    target_indices = _get_target_indices(cl_prototypes)

    for monomer, prototype in cl_prototypes.items():
        if target_indices.count(
            prototype["target_index"]
        ) > 1 or _cl_has_cycle(monomer, cl_prototypes):
            # compute how much fillin the next best target_index would yield

            # we exclude already existing target indices to avoid that
            # updating the target index removes uniqueness from already unique
            # target indices, this may slightly reduce chances of finding a
            # solution but prevents infinite loops
            for target_index in list(set(target_indices)):
                try:
                    local_idx = prototype["possible_indices"].index(
                        target_index
                    )
                except ValueError:
                    local_idx = None

                if local_idx:
                    del prototype["possible_indices"][local_idx]
                    del prototype["appearance_counts"][local_idx]

            if len(prototype["possible_indices"]) == 0:
                prototype["diff_fillin"] = -1
                continue

            idx = np.argmin(prototype["appearance_counts"])

            prototype["local_index"] = idx
            prototype["alternate_target_index"] = prototype[
                "possible_indices"
            ][idx]
            prototype["alternate_appearance_count"] = prototype[
                "appearance_counts"
            ][idx]

            prototype["alternate_fillin"] = (
                prototype["alternate_appearance_count"]
                * prototype["species_count"]
            )

            prototype["diff_fillin"] = (
                prototype["alternate_fillin"] - prototype["fillin"]
            )
        else:
            prototype["diff_fillin"] = -1

    if all(
        prototype["diff_fillin"] == -1 for prototype in cl_prototypes.values()
    ):
        raise RuntimeError(
            "Could not compute a valid set of conservation "
            "laws for this model!"
        )

    # this puts prototypes with high diff_fillin last
    cl_prototypes = sorted(
        cl_prototypes.items(), key=lambda kv: kv[1]["diff_fillin"]
    )
    cl_prototypes = {proto[0]: proto[1] for proto in cl_prototypes}

    for monomer in cl_prototypes:
        prototype = cl_prototypes[monomer]
        # we check that we
        # A) have an alternative index computed, i.e. that
        # that monomer originally had a non-unique target_index
        # B) that the target_index still is not unique or part of a cyclic
        # dependency. due to the sorting, this will always be the monomer
        # with the highest diff_fillin (note that the target index counts
        # are recomputed on the fly)

        if prototype["diff_fillin"] > -1 and (
            _get_target_indices(cl_prototypes).count(prototype["target_index"])
            > 1
            or _cl_has_cycle(monomer, cl_prototypes)
        ):
            prototype["fillin"] = prototype["alternate_fillin"]
            prototype["target_index"] = prototype["alternate_target_index"]
            prototype["appearance_count"] = prototype[
                "alternate_appearance_count"
            ]

            del prototype["possible_indices"][prototype["local_index"]]
            del prototype["appearance_counts"][prototype["local_index"]]


def _get_target_indices(cl_prototypes: CL_Prototype) -> list[list[int]]:
    """
    Computes the list target indices for the current
    conservation law prototype

    :param cl_prototypes:
        dict that contains target indices for every monomer

    :return:
        List of lists of target indices
    """
    return [prototype["target_index"] for prototype in cl_prototypes.values()]


def _construct_conservation_from_prototypes(
    cl_prototypes: CL_Prototype, pysb_model: pysb.Model
) -> list[ConservationLaw]:
    """
    Computes the algebraic expression for the total amount of a given
    monomer

    :param cl_prototypes:
        see return of :func:`_generate_cl_prototypes`

    :param pysb_model:
        pysb model

    :return:
        list of dicts describing conservation laws
    """
    conservation_laws = []
    for monomer_name in cl_prototypes:
        target_index = cl_prototypes[monomer_name]["target_index"]
        coefficients = dict()

        for ix, specie in enumerate(pysb_model.species):
            count = extract_monomers(specie).count(monomer_name)
            if count > 0:
                coefficients[sp.Symbol(f"__s{ix}")] = count

        conservation_laws.append(
            {
                "state": sp.Symbol(f"__s{target_index}"),
                "total_abundance": sp.Symbol(f"tcl__s{target_index}"),
                "coefficients": coefficients,
            }
        )

    return conservation_laws


def _add_conservation_for_constant_species(
    ode_model: DEModel, conservation_laws: list[ConservationLaw]
) -> None:
    """
    Computes the algebraic expression for the total amount of a given
    monomer

    :param ode_model:
        DEModel instance to which the conservation laws will be added

    :param conservation_laws:
        see return of :func:`_construct_conservation_from_prototypes`

    """

    for ix in range(ode_model.num_states_rdata()):
        if ode_model.state_is_constant(ix):
            conservation_laws.append(
                {
                    "state": sp.Symbol(f"__s{ix}"),
                    "total_abundance": sp.Symbol(f"tcl__s{ix}"),
                    "coefficients": {sp.Symbol(f"__s{ix}"): 1.0},
                }
            )


def _flatten_conservation_laws(
    conservation_laws: list[ConservationLaw],
) -> None:
    """
    Flatten the conservation laws such that the state_expr not longer
    depend on any states that are replaced by conservation laws

    :param conservation_laws:
        see return of :func:`_construct_conservation_from_prototypes`
    """
    conservation_law_subs = _get_conservation_law_subs(conservation_laws)

    while conservation_law_subs:
        for cl in conservation_laws:
            # only update if we changed something
            if any(
                _apply_conseration_law_sub(cl, sub)
                for sub in conservation_law_subs
            ):
                conservation_law_subs = _get_conservation_law_subs(
                    conservation_laws
                )


def _apply_conseration_law_sub(
    cl: ConservationLaw, sub: tuple[sp.Symbol, ConservationLaw]
) -> bool:
    """
    Applies a substitution to a conservation law by replacing the
    coefficient of the state of the

    :param cl:
        conservation law

    :param sub:
        substitution to apply, tuple of (state to be replaced, conservation
        law)

    :return: boolean flag indicating whether the substitution was applied
    """
    if not _state_in_cl_formula(sub[0], cl):
        return False

    coeff = cl["coefficients"].pop(sub[0], 0.0)
    # x_j = T/b_j - sum_{i≠j}(x_i * b_i) / b_j
    # don't need to account for totals here as we can simply
    # absorb that into the new total
    for k, v in sub[1].items():
        if k == sub[0]:
            continue
        update = -coeff * v / sub[1][sub[0]]

        if k in cl["coefficients"]:
            cl["coefficients"][k] += update
        else:
            cl["coefficients"][k] = update

    return True


def _state_in_cl_formula(state: sp.Symbol, cl: ConservationLaw) -> bool:
    """
    Checks whether state appears in the formula the provided cl

    :param state:
        state

    :param cl:
        conservation law

    :return:
        boolean indicator
    """
    if cl["state"] == state:
        return False

    return cl["coefficients"].get(state, 0.0) != 0.0


def _get_conservation_law_subs(
    conservation_laws: list[ConservationLaw],
) -> list[tuple[sp.Symbol, dict[sp.Symbol, sp.Expr]]]:
    """
    Computes a list of (state, coeffs) tuples for conservation laws that still
    appear in other conservation laws

    :param conservation_laws:
        see return of :func:`_flatten_conservation_laws`

    :return:
        list of tuples containing substitution rules to be used with sympy
        subs
    """
    return [
        (cl["state"], cl["coefficients"])
        for cl in conservation_laws
        if any(
            _state_in_cl_formula(cl["state"], other_cl)
            for other_cl in conservation_laws
        )
    ]


def has_fixed_parameter_ic(
    specie: pysb.core.ComplexPattern,
    pysb_model: pysb.Model,
    ode_model: DEModel,
) -> bool:
    """
    Wrapper to interface
    :meth:`de_export.DEModel.state_has_fixed_parameter_initial_condition`
    from a pysb specie/model arguments

    :param specie:
        pysb species

    :param pysb_model:
        pysb model

    :param ode_model:
        ODE model

    :return:
        ``False`` if the species does not have an initial condition at all.
        Otherwise the return value of
        :meth:`de_export.DEModel.state_has_fixed_parameter_initial_condition`
    """
    # ComplexPatterns are not hashable, so we have to compare by string
    ic_index = next(
        (
            ic
            for ic, condition in enumerate(pysb_model.initials)
            if pysb.pattern.match_complex_pattern(
                condition[0], specie, exact=True
            )
        ),
        None,
    )
    if ic_index is None:
        return False
    else:
        return ode_model.state_has_fixed_parameter_initial_condition(ic_index)


def extract_monomers(
    complex_patterns: Union[pysb.ComplexPattern, list[pysb.ComplexPattern]],
) -> list[str]:
    """
    Constructs a list of monomer names contained in complex patterns.
    Multiplicity of names corresponds to the stoichiometry in the complex.

    :param complex_patterns:
        (list of) complex pattern(s)

    :return:
        list of monomer names
    """
    if not isinstance(complex_patterns, list):
        complex_patterns = [complex_patterns]
    return [
        mp.monomer.name
        for cp in complex_patterns
        if cp is not None
        for mp in cp.monomer_patterns
    ]


def _get_unconserved_monomers(
    rule: pysb.Rule, pysb_model: pysb.Model
) -> set[str]:
    """
    Constructs the set of monomer names for which the specified rule changes
    the stoichiometry of the monomer in the specified model.

    :param rule:
        the pysb rule

    :param pysb_model:
        pysb model

    :return:
        set of monomer names for which the stoichiometry is not conserved
    """
    unconserved_monomers = set()

    if (
        not rule.delete_molecules
        and len(rule.product_pattern.complex_patterns) == 0
    ):
        # if delete_molecules is not True but we have a degradation rule,
        # we have to actually go through the reactions that are created by
        # the rule
        for reaction in [
            r for r in pysb_model.reactions if rule.name in r["rule"]
        ]:
            unconserved_monomers |= _get_changed_stoichiometries(
                [pysb_model.species[ix] for ix in reaction["reactants"]],
                [pysb_model.species[ix] for ix in reaction["products"]],
            )
    else:
        # otherwise we can simply extract all information for the rule
        # itself, which is computationally much more efficient
        unconserved_monomers |= _get_changed_stoichiometries(
            rule.reactant_pattern.complex_patterns,
            rule.product_pattern.complex_patterns,
        )

    return unconserved_monomers


def _get_changed_stoichiometries(
    reactants: Union[pysb.ComplexPattern, list[pysb.ComplexPattern]],
    products: Union[pysb.ComplexPattern, list[pysb.ComplexPattern]],
) -> set[str]:
    """
    Constructs the set of monomer names which have different
    stoichiometries in reactants and products.

    :param reactants:
        (list of) complex pattern(s)
    :param products:
        (list of) complex pattern(s)

    :returns:
        set of monomer name for which the stoichiometry changed
    """

    changed_stoichiometries = set()

    reactant_monomers = extract_monomers(reactants)

    product_monomers = extract_monomers(products)

    for monomer in set(reactant_monomers + product_monomers):
        if reactant_monomers.count(monomer) != product_monomers.count(monomer):
            changed_stoichiometries.add(monomer)

    return changed_stoichiometries


def pysb_model_from_path(pysb_model_file: Union[str, Path]) -> pysb.Model:
    """Load a pysb model module and return the :class:`pysb.Model` instance

    :param pysb_model_file: Full or relative path to the PySB model module
    :return: The pysb Model instance
    """

    pysb_model_module_name = os.path.splitext(
        os.path.split(pysb_model_file)[-1]
    )[0]

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        pysb_model_module_name, pysb_model_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pysb_model_module_name] = module
    spec.loader.exec_module(module)

    return module.model
