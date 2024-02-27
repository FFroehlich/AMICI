"""
SBML Utilities
--------------
This module provides helper functions for working with SBML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from typing import Any, Union

    SbmlID = Union[str, sp.Symbol]

import xml.dom.minidom

import libsbml
from sympy.core.parameters import evaluate
from sympy.printing.mathml import MathMLContentPrinter

from .import_utils import (
    SBMLException,
    _check_unsupported_functions,
    _parse_special_functions,
    amici_time_symbol,
    sbml_time_symbol,
)


class SbmlInvalidIdSyntax(SBMLException):
    pass


class SbmlDuplicateComponentIdError(SBMLException):
    pass


class SbmlMissingComponentIdError(SBMLException):
    pass


class SbmlMathError(SBMLException):
    pass


class SbmlAnnotationError(SBMLException):
    pass


def create_sbml_model(
    model_id: str, level: int = 2, version: int = 5
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """Helper for creating an empty SBML model.

    :param model_id:
        SBML ID of the new model.

    :param level:
        Level of the new SBML document.

    :param version:
        Version of the new SBML document.

    :return:
        A tuple containing the newly created :py:class:`libsbml.SBMLDocument`
        and :py:class:`libsbml.Model`.
    """
    doc = libsbml.SBMLDocument(level, version)
    model = doc.createModel()
    model.setId(model_id)
    return doc, model


def add_compartment(
    model: libsbml.Model,
    compartment_id: SbmlID,
    *,
    size: float = 1.0,
) -> libsbml.Species:
    """Helper for adding a compartment to a SBML model.

    :param model:
        SBML model to which the compartment is to be added.

    :param compartment_id:
        SBML ID of the new compartment.

    :param size:
        Size of the new compartment. Defaults to `1.0`.

    :return:
        The new compartment as a :py:class:`libsbml.Compartment` object.
    """
    compartment_id = str(compartment_id)

    # Check whether a compartment with the same ID already exists
    # TODO the resulting SBML may still be invalid
    #      if other types of objects (e.g., parameter) have the same ID
    if model.getCompartment(compartment_id):
        raise SbmlDuplicateComponentIdError(
            f"A compartment with ID {compartment_id} has already been defined"
        )

    cmp = model.createCompartment()
    if cmp.setId(compartment_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{compartment_id} is not a valid SBML ID")
    cmp.setSize(size)

    return cmp


def add_species(
    model: libsbml.Model,
    species_id: SbmlID,
    *,
    compartment_id: str | None = None,
    name: bool | str = False,
    initial_amount: float = 0.0,
    units: str | None = None,
) -> libsbml.Species:
    """Helper for adding a species to a SBML model.

    :param model:
        SBML model to which the species is to be added.

    :param species_id:
        SBML ID of the new species.

    :param compartment_id:
        Compartment ID for the new species.
        If there is only one compartment it can be auto-selected.

    :param initial_amount:
        Initial amount of the new species.

    :param units:
        Units attribute for the new species.

    :return:
        The new species as a :py:class:`libsbml.Species` object.
    """
    species_id = str(species_id)
    if name is True:
        name = species_id

    # Check whether an element with the same ID already exists
    if model.getElementBySId(species_id):
        raise SbmlDuplicateComponentIdError(
            f"An element with ID {species_id} has already been defined."
        )

    if compartment_id is None:
        compartments = model.getListOfCompartments()
        if len(compartments) != 1:
            raise ValueError(
                "Compartment auto-selection is possible "
                "only if there is one and only one compartment."
            )
        compartment_id = compartments[0].getId()
    elif not model.getCompartment(compartment_id):
        raise SbmlMissingComponentIdError(
            f"No compartment with ID {compartment_id}."
        )

    sp = model.createSpecies()
    if sp.setIdAttribute(species_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{species_id} is not a valid SBML ID.")
    sp.setCompartment(compartment_id)
    sp.setInitialAmount(float(initial_amount))
    if units is not None:
        sp.setUnits(str(units))
    if isinstance(name, str):
        sp.setName(name)

    return sp


def add_parameter(
    model: libsbml.Model,
    parameter_id: SbmlID,
    *,
    name: bool | str = False,
    value: float | None = None,
    units: str | None = None,
    constant: bool | None = None,
) -> libsbml.Parameter:
    """Helper for adding a parameter to a SBML model.

    :param model:
        SBML model to which the parameter is to be added.

    :param parameter_id:
        SBML ID of the new parameter.

    :param name:
        SBML name of the new parameter.

    :param value:
        Value attribute for the new parameter.

    :param units:
        Units attribute for the new parameter.

    :param constant:
        Constant attribute for the new parameter.

    :return:
        The new parameter as a :py:class:`libsbml.Parameter` object.
    """
    parameter_id = str(parameter_id)
    if name is True:
        name = parameter_id

    # Check whether an element with the same ID already exists
    if model.getElementBySId(parameter_id):
        raise SbmlDuplicateComponentIdError(
            f"An element with ID {parameter_id} has already been defined."
        )

    par = model.createParameter()
    if par.setIdAttribute(parameter_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{parameter_id} is not a valid SBML ID.")
    if units is not None:
        par.setUnits(str(units))
    if constant is not None:
        par.setConstant(bool(constant))
    if value is not None:
        par.setValue(float(value))
    if isinstance(name, str):
        par.setName(name)

    return par


def add_assignment_rule(
    model: libsbml.Model,
    variable_id: SbmlID,
    formula,
    rule_id: str | None = None,
) -> libsbml.AssignmentRule:
    """Helper for adding an assignment rule to a SBML model.

    :param model:
        SBML model to which the assignment rule is to be added.

    :param variable_id:
        SBML ID of the quantity for which the assignment rule is to be added.

    :param formula:
        Formula for the assignment rule (it will be sympified).

    :param rule_id:
        SBML ID of the new assignment rule.
        Defaults to `'assignment_' + variableId`.

    :return:
        The assignment rule as a :py:class:`libsbml.AssignmentRule` object.
    """
    variable_id = str(variable_id)
    if rule_id is None:
        rule_id = "assignment_" + variable_id

    # Check whether rules exists for this parameter or with the same name
    if model.getRuleByVariable(variable_id):
        raise SbmlDuplicateComponentIdError(
            f"A rule for parameter {variable_id} has already been defined."
        )
    if model.getElementBySId(rule_id):
        raise SbmlDuplicateComponentIdError(
            f"An element with SBML ID {rule_id} has already been defined."
        )

    rule = model.createAssignmentRule()
    if rule.setVariable(variable_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{variable_id} is not a valid SBML ID.")
    if rule.setIdAttribute(rule_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{rule_id} is not a valid SBML ID.")
    set_sbml_math(rule, formula)

    return rule


def add_rate_rule(
    model: libsbml.Model,
    variable_id: SbmlID,
    formula,
    rule_id: str | None = None,
) -> libsbml.RateRule:
    """
    Helper for adding a rate rule to a SBML model.

    :param model:
        SBML model to which the rate rule is to be added.

    :param variable_id:
        SBML ID of the quantity for which the rate rule is to be added.

    :param formula:
        Formula for the rate rule (it will be sympified).

    :param rule_id:
        SBML ID of the new rate rule.
        Defaults to `'rate_' + variableId`.

    :return:
        The new rate rule as a :py:class:`libsbml.RateRule` object.
    """
    variable_id = str(variable_id)
    if rule_id is None:
        rule_id = "rate_" + variable_id

    # Check whether rules exists for this parameter or with the same name
    if model.getRuleByVariable(variable_id):
        raise SbmlDuplicateComponentIdError(
            f"A rule for parameter {variable_id} has already been defined."
        )
    if model.getElementBySId(rule_id):
        raise SbmlDuplicateComponentIdError(
            f"An element with SBML ID {rule_id} has already been defined."
        )

    rule = model.createRateRule()
    if rule.setVariable(variable_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{variable_id} is not a valid SBML ID.")
    if rule.setIdAttribute(rule_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{rule_id} is not a valid SBML ID.")
    set_sbml_math(rule, formula)

    return rule


def add_inflow(
    model: libsbml.Model,
    species_id: SbmlID,
    rate,
    *,
    reaction_id: str | None = None,
    reversible: bool = False,
) -> libsbml.Reaction:
    species_id = str(species_id)
    if reaction_id is None:
        reaction_id = f"inflow_of_{species_id}"

    if model.getElementBySId(reaction_id):
        raise SbmlDuplicateComponentIdError(
            f"An element with SBML ID {reaction_id} has already been defined."
        )

    reaction = model.createReaction()
    if reaction.setId(reaction_id) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlInvalidIdSyntax(f"{reaction_id} is not a valid SBML ID.")
    reaction.setReversible(reversible)

    spr = reaction.createProduct()
    spr.setSpecies(species_id)

    kl = reaction.createKineticLaw()
    compartment_id = model.getSpecies(species_id).getCompartment()
    set_sbml_math(kl, sp.Symbol(compartment_id) * rate)

    return reaction


def get_sbml_units(model: libsbml.Model, x: SbmlID | sp.Basic) -> None | str:
    """Try to get the units for expression `x`.

    :param model:
        SBML model.
    :param x:
        Expression to get the units of.
    :return:
        A string if the units could be determined, otherwise `None`.
    """
    # TODO can the SBML unit inference machinery be used?
    x = sp.sympify(x)
    if not x.is_Symbol:
        return None
    if x.name == sbml_time_symbol.name:
        if model.isSetTimeUnits():
            return model.getTimeUnits()
        return None
    par = model.getParameter(x.name)
    if par is None:
        return None
    units = par.getUnits()
    if units == "":
        return None
    return units


def pretty_xml(ugly_xml: str) -> str:
    "Prettifies an XML document (given as a string)."
    dom = xml.dom.minidom.parseString(ugly_xml)
    pretty_xml = dom.toprettyxml()
    # We must delete the first line (xml header)
    return pretty_xml[pretty_xml.index("\n") + 1 :]


class MathMLSbmlPrinter(MathMLContentPrinter):
    """Prints a SymPy expression to a MathML expression parsable by libSBML.

    Differences from `sympy.MathMLContentPrinter`:
    1. underscores in symbol names are not converted to subscripts
    2. symbols with name 'time' are converted to the SBML time symbol
    """

    def _print_Symbol(self, sym: sp.Symbol) -> xml.dom.minidom.Element:
        ci = self.dom.createElement(self.mathml_tag(sym))
        ci.appendChild(self.dom.createTextNode(sym.name))
        return ci

    def doprint(self, expr, *, pretty: bool = False) -> str:
        mathml = '<math xmlns="http://www.w3.org/1998/Math/MathML">'
        mathml += super().doprint(expr)
        mathml += "</math>"
        mathml = mathml.replace(
            "<ci>time</ci>",
            '<csymbol encoding="text" definitionURL='
            '"http://www.sbml.org/sbml/symbols/time"> time </csymbol>',
        )
        return pretty_xml(mathml) if pretty else mathml


def sbml_mathml(
    expr, *, replace_time: bool = False, pretty: bool = False, **settings
) -> str:
    """Prints a SymPy expression to a MathML expression parsable by libSBML.

    :param expr:
        expression to be converted to MathML (will be sympified).

    :param replace_time:
        replace the AMICI time symbol with the SBML time symbol.

    :param pretty:
        prettify the resulting MathML.
    """
    with evaluate(False):
        expr = sp.sympify(expr)
        if replace_time:
            expr = expr.subs(amici_time_symbol, sbml_time_symbol)
    return MathMLSbmlPrinter(settings).doprint(expr, pretty=pretty)


def sbml_math_ast(expr, **kwargs) -> libsbml.ASTNode:
    """Convert a SymPy expression to SBML math AST.

    :param expr:
        expression to be converted (will be sympified).

    :param kwargs:
        extra options for MathML conversion.
    """
    mathml = sbml_mathml(expr, **kwargs)
    ast = libsbml.readMathMLFromString(mathml)
    if ast is None:
        raise SbmlMathError(
            f"error while converting the following expression to SBML AST.\n"
            f"expression:\n{expr}\n"
            f"MathML:\n{pretty_xml(mathml)}"
        )
    return ast


def set_sbml_math(obj: libsbml.SBase, expr, **kwargs) -> None:
    """Set the math attribute of a SBML node using a SymPy expression.

    :param obj:
        SBML node supporting `setMath` method.

    :param expr:
        expression to which the math attribute of `obj` should be se to
        (will be sympified).

    :param kwargs:
        extra options for MathML conversion.
    """
    mathml = sbml_math_ast(expr, **kwargs)
    if obj.setMath(mathml) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise SbmlMathError(
            f"Could not set math attribute of SBML object {obj}\n"
            f"expression:\n{expr}\n"
            f"MathML:\n{pretty_xml(mathml)}"
        )


def mathml2sympy(
    mathml: str,
    *,
    evaluate: bool = False,
    locals: dict[str, Any] | None = None,
    expression_type: str = "mathml2sympy",
) -> sp.Basic:
    ast = libsbml.readMathMLFromString(mathml)
    if ast is None:
        raise ValueError(
            f"libSBML could not parse MathML string:\n{pretty_xml(mathml)}"
        )

    formula = _parse_logical_operators(libsbml.formulaToL3String(ast))

    if evaluate:
        expr = sp.sympify(formula, locals=locals)
    else:
        with sp.core.parameters.evaluate(False):
            expr = sp.sympify(formula, locals=locals)

    expr = _parse_special_functions(expr)

    if expression_type is not None:
        _check_unsupported_functions(expr, expression_type)

    return expr


def _parse_logical_operators(
    math_str: str | float | None,
) -> str | float | None:
    """
    Parses a math string in order to replace logical operators by a form
    parsable for sympy

    :param math_str:
        str with mathematical expression
    :param math_str:
        parsed math_str
    """
    if not isinstance(math_str, str):
        return math_str

    if " xor(" in math_str or " Xor(" in math_str:
        raise SBMLException(
            "Xor is currently not supported as logical " "operation."
        )

    return (math_str.replace("&&", "&")).replace("||", "|")
