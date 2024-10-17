"""Info about C++ functions in the generated model code."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class _FunctionInfo:
    """Information on a model-specific generated C++ function

    :ivar ode_arguments: argument list of the ODE function.
        input variables should be ``const``.
    :ivar dae_arguments: argument list of the DAE function, if different from
        ODE function. input variables should be ``const``.
    :ivar return_type: the return type of the function
    :ivar assume_pow_positivity:
        identifies the functions on which ``assume_pow_positivity`` will have
        an effect when specified during model generation. generally these are
        functions that are used for solving the ODE, where negative values may
        negatively affect convergence of the integration algorithm
    :ivar sparse:
        specifies whether the result of this function will be stored in sparse
        format. sparse format means that the function will only return an
        array of nonzero values and not a full matrix.
    :ivar generate_body:
        indicates whether a model-specific implementation is to be generated
    :ivar body:
        the actual function body. will be filled later
    """

    ode_arguments: str = ""
    dae_arguments: str = ""
    return_type: str = "void"
    assume_pow_positivity: bool = False
    sparse: bool = False
    generate_body: bool = True
    body: str = ""

    def arguments(self, ode: bool = True) -> str:
        """Get the arguments for the ODE or DAE function"""
        if ode or not self.dae_arguments:
            return self.ode_arguments
        return self.dae_arguments


# Information on a model-specific generated C++ function
# prototype for generated C++ functions, keys are the names of functions
functions = {
    "Jy": _FunctionInfo(
        "realtype *Jy, const int iy, const realtype *p, "
        "const realtype *k, const realtype *y, const realtype *sigmay, "
        "const realtype *my"
    ),
    "dJydsigma": _FunctionInfo(
        "realtype *dJydsigma, const int iy, const realtype *p, "
        "const realtype *k, const realtype *y, const realtype *sigmay, "
        "const realtype *my"
    ),
    "dJydy": _FunctionInfo(
        "realtype *dJydy, const int iy, const realtype *p, "
        "const realtype *k, const realtype *y, "
        "const realtype *sigmay, const realtype *my",
        sparse=True,
    ),
    "Jz": _FunctionInfo(
        "realtype *Jz, const int iz, const realtype *p, const realtype *k, "
        "const realtype *z, const realtype *sigmaz, const realtype *mz"
    ),
    "dJzdsigma": _FunctionInfo(
        "realtype *dJzdsigma, const int iz, const realtype *p, "
        "const realtype *k, const realtype *z, const realtype *sigmaz, "
        "const realtype *mz"
    ),
    "dJzdz": _FunctionInfo(
        "realtype *dJzdz, const int iz, const realtype *p, "
        "const realtype *k, const realtype *z, const realtype *sigmaz, "
        "const double *mz",
    ),
    "Jrz": _FunctionInfo(
        "realtype *Jrz, const int iz, const realtype *p, "
        "const realtype *k, const realtype *rz, const realtype *sigmaz"
    ),
    "dJrzdsigma": _FunctionInfo(
        "realtype *dJrzdsigma, const int iz, const realtype *p, "
        "const realtype *k, const realtype *rz, const realtype *sigmaz"
    ),
    "dJrzdz": _FunctionInfo(
        "realtype *dJrzdz, const int iz, const realtype *p, "
        "const realtype *k, const realtype *rz, const realtype *sigmaz",
    ),
    "root": _FunctionInfo(
        "realtype *root, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *tcl"
    ),
    "dwdp": _FunctionInfo(
        "realtype *dwdp, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w, const realtype *tcl, const realtype *dtcldp, "
        "const realtype *spl, const realtype *sspl, bool include_static",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "dwdx": _FunctionInfo(
        "realtype *dwdx, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w, const realtype *tcl, const realtype *spl, "
        "bool include_static",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "create_splines": _FunctionInfo(
        "const realtype *p, const realtype *k",
        return_type="std::vector<HermiteSpline>",
    ),
    "spl": _FunctionInfo(generate_body=False),
    "sspl": _FunctionInfo(generate_body=False),
    "spline_values": _FunctionInfo(
        "const realtype *p, const realtype *k", generate_body=False
    ),
    "spline_slopes": _FunctionInfo(
        "const realtype *p, const realtype *k", generate_body=False
    ),
    "dspline_valuesdp": _FunctionInfo(
        "realtype *dspline_valuesdp, const realtype *p, const realtype *k, "
        "const int ip"
    ),
    "dspline_slopesdp": _FunctionInfo(
        "realtype *dspline_slopesdp, const realtype *p, const realtype *k, "
        "const int ip"
    ),
    "dwdw": _FunctionInfo(
        "realtype *dwdw, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w, const realtype *tcl, bool include_static",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "dxdotdw": _FunctionInfo(
        "realtype *dxdotdw, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w",
        "realtype *dxdotdw, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *dx, const realtype *w",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "dxdotdx_explicit": _FunctionInfo(
        "realtype *dxdotdx_explicit, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *w",
        "realtype *dxdotdx_explicit, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *dx, const realtype *w",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "dxdotdp_explicit": _FunctionInfo(
        "realtype *dxdotdp_explicit, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *w",
        "realtype *dxdotdp_explicit, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *dx, const realtype *w",
        assume_pow_positivity=True,
        sparse=True,
    ),
    "dydx": _FunctionInfo(
        "realtype *dydx, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w, const realtype *dwdx",
    ),
    "dydp": _FunctionInfo(
        "realtype *dydp, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const int ip, const realtype *w, const realtype *tcl, "
        "const realtype *dtcldp, const realtype *spl, const realtype *sspl"
    ),
    "dzdx": _FunctionInfo(
        "realtype *dzdx, const int ie, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h",
    ),
    "dzdp": _FunctionInfo(
        "realtype *dzdp, const int ie, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const int ip",
    ),
    "drzdx": _FunctionInfo(
        "realtype *drzdx, const int ie, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h",
    ),
    "drzdp": _FunctionInfo(
        "realtype *drzdp, const int ie, const realtype t, "
        "const realtype *x, const realtype *p, const realtype *k, "
        "const realtype *h, const int ip",
    ),
    "dsigmaydy": _FunctionInfo(
        "realtype *dsigmaydy, const realtype t, const realtype *p, "
        "const realtype *k, const realtype *y"
    ),
    "dsigmaydp": _FunctionInfo(
        "realtype *dsigmaydp, const realtype t, const realtype *p, "
        "const realtype *k, const realtype *y, const int ip",
    ),
    "sigmay": _FunctionInfo(
        "realtype *sigmay, const realtype t, const realtype *p, "
        "const realtype *k, const realtype *y",
    ),
    "dsigmazdp": _FunctionInfo(
        "realtype *dsigmazdp, const realtype t, const realtype *p,"
        " const realtype *k, const int ip",
    ),
    "sigmaz": _FunctionInfo(
        "realtype *sigmaz, const realtype t, const realtype *p, "
        "const realtype *k",
    ),
    "sroot": _FunctionInfo(
        "realtype *stau, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *sx, const int ip, const int ie, "
        "const realtype *tcl",
        generate_body=False,
    ),
    "drootdt": _FunctionInfo(generate_body=False),
    "drootdt_total": _FunctionInfo(generate_body=False),
    "drootdp": _FunctionInfo(generate_body=False),
    "drootdx": _FunctionInfo(generate_body=False),
    "stau": _FunctionInfo(
        "realtype *stau, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *tcl, const realtype *sx, const int ip, "
        "const int ie"
    ),
    "deltax": _FunctionInfo(
        "double *deltax, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const int ie, const realtype *xdot, const realtype *xdot_old"
    ),
    "ddeltaxdx": _FunctionInfo(generate_body=False),
    "ddeltaxdt": _FunctionInfo(generate_body=False),
    "ddeltaxdp": _FunctionInfo(generate_body=False),
    "deltasx": _FunctionInfo(
        "realtype *deltasx, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w, const int ip, const int ie, "
        "const realtype *xdot, const realtype *xdot_old, "
        "const realtype *sx, const realtype *stau, const realtype *tcl"
    ),
    "deltaxB": _FunctionInfo(
        "realtype *deltaxB, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const int ie, const realtype *xdot, const realtype *xdot_old, "
        "const realtype *xB, const realtype *xBdot, const realtype *tcl"
    ),
    "deltaqB": _FunctionInfo(
        "realtype *deltaqB, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const int ip, const int ie, const realtype *xdot, "
        "const realtype *xdot_old, const realtype *xB, "
        "const realtype *xBdot"
    ),
    "w": _FunctionInfo(
        "realtype *w, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *tcl, const realtype *spl, "
        "bool include_static",
        assume_pow_positivity=True,
    ),
    "x0": _FunctionInfo(
        "realtype *x0, const realtype t, const realtype *p, "
        "const realtype *k"
    ),
    "x0_fixedParameters": _FunctionInfo(
        "realtype *x0_fixedParameters, const realtype t, "
        "const realtype *p, const realtype *k, "
        "gsl::span<const int> reinitialization_state_idxs",
    ),
    "sx0": _FunctionInfo(
        "realtype *sx0, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const int ip",
    ),
    "sx0_fixedParameters": _FunctionInfo(
        "realtype *sx0_fixedParameters, const realtype t, "
        "const realtype *x0, const realtype *p, const realtype *k, "
        "const int ip, gsl::span<const int> reinitialization_state_idxs",
    ),
    "xdot": _FunctionInfo(
        "realtype *xdot, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *w",
        "realtype *xdot, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h, "
        "const realtype *dx, const realtype *w",
        assume_pow_positivity=True,
    ),
    "xdot_old": _FunctionInfo(generate_body=False),
    "y": _FunctionInfo(
        "realtype *y, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, "
        "const realtype *h, const realtype *w",
    ),
    "x_rdata": _FunctionInfo(
        "realtype *x_rdata, const realtype *x, const realtype *tcl, "
        "const realtype *p, const realtype *k"
    ),
    "total_cl": _FunctionInfo(
        "realtype *total_cl, const realtype *x_rdata, "
        "const realtype *p, const realtype *k"
    ),
    "dtotal_cldp": _FunctionInfo(
        "realtype *dtotal_cldp, const realtype *x_rdata, "
        "const realtype *p, const realtype *k, const int ip"
    ),
    "dtotal_cldx_rdata": _FunctionInfo(
        "realtype *dtotal_cldx_rdata, const realtype *x_rdata, "
        "const realtype *p, const realtype *k, const realtype *tcl",
        sparse=True,
    ),
    "x_solver": _FunctionInfo("realtype *x_solver, const realtype *x_rdata"),
    "dx_rdatadx_solver": _FunctionInfo(
        "realtype *dx_rdatadx_solver, const realtype *x, "
        "const realtype *tcl, const realtype *p, const realtype *k",
        sparse=True,
    ),
    "dx_rdatadp": _FunctionInfo(
        "realtype *dx_rdatadp, const realtype *x, "
        "const realtype *tcl, const realtype *p, const realtype *k, "
        "const int ip"
    ),
    "dx_rdatadtcl": _FunctionInfo(
        "realtype *dx_rdatadtcl, const realtype *x, "
        "const realtype *tcl, const realtype *p, const realtype *k",
        sparse=True,
    ),
    "z": _FunctionInfo(
        "realtype *z, const int ie, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h"
    ),
    "rz": _FunctionInfo(
        "realtype *rz, const int ie, const realtype t, const realtype *x, "
        "const realtype *p, const realtype *k, const realtype *h"
    ),
}

#: list of sparse functions
sparse_functions = [
    func_name for func_name, func_info in functions.items() if func_info.sparse
]

#: list of nobody functions
nobody_functions = [
    func_name
    for func_name, func_info in functions.items()
    if not func_info.generate_body
]

#: list of sensitivity functions
sensi_functions = [
    func_name
    for func_name, func_info in functions.items()
    if "const int ip" in func_info.arguments()
]

#: list of sparse sensitivity functions
sparse_sensi_functions = [
    func_name
    for func_name, func_info in functions.items()
    if "const int ip" not in func_info.arguments()
    and func_name.endswith("dp")
    or func_name.endswith("dp_explicit")
]

#: list of event functions
event_functions = [
    func_name
    for func_name, func_info in functions.items()
    if "const int ie" in func_info.arguments()
    and "const int ip" not in func_info.arguments()
]

#: list of event sensitivity functions
event_sensi_functions = [
    func_name
    for func_name, func_info in functions.items()
    if "const int ie" in func_info.arguments()
    and "const int ip" in func_info.arguments()
]

#: list of multiobs functions
multiobs_functions = [
    func_name
    for func_name, func_info in functions.items()
    if "const int iy" in func_info.arguments()
    or "const int iz" in func_info.arguments()
]


def var_in_function_signature(name: str, varname: str, ode: bool) -> bool:
    """
    Checks if the values for a symbolic variable is passed in the signature
    of a function

    :param name:
        name of the function
    :param varname:
        name of the symbolic variable
    :param ode:
        whether to check the ODE or DAE signature

    :return:
        boolean indicating whether the variable occurs in the function
        signature
    """
    return name in functions and re.search(
        rf"const (realtype|double) \*{varname}[0]*(,|$)+",
        functions[name].arguments(ode=ode),
    )
