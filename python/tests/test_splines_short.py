import numpy as np
import sympy as sp
from amici.sbml_utils import amici_time_symbol
from amici.splines import CubicHermiteSpline, UniformGrid
from splines_utils import (
    example_spline_1,
    check_splines_full,
)
    

def test_spline_piecewise(**kwargs):
    spline, params, tols = example_spline_1()
    check_splines_full(spline, params, tols, **kwargs)


def test_two_splines(**kwargs):
    spline0, params0, tols0 = example_spline_1(
        0, num_nodes=4, fixed_values=[0, 2], extrapolate='linear'
    )
    spline1, params1, tols1 = example_spline_1(
        1, num_nodes=5, scale=1.5, offset=5, extrapolate='linear'
    )

    splines = [spline0, spline1]

    params = dict(params0)
    params.update(params1)

    if isinstance(tols0, dict):
        tols0 = (tols0, tols0, tols0)
    if isinstance(tols1, dict):
        tols1 = (tols1, tols1, tols1)

    tols = []
    for (t0, t1) in zip(tols0, tols1):
        keys = set().union(t0.keys(), t1.keys())
        t = {
            key: max(
                t0.get(key, 0.0),
                t1.get(key, 0.0),
            ) for key in keys
        }
        tols.append(t)

    tols[1]['x_rtol']   = max(1e-9, tols[1].get('x_rtol', -np.inf))
    tols[1]['x_atol']   = max(5e-9,  tols[1].get('x_atol', -np.inf))
    tols[1]['sx_rtol']  = max(1e-5,  tols[1].get('llh_rtol', -np.inf))
    tols[1]['sx_atol']  = max(5e-9, tols[1].get('sx_atol', -np.inf))
    tols[1]['llh_rtol'] = max(5e-14, tols[1].get('llh_rtol', -np.inf))
    tols[1]['sllh_atol'] = max(5e-5,  tols[1].get('sllh_atol', -np.inf))

    tols[2]['x_rtol']    = max(5e-10, tols[2].get('x_rtol', -np.inf))
    tols[2]['x_atol']    = max(1e-8,  tols[2].get('x_atol', -np.inf))
    tols[2]['llh_rtol']  = max(5e-14, tols[2].get('llh_rtol', -np.inf))
    tols[2]['sllh_atol'] = max(5e-5,  tols[2].get('sllh_atol', -np.inf))

    check_splines_full(splines, params, tols, check_piecewise=False, **kwargs)


def test_splines_plist():
    # Dummy spline #1
    xx = UniformGrid(0, 5, length=3)
    yy = np.asarray([0.0, 1.0, 0.5])
    spline1 = CubicHermiteSpline(
        f'y1', amici_time_symbol,
        xx, yy,
        bc='auto', extrapolate=(None, 'constant'),
    )
    # Dummy spline #2
    xx = UniformGrid(0, 5, length=4)
    yy = np.asarray([0.0, 0.5, -0.5, 0.5])
    spline2 = CubicHermiteSpline(
        f'y2', amici_time_symbol,
        xx, yy,
        bc='auto', extrapolate=(None, 'constant'),
    )
    # Real spline #3
    xx = UniformGrid(0, 5, length=6)
    p1, p2, p3, p4, p5 = sp.symbols('p1 p2 p3 p4 p5')
    yy = np.asarray([p1 + p2, p2 * p3, p4, sp.cos(p1 + p3), p4 * sp.log(p1), p3])
    dd = np.asarray([-0.75, -0.875, p5, 0.125, 1.15057181, 0.0])
    params = {
        p1: 1.0, p2: 0.5, p3: 1.5, p4: -0.25, p5: -0.5
    }
    # print([y.subs(params).evalf() for y in yy])
    spline3 = CubicHermiteSpline(
        f'y3', amici_time_symbol,
        xx, yy, dd,
        bc='auto', extrapolate=(None, 'constant'),
    )
    # Dummy spline 4
    xx = UniformGrid(0, 5, length=3)
    yy = np.asarray([0.0, -0.5, 0.5])
    spline4 = CubicHermiteSpline(
        f'y4', amici_time_symbol,
        xx, yy,
        bc='auto', extrapolate=(None, 'constant'),
    )
    tols = dict(
        x_rtol=1e-6,
        x_atol=1e-11,
        sx_rtol=1e-6,
        sx_atol=5e-11,
        llh_rtol=1e-14,
        sllh_atol=5e-9,
    )
    check_splines_full(
        [spline1, spline2, spline3, spline4], params, tols,
        check_piecewise=False,
        check_forward=False,
        check_adjoint=True, # plist cannot be checked, but complex parameter dependence can
        parameter_lists=[[0, 1, 4], [2, 3]],
    )
    # Debug
    # # res = check_splines(
    #     [spline1, spline2, spline3, spline4], params,
    #     use_adjoint=False,
    #     parameter_lists=[[0, 1, 4], [2, 3]],
    #     #folder='debug',
    #     #debug='print',
    # )
