"""
Plotting
--------
Plotting related functions
"""
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from . import Model, ReturnDataView
from .numpy import StrOrExpr, evaluate


def plot_state_trajectories(
    rdata: ReturnDataView,
    state_indices: Optional[Iterable[int]] = None,
    ax: Optional[Axes] = None,
    model: Model = None,
    prefer_names: bool = True,
) -> None:
    """
    Plot state trajectories

    :param rdata:
        AMICI simulation results as returned by
        :func:`amici.amici.runAmiciSimulation`

    :param state_indices:
        Indices of states for which trajectories are to be plotted

    :param ax:
        matplotlib Axes instance to plot into

    :param model:
        amici model instance

    :param prefer_names:
        Whether state names should be preferred over IDs, if available.
    """
    if not ax:
        fig, ax = plt.subplots()
    if not state_indices:
        state_indices = range(rdata["x"].shape[1])
    for ix in state_indices:
        if model is None:
            label = f"$x_{{{ix}}}$"
        elif prefer_names and model.getStateNames()[ix]:
            label = model.getStateNames()[ix]
        else:
            label = model.getStateIds()[ix]
        ax.plot(rdata["t"], rdata["x"][:, ix], label=label)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x(t)$")
        ax.legend()
        ax.set_title("State trajectories")


def plot_observable_trajectories(
    rdata: ReturnDataView,
    observable_indices: Optional[Iterable[int]] = None,
    ax: Optional[Axes] = None,
    model: Model = None,
    prefer_names: bool = True,
) -> None:
    """
    Plot observable trajectories

    :param rdata:
        AMICI simulation results as returned by
        :func:`amici.amici.runAmiciSimulation`

    :param observable_indices:
        Indices of observables for which trajectories are to be plotted

    :param ax:
        matplotlib Axes instance to plot into

    :param model:
        amici model instance

    :param prefer_names:
        Whether observables names should be preferred over IDs, if available.
    """
    if not ax:
        fig, ax = plt.subplots()
    if not observable_indices:
        observable_indices = range(rdata["y"].shape[1])
    for iy in observable_indices:
        if model is None:
            label = f"$y_{{{iy}}}$"
        elif prefer_names and model.getObservableNames()[iy]:
            label = model.getObservableNames()[iy]
        else:
            label = model.getObservableIds()[iy]
        ax.plot(rdata["t"], rdata["y"][:, iy], label=label)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y(t)$")
        ax.legend()
        ax.set_title("Observable trajectories")


def plot_jacobian(rdata: ReturnDataView):
    """Plot Jacobian as heatmap."""
    df = pd.DataFrame(
        data=rdata.J,
        index=rdata._swigptr.state_ids_solver,
        columns=rdata._swigptr.state_ids_solver,
    )
    sns.heatmap(df, center=0.0)
    plt.title("Jacobian")


# backwards compatibility
plotStateTrajectories = plot_state_trajectories
plotObservableTrajectories = plot_observable_trajectories


def plot_expressions(
    exprs: Union[Sequence[StrOrExpr], StrOrExpr], rdata: ReturnDataView
) -> None:
    """Plot the given expressions evaluated on the given simulation outputs.

    :param exprs:
        A symbolic expression, e.g. a sympy expression or a string that can be sympified.
        Can include state variable, expression, and observable IDs, depending on whether
        the respective data is available in the simulation results.
        Parameters are not yet supported.
    :param rdata:
        The simulation results.
    """
    if not isinstance(exprs, Sequence) or isinstance(exprs, str):
        exprs = [exprs]

    for expr in exprs:
        plt.plot(rdata.t, evaluate(expr, rdata), label=str(expr))

    plt.legend()
    plt.gca().set_xlabel("$t$")
