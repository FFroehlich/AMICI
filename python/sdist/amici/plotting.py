"""
Plotting
--------
Plotting related functions
"""
from typing import Optional, Union
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from . import Model, ReturnDataView
from .numpy import StrOrExpr, evaluate


def plot_state_trajectories(
    rdata: ReturnDataView,
    state_indices: Optional[Sequence[int]] = None,
    ax: Optional[Axes] = None,
    model: Model = None,
    prefer_names: bool = True,
    marker=None,
) -> None:
    """
    Plot state trajectories.

    :param rdata:
        AMICI simulation results as returned by
        :func:`amici.amici.runAmiciSimulation`.
    :param state_indices:
        Indices of state variables for which trajectories are to be plotted.
    :param ax:
        :class:`matplotlib.pyplot.Axes` instance to plot into.
    :param model:
        The model *rdata* was generated from.
    :param prefer_names:
        Whether state names should be preferred over IDs, if available.
    :param marker:
        Point marker for plotting (see
        `matplotlib documentation <https://matplotlib.org/stable/api/markers_api.html>`_).
    """
    if not ax:
        fig, ax = plt.subplots()
    if not state_indices:
        state_indices = range(rdata["x"].shape[1])

    if marker is None:
        # Show marker if only one time point is available,
        #  otherwise nothing will be shown
        marker = "o" if len(rdata.t) == 1 else None

    if model is None and rdata.ptr.state_ids is None:
        labels = [f"$x_{{{ix}}}$" for ix in state_indices]
    elif model is not None and prefer_names:
        labels = np.asarray(model.getStateNames())[list(state_indices)]
        labels = [
            l if l else model.getStateIds()[ix] for ix, l in enumerate(labels)
        ]
    elif model is not None:
        labels = np.asarray(model.getStateIds())[list(state_indices)]
    else:
        labels = np.asarray(rdata.ptr.state_ids)[list(state_indices)]

    for ix, label in zip(state_indices, labels):
        ax.plot(rdata["t"], rdata["x"][:, ix], marker=marker, label=label)
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
    marker=None,
) -> None:
    """
    Plot observable trajectories.

    :param rdata:
        AMICI simulation results as returned by
        :func:`amici.amici.runAmiciSimulation`.
    :param observable_indices:
        Indices of observables for which trajectories are to be plotted.
    :param ax:
        :class:`matplotlib.pyplot.Axes` instance to plot into.
    :param model:
        The model *rdata* was generated from.
    :param prefer_names:
        Whether observable names should be preferred over IDs, if available.
    :param marker:
        Point marker for plotting (see
        `matplotlib documentation <https://matplotlib.org/stable/api/markers_api.html>`_).

    """
    if not ax:
        fig, ax = plt.subplots()
    if not observable_indices:
        observable_indices = range(rdata["y"].shape[1])

    if marker is None:
        # Show marker if only one time point is available,
        #  otherwise nothing will be shown
        marker = "o" if len(rdata.t) == 1 else None

    if model is None and rdata.ptr.observable_ids is None:
        labels = [f"$y_{{{iy}}}$" for iy in observable_indices]
    elif model is not None and prefer_names:
        labels = np.asarray(model.getObservableNames())[
            list(observable_indices)
        ]
        labels = [
            l if l else model.getObservableIds()[ix]
            for ix, l in enumerate(labels)
        ]
    elif model is not None:
        labels = np.asarray(model.getObservableIds())[list(observable_indices)]
    else:
        labels = np.asarray(rdata.ptr.observable_ids)[list(observable_indices)]

    for iy, label in zip(observable_indices, labels):
        ax.plot(rdata["t"], rdata["y"][:, iy], marker=marker, label=label)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y(t)$")
        ax.legend()
        ax.set_title("Observable trajectories")


def plot_jacobian(rdata: ReturnDataView):
    """Plot Jacobian as heatmap."""
    df = pd.DataFrame(
        data=rdata.J,
        index=rdata.ptr.state_ids_solver,
        columns=rdata.ptr.state_ids_solver,
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
        A symbolic expression, e.g., a sympy expression or a string that can be
        sympified. It Can include state variable, expression, and
        observable IDs, depending on whether the respective data is available
        in the simulation results. Parameters are not yet supported.
    :param rdata:
        The simulation results.
    """
    if not isinstance(exprs, Sequence) or isinstance(exprs, str):
        exprs = [exprs]

    for expr in exprs:
        plt.plot(rdata.t, evaluate(expr, rdata), label=str(expr))

    plt.legend()
    plt.gca().set_xlabel("$t$")
