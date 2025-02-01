"""Model simulation using JAX."""

# ruff: noqa: F821 F722

from abc import abstractmethod
from pathlib import Path
import enum

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax
import jaxtyping as jt

from collections.abc import Callable


class ReturnValue(enum.Enum):
    llh = "log-likelihood"
    nllhs = "pointwise negative log-likelihood"
    x0 = "full initial state vector"
    x0_solver = "reduced initial state vector"
    x = "full state vector"
    x_solver = "reduced state vector"
    y = "observables"
    sigmay = "standard deviations of the observables"
    tcl = "total values for conservation laws"
    res = "residuals"
    chi2 = "sum(((observed - simulated) / sigma ) ** 2)"


class JAXModel(eqx.Module):
    """
    JAXModel provides an abstract base class for a JAX-based implementation of an AMICI model. The class implements
    routines for simulation and evaluation of derived quantities, model specific implementations need to be provided by
    classes inheriting from JAXModel.

    :ivar api_version:
        API version of the derived class. Needs to match the API version of the base class (MODEL_API_VERSION).
    :ivar MODEL_API_VERSION:
        API version of the base class.
    :ivar jax_py_file:
        Path to the JAX model file.
    """

    MODEL_API_VERSION = "0.0.3"
    api_version: str
    jax_py_file: Path

    def __init__(self):
        if self.api_version != self.MODEL_API_VERSION:
            raise ValueError(
                "JAXModel API version mismatch, please regenerate the model class."
            )
        super().__init__()

    @abstractmethod
    def _xdot(
        self,
        t: jnp.float_,
        x: jt.Float[jt.Array, "nxs"],
        args: tuple[jt.Float[jt.Array, "np"], jt.Float[jt.Array, "ncl"]],
    ) -> jt.Float[jt.Array, "nxs"]:
        """
        Right-hand side of the ODE system.

        :param t: time point
        :param x: state vector
        :param args: tuple of parameters and total values for conservation laws
        :return:
            Temporal derivative of the state vector x at time point t.
        """
        ...

    @abstractmethod
    def _w(
        self,
        t: jt.Float[jt.Array, ""],
        x: jt.Float[jt.Array, "nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
    ) -> jt.Float[jt.Array, "nw"]:
        """
        Compute the expressions, i.e. derived quantities that are used in other parts of the model.

        :param t: time point
        :param x: state vector
        :param p: parameters
        :param tcl: total values for conservation laws
        :return:
            Expression values.
        """
        ...

    @abstractmethod
    def _x0(self, p: jt.Float[jt.Array, "np"]) -> jt.Float[jt.Array, "nx"]:
        """
        Compute the initial state vector.

        :param p: parameters
        """
        ...

    @abstractmethod
    def _x_solver(
        self, x: jt.Float[jt.Array, "nx"]
    ) -> jt.Float[jt.Array, "nxs"]:
        """
        Transform the full state vector to the reduced state vector for ODE solving.

        :param x:
            full state vector
        :return:
            reduced state vector
        """
        ...

    @abstractmethod
    def _x_rdata(
        self, x: jt.Float[jt.Array, "nxs"], tcl: jt.Float[jt.Array, "ncl"]
    ) -> jt.Float[jt.Array, "nx"]:
        """
        Compute the full state vector from the reduced state vector and conservation laws.

        :param x:
            reduced state vector
        :param tcl:
            total values for conservation laws
        :return:
            full state vector
        """
        ...

    @abstractmethod
    def _tcl(
        self, x: jt.Float[jt.Array, "nx"], p: jt.Float[jt.Array, "np"]
    ) -> jt.Float[jt.Array, "ncl"]:
        """
        Compute the total values for conservation laws.

        :param x:
            state vector
        :param p:
            parameters
        :return:
            total values for conservation laws
        """
        ...

    @abstractmethod
    def _y(
        self,
        t: jt.Float[jt.Scalar, ""],
        x: jt.Float[jt.Array, "nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        op: jt.Float[jt.Array, "ny"],
    ) -> jt.Float[jt.Array, "ny"]:
        """
        Compute the observables.

        :param t:
            time point
        :param x:
            state vector
        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param op:
            observables parameters
        :return:
            observables
        """
        ...

    @abstractmethod
    def _sigmay(
        self,
        y: jt.Float[jt.Array, "ny"],
        p: jt.Float[jt.Array, "np"],
        np: jt.Float[jt.Array, "ny"],
    ) -> jt.Float[jt.Array, "ny"]:
        """
        Compute the standard deviations of the observables.

        :param y:
            observables
        :param p:
            parameters
        :param np:
            noise parameters
        :return:
            standard deviations of the observables
        """
        ...

    @abstractmethod
    def _nllh(
        self,
        t: jt.Float[jt.Scalar, ""],
        x: jt.Float[jt.Array, "nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        my: jt.Float[jt.Array, ""],
        iy: jt.Int[jt.Array, ""],
        op: jt.Float[jt.Array, "ny"],
        np: jt.Float[jt.Array, "ny"],
    ) -> jt.Float[jt.Scalar, ""]:
        """
        Compute the negative log-likelihood of the observable for the specified observable index.

        :param t:
            time point
        :param x:
            state vector
        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param my:
            observed data
        :param iy:
            observable index
        :param op:
            observables parameters
        :param np:
            noise parameters
        :return:
            log-likelihood of the observable
        """
        ...

    @property
    @abstractmethod
    def state_ids(self) -> list[str]:
        """
        Get the state ids of the model.

        :return:
            State ids
        """
        ...

    @property
    @abstractmethod
    def observable_ids(self) -> list[str]:
        """
        Get the observable ids of the model.

        :return:
            Observable ids
        """
        ...

    @property
    @abstractmethod
    def parameter_ids(self) -> list[str]:
        """
        Get the parameter ids of the model.

        :return:
            Parameter ids
        """
        ...

    def _eq(
        self,
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        x0: jt.Float[jt.Array, "nxs"],
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: jnp.int_,
    ) -> tuple[jt.Float[jt.Array, "1 nxs"], dict]:
        """
        Solve the steady state equation.

        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param x0:
            initial state vector
        :param solver:
            ODE solver
        :param controller:
            step size controller
        :param max_steps:
            maximum number of steps
        :return:
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._xdot),
            solver,
            args=(p, tcl),
            t0=0.0,
            t1=jnp.inf,
            dt0=None,
            y0=x0,
            stepsize_controller=controller,
            max_steps=max_steps,
            adjoint=diffrax.DirectAdjoint(),
            event=diffrax.Event(
                cond_fn=steady_state_event,
            ),
            throw=False,
        )
        # If the event was triggered, the event mask is True and the solution is the steady state. Otherwise, the
        # solution is the last state and the event mask is False. In the latter case, we return inf for the steady
        # state.
        ys = jnp.where(
            sol.event_mask,
            sol.ys[-1, :],
            jnp.inf * jnp.ones_like(sol.ys[-1, :]),
        )
        return ys, sol.stats

    def _solve(
        self,
        p: jt.Float[jt.Array, "np"],
        ts: jt.Float[jt.Array, "nt_dyn"],
        tcl: jt.Float[jt.Array, "ncl"],
        x0: jt.Float[jt.Array, "nxs"],
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        max_steps: jnp.int_,
        adjoint: diffrax.AbstractAdjoint,
    ) -> tuple[jt.Float[jt.Array, "nt nxs"], dict]:
        """
        Solve the ODE system.

        :param p:
            parameters
        :param ts:
            time points at which solutions are evaluated
        :param tcl:
            total values for conservation laws
        :param x0:
            initial state vector
        :param solver:
            ODE solver
        :param controller:
            step size controller
        :param max_steps:
            maximum number of steps
        :param adjoint:
            adjoint method
        :return:
            solution at time points ts and statistics
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._xdot),
            solver,
            args=(p, tcl),
            t0=0.0,
            t1=ts[-1],
            dt0=None,
            y0=x0,
            stepsize_controller=controller,
            max_steps=max_steps,
            adjoint=adjoint,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
        )
        return sol.ys, sol.stats

    def _x_rdatas(
        self, x: jt.Float[jt.Array, "nt nxs"], tcl: jt.Float[jt.Array, "ncl"]
    ) -> jt.Float[jt.Array, "nt nx"]:
        """
        Compute the full state vector from the reduced state vector and conservation laws.

        :param x:
            reduced state vector
        :param tcl:
            total values for conservation laws
        :return:
            full state vector
        """
        return jax.vmap(self._x_rdata, in_axes=(0, None))(x, tcl)

    def _nllhs(
        self,
        ts: jt.Float[jt.Array, "nt nx"],
        xs: jt.Float[jt.Array, "nt nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        mys: jt.Float[jt.Array, "nt"],
        iys: jt.Int[jt.Array, "nt"],
        ops: jt.Float[jt.Array, "nt *nop"],
        nps: jt.Float[jt.Array, "nt *nnp"],
    ) -> jt.Float[jt.Array, "nt"]:
        """
        Compute the negative log-likelihood for each observable.

        :param ts:
            time points
        :param xs:
            state vectors
        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param mys:
            observed data
        :param iys:
            observable indices
        :param ops:
            observables parameters
        :param nps:
            noise parameters
        :return:
            negative log-likelihoods of the observables
        """
        return jax.vmap(self._nllh, in_axes=(0, 0, None, None, 0, 0, 0, 0))(
            ts, xs, p, tcl, mys, iys, ops, nps
        )

    def _ys(
        self,
        ts: jt.Float[jt.Array, "nt"],
        xs: jt.Float[jt.Array, "nt nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        iys: jt.Float[jt.Array, "nt"],
        ops: jt.Float[jt.Array, "nt *nop"],
    ) -> jt.Int[jt.Array, "nt"]:
        """
        Compute the observables.

        :param ts:
            time points
        :param xs:
            state vectors
        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param iys:
            observable indices
        :param ops:
            observables parameters
        :return:
            observables
        """
        return jax.vmap(
            lambda t, x, p, tcl, iy, op: self._y(t, x, p, tcl, op)
            .at[iy]
            .get(),
            in_axes=(0, 0, None, None, 0, 0),
        )(ts, xs, p, tcl, iys, ops)

    def _sigmays(
        self,
        ts: jt.Float[jt.Array, "nt"],
        xs: jt.Float[jt.Array, "nt nxs"],
        p: jt.Float[jt.Array, "np"],
        tcl: jt.Float[jt.Array, "ncl"],
        iys: jt.Int[jt.Array, "nt"],
        ops: jt.Float[jt.Array, "nt *nop"],
        nps: jt.Float[jt.Array, "nt *nnp"],
    ):
        """
        Compute the standard deviations of the observables.

        :param ts:
            time points
        :param xs:
            state vectors
        :param p:
            parameters
        :param tcl:
            total values for conservation laws
        :param iys:
            observable indices
        :param ops:
            observables parameters
        :param nps:
            noise parameters
        :return:
            standard deviations of the observables
        """
        return jax.vmap(
            lambda t, x, p, tcl, iy, op, np: self._sigmay(
                self._y(t, x, p, tcl, op), p, np
            )
            .at[iy]
            .get(),
            in_axes=(0, 0, None, None, 0, 0, 0),
        )(ts, xs, p, tcl, iys, ops, nps)

    @eqx.filter_jit
    def simulate_condition(
        self,
        p: jt.Float[jt.Array, "np"],
        ts_dyn: jt.Float[jt.Array, "nt_dyn"],
        ts_posteq: jt.Float[jt.Array, "nt_posteq"],
        my: jt.Float[jt.Array, "nt"],
        iys: jt.Int[jt.Array, "nt"],
        iy_trafos: jt.Int[jt.Array, "nt"],
        ops: jt.Float[jt.Array, "nt *nop"],
        nps: jt.Float[jt.Array, "nt *nnp"],
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        adjoint: diffrax.AbstractAdjoint,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: int | jnp.int_,
        x_preeq: jt.Float[jt.Array, "*nx"] = jnp.array([]),
        mask_reinit: jt.Bool[jt.Array, "*nx"] = jnp.array([]),
        x_reinit: jt.Float[jt.Array, "*nx"] = jnp.array([]),
        ts_mask: jt.Bool[jt.Array, "nt"] = jnp.array([]),
        ret: ReturnValue = ReturnValue.llh,
    ) -> tuple[jt.Float[jt.Array, "nt *nx"] | jnp.float_, dict]:
        r"""
        Simulate a condition.

        :param p:
            parameters for simulation ordered according to ids in :ivar parameter_ids:
        :param ts_dyn:
            time points for dynamic simulation. Sorted in monotonically increasing order but duplicate time points are
            allowed to facilitate the evaluation of multiple observables at specific time points.
        :param ts_posteq:
            time points for post-equilibration. Usually valued \Infty, but needs to be shaped according to
            the number of observables that are evaluated after post-equilibration.
        :param my:
            observed data
        :param iys:
            indices of the observables according to ordering in :ivar observable_ids:
        :param iy_trafos:
            indices of transformations for observables
        :param ops:
            observables parameters
        :param nps:
            noise parameters
        :param solver:
            ODE solver
        :param controller:
            step size controller
        :param adjoint:
            adjoint method. Recommended values are `diffrax.DirectAdjoint()` for jax.jacfwd (with vector-valued
            outputs) and  `diffrax.RecursiveCheckpointAdjoint()` for jax.grad (for scalar-valued outputs).
        :param steady_state_event:
            event function for steady state. See :func:`diffrax.steady_state_event` for details.
        :param max_steps:
            maximum number of solver steps
        :param x_preeq:
            initial state vector for pre-equilibration. If not provided, the initial state vector is computed using
            :meth:`_x0`.
        :param mask_reinit:
            mask for re-initialization. If `True`, the corresponding state variable is re-initialized.
        :param x_reinit:
            re-initialized state vector. If not provided, the state vector is not re-initialized.
        :param ts_mask:
            mask to remove (padded) time points. If `True`, the corresponding time point is used for the evaluation of
            the output. Only applied if ret is ReturnValue.llh, ReturnValue.nllhs, ReturnValue.res, or ReturnValue.chi2.
        :param ret:
            which output to return. See :class:`ReturnValue` for available options.
        :return:
            output according to `ret` and general results/statistics
        """
        if x_preeq.shape[0]:
            x = x_preeq
        else:
            x = self._x0(p)

        if not ts_mask.shape[0]:
            ts_mask = jnp.ones_like(my, dtype=jnp.bool_)

        # Re-initialization
        if x_reinit.shape[0]:
            x = jnp.where(mask_reinit, x_reinit, x)
        x_solver = self._x_solver(x)
        tcl = self._tcl(x, p)

        # Dynamic simulation
        if ts_dyn.shape[0]:
            x_dyn, stats_dyn = self._solve(
                p,
                ts_dyn,
                tcl,
                x_solver,
                solver,
                controller,
                max_steps,
                adjoint,
            )
            x_solver = x_dyn[-1, :]
        else:
            x_dyn = jnp.repeat(
                x_solver.reshape(1, -1), ts_dyn.shape[0], axis=0
            )
            stats_dyn = None

        # Post-equilibration
        if ts_posteq.shape[0]:
            x_solver, stats_posteq = self._eq(
                p,
                tcl,
                x_solver,
                solver,
                controller,
                steady_state_event,
                max_steps,
            )
        else:
            stats_posteq = None

        x_posteq = jnp.repeat(
            x_solver.reshape(1, -1), ts_posteq.shape[0], axis=0
        )

        ts = jnp.concatenate((ts_dyn, ts_posteq), axis=0)

        x = jnp.concatenate((x_dyn, x_posteq), axis=0)

        nllhs = self._nllhs(ts, x, p, tcl, my, iys, ops, nps)
        nllhs = jnp.where(ts_mask, nllhs, 0.0)
        llh = -jnp.sum(nllhs)

        stats = dict(
            ts=ts,
            x=x,
            llh=llh,
            stats_dyn=stats_dyn,
            stats_posteq=stats_posteq,
        )
        if ret == ReturnValue.llh:
            output = llh
        elif ret == ReturnValue.nllhs:
            output = nllhs
        elif ret == ReturnValue.x:
            output = self._x_rdatas(x, tcl)
        elif ret == ReturnValue.x_solver:
            output = x
        elif ret == ReturnValue.y:
            output = self._ys(ts, x, p, tcl, iys, ops)
        elif ret == ReturnValue.sigmay:
            output = self._sigmays(ts, x, p, tcl, iys, ops, nps)
        elif ret == ReturnValue.x0:
            output = self._x_rdata(x[0, :], tcl)
        elif ret == ReturnValue.x0_solver:
            output = x[0, :]
        elif ret == ReturnValue.tcl:
            output = tcl
        elif ret in (ReturnValue.res, ReturnValue.chi2):
            obs_trafo = jax.vmap(
                lambda y, iy_trafo: jnp.array(
                    # needs to follow order in amici.jax.petab.SCALE_TO_INT
                    [y, safe_log(y), safe_log(y) / jnp.log(10)]
                )
                .at[iy_trafo]
                .get(),
            )
            ys_obj = obs_trafo(self._ys(ts, x, p, tcl, iys, ops), iy_trafos)
            m_obj = obs_trafo(my, iy_trafos)
            if ret == ReturnValue.chi2:
                sigma_obj = self._sigmays(ts, x, p, tcl, iys, ops, nps)
                chi2 = jnp.square((ys_obj - m_obj) / sigma_obj)
                chi2 = jnp.where(ts_mask, chi2, 0.0)
                output = jnp.sum(chi2)
            else:
                output = ys_obj - m_obj
                output = jnp.where(ts_mask, output, 0.0)
        else:
            raise NotImplementedError(f"Return value {ret} not implemented.")

        return output, stats

    @eqx.filter_jit
    def preequilibrate_condition(
        self,
        p: jt.Float[jt.Array, "np"],
        x_reinit: jt.Float[jt.Array, "*nx"],
        mask_reinit: jt.Bool[jt.Array, "*nx"],
        solver: diffrax.AbstractSolver,
        controller: diffrax.AbstractStepSizeController,
        steady_state_event: Callable[
            ..., diffrax._custom_types.BoolScalarLike
        ],
        max_steps: int | jnp.int_,
    ) -> tuple[jt.Float[jt.Array, "nx"], dict]:
        r"""
        Simulate a condition.

        :param p:
            parameters for simulation ordered according to ids in :ivar parameter_ids:
        :param x_reinit:
            re-initialized state vector. If not provided, the state vector is not re-initialized.
        :param mask_reinit:
            mask for re-initialization. If `True`, the corresponding state variable is re-initialized.
        :param solver:
            ODE solver
        :param controller:
            step size controller
        :param max_steps:
            maximum number of solver steps
        :return:
            pre-equilibrated state variables and statistics
        """
        # Pre-equilibration
        x0 = self._x0(p)
        if x_reinit.shape[0]:
            x0 = jnp.where(mask_reinit, x_reinit, x0)
        tcl = self._tcl(x0, p)
        current_x = self._x_solver(x0)
        current_x, stats_preeq = self._eq(
            p,
            tcl,
            current_x,
            solver,
            controller,
            steady_state_event,
            max_steps,
        )

        return self._x_rdata(current_x, tcl), dict(stats_preeq=stats_preeq)


def safe_log(x: jnp.float_) -> jnp.float_:
    """
    Safe logarithm that returns `jnp.log(jnp.finfo(jnp.float_).eps)` for x <= 0.

    :param x:
        input
    :return:
        logarithm of x
    """
    # see https://docs.kidger.site/equinox/api/debug/, need double jnp.where to guard
    # against nans in forward & backward passes
    safe_x = jnp.where(
        x > jnp.finfo(jnp.float_).eps, x, jnp.finfo(jnp.float_).eps
    )
    return jnp.where(
        x > 0, jnp.log(safe_x), jnp.log(jnp.finfo(jnp.float_).eps)
    )


def safe_div(x: jnp.float_, y: jnp.float_) -> jnp.float_:
    """
    Safe division that returns `x/jnp.finfo(jnp.float_).eps` for `y == 0`.

    :param x:
        numerator
    :param y:
        denominator
    :return:
        x / y
    """
    # see https://docs.kidger.site/equinox/api/debug/, need double jnp.where to guard
    # against nans in forward & backward passes
    safe_y = jnp.where(y != 0, y, jnp.finfo(jnp.float_).eps)
    return jnp.where(y != 0, x / safe_y, x / jnp.finfo(jnp.float_).eps)
