from abc import abstractmethod
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import jax
from collections.abc import Iterable

import amici

jax.config.update("jax_enable_x64", True)


class JAXModel(eqx.Module):
    _unscale_funs = {
        amici.ParameterScaling.none: lambda x: x,
        amici.ParameterScaling.ln: lambda x: jnp.exp(x),
        amici.ParameterScaling.log10: lambda x: jnp.power(10, x),
    }
    solver: diffrax.AbstractSolver
    controller: diffrax.AbstractStepSizeController
    atol: float
    rtol: float
    pcoeff: float
    icoeff: float
    dcoeff: float
    maxsteps: int
    term: diffrax.ODETerm
    sensi_order: amici.SensitivityOrder

    def __init__(self):
        self.solver = diffrax.Kvaerno5()
        self.atol: float = 1e-8
        self.rtol: float = 1e-8
        self.pcoeff: float = 0.4
        self.icoeff: float = 0.3
        self.dcoeff: float = 0.0
        self.maxsteps: int = 2**10
        self.controller = diffrax.PIDController(
            rtol=self.rtol,
            atol=self.atol,
            pcoeff=self.pcoeff,
            icoeff=self.icoeff,
            dcoeff=self.dcoeff,
        )
        self.term = diffrax.ODETerm(self.xdot)
        self.sensi_order = amici.SensitivityOrder.none

    @staticmethod
    @abstractmethod
    def xdot(t, x, args):
        ...

    @staticmethod
    @abstractmethod
    def _w(t, x, p, k, tcl):
        ...

    @staticmethod
    @abstractmethod
    def x0(p, k):
        ...

    @staticmethod
    @abstractmethod
    def x_solver(x):
        ...

    @staticmethod
    @abstractmethod
    def x_rdata(x, tcl):
        ...

    @staticmethod
    @abstractmethod
    def tcl(x, p, k):
        ...

    @staticmethod
    @abstractmethod
    def y(t, x, p, k, tcl):
        ...

    @staticmethod
    @abstractmethod
    def sigmay(y, p, k):
        ...

    @staticmethod
    @abstractmethod
    def Jy(y, my, sigmay):
        ...

    def unscale_p(self, p, pscale):
        return jax.vmap(
            lambda p_i, pscale_i: jnp.stack(
                (p_i, jnp.exp(p_i), jnp.power(10, p_i))
            )
            .at[pscale_i]
            .get()
        )(p, pscale)

    def _solve(self, ts, p, k):
        x0 = self.x0(p, k)
        tcl = self.tcl(x0, p, k)
        sol = diffrax.diffeqsolve(
            self.term,
            self.solver,
            args=(p, k, tcl),
            t0=0.0,
            t1=ts[-1],
            dt0=None,
            y0=self.x_solver(x0),
            stepsize_controller=self.controller,
            max_steps=self.maxsteps,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys, tcl, sol.stats

    def _obs(self, ts, x, p, k, tcl):
        return jax.vmap(self.y, in_axes=(0, 0, None, None, None))(
            ts, x, p, k, tcl
        )

    def _sigmay(self, obs, p, k):
        return jax.vmap(self.sigmay, in_axes=(0, None, None))(obs, p, k)

    def _x_rdata(self, x, tcl):
        return jax.vmap(self.x_rdata, in_axes=(0, None))(x, tcl)

    def _loss(self, obs: jnp.ndarray, sigmay: jnp.ndarray, my: np.ndarray):
        loss_fun = jax.vmap(self.Jy, in_axes=(0, 0, 0))
        return -jnp.sum(loss_fun(obs, my, sigmay))

    def _run(
        self,
        ts: np.ndarray,
        p: np.ndarray,
        k: jnp.ndarray,
        my: jnp.ndarray,
        pscale: np.ndarray,
    ):
        ps = self.unscale_p(p, pscale)
        x, tcl, stats = self._solve(ts, ps, k)
        obs = self._obs(ts, x, ps, k, tcl)
        my_r = my.reshape((len(ts), -1))
        sigmay = self._sigmay(obs, ps, k)
        llh = self._loss(obs, sigmay, my_r)
        x_rdata = self._x_rdata(x, tcl)
        return llh, (x_rdata, obs, stats)

    @eqx.filter_jit
    def run(
        self,
        ts: np.ndarray,
        p: jnp.ndarray,
        k: np.ndarray,
        my: np.ndarray,
        pscale: np.ndarray,
    ):
        return self._run(ts, p, k, my, pscale)

    @eqx.filter_jit
    def srun(
        self,
        ts: np.ndarray,
        p: jnp.ndarray,
        k: np.ndarray,
        my: np.ndarray,
        pscale: np.ndarray,
    ):
        (llh, (x, obs, stats)), sllh = (
            jax.value_and_grad(self._run, 1, True)
        )(ts, p, k, my, pscale)
        return llh, sllh, (x, obs, stats)

    @eqx.filter_jit
    def s2run(
        self,
        ts: np.ndarray,
        p: jnp.ndarray,
        k: np.ndarray,
        my: np.ndarray,
        pscale: np.ndarray,
    ):
        (llh, (_, _, _)), sllh = (jax.value_and_grad(self._run, 1, True))(
            ts, p, k, my, pscale
        )
        s2llh, (x, obs, stats) = jax.jacfwd(
            jax.grad(self._run, 1, True), 1, True
        )(ts, p, k, my, pscale)
        return llh, sllh, s2llh, (x, obs, stats)

    def run_simulation(self, edata: amici.ExpData):
        ts = np.asarray(edata.getTimepoints())
        p = jnp.asarray(edata.parameters)
        k = np.asarray(edata.fixedParameters)
        my = np.asarray(edata.getObservedData())
        pscale = np.asarray(edata.pscale)

        rdata_kwargs = dict()

        if self.sensi_order == amici.SensitivityOrder.none:
            (
                rdata_kwargs["llh"],
                (rdata_kwargs["x"], rdata_kwargs["y"], rdata_kwargs["stats"]),
            ) = self.run(ts, p, k, my, pscale)
        elif self.sensi_order == amici.SensitivityOrder.first:
            (
                rdata_kwargs["llh"],
                rdata_kwargs["sllh"],
                (rdata_kwargs["x"], rdata_kwargs["y"], rdata_kwargs["stats"]),
            ) = self.srun(ts, p, k, my, pscale)
        elif self.sensi_order == amici.SensitivityOrder.second:
            (
                rdata_kwargs["llh"],
                rdata_kwargs["sllh"],
                rdata_kwargs["s2llh"],
                (rdata_kwargs["x"], rdata_kwargs["y"], rdata_kwargs["stats"]),
            ) = self.s2run(ts, p, k, my, pscale)

        for field in rdata_kwargs.keys():
            if field == "llh":
                rdata_kwargs[field] = np.float64(rdata_kwargs[field])
            elif field not in ["sllh", "s2llh"]:
                rdata_kwargs[field] = np.asarray(rdata_kwargs[field]).T
                if rdata_kwargs[field].ndim == 1:
                    rdata_kwargs[field] = np.expand_dims(
                        rdata_kwargs[field], 1
                    )

        return ReturnDataJAX(**rdata_kwargs)

    def run_simulations(
        self,
        edatas: Iterable[amici.ExpData],
        num_threads: int = 1,
    ):
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                results = pool.map(self.run_simulation, edatas)
        else:
            results = map(self.run_simulation, edatas)
        return list(results)


@dataclass
class ReturnDataJAX(dict):
    x: np.array = None
    sx: np.array = None
    y: np.array = None
    sy: np.array = None
    sigmay: np.array = None
    ssigmay: np.array = None
    llh: np.array = None
    sllh: np.array = None
    stats: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
