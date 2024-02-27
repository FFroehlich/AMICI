#!/usr/bin/env python3
import sys

import numpy as np
from example import AmiciExample


class ExampleSteadystate(AmiciExample):
    def __init__(self):
        AmiciExample.__init__(self)

        self.numX = 3
        self.numP = 5
        self.numK = 4

        self.modelOptions["theta"] = np.log10([1, 0.5, 0.4, 2, 0.1])
        self.modelOptions["kappa"] = [0.1, 0.4, 0.7, 1]
        self.modelOptions["ts"] = np.linspace(0, 100, 50)
        self.modelOptions["pscale"] = 2


def writeNoSensi(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.append(np.linspace(0, 100, 50), np.inf)
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/nosensi/")


def writeSensiForward(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.append(np.linspace(0, 100, 50), np.inf)
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiforward/")


def writeSensiForwardPlist(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.append(np.linspace(0, 100, 50), np.inf)
    ex.solverOptions["sens_ind"] = [3, 1, 2, 4]
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiforwardplist/")


def writeSensiForwardDense(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.append(np.linspace(0, 100, 50), np.inf)
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["linsol"] = 1
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0
    ex.writeToFile(filename, "/model_steadystate/sensiforwarddense/")


def writeSensiForwardErrorInt(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.append(np.linspace(0, 100, 50), np.inf)
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 100
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiforwarderrorint/")


def writeSensiForwardErrorNewt(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = [0, np.inf]
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 100
    ex.solverOptions["newton_maxsteps"] = 2
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiforwarderrornewt/")


def writeSensiFwdNewtonPreeq(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 1
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 20
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensifwdnewtonpreeq/")


def writeSensiAdjNewtonPreeq(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 2
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 20
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiadjnewtonpreeq/")


def writeSensiFwdSimPreeq(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 1
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensifwdsimpreeq/")


def writeSensiFwdSimPreeqFSA(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.modelOptions["steadyStateSensitivityMode"] = 1
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 1
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensifwdsimpreeqFSA/")


def writeSensiAdjSimPreeq(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 2
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiadjsimpreeq/")


def writeSensiAdjSimPreeqFSA(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["conditionPreequilibration"] = [0.1, 0.4, 0.7, 0.5]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.modelOptions["steadyStateSensitivityMode"] = 1
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 2
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiadjsimpreeqFSA/")


def writeSensiFwdByhandPreeq(filename):
    ex = ExampleSteadystate()

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 1
    ex.solverOptions["x0"] = np.array(
        [0.532609637980272, 0.625849232840357, 0.066666666666667]
    )
    ex.solverOptions["sx0"] = np.array(
        [
            [-0.425457638009506, 0.499939012301881, 0],
            [-0.375463736779318, -0.999878024603762, -0.000000000000000],
            [0.375463736779318, -0.441193089396203, 0],
            [0.300370989423454, 0.799902419683010, 0.000000000000000],
            [0.425457638009506, 0.941132101698086, 0.153505672866270],
        ]
    )
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensifwdbyhandpreeq/")


def writeSensiAdjByhandPreeq(filename):
    ex = ExampleSteadystate()

    ex.data = {}
    ex.data["Y"] = np.array(
        [
            [0.8410, 1.6662, 0.3813],
            [0.7834, 1.6230, 0.2233],
            [0.8187, 1.5386, 0.2208],
            [0.7906, 1.4476, 0.2358],
            [0.7184, 1.3543, 0.1409],
            [0.6627, 1.2840, 0.1268],
            [0.7099, 1.1786, 0.1289],
            [0.7104, 1.1362, 0.1768],
            [0.7089, 1.0326, 0.1127],
            [0.6035, 0.9752, 0.0923],
        ]
    )
    ex.data["Sigma_Y"] = np.ones(shape=ex.data["Y"].shape)
    ex.data["Sigma_Z"] = []
    ex.data["Z"] = []
    ex.data["condition"] = ex.modelOptions["kappa"]
    ex.data["t"] = np.linspace(0, 5, 10)

    ex.modelOptions["ts"] = np.linspace(0, 5, 10)
    ex.solverOptions["sensi"] = 1
    ex.solverOptions["sensi_meth"] = 2
    ex.solverOptions["x0"] = np.array(
        [0.532609637980272, 0.625849232840357, 0.066666666666667]
    )
    ex.solverOptions["sx0"] = np.array(
        [
            [-0.425457638009506, 0.499939012301881, 0],
            [-0.375463736779318, -0.999878024603762, -0.000000000000000],
            [0.375463736779318, -0.441193089396203, 0],
            [0.300370989423454, 0.799902419683010, 0.000000000000000],
            [0.425457638009506, 0.941132101698086, 0.153505672866270],
        ]
    )
    ex.solverOptions["atol"] = 10**-16
    ex.solverOptions["rtol"] = 10**-12
    ex.solverOptions["sens_ind"] = np.arange(0, ex.numP)
    ex.solverOptions["linsol"] = 9
    ex.solverOptions["maxsteps"] = 10000
    ex.solverOptions["newton_maxsteps"] = 0
    ex.solverOptions["ss_tol_factor"] = 1.0
    ex.solverOptions["ss_tol_sensi_factor"] = 1.0

    ex.writeToFile(filename, "/model_steadystate/sensiadjbyhandpreeq/")


def main():
    if len(sys.argv) < 2:
        print("Error: Must provide output file as first and only argument.")
        sys.exit(1)
    filename = sys.argv[1]

    writeNoSensi(filename)
    writeSensiForward(filename)
    writeSensiForwardPlist(filename)
    writeSensiForwardDense(filename)
    writeSensiForwardErrorInt(filename)
    writeSensiForwardErrorNewt(filename)
    writeSensiFwdNewtonPreeq(filename)
    writeSensiAdjNewtonPreeq(filename)
    writeSensiFwdSimPreeq(filename)
    writeSensiFwdSimPreeqFSA(filename)
    writeSensiAdjSimPreeq(filename)
    writeSensiAdjSimPreeqFSA(filename)
    writeSensiFwdByhandPreeq(filename)
    writeSensiAdjByhandPreeq(filename)


if __name__ == "__main__":
    main()
