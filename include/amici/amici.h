#ifndef amici_h
#define amici_h

#include "amici/edata.h"
#include "amici/model.h"
#include "amici/rdata.h"
#include "amici/solver.h"

namespace amici {

/**
 * @brief Core integration routine. Initializes the solver and runs the forward
 * and backward problem.
 *
 * @param solver Solver instance
 * @param edata pointer to experimental data object
 * @param model model specification object
 * @param rethrow rethrow integration exceptions?
 * @return rdata pointer to return data object
 */
std::unique_ptr<ReturnData> runAmiciSimulation(
    Solver& solver, ExpData const* edata, Model& model, bool rethrow = false
);

/**
 * @brief Same as runAmiciSimulation, but for multiple ExpData instances. When
 * compiled with OpenMP support, this function runs multi-threaded.
 *
 * @param solver Solver instance
 * @param edatas experimental data objects
 * @param model model specification object
 * @param failfast flag to allow early termination
 * @param num_threads number of threads for parallel execution
 * @return vector of pointers to return data objects
 */
std::vector<std::unique_ptr<ReturnData>> runAmiciSimulations(
    Solver const& solver, std::vector<ExpData*> const& edatas,
    Model const& model, bool failfast, int num_threads
);

/**
 * @brief Same as runAmiciSimulation, but for a timeseries of ExpData instances.
 *
 * @param solver Solver instance
 * @param edatas experimental data objects
 * @param model model specification object
 * @param failfast flag to allow early termination
 * @return vector of pointers to return data objects
 */
std::vector<std::unique_ptr<ReturnData>> runAmiciSimulationsTimeseries(
    Solver const& solver, std::vector<ExpData*> const& edatas,
    Model const& model, bool failfast
);

/**
 * @brief Get the string representation of the given simulation status code
 * (see ReturnData::status).
 * @param status Status code
 * @return Name of the variable representing this status code.
 */
std::string simulation_status_to_str(int status);

} // namespace amici

#endif /* amici_h */
