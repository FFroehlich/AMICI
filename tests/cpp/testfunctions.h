#ifndef TESTFUNCTIONS_H
#define TESTFUNCTIONS_H

#include <amici/amici.h>
#include <amici/hdf5.h>

#include <H5Cpp.h>

#ifndef __APPLE__
#include <iostream>
#endif

#include <sstream>
#include <string>

namespace amici {

class ReturnData;
class ExpData;

#if !defined(NEW_OPTION_FILE) || !defined(HDFFILE) || !defined(HDFFILEWRITE)
#error "Must define NEW_OPTION_FILE HDFFILE HDFFILEWRITE"
#endif

#define TEST_ATOL 1e-10
#define TEST_RTOL 1e-05

/**
 * @brief helper function to initialize default names/ids
 * @param name name of variables
 * @param length number of variables
 * @return default names/ids
 */
std::vector<std::string>
getVariableNames(const char* name, int length);

/**
 * @brief The Model_Test class is a model-unspecific implementation
 of model designed for unit testing.
 */
class Model_Test : public Model
{
  public:
    /** constructor with model dimensions
     * @param model_dimensions ModelDimensions model_dimensions,
     * @param o2mode second order sensitivity mode
     * @param simulation_parameters Simulation parameters
     * @param idlist indexes indicating algebraic components (DAE only)
     * @param z2event mapping of event outputs to events
     */
    Model_Test(ModelDimensions const& model_dimensions,
               SimulationParameters simulation_parameters,
               const SecondOrderMode o2mode,
               const std::vector<realtype> idlist,
               const std::vector<int> z2event)
      : Model(model_dimensions,
              simulation_parameters,
              o2mode,
              idlist,
              z2event)
    {}

    /** default constructor */
    Model_Test()
        : Model(
              ModelDimensions(),
              SimulationParameters(),
              SecondOrderMode::none,
              std::vector<realtype>(),
              std::vector<int>())
    {}

    Model *clone() const override { return new Model_Test(*this); }

    std::unique_ptr<Solver> getSolver() override {
        throw AmiException("not implemented");
    }
    void froot(const realtype /*t*/, const AmiVector & /*x*/,
               const AmiVector & /*dx*/,
               gsl::span<realtype> /*root*/) override {
        throw AmiException("not implemented");
    }
    void fxdot(const realtype /*t*/, const AmiVector & /*x*/,
               const AmiVector & /*dx*/, AmiVector & /*xdot*/) override {
        throw AmiException("not implemented");
    }
    void fsxdot(const realtype /*t*/, const AmiVector & /*x*/,
                const AmiVector & /*dx*/, const int /*ip*/,
                const AmiVector & /*sx*/, const AmiVector & /*sdx*/,
                AmiVector & /*sxdot*/) override {
        throw AmiException("not implemented");
    }
    void fJ(const realtype /*t*/, const realtype /*cj*/,
            const AmiVector & /*x*/, const AmiVector & /*dx*/,
            const AmiVector & /*xdot*/, SUNMatrix /*J*/) override {
        throw AmiException("not implemented");
    }
    void fJSparse(const realtype /*t*/, const realtype /*cj*/,
                  const AmiVector & /*x*/, const AmiVector & /*dx*/,
                  const AmiVector & /*xdot*/, SUNMatrix /*J*/) override {
        throw AmiException("not implemented");
    }
    void fJB(const realtype /*t*/, realtype /*cj*/, const AmiVector & /*x*/,
             const AmiVector & /*dx*/, const AmiVector & /*xB*/,
             const AmiVector & /*dxB*/, const AmiVector & /*xBdot*/,
             SUNMatrix /*JB*/) override {
        throw AmiException("not implemented");
    }
    void fJSparseB(const realtype /*t*/, realtype /*cj*/,
                   const AmiVector & /*x*/, const AmiVector & /*dx*/,
                   const AmiVector & /*xB*/, const AmiVector & /*dxB*/,
                   const AmiVector & /*xBdot*/, SUNMatrix /*JB*/) override {
        throw AmiException("not implemented");
    }
    void fJDiag(const realtype /*t*/, AmiVector & /*Jdiag*/,
                const realtype /*cj*/, const AmiVector & /*x*/,
                const AmiVector & /*dx*/) override {
        throw AmiException("not implemented");
    }
    void fdxdotdp(const realtype /*t*/, const AmiVector & /*x*/,
                  const AmiVector & /*dx*/) override {
        throw AmiException("not implemented");
    }
    void fJv(const realtype /*t*/, const AmiVector & /*x*/,
             const AmiVector & /*dx*/, const AmiVector & /*xdot*/,
             const AmiVector & /*v*/, AmiVector & /*nJv*/,
             const realtype /*cj*/) override {
        throw AmiException("not implemented");
    }

    void fxBdot_ss(const realtype /*t*/, const AmiVector & /*xB*/,
                   const AmiVector & /*dxB*/, AmiVector & /*xBdot*/
                   ) override {
        throw AmiException("not implemented");
    }

    void fxBdot(realtype /*t*/, const AmiVector &/*x*/,
                const AmiVector &/*dx*/, const AmiVector &/*xB*/,
                const AmiVector &/*dxB*/, AmiVector &/*xBdot*/) override {
        throw AmiException("not implemented");
    }

    void fJSparseB_ss(SUNMatrix /*JB*/) override {
        throw AmiException("not implemented");
    }

    void writeSteadystateJB(const realtype /*t*/, realtype /*cj*/,
                            const AmiVector & /*x*/, const AmiVector & /*dx*/,
                            const AmiVector & /*xB*/, const AmiVector & /*dxB*/,
                            const AmiVector & /*xBdot*/) override {
        throw AmiException("not implemented");
    }

    std::vector<std::string> getParameterNames() const override {
        return getVariableNames("p", np());
    }

    std::vector<std::string> getStateNames() const override {
        return getVariableNames("x", nx_rdata);
    }

    std::vector<std::string> getFixedParameterNames() const override {
        return getVariableNames("k", nk());
    }

    std::vector<std::string> getObservableNames() const override {
        return getVariableNames("y", ny);
    }

    std::vector<std::string> getParameterIds() const override {
        return getVariableNames("p", np());
    }

    std::vector<std::string> getStateIds() const override {
        return getVariableNames("x", nx_rdata);
    }

    std::vector<std::string> getFixedParameterIds() const override {
        return getVariableNames("k", nk());
    }

    std::vector<std::string> getObservableIds() const override {
        return getVariableNames("y", ny);
    }
};

void simulateWithDefaultOptions();

void simulateVerifyWrite(const std::string &path);

void simulateVerifyWrite(std::string const &path, double atol, double rtol);

void simulateVerifyWrite(const std::string &hdffileOptions,
                         const std::string &hdffileResults,
                         const std::string &hdffilewrite,
                         const std::string &path, double atol, double rtol);

std::unique_ptr<ExpData> getTestExpData(const Model &model);

bool withinTolerance(double expected, double actual, double atol, double rtol,
                     int index, const char *name);

void checkEqualArray(const double *expected, const double *actual, int length,
                     double atol, double rtol, const char *name);

void checkEqualArray(std::vector<double> const &expected,
                     std::vector<double> const &actual, double atol,
                     double rtol, std::string const &name);

// TODO: delete after transitioning to C++-written test results
void verifyReturnDataMatlab(const std::string &hdffile,
                            const std::string &resultPath,
                            const ReturnData *rdata, const Model *model,
                            double atol, double rtol);

// TODO: delete after transitioning to C++-written test results
void verifyReturnDataSensitivitiesMatlab(const H5::H5File &file_id,
                                         const std::string &resultPath,
                                         const ReturnData *rdata,
                                         const Model *model, double atol,
                                         double rtol);

void verifyReturnData(const std::string &hdffile, const std::string &resultPath,
                      const ReturnData *rdata, const Model *model, double atol,
                      double rtol);

void verifyReturnDataSensitivities(const H5::H5File &file_id,
                                   const std::string &resultPath,
                                   const ReturnData *rdata, const Model *model,
                                   double atol, double rtol);

void printBacktrace(int depth);

} // namespace amici

#endif
