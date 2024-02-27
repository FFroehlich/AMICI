#include "testfunctions.h"

#include <amici/amici.h>
#include <amici/model_ode.h>
#include <amici/symbolic_functions.h>

#include <exception>
#include <vector>

#include <gtest/gtest.h>

namespace amici {
namespace generic_model {
std::unique_ptr<Model> getModel();
} // namespace generic_model
} // namespace amici

using namespace amici;

namespace  {

class ExpDataTest : public ::testing::Test {
  protected:
    void SetUp() override {
        model->setTimepoints(timepoints);
        model->setNMaxEvent(nmaxevent);
        testModel.setTimepoints(timepoints);
        testModel.setNMaxEvent(nmaxevent);
    }

    int nx = 1, ny = 2, nz = 3, nmaxevent = 4;
    std::vector<realtype> timepoints = { 1, 2, 3, 4 };

    std::unique_ptr<Model> model = generic_model::getModel();

    Model_Test testModel = Model_Test(
        ModelDimensions(
            nx,        // nx_rdata
            nx,        // nxtrue_rdata
            nx,        // nx_solver
            nx,        // nxtrue_solver
            0,         // nx_solver_reinit
            1,  // np
            3,  // nk
            ny,        // ny
            ny,        // nytrue
            nz,        // nz
            nz,        // nztrue
            nmaxevent, // ne
            0,         // ne_solver
            0,         // nspl
            0,         // nJ
            0,         // nw
            0,         // ndwdx
            0,         // ndwdp
            0,         // dwdw
            0,         // ndxdotdw
            {},         // ndJydy
            0,         // ndxrdatadxsolver
            0,         // ndxrdatadtcl
            0,         // ndtotal_cldx_rdata
            0,         // nnz
            0,         // ubw
            0          // lbw
            ),
        SimulationParameters(
            std::vector<realtype>(3, 0.0),
            std::vector<realtype>(1, 0.0),
            std::vector<int>(2, 1)
            ),
        SecondOrderMode::none,
        std::vector<realtype>(),
        std::vector<int>());
};

TEST_F(ExpDataTest, DefaultConstructable)
{
    ExpData edata{};
    ASSERT_EQ(edata.nytrue(), 0);
    ASSERT_EQ(edata.nztrue(), 0);
    ASSERT_EQ(edata.nmaxevent(), 0);
}
TEST_F(ExpDataTest, ModelCtor)
{
    ExpData edata(model->nytrue, model->nztrue, model->nMaxEvent());
    ASSERT_EQ(edata.nytrue(), model->nytrue);
    ASSERT_EQ(edata.nztrue(), model->nztrue);
    ASSERT_EQ(edata.nmaxevent(), model->nMaxEvent());
}

TEST_F(ExpDataTest, DimensionCtor)
{
    ExpData edata(model->nytrue, model->nztrue, model->nMaxEvent(), timepoints);
    ASSERT_EQ(edata.nytrue(), model->nytrue);
    ASSERT_EQ(edata.nztrue(), model->nztrue);
    ASSERT_EQ(edata.nmaxevent(), model->nMaxEvent());
    ASSERT_EQ(edata.nt(), model->nt());
    checkEqualArray(
        timepoints, edata.getTimepoints(), TEST_ATOL, TEST_RTOL, "ts");
}

TEST_F(ExpDataTest, MeasurementCtor)
{
    std::vector<realtype> y(ny * timepoints.size(), 0.0);
    std::vector<realtype> y_std(ny * timepoints.size(), 0.1);
    std::vector<realtype> z(nz * nmaxevent, 0.0);
    std::vector<realtype> z_std(nz * nmaxevent, 0.1);

    ExpData edata(testModel.nytrue,
                  testModel.nztrue,
                  testModel.nMaxEvent(),
                  timepoints,
                  y,
                  y_std,
                  z,
                  z_std);
    ASSERT_EQ(edata.nytrue(), testModel.nytrue);
    ASSERT_EQ(edata.nztrue(), testModel.nztrue);
    ASSERT_EQ(edata.nmaxevent(), testModel.nMaxEvent());
    ASSERT_EQ(edata.nt(), testModel.nt());
    checkEqualArray(
        timepoints, edata.getTimepoints(), TEST_ATOL, TEST_RTOL, "ts");
    checkEqualArray(
        y, edata.getObservedData(), TEST_ATOL, TEST_RTOL, "observedData");
    checkEqualArray(y_std,
                    edata.getObservedDataStdDev(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedDataStdDev");
    checkEqualArray(
        z, edata.getObservedEvents(), TEST_ATOL, TEST_RTOL, "observedEvents");
    checkEqualArray(z_std,
                    edata.getObservedEventsStdDev(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedEventsStdDev");

    ExpData edata_copy(edata);
    ASSERT_EQ(edata.nytrue(), edata_copy.nytrue());
    ASSERT_EQ(edata.nztrue(), edata_copy.nztrue());
    ASSERT_EQ(edata.nmaxevent(), edata_copy.nmaxevent());
    ASSERT_EQ(edata.nt(), edata_copy.nt());
    checkEqualArray(edata_copy.getTimepoints(),
                    edata.getTimepoints(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "ts");
    checkEqualArray(edata_copy.getObservedData(),
                    edata.getObservedData(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedData");
    checkEqualArray(edata_copy.getObservedDataStdDev(),
                    edata.getObservedDataStdDev(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedDataStdDev");
    checkEqualArray(edata_copy.getObservedEvents(),
                    edata.getObservedEvents(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedEvents");
    checkEqualArray(edata_copy.getObservedEventsStdDev(),
                    edata.getObservedEventsStdDev(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "observedEventsStdDev");
}

TEST_F(ExpDataTest, CopyConstructable)
{
    testModel.setTimepoints(timepoints);
    auto edata = ExpData(testModel);
    ASSERT_EQ(edata.nytrue(), testModel.nytrue);
    ASSERT_EQ(edata.nztrue(), testModel.nztrue);
    ASSERT_EQ(edata.nmaxevent(), testModel.nMaxEvent());
    ASSERT_EQ(edata.nt(), testModel.nt());
    checkEqualArray(testModel.getTimepoints(),
                    edata.getTimepoints(),
                    TEST_ATOL,
                    TEST_RTOL,
                    "ts");
}


TEST_F(ExpDataTest, Equality)
{
    auto edata = ExpData(testModel);
    auto edata2(edata);
    ASSERT_TRUE(edata == edata2);

    edata2.id = "different";
    ASSERT_FALSE(edata == edata2);
}

TEST_F(ExpDataTest, DimensionChecks)
{
    std::vector<realtype> bad_std(ny, -0.1);
    std::vector<realtype> y(ny * timepoints.size(), 0.0);
    std::vector<realtype> y_std(ny * timepoints.size(), 0.1);
    std::vector<realtype> z(nz * nmaxevent, 0.0);
    std::vector<realtype> z_std(nz * nmaxevent, 0.1);

    ASSERT_THROW(ExpData(testModel.nytrue,
                         testModel.nztrue,
                         testModel.nMaxEvent(),
                         timepoints,
                         z,
                         z_std,
                         z,
                         z_std),
                 AmiException);

    ASSERT_THROW(ExpData(testModel.nytrue,
                         testModel.nztrue,
                         testModel.nMaxEvent(),
                         timepoints,
                         z,
                         bad_std,
                         z,
                         z_std),
                 AmiException);

    ExpData edata(testModel);

    std::vector<realtype> bad_y(ny * timepoints.size() + 1, 0.0);
    std::vector<realtype> bad_y_std(ny * timepoints.size() + 1, 0.1);
    std::vector<realtype> bad_z(nz * nmaxevent + 1, 0.0);
    std::vector<realtype> bad_z_std(nz * nmaxevent + 1, 0.1);

    ASSERT_THROW(edata.setObservedData(bad_y), AmiException);
    ASSERT_THROW(edata.setObservedDataStdDev(bad_y_std), AmiException);
    ASSERT_THROW(edata.setObservedEvents(bad_z), AmiException);
    ASSERT_THROW(edata.setObservedEventsStdDev(bad_y_std), AmiException);

    std::vector<realtype> bad_single_y(edata.nt() + 1, 0.0);
    std::vector<realtype> bad_single_y_std(edata.nt() + 1, 0.1);
    std::vector<realtype> bad_single_z(edata.nmaxevent() + 1, 0.0);
    std::vector<realtype> bad_single_z_std(edata.nmaxevent() + 1, 0.1);

    ASSERT_THROW(edata.setObservedData(bad_single_y, 0),
                 AmiException);
    ASSERT_THROW(edata.setObservedDataStdDev(bad_single_y_std, 0),
                 AmiException);
    ASSERT_THROW(edata.setObservedEvents(bad_single_z, 0),
                 AmiException);
    ASSERT_THROW(edata.setObservedEventsStdDev(bad_single_y_std, 0),
                 AmiException);

    ASSERT_THROW(edata.setTimepoints(std::vector<realtype>{ 0.0, 1.0, 0.5 }),
                 AmiException);
}

TEST_F(ExpDataTest, SettersGetters)
{
    ExpData edata(testModel);

    std::vector<realtype> y(ny * timepoints.size(), 0.0);
    std::vector<realtype> y_std(ny * timepoints.size(), 0.1);
    std::vector<realtype> z(nz * nmaxevent, 0.0);
    std::vector<realtype> z_std(nz * nmaxevent, 0.1);

    edata.setObservedData(y);
    checkEqualArray(
        edata.getObservedData(), y, TEST_ATOL, TEST_RTOL, "ObservedData");
    edata.setObservedDataStdDev(y_std);
    checkEqualArray(edata.getObservedDataStdDev(),
                    y_std,
                    TEST_ATOL,
                    TEST_RTOL,
                    "ObservedDataStdDev");
    edata.setObservedEvents(z);
    checkEqualArray(
        edata.getObservedEvents(), z, TEST_ATOL, TEST_RTOL, "ObservedEvents");
    edata.setObservedEventsStdDev(z_std);
    checkEqualArray(edata.getObservedEventsStdDev(),
                    z_std,
                    TEST_ATOL,
                    TEST_RTOL,
                    "ObservedEventsStdDev");

    std::vector<realtype> single_y(edata.nt(), 0.0);
    std::vector<realtype> single_y_std(edata.nt(), 0.1);

    for (int iy = 0; iy < ny; ++iy) {
        edata.setObservedData(single_y, iy);
        edata.setObservedDataStdDev(single_y_std, iy);
    }
    ASSERT_THROW(edata.setObservedData(single_y, ny), std::exception);
    ASSERT_THROW(edata.setObservedData(single_y, -1), std::exception);
    ASSERT_THROW(edata.setObservedDataStdDev(single_y_std, ny), std::exception);
    ASSERT_THROW(edata.setObservedDataStdDev(single_y_std, -1), std::exception);

    std::vector<realtype> single_z(edata.nmaxevent(), 0.0);
    std::vector<realtype> single_z_std(edata.nmaxevent(), 0.1);

    for (int iz = 0; iz < nz; ++iz) {
        edata.setObservedEvents(single_z, iz);
        edata.setObservedEventsStdDev(single_z_std, iz);
    }

    ASSERT_THROW(edata.setObservedEvents(single_z, nz), std::exception);
    ASSERT_THROW(edata.setObservedEvents(single_z, -1), std::exception);
    ASSERT_THROW(edata.setObservedEventsStdDev(single_z_std, nz),
                 std::exception);
    ASSERT_THROW(edata.setObservedEventsStdDev(single_z_std, -1),
                 std::exception);

    ASSERT_TRUE(edata.getObservedDataPtr(0));
    ASSERT_TRUE(edata.getObservedDataStdDevPtr(0));
    ASSERT_TRUE(edata.getObservedEventsPtr(0));
    ASSERT_TRUE(edata.getObservedEventsStdDevPtr(0));

    std::vector<realtype> empty(0, 0.0);

    edata.setObservedData(empty);
    edata.setObservedDataStdDev(empty);
    edata.setObservedEvents(empty);
    edata.setObservedEventsStdDev(empty);

    ASSERT_TRUE(!edata.getObservedDataPtr(0));
    ASSERT_TRUE(!edata.getObservedDataStdDevPtr(0));
    ASSERT_TRUE(!edata.getObservedEventsPtr(0));
    ASSERT_TRUE(!edata.getObservedEventsStdDevPtr(0));

    checkEqualArray(
        edata.getObservedData(), empty, TEST_ATOL, TEST_RTOL, "ObservedData");
    checkEqualArray(edata.getObservedDataStdDev(),
                    empty,
                    TEST_ATOL,
                    TEST_RTOL,
                    "ObservedDataStdDev");
    checkEqualArray(
        edata.getObservedEvents(), empty, TEST_ATOL, TEST_RTOL, "ObservedEvents");
    checkEqualArray(edata.getObservedEventsStdDev(),
                    empty,
                    TEST_ATOL,
                    TEST_RTOL,
                    "ObservedEventsStdDev");
}

} // namespace
