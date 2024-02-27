#include <amici/model.h>
#include <amici/serialization.h>
#include <amici/solver_cvodes.h>

#include "testfunctions.h"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

void
checkReturnDataEqual(amici::ReturnData const& r, amici::ReturnData const& s)
{
    ASSERT_EQ(r.id, s.id);
    ASSERT_EQ(r.np, s.np);
    ASSERT_EQ(r.nk, s.nk);
    ASSERT_EQ(r.nx, s.nx);
    ASSERT_EQ(r.nxtrue, s.nxtrue);
    ASSERT_EQ(r.nx_solver, s.nx_solver);
    ASSERT_EQ(r.nx_solver_reinit, s.nx_solver_reinit);
    ASSERT_EQ(r.ny, s.ny);
    ASSERT_EQ(r.nytrue, s.nytrue);
    ASSERT_EQ(r.nz, s.nz);
    ASSERT_EQ(r.nztrue, s.nztrue);
    ASSERT_EQ(r.ne, s.ne);
    ASSERT_EQ(r.nJ, s.nJ);
    ASSERT_EQ(r.nplist, s.nplist);
    ASSERT_EQ(r.nmaxevent, s.nmaxevent);
    ASSERT_EQ(r.nt, s.nt);
    ASSERT_EQ(r.newton_maxsteps, s.newton_maxsteps);
    ASSERT_EQ(r.pscale, s.pscale);
    ASSERT_EQ(static_cast<int>(r.o2mode), static_cast<int>(s.o2mode));
    ASSERT_EQ(static_cast<int>(r.sensi), static_cast<int>(s.sensi));
    ASSERT_EQ(static_cast<int>(r.sensi_meth), static_cast<int>(s.sensi_meth));

    using amici::checkEqualArray;
    checkEqualArray(r.ts, s.ts, 1e-16, 1e-16, "ts");
    checkEqualArray(r.xdot, s.xdot, 1e-16, 1e-16, "xdot");
    checkEqualArray(r.J, s.J, 1e-16, 1e-16, "J");
    checkEqualArray(r.z, s.z, 1e-16, 1e-16, "z");
    checkEqualArray(r.sigmaz, s.sigmaz, 1e-16, 1e-16, "sigmaz");
    checkEqualArray(r.sz, s.sz, 1e-16, 1e-16, "sz");
    checkEqualArray(r.ssigmaz, s.ssigmaz, 1e-16, 1e-16, "ssigmaz");
    checkEqualArray(r.rz, s.rz, 1e-16, 1e-16, "rz");
    checkEqualArray(r.srz, s.srz, 1e-16, 1e-16, "srz");
    checkEqualArray(r.s2rz, s.s2rz, 1e-16, 1e-16, "s2rz");
    checkEqualArray(r.x, s.x, 1e-16, 1e-16, "x");
    checkEqualArray(r.sx, s.sx, 1e-16, 1e-16, "sx");

    checkEqualArray(r.y, s.y, 1e-16, 1e-16, "y");
    checkEqualArray(r.sigmay, s.sigmay, 1e-16, 1e-16, "sigmay");
    checkEqualArray(r.sy, s.sy, 1e-16, 1e-16, "sy");
    checkEqualArray(r.ssigmay, s.ssigmay, 1e-16, 1e-16, "ssigmay");

    ASSERT_EQ(r.numsteps, s.numsteps);
    ASSERT_EQ(r.numstepsB, s.numstepsB);
    ASSERT_EQ(r.numrhsevals, s.numrhsevals);
    ASSERT_EQ(r.numrhsevalsB, s.numrhsevalsB);
    ASSERT_EQ(r.numerrtestfails, s.numerrtestfails);
    ASSERT_EQ(r.numerrtestfailsB, s.numerrtestfailsB);
    ASSERT_EQ(r.numnonlinsolvconvfails, s.numnonlinsolvconvfails);
    ASSERT_EQ(r.numnonlinsolvconvfailsB, s.numnonlinsolvconvfailsB);
    ASSERT_EQ(r.order, s.order);
    ASSERT_EQ(r.cpu_time, s.cpu_time);
    ASSERT_EQ(r.cpu_timeB, s.cpu_timeB);

    ASSERT_EQ(r.preeq_status, s.preeq_status);
    ASSERT_TRUE(r.preeq_t == s.preeq_t ||
                (std::isnan(r.preeq_t) && std::isnan(s.preeq_t)));
    ASSERT_TRUE(r.preeq_wrms == s.preeq_wrms ||
                (std::isnan(r.preeq_wrms) && std::isnan(s.preeq_wrms)));
    ASSERT_EQ(r.preeq_numsteps, s.preeq_numsteps);
    EXPECT_NEAR(r.preeq_cpu_time, s.preeq_cpu_time, 1e-16);

    ASSERT_EQ(r.posteq_status, s.posteq_status);
    ASSERT_TRUE(r.posteq_t == s.posteq_t ||
                (std::isnan(r.posteq_t) && std::isnan(s.posteq_t)));
    ASSERT_TRUE(r.posteq_wrms == s.posteq_wrms ||
                (std::isnan(r.posteq_wrms) && std::isnan(s.posteq_wrms)));
    ASSERT_EQ(r.posteq_numsteps, s.posteq_numsteps);
    EXPECT_NEAR(r.posteq_cpu_time, s.posteq_cpu_time, 1e-16);

    checkEqualArray(r.x0, s.x0, 1e-16, 1e-16, "x0");
    checkEqualArray(r.sx0, s.sx0, 1e-16, 1e-16, "sx0");

    ASSERT_TRUE(r.llh == s.llh || (std::isnan(r.llh) && std::isnan(s.llh)));
    ASSERT_TRUE(r.chi2 == s.chi2 || (std::isnan(r.llh) && std::isnan(s.llh)));
    ASSERT_EQ(r.status, s.status);

    checkEqualArray(r.sllh, s.sllh, 1e-5, 1e-5, "sllh");
    checkEqualArray(r.s2llh, s.s2llh, 1e-5, 1e-5, "s2llh");
}

class SolverSerializationTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // set non-default values for all members
        solver.setAbsoluteTolerance(1e-4);
        solver.setRelativeTolerance(1e-5);
        solver.setAbsoluteToleranceQuadratures(1e-6);
        solver.setRelativeToleranceQuadratures(1e-7);
        solver.setAbsoluteToleranceSteadyState(1e-8);
        solver.setRelativeToleranceSteadyState(1e-9);
        solver.setSensitivityMethod(amici::SensitivityMethod::adjoint);
        solver.setSensitivityOrder(amici::SensitivityOrder::second);
        solver.setMaxSteps(1e1);
        solver.setMaxStepsBackwardProblem(1e2);
        solver.setNewtonMaxSteps(1e3);
        solver.setStateOrdering(static_cast<int>(amici::SUNLinSolKLU::StateOrdering::COLAMD));
        solver.setInterpolationType(amici::InterpolationType::polynomial);
        solver.setStabilityLimitFlag(false);
        solver.setLinearSolver(amici::LinearSolver::dense);
        solver.setLinearMultistepMethod(amici::LinearMultistepMethod::adams);
        solver.setNonlinearSolverIteration(amici::NonlinearSolverIteration::newton);
        solver.setInternalSensitivityMethod(amici::InternalSensitivityMethod::staggered);
        solver.setReturnDataReportingMode(amici::RDataReporting::likelihood);
    }

    amici::CVodeSolver solver;
};

TEST(ModelSerializationTest, ToFile)
{
    int np = 1;
    int nk = 2;
    int nx = 3;
    int ny = 4;
    int nz = 5;
    int ne = 6;
    amici::CVodeSolver solver;
    amici::Model_Test m = amici::Model_Test(
        amici::ModelDimensions(
            nx,        // nx_rdata
            nx,        // nxtrue_rdata
            nx,        // nx_solver
            nx,        // nxtrue_solver
            0,         // nx_solver_reinit
            np,         // np
            nk,        // nk
            ny,        // ny
            ny,        // nytrue
            nz,        // nz
            nz,        // nztrue
            ne,        // ne
            0,         // ne_solver
            0,         // nspl
            0,         // nJ
            9,         // nw
            2,         // ndwdx
            2,         // ndwdp
            2,         // dwdw
            13,         // ndxdotdw
            {},         // ndJydy
            9,         // ndxrdatadxsolver
            0,         // ndxrdatadtcl
            0,          // ndtotal_cldx_rdata
            17,         // nnz
            18,         // ubw
            19          // lbw
            ),
        amici::SimulationParameters(
            std::vector<realtype>(nk, 0.0),
            std::vector<realtype>(np, 0.0),
            std::vector<int>(np, 0)
            ),
        amici::SecondOrderMode::none,
        std::vector<realtype>(nx, 0.0),
        std::vector<int>(nz, 0));

    {
        std::ofstream ofs("sstore.dat");
        boost::archive::text_oarchive oar(ofs);
        // oar & static_cast<amici::Solver&>(solver);
        oar& static_cast<amici::Model&>(m);
    }
    {
        std::ifstream ifs("sstore.dat");
        boost::archive::text_iarchive iar(ifs);
        amici::CVodeSolver v;
        amici::Model_Test n;
        // iar &static_cast<amici::Solver&>(v);
        iar& static_cast<amici::Model&>(n);
        // CHECK_TRUE(solver == v);
        ASSERT_EQ(m, n);
    }
}

TEST(ReturnDataSerializationTest, ToString)
{
    int np = 1;
    int nk = 2;
    int nx = 3;
    int ny = 4;
    int nz = 5;
    int ne = 6;
    amici::CVodeSolver solver;
    amici::Model_Test m = amici::Model_Test(
        amici::ModelDimensions(
            nx,        // nx_rdata
            nx,        // nxtrue_rdata
            nx,        // nx_solver
            nx,        // nxtrue_solver
            0,         // nx_solver_reinit
            np,        // np
            nk,        // nk
            ny,        // ny
            ny,        // nytrue
            nz,        // nz
            nz,        // nztrue
            ne,        // ne
            0,         // ne_solver
            0,         // nspl
            0,         // nJ
            9,         // nw
            10,        // ndwdx
            2,         // ndwdp
            12,        // dwdw
            13,        // ndxdotdw
            {},        // ndJydy
            9,         // ndxrdatadxsolver
            0,         // ndxrdatadtcl
            0,         // ndtotal_cldx_rdata
            17,        // nnz
            18,        // ubw
            19         // lbw
            ),
        amici::SimulationParameters(
            std::vector<realtype>(nk, 0.0),
            std::vector<realtype>(np, 0.0),
            std::vector<int>(np, 0)
            ),
        amici::SecondOrderMode::none,
        std::vector<realtype>(nx, 0.0),
        std::vector<int>(nz, 0));

    amici::ReturnData r(solver, m);
    r.id = "some_id";
    std::string serialized = amici::serializeToString(r);

    checkReturnDataEqual(
        r, amici::deserializeFromString<amici::ReturnData>(serialized));
}

TEST_F(SolverSerializationTest, ToChar)
{
    int length;
    char* buf = amici::serializeToChar(solver, &length);

    amici::CVodeSolver v =
        amici::deserializeFromChar<amici::CVodeSolver>(buf, length);

    delete[] buf;
    ASSERT_EQ(solver, v);
}

TEST_F(SolverSerializationTest, ToStdVec)
{

    auto buf = amici::serializeToStdVec(solver);
    amici::CVodeSolver v =
        amici::deserializeFromChar<amici::CVodeSolver>(buf.data(), buf.size());

    ASSERT_EQ(solver, v);
}
