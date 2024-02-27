#include "testfunctions.h"

#include <amici/amici.h>
#include <amici/forwardproblem.h>
#include <amici/model_ode.h>
#include <amici/solver_cvodes.h>
#include <amici/solver_idas.h>
#include <amici/symbolic_functions.h>
#include <amici/misc.h>

#include <cmath>
#include <cstring>
#include <exception>
#include <vector>

#include <gtest/gtest.h>

namespace amici {
namespace generic_model {

std::unique_ptr<Model> getModel()
{
    return std::make_unique<Model_Test>();
}

} // namespace generic_model
} // namespace amici

using namespace amici;

namespace {

void
testSolverGetterSetters(CVodeSolver solver,
                        SensitivityMethod sensi_meth,
                        SensitivityOrder sensi,
                        InternalSensitivityMethod ism,
                        InterpolationType interp,
                        NonlinearSolverIteration iter,
                        LinearMultistepMethod lmm,
                        int steps,
                        int badsteps,
                        double tol,
                        double badtol);

class ModelTest : public ::testing::Test {
  protected:
    int nx = 1, ny = 2, nz = 3, nmaxevent = 4;
    std::vector<realtype> p{ 1.0 };
    std::vector<realtype> k{ 0.5, 0.4, 0.7 };
    std::vector<int> plist{ 1 };
    std::vector<realtype> idlist{ 0 };
    std::vector<int> z2event{ 0, 0, 0 };
    Model_Test model = Model_Test(
        ModelDimensions(
            nx,        // nx_rdata
            nx,        // nxtrue_rdata
            nx,        // nx_solver
            nx,        // nxtrue_solver
            0,         // nx_solver_reinit
            static_cast<int>(p.size()),  // np
            static_cast<int>(k.size()),  // nk
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
            {},        // ndJydy
            0,         // ndxrdatadxsolver
            0,         // ndxrdatadtcl
            0,         // ndtotal_cldx_rdata
            0,         // nnz
            0,         // ubw
            0          // lbw
            ),
        SimulationParameters(k, p, plist),
        SecondOrderMode::none,
        idlist,
        z2event);
    std::vector<double> unscaled{ NAN };
};


TEST_F(ModelTest, LinScaledParameterIsNotTransformed)
{
    model.setParameterScale(ParameterScaling::none);

    ASSERT_EQ(p[0], model.getParameters()[0]);
}

TEST_F(ModelTest, LogScaledParameterIsTransformed)
{
    model.setParameterScale(ParameterScaling::ln);

    ASSERT_NEAR(std::log(p[0]), model.getParameters()[0], 1e-16);
}

TEST_F(ModelTest, Log10ScaledParameterIsTransformed)
{
    model.setParameterScale(ParameterScaling::log10);

    ASSERT_NEAR(std::log10(p[0]), model.getParameters()[0], 1e-16);
}

TEST_F(ModelTest, ParameterScaleTooShort)
{
    std::vector<ParameterScaling> pscale(p.size() - 1,
                                         ParameterScaling::log10);
    ASSERT_THROW(model.setParameterScale(pscale), AmiException);

}

TEST_F(ModelTest, ParameterScaleTooLong)
{
    std::vector<ParameterScaling> pscale (p.size() + 1,
                                         ParameterScaling::log10);
    ASSERT_THROW(model.setParameterScale(pscale), AmiException);
}

TEST_F(ModelTest, UnsortedTimepointsThrow){
    ASSERT_THROW(model.setTimepoints(std::vector<realtype>{ 0.0, 1.0, 0.5 }),
                 AmiException);
}

TEST_F(ModelTest, ParameterNameIdGetterSetter)
{
    model.setParameterById("p0", 3.0);
    ASSERT_NEAR(model.getParameterById("p0"), 3.0, 1e-16);
    ASSERT_THROW(model.getParameterById("p1"), AmiException);
    ASSERT_NEAR(
        model.setParametersByIdRegex("p[\\d]+", 5.0), p.size(), 1e-16);
    for (const auto& ip : model.getParameters())
        ASSERT_NEAR(ip, 5.0, 1e-16);
    ASSERT_THROW(model.setParametersByIdRegex("k[\\d]+", 5.0), AmiException);

    model.setParameterByName("p0", 3.0);
    ASSERT_NEAR(model.getParameterByName("p0"), 3.0, 1e-16);
    ASSERT_THROW(model.getParameterByName("p1"), AmiException);
    ASSERT_NEAR(
        model.setParametersByNameRegex("p[\\d]+", 5.0), p.size(), 1e-16);
    for (const auto& ip : model.getParameters())
        ASSERT_NEAR(ip, 5.0, 1e-16);
    ASSERT_THROW(model.setParametersByNameRegex("k[\\d]+", 5.0), AmiException);

    model.setFixedParameterById("k0", 3.0);
    ASSERT_NEAR(model.getFixedParameterById("k0"), 3.0, 1e-16);
    ASSERT_THROW(model.getFixedParameterById("k4"), AmiException);
    ASSERT_NEAR(
        model.setFixedParametersByIdRegex("k[\\d]+", 5.0), k.size(), 1e-16);
    for (const auto& ik : model.getFixedParameters())
        ASSERT_NEAR(ik, 5.0, 1e-16);
    ASSERT_THROW(model.setFixedParametersByIdRegex("p[\\d]+", 5.0), AmiException);

    model.setFixedParameterByName("k0", 3.0);
    ASSERT_NEAR(model.getFixedParameterByName("k0"), 3.0, 1e-16);
    ASSERT_THROW(model.getFixedParameterByName("k4"), AmiException);
    ASSERT_NEAR(
        model.setFixedParametersByNameRegex("k[\\d]+", 5.0), k.size(), 1e-16);
    for (const auto& ik : model.getFixedParameters())
        ASSERT_NEAR(ik, 5.0, 1e-16);
    ASSERT_THROW(model.setFixedParametersByNameRegex("p[\\d]+", 5.0),
                 AmiException);
}

TEST_F(ModelTest, ReinitializeFixedParameterInitialStates)
{
    ASSERT_THROW(model.setReinitializeFixedParameterInitialStates(true),
                 AmiException);
    model.setReinitializeFixedParameterInitialStates(false);
    ASSERT_TRUE(!model.getReinitializeFixedParameterInitialStates());
    AmiVector x(nx);
    AmiVectorArray sx(model.np(), nx);
}

TEST(SymbolicFunctionsTest, Sign)
{
    ASSERT_EQ(-1, sign(-2));
    ASSERT_EQ(0, sign(0));
    ASSERT_EQ(1, sign(2));
}

TEST(SymbolicFunctionsTest, Heaviside)
{
    ASSERT_EQ(0, heaviside(-1, 0.5));
    ASSERT_EQ(0.5, heaviside(0, 0.5));
    ASSERT_EQ(1, heaviside(1, 0.5));
}

TEST(SymbolicFunctionsTest, Min)
{
    ASSERT_EQ(-1, min(-1, 2, 0));
    ASSERT_EQ(-2, min(1, -2, 0));
    ASSERT_TRUE(isNaN(min(getNaN(), getNaN(), 0)));
    ASSERT_EQ(-1, min(-1, getNaN(), 0));
    ASSERT_EQ(-1, min(getNaN(), -1, 0));
}

TEST(SymbolicFunctionsTest, Max)
{
    ASSERT_EQ(2, max(-1, 2, 0));
    ASSERT_EQ(1, max(1, -2, 0));
    ASSERT_TRUE(isNaN(max(getNaN(), getNaN(), 0)));
    ASSERT_EQ(-1, max(-1, getNaN(), 0));
    ASSERT_EQ(-1, max(getNaN(), -1, 0));
}

TEST(SymbolicFunctionsTest, DMin)
{
    ASSERT_EQ(0, Dmin(1, -1, -2, 0));
    ASSERT_EQ(1, Dmin(1, -1, 2, 0));
    ASSERT_EQ(1, Dmin(2, -1, -2, 0));
    ASSERT_EQ(0, Dmin(2, -1, 2, 0));
}

TEST(SymbolicFunctionsTest, DMax)
{
    ASSERT_EQ(1, Dmax(1, -1, -2, 0));
    ASSERT_EQ(0, Dmax(1, -1, 2, 0));
    ASSERT_EQ(0, Dmax(2, -1, -2, 0));
    ASSERT_EQ(1, Dmax(2, -1, 2, 0));
}

TEST(SymbolicFunctionsTest, pos_pow)
{
    ASSERT_EQ(0, pos_pow(-0.1, 3));
    ASSERT_EQ(pow(0.1, 3), pos_pow(0.1, 3));
}

TEST(SolverTestBasic, Equality)
{
    IDASolver i1, i2;
    CVodeSolver c1, c2;

    ASSERT_EQ(i1, i2);
    ASSERT_EQ(c1, c2);
    ASSERT_FALSE(i1 == c1);
}

TEST(SolverTestBasic, Clone)
{
    IDASolver i1;
    std::unique_ptr<Solver> i2(i1.clone());
    ASSERT_EQ(i1, *i2);

    CVodeSolver c1;
    std::unique_ptr<Solver> c2(c1.clone());
    ASSERT_TRUE(c1 == *c2);
    ASSERT_FALSE(*i2 == *c2);
}

TEST(SolverIdasTest, DefaultConstructableAndNotLeaky)
{
    IDASolver solver;
}

TEST(SolverIdasTest, CopyCtor)
{
    IDASolver solver1;
    IDASolver solver2(solver1);
}



class SolverTest : public ::testing::Test {
  protected:
    void SetUp() override {
        tol = 0.01;
        badtol = -0.01;
        sensi_meth = SensitivityMethod::adjoint;
        sensi = SensitivityOrder::first;
        steps = 1000;
        badsteps = -1;
        lmm = LinearMultistepMethod::adams;
        iter = NonlinearSolverIteration::fixedpoint;
        ism = InternalSensitivityMethod::staggered1;
        interp = InterpolationType::polynomial;
    }

    int nx = 1, ny = 2, nz = 3, ne = 0;
    double tol, badtol;
    std::vector<realtype> timepoints = { 1, 2, 3, 4 };

    std::unique_ptr<Model> model = generic_model::getModel();
    SensitivityMethod sensi_meth;
    SensitivityOrder sensi;
    int steps, badsteps;
    LinearMultistepMethod lmm;
    NonlinearSolverIteration iter;
    InternalSensitivityMethod ism;
    InterpolationType interp;

    Model_Test testModel = Model_Test(
        ModelDimensions(
            nx,        // nx_rdata
            nx,        // nxtrue_rdata
            nx,        // nx_solver
            nx,        // nxtrue_solver
            0,         // nx_solver_reinit
            1,         // np
            3,         // nk
            ny,        // ny
            ny,        // nytrue
            nz,        // nz
            nz,        // nztrue
            ne,        // ne
            0,         // ne_solver
            0,         // nspl
            0,         // nJ
            0,         // nw
            0,         // ndwdx
            0,         // ndwdp
            0,         // dwdw
            0,         // ndxdotdw
            {},        // ndJydy
            0,         // ndxrdatadxsolver
            0,         // ndxrdatadtcl
            0,         // ndtotal_cldx_rdata
            1,         // nnz
            0,         // ubw
            0          // lbw
            ),
        SimulationParameters(
            std::vector<realtype>(3, 0.0),
            std::vector<realtype>(1, 0.0),
            std::vector<int>(2, 1)
            ),
        SecondOrderMode::none,
        std::vector<realtype>(0, 0.0),
        std::vector<int>());

    CVodeSolver solver = CVodeSolver();
};

TEST_F(SolverTest, SettersGettersNoSetup)
{
    testSolverGetterSetters(solver,
                            sensi_meth,
                            sensi,
                            ism,
                            interp,
                            iter,
                            lmm,
                            steps,
                            badsteps,
                            tol,
                            badtol);
}

TEST_F(SolverTest, SettersGettersWithSetup)
{

    solver.setSensitivityMethod(sensi_meth);
    ASSERT_EQ(static_cast<int>(solver.getSensitivityMethod()),
              static_cast<int>(sensi_meth));

    auto rdata = std::make_unique<ReturnData>(solver, testModel);
    AmiVector x(nx), dx(nx);
    AmiVectorArray sx(nx, 1), sdx(nx, 1);

    testModel.setInitialStates(std::vector<realtype>{ 0 });

    solver.setup(0, &testModel, x, dx, sx, sdx);

    testSolverGetterSetters(solver,
                            sensi_meth,
                            sensi,
                            ism,
                            interp,
                            iter,
                            lmm,
                            steps,
                            badsteps,
                            tol,
                            badtol);
}

void
testSolverGetterSetters(CVodeSolver solver,
                        SensitivityMethod sensi_meth,
                        SensitivityOrder sensi,
                        InternalSensitivityMethod ism,
                        InterpolationType interp,
                        NonlinearSolverIteration iter,
                        LinearMultistepMethod lmm,
                        int steps,
                        int badsteps,
                        double tol,
                        double badtol)
{

    solver.setSensitivityMethod(sensi_meth);
    ASSERT_EQ(static_cast<int>(solver.getSensitivityMethod()),
              static_cast<int>(sensi_meth));

    solver.setSensitivityOrder(sensi);
    ASSERT_EQ(static_cast<int>(solver.getSensitivityOrder()),
              static_cast<int>(sensi));

    solver.setInternalSensitivityMethod(ism);
    ASSERT_EQ(static_cast<int>(solver.getInternalSensitivityMethod()),
              static_cast<int>(ism));

    solver.setInterpolationType(interp);
    ASSERT_EQ(static_cast<int>(solver.getInterpolationType()),
              static_cast<int>(interp));

    solver.setNonlinearSolverIteration(iter);
    ASSERT_EQ(static_cast<int>(solver.getNonlinearSolverIteration()),
              static_cast<int>(iter));

    solver.setLinearMultistepMethod(lmm);
    ASSERT_EQ(static_cast<int>(solver.getLinearMultistepMethod()),
              static_cast<int>(lmm));

        solver.setStabilityLimitFlag(true);
    ASSERT_EQ(solver.getStabilityLimitFlag(), true);

    ASSERT_THROW(solver.setNewtonMaxSteps(badsteps), AmiException);
    solver.setNewtonMaxSteps(steps);
    ASSERT_EQ(solver.getNewtonMaxSteps(), steps);

    ASSERT_THROW(solver.setMaxSteps(badsteps), AmiException);
    solver.setMaxSteps(steps);
    ASSERT_EQ(solver.getMaxSteps(), steps);

    ASSERT_THROW(solver.setMaxStepsBackwardProblem(badsteps), AmiException);
    solver.setMaxStepsBackwardProblem(steps);
    ASSERT_EQ(solver.getMaxStepsBackwardProblem(), steps);

    ASSERT_THROW(solver.setRelativeTolerance(badtol), AmiException);
    solver.setRelativeTolerance(tol);
    ASSERT_EQ(solver.getRelativeTolerance(), tol);

    ASSERT_THROW(solver.setAbsoluteTolerance(badtol), AmiException);
    solver.setAbsoluteTolerance(tol);
    ASSERT_EQ(solver.getAbsoluteTolerance(), tol);

    ASSERT_THROW(solver.setRelativeToleranceQuadratures(badtol), AmiException);
    solver.setRelativeToleranceQuadratures(tol);
    ASSERT_EQ(solver.getRelativeToleranceQuadratures(), tol);

    ASSERT_THROW(solver.setAbsoluteToleranceQuadratures(badtol), AmiException);
    solver.setAbsoluteToleranceQuadratures(tol);
    ASSERT_EQ(solver.getAbsoluteToleranceQuadratures(), tol);

    ASSERT_THROW(solver.setRelativeToleranceSteadyState(badtol), AmiException);
    solver.setRelativeToleranceSteadyState(tol);
    ASSERT_EQ(solver.getRelativeToleranceSteadyState(), tol);

    ASSERT_THROW(solver.setAbsoluteToleranceSteadyState(badtol), AmiException);
    solver.setAbsoluteToleranceSteadyState(tol);
    ASSERT_EQ(solver.getAbsoluteToleranceSteadyState(), tol);
}

TEST_F(SolverTest, SteadyStateToleranceFactor)
{
    CVodeSolver s;
    // test with unset steadystate tolerances
    ASSERT_DOUBLE_EQ(
        s.getRelativeToleranceSteadyState(),
        s.getSteadyStateToleranceFactor() * s.getRelativeTolerance());
    ASSERT_DOUBLE_EQ(
        s.getAbsoluteToleranceSteadyState(),
        s.getSteadyStateToleranceFactor() * s.getAbsoluteTolerance());
    ASSERT_DOUBLE_EQ(
        s.getRelativeToleranceSteadyStateSensi(),
        s.getSteadyStateSensiToleranceFactor() * s.getRelativeTolerance());
    ASSERT_DOUBLE_EQ(
        s.getAbsoluteToleranceSteadyState(),
        s.getSteadyStateSensiToleranceFactor() * s.getAbsoluteTolerance());

    // test with changed steadystate tolerance factor
    s.setSteadyStateToleranceFactor(5);
    ASSERT_DOUBLE_EQ(
        s.getRelativeToleranceSteadyState(),
        s.getSteadyStateToleranceFactor() * s.getRelativeTolerance());
    ASSERT_DOUBLE_EQ(
        s.getAbsoluteToleranceSteadyState(),
        s.getSteadyStateToleranceFactor() * s.getAbsoluteTolerance());
    s.setSteadyStateSensiToleranceFactor(5);
    ASSERT_DOUBLE_EQ(
        s.getRelativeToleranceSteadyStateSensi(),
        s.getSteadyStateSensiToleranceFactor() * s.getRelativeTolerance());
    ASSERT_DOUBLE_EQ(
        s.getAbsoluteToleranceSteadyState(),
        s.getSteadyStateSensiToleranceFactor() * s.getAbsoluteTolerance());


    // test with steadystate tolerance override tolerance factor
    s.setRelativeToleranceSteadyState(2);
    ASSERT_NE(s.getRelativeToleranceSteadyState(),
        s.getSteadyStateToleranceFactor() * s.getRelativeTolerance());
    ASSERT_EQ(s.getRelativeToleranceSteadyState(), 2);
    s.setAbsoluteToleranceSteadyState(3);
    ASSERT_NE(s.getAbsoluteToleranceSteadyState(),
              s.getSteadyStateToleranceFactor() * s.getAbsoluteTolerance());
    ASSERT_EQ(s.getAbsoluteToleranceSteadyState(), 3);
    s.setRelativeToleranceSteadyStateSensi(4);
    ASSERT_NE(s.getRelativeToleranceSteadyStateSensi(),
              s.getSteadyStateSensiToleranceFactor() * s.getRelativeTolerance());
    ASSERT_EQ(s.getRelativeToleranceSteadyStateSensi(), 4);
    s.setAbsoluteToleranceSteadyStateSensi(5);
    ASSERT_NE(s.getAbsoluteToleranceSteadyStateSensi(),
              s.getSteadyStateSensiToleranceFactor() * s.getAbsoluteTolerance());
    ASSERT_EQ(s.getAbsoluteToleranceSteadyStateSensi(), 5);
}

class AmiVectorTest : public ::testing::Test {
  protected:
    std::vector<double> vec1{ 1, 2, 4, 3 };
    std::vector<double> vec2{ 4, 1, 2, 3 };
    std::vector<double> vec3{ 4, 4, 2, 1 };
};

TEST_F(AmiVectorTest, Vector)
{
    AmiVector av(vec1);
    N_Vector nvec = av.getNVector();
    for (int i = 0; i < av.getLength(); ++i)
        ASSERT_EQ(av.at(i), NV_Ith_S(nvec, i));
}

TEST_F(AmiVectorTest, VectorArray)
{
    AmiVectorArray ava(4, 3);
    AmiVector av1(vec1), av2(vec2), av3(vec3);
    std::vector<AmiVector> avs{ av1, av2, av3 };
    for (int i = 0; i < ava.getLength(); ++i)
        ava[i] = avs.at(i);

    std::vector<double> badLengthVector(13, 0.0);
    std::vector<double> flattened(12, 0.0);

    ASSERT_THROW(ava.flatten_to_vector(badLengthVector), AmiException);
    ava.flatten_to_vector(flattened);
    for (int i = 0; i < ava.getLength(); ++i) {
        const AmiVector av = ava[i];
        for (int j = 0; j < av.getLength(); ++j)
            ASSERT_EQ(flattened.at(i * av.getLength() + j), av.at(j));
    }
}

class SunMatrixWrapperTest : public ::testing::Test {
  protected:
    void SetUp() override {
        A.set_data(0, 0, 0.69);
        A.set_data(1, 0, 0.32);
        A.set_data(2, 0, 0.95);
        A.set_data(0, 1, 0.03);
        A.set_data(1, 1, 0.44);
        A.set_data(2, 1, 0.38);

        B.set_indexptr(0, 0);
        B.set_indexptr(1, 2);
        B.set_indexptr(2, 4);
        B.set_indexptr(3, 5);
        B.set_indexptr(4, 7);
        B.set_data(0, 3);
        B.set_data(1, 1);
        B.set_data(2, 3);
        B.set_data(3, 7);
        B.set_data(4, 1);
        B.set_data(5, 2);
        B.set_data(6, 9);
        B.set_indexval(0, 1);
        B.set_indexval(1, 3);
        B.set_indexval(2, 0);
        B.set_indexval(3, 2);
        B.set_indexval(4, 0);
        B.set_indexval(5, 1);
        B.set_indexval(6, 3);
    }

    //inputs
    std::vector<double> a{0.82, 0.91, 0.13};
    std::vector<double> b{0.77, 0.80};
    SUNMatrixWrapper A = SUNMatrixWrapper(3, 2);
    SUNMatrixWrapper B = SUNMatrixWrapper(4, 4, 7, CSC_MAT);
    // result
    std::vector<double> d{1.3753, 1.5084, 1.1655};
};

TEST_F(SunMatrixWrapperTest, SparseMultiply)
{

    auto A_sparse = SUNMatrixWrapper(A, 0.0, CSC_MAT);
    auto c(a); //copy c
    A_sparse.multiply(c, b);
    checkEqualArray(d, c, TEST_ATOL, TEST_RTOL, "multiply");
}

TEST_F(SunMatrixWrapperTest, SparseMultiplyEmpty)
{
    // Ensure empty Matrix vector multiplication succeeds
    auto A_sparse = SUNMatrixWrapper(1, 1, 0, CSR_MAT);
    std::vector<double> b {0.1};
    std::vector<double> c {0.1};
    A_sparse.multiply(c, b);
    ASSERT_TRUE(c[0] == 0.1);

    A_sparse = SUNMatrixWrapper(1, 1, 0, CSC_MAT);
    A_sparse.multiply(c, b);
    ASSERT_TRUE(c[0] == 0.1);
}

TEST_F(SunMatrixWrapperTest, DenseMultiply)
{
    auto c(a); //copy c
    A.multiply(c, b);
    checkEqualArray(d, c, TEST_ATOL, TEST_RTOL, "multiply");
}

TEST_F(SunMatrixWrapperTest, StdVectorCtor)
{
    auto b_amivector = AmiVector(b);
    auto a_amivector = AmiVector(a);
}

TEST_F(SunMatrixWrapperTest, TransformThrows)
{
    ASSERT_THROW(SUNMatrixWrapper(A, 0.0, 13), std::invalid_argument);
    auto A_sparse = SUNMatrixWrapper(A, 0.0, CSR_MAT);
    ASSERT_THROW(SUNMatrixWrapper(A_sparse, 0.0, CSR_MAT),
                 std::invalid_argument);
}

TEST_F(SunMatrixWrapperTest, BlockTranspose)
{
    SUNMatrixWrapper B_sparse(4, 4, 7, CSR_MAT);
    ASSERT_THROW(B.transpose(B_sparse, 1.0, 4), std::domain_error);

    B_sparse = SUNMatrixWrapper(4, 4, 7, CSC_MAT);
    B.transpose(B_sparse, -1.0, 2);
    for (int idx = 0; idx < 7; idx++) {
        ASSERT_EQ(SM_INDEXVALS_S(B.get())[idx],
                  SM_INDEXVALS_S(B_sparse.get())[idx]);
        if (idx == 1) {
            ASSERT_EQ(SM_DATA_S(B.get())[idx],
                      -SM_DATA_S(B_sparse.get())[3]);
        } else if (idx == 3) {
            ASSERT_EQ(SM_DATA_S(B.get())[idx],
                      -SM_DATA_S(B_sparse.get())[1]);
        } else {
            ASSERT_EQ(SM_DATA_S(B.get())[idx],
                      -SM_DATA_S(B_sparse.get())[idx]);
        }
    }
    for (int icol = 0; icol <= 4; icol++)
        ASSERT_EQ(SM_INDEXPTRS_S(B.get())[icol],
                  SM_INDEXPTRS_S(B_sparse.get())[icol]);
}

TEST(UnravelIndex, UnravelIndex)
{
    EXPECT_EQ(unravel_index(0, 2), std::make_pair((size_t) 0, (size_t) 0));
    EXPECT_EQ(unravel_index(1, 2), std::make_pair((size_t) 0, (size_t) 1));
    EXPECT_EQ(unravel_index(2, 2), std::make_pair((size_t) 1, (size_t) 0));
    EXPECT_EQ(unravel_index(3, 2), std::make_pair((size_t) 1, (size_t) 1));
    EXPECT_EQ(unravel_index(4, 2), std::make_pair((size_t) 2, (size_t) 0));
    EXPECT_EQ(unravel_index(5, 2), std::make_pair((size_t) 2, (size_t) 1));
    EXPECT_EQ(unravel_index(6, 2), std::make_pair((size_t) 3, (size_t) 0));
}

TEST(UnravelIndex, UnravelIndexSunMatDense)
{
    SUNMatrixWrapper A = SUNMatrixWrapper(3, 2);

    A.set_data(0, 0, 0);
    A.set_data(1, 0, 1);
    A.set_data(2, 0, 2);
    A.set_data(0, 1, 3);
    A.set_data(1, 1, 4);
    A.set_data(2, 1, 5);

    for(int i = 0; i < 6; ++i) {
        auto idx = unravel_index(i, A);
        EXPECT_EQ(A.get_data(idx.first, idx.second), i);
    }
}

TEST(UnravelIndex, UnravelIndexSunMatSparse)
{
    SUNMatrixWrapper D = SUNMatrixWrapper(4, 2);

    // [0, 3]
    // [0, 0]
    // [1, 0]
    // [2, 0]
    // data [1, 2, 3]
    // colptrs [0, 2, 3]
    // rowidxs [2, 3, 0]
    D.set_data(0, 0, 0);
    D.set_data(1, 0, 0);
    D.set_data(2, 0, 1);
    D.set_data(3, 0, 2);
    D.set_data(0, 1, 3);
    D.set_data(1, 1, 0);
    D.set_data(2, 1, 0);
    D.set_data(3, 1, 0);

    auto S = SUNSparseFromDenseMatrix(D, 1e-15, CSC_MAT);

    EXPECT_EQ(unravel_index(0, S), std::make_pair((sunindextype) 2, (sunindextype) 0));
    EXPECT_EQ(unravel_index(1, S), std::make_pair((sunindextype) 3, (sunindextype) 0));
    EXPECT_EQ(unravel_index(2, S), std::make_pair((sunindextype) 0, (sunindextype) 1));

    SUNMatDestroy(S);
}


TEST(UnravelIndex, UnravelIndexSunMatSparseMissingIndices)
{
    // Sparse matrix without any indices set
    SUNMatrixWrapper mat = SUNMatrixWrapper(2, 3, 2, CSC_MAT);
    EXPECT_EQ(unravel_index(0, mat), std::make_pair((sunindextype) -1, (sunindextype) -1));
    EXPECT_EQ(unravel_index(1, mat), std::make_pair((sunindextype) -1, (sunindextype) -1));
}


TEST(ReturnCodeToStr, ReturnCodeToStr)
{
    EXPECT_EQ("AMICI_SUCCESS", simulation_status_to_str(AMICI_SUCCESS));
    EXPECT_EQ("AMICI_UNRECOVERABLE_ERROR",
              simulation_status_to_str(AMICI_UNRECOVERABLE_ERROR));
}

TEST(SpanEqual, SpanEqual)
{
    std::vector<realtype> a {1, 2, 3};
    std::vector<realtype> b {1, 2, NAN};

    EXPECT_TRUE(is_equal(a, a));
    EXPECT_TRUE(is_equal(b, b));
    EXPECT_FALSE(is_equal(a, b));
}

TEST(CpuTimer, CpuTimer)
{
    amici::CpuTimer timer;
    auto elapsed = timer.elapsed_seconds();
    EXPECT_LE(0.0, elapsed);
    EXPECT_GT(1.0, elapsed);

    elapsed = timer.elapsed_milliseconds();
    EXPECT_LT(0.0, elapsed);
    EXPECT_GT(1000.0, elapsed);
}

} // namespace
