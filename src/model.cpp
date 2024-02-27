#include <amici/amici.h>
#include <amici/cblas.h>
#include <amici/exception.h>
#include <amici/misc.h>
#include <amici/model.h>
#include <amici/symbolic_functions.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <regex>
#include <typeinfo>
#include <utility>

namespace amici {

/**
 * @brief Maps ModelQuantity items to their string value
 */
std::map<ModelQuantity, std::string> const model_quantity_to_str{
    {ModelQuantity::J, "J"},
    {ModelQuantity::JB, "JB"},
    {ModelQuantity::Jv, "Jv"},
    {ModelQuantity::JvB, "JvB"},
    {ModelQuantity::JDiag, "JDiag"},
    {ModelQuantity::sx, "sx"},
    {ModelQuantity::sy, "sy"},
    {ModelQuantity::sz, "sz"},
    {ModelQuantity::srz, "srz"},
    {ModelQuantity::ssigmay, "ssigmay"},
    {ModelQuantity::ssigmaz, "ssigmaz"},
    {ModelQuantity::xdot, "xdot"},
    {ModelQuantity::sxdot, "sxdot"},
    {ModelQuantity::xBdot, "xBdot"},
    {ModelQuantity::x0, "x0"},
    {ModelQuantity::x0_rdata, "x0_rdata"},
    {ModelQuantity::x, "x"},
    {ModelQuantity::x_rdata, "x_rdata"},
    {ModelQuantity::dwdw, "dwdw"},
    {ModelQuantity::dwdx, "dwdx"},
    {ModelQuantity::dwdp, "dwdp"},
    {ModelQuantity::y, "y"},
    {ModelQuantity::dydp, "dydp"},
    {ModelQuantity::dydx, "dydx"},
    {ModelQuantity::w, "w"},
    {ModelQuantity::root, "root"},
    {ModelQuantity::qBdot, "qBdot"},
    {ModelQuantity::qBdot_ss, "qBdot_ss"},
    {ModelQuantity::xBdot_ss, "xBdot_ss"},
    {ModelQuantity::JSparseB_ss, "JSparseB_ss"},
    {ModelQuantity::deltax, "deltax"},
    {ModelQuantity::deltasx, "deltasx"},
    {ModelQuantity::deltaxB, "deltaxB"},
    {ModelQuantity::k, "k"},
    {ModelQuantity::p, "p"},
    {ModelQuantity::ts, "ts"},
    {ModelQuantity::dJydy, "dJydy"},
    {ModelQuantity::dJydy_matlab, "dJydy"},
    {ModelQuantity::deltaqB, "deltaqB"},
    {ModelQuantity::dsigmaydp, "dsigmaydp"},
    {ModelQuantity::dsigmaydy, "dsigmaydy"},
    {ModelQuantity::dsigmazdp, "dsigmazdp"},
    {ModelQuantity::dJydsigma, "dJydsigma"},
    {ModelQuantity::dJydx, "dJydx"},
    {ModelQuantity::dJrzdx, "dJrzdx"},
    {ModelQuantity::dJzdx, "dJzdx"},
    {ModelQuantity::dzdp, "dzdp"},
    {ModelQuantity::dzdx, "dzdx"},
    {ModelQuantity::dJrzdsigma, "dJrzdsigma"},
    {ModelQuantity::dJrzdz, "dJrzdz"},
    {ModelQuantity::dJzdsigma, "dJzdsigma"},
    {ModelQuantity::dJzdz, "dJzdz"},
    {ModelQuantity::drzdp, "drzdp"},
    {ModelQuantity::drzdx, "drzdx"},

};

static void setNaNtoZero(std::vector<realtype>& vec) {
    std::for_each(vec.begin(), vec.end(), [](double& val) {
        if (std::isnan(val)) {
            val = 0.0;
        }
    });
}

/**
 * @brief local helper function to get parameters
 * @param ids vector of name/ids of (fixed)Parameters
 * @param values values of the (fixed)Parameters
 * @param id name/id to look for in the vector
 * @param variable_name string indicating what variable we are looking at
 * @param id_name string indicating whether name or id was specified
 * @return value of the selected parameter
 */
static realtype getValueById(
    std::vector<std::string> const& ids, std::vector<realtype> const& values,
    std::string const& id, char const* variable_name, char const* id_name
) {
    auto it = std::find(ids.begin(), ids.end(), id);
    if (it != ids.end())
        return values.at(it - ids.begin());

    throw AmiException(
        "Could not find %s with specified %s", variable_name, id_name
    );
}

/**
 * @brief local helper function to set parameters
 * @param ids vector of names/ids of (fixed)Parameters
 * @param values values of the (fixed)Parameters
 * @param value for the selected parameter
 * @param id name/id to look for in the vector
 * @param variable_name string indicating what variable we are looking at
 * @param id_name string indicating whether name or id was specified
 */
static void setValueById(
    std::vector<std::string> const& ids, std::vector<realtype>& values,
    realtype value, std::string const& id, char const* variable_name,
    char const* id_name
) {
    auto it = std::find(ids.begin(), ids.end(), id);
    if (it != ids.end())
        values.at(it - ids.begin()) = value;
    else
        throw AmiException(
            "Could not find %s with specified %s", variable_name, id_name
        );
}

/**
 * @brief local helper function to set parameters via regex
 * @param ids vector of names/ids of (fixed)Parameters
 * @param values values of the (fixed)Parameters
 * @param value for the selected parameter
 * @param regex string according to which names/ids are to be matched
 * @param variable_name string indicating what variable we are looking at
 * @param id_name string indicating whether name or id was specified
 * @return number of matched names/ids
 */

static int setValueByIdRegex(
    std::vector<std::string> const& ids, std::vector<realtype>& values,
    realtype value, std::string const& regex, char const* variable_name,
    char const* id_name
) {
    try {
        std::regex pattern(regex);
        int n_found = 0;
        for (auto const& id : ids) {
            if (std::regex_match(id, pattern)) {
                values.at(&id - &ids[0]) = value;
                ++n_found;
            }
        }

        if (n_found == 0)
            throw AmiException(
                "Could not find %s with specified %s (%s)", variable_name,
                id_name, regex.c_str()
            );

        return n_found;
    } catch (std::regex_error const& e) {
        auto err_string = regexErrorToString(e.code());
        throw AmiException(
            "Specified regex pattern %s could not be compiled:"
            " %s (%s)",
            regex.c_str(), e.what(), err_string.c_str()
        );
    }
}

Model::Model(
    ModelDimensions const& model_dimensions,
    SimulationParameters simulation_parameters, SecondOrderMode o2mode,
    std::vector<realtype> idlist, std::vector<int> z2event,
    bool const pythonGenerated, int const ndxdotdp_explicit,
    int const ndxdotdx_explicit, int const w_recursion_depth,
    std::map<realtype, std::vector<int>> state_independent_events
)
    : ModelDimensions(model_dimensions)
    , pythonGenerated(pythonGenerated)
    , o2mode(o2mode)
    , idlist(std::move(idlist))
    , state_independent_events_(std::move(state_independent_events))
    , derived_state_(model_dimensions)
    , z2event_(std::move(z2event))
    , state_is_non_negative_(nx_solver, false)
    , w_recursion_depth_(w_recursion_depth)
    , simulation_parameters_(std::move(simulation_parameters)) {
    Expects(
        model_dimensions.np
        == gsl::narrow<int>(simulation_parameters_.parameters.size())
    );
    Expects(
        model_dimensions.nk
        == gsl::narrow<int>(simulation_parameters_.fixedParameters.size())
    );

    simulation_parameters.pscale = std::vector<ParameterScaling>(
        model_dimensions.np, ParameterScaling::none
    );

    state_.h.resize(ne, 0.0);
    state_.total_cl.resize(nx_rdata - nx_solver, 0.0);
    state_.stotal_cl.resize((nx_rdata - nx_solver) * np(), 0.0);
    state_.unscaledParameters.resize(np());
    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
    state_.fixedParameters = simulation_parameters_.fixedParameters;
    state_.plist = simulation_parameters_.plist;

    root_initial_values_.resize(ne, true);

    /* If Matlab wrapped: dxdotdp is a full AmiVector,
       if Python wrapped: dxdotdp_explicit and dxdotdp_implicit are CSC matrices
     */
    if (pythonGenerated) {

        dwdw_ = SUNMatrixWrapper(nw, nw, ndwdw, CSC_MAT);
        // size dynamically adapted for dwdx_ and dwdp_
        derived_state_.dwdx_ = SUNMatrixWrapper(nw, nx_solver, 0, CSC_MAT);
        derived_state_.dwdp_ = SUNMatrixWrapper(nw, np(), 0, CSC_MAT);

        for (int irec = 0; irec <= w_recursion_depth_; ++irec) {
            /* for the first element we know the exact size, while for all
               others we guess the size*/
            dwdp_hierarchical_.emplace_back(
                SUNMatrixWrapper(nw, np(), irec * ndwdw + ndwdp, CSC_MAT)
            );
            dwdx_hierarchical_.emplace_back(
                SUNMatrixWrapper(nw, nx_solver, irec * ndwdw + ndwdx, CSC_MAT)
            );
        }
        assert(
            gsl::narrow<int>(dwdp_hierarchical_.size())
            == w_recursion_depth_ + 1
        );
        assert(
            gsl::narrow<int>(dwdx_hierarchical_.size())
            == w_recursion_depth_ + 1
        );

        derived_state_.dxdotdp_explicit
            = SUNMatrixWrapper(nx_solver, np(), ndxdotdp_explicit, CSC_MAT);
        // guess size, will be dynamically reallocated
        derived_state_.dxdotdp_implicit
            = SUNMatrixWrapper(nx_solver, np(), ndwdp + ndxdotdw, CSC_MAT);
        derived_state_.dxdotdx_explicit = SUNMatrixWrapper(
            nx_solver, nx_solver, ndxdotdx_explicit, CSC_MAT
        );
        // guess size, will be dynamically reallocated
        derived_state_.dxdotdx_implicit
            = SUNMatrixWrapper(nx_solver, nx_solver, ndwdx + ndxdotdw, CSC_MAT);
        // dynamically allocate on first call
        derived_state_.dxdotdp_full
            = SUNMatrixWrapper(nx_solver, np(), 0, CSC_MAT);

        for (int iytrue = 0; iytrue < nytrue; ++iytrue)
            derived_state_.dJydy_.emplace_back(
                SUNMatrixWrapper(nJ, ny, ndJydy.at(iytrue), CSC_MAT)
            );
    } else {
        derived_state_.dwdx_ = SUNMatrixWrapper(nw, nx_solver, ndwdx, CSC_MAT);
        derived_state_.dwdp_ = SUNMatrixWrapper(nw, np(), ndwdp, CSC_MAT);
        derived_state_.dJydy_matlab_
            = std::vector<realtype>(nJ * nytrue * ny, 0.0);
    }
    requireSensitivitiesForAllParameters();
}

bool operator==(Model const& a, Model const& b) {
    if (typeid(a) != typeid(b))
        return false;

    return (static_cast<ModelDimensions const&>(a)
            == static_cast<ModelDimensions const&>(b))
           && (a.o2mode == b.o2mode) && (a.z2event_ == b.z2event_)
           && (a.idlist == b.idlist)
           && (a.simulation_parameters_ == b.simulation_parameters_)
           && (a.x0data_ == b.x0data_) && (a.sx0data_ == b.sx0data_)
           && (a.nmaxevent_ == b.nmaxevent_)
           && (a.state_is_non_negative_ == b.state_is_non_negative_)
           && (a.sigma_res_ == b.sigma_res_) && (a.min_sigma_ == b.min_sigma_)
           && a.state_ == b.state_;
}

bool operator==(ModelDimensions const& a, ModelDimensions const& b) {
    if (typeid(a) != typeid(b))
        return false;
    return (a.nx_rdata == b.nx_rdata) && (a.nxtrue_rdata == b.nxtrue_rdata)
           && (a.nx_solver == b.nx_solver)
           && (a.nxtrue_solver == b.nxtrue_solver)
           && (a.nx_solver_reinit == b.nx_solver_reinit) && (a.np == b.np)
           && (a.nk == b.nk) && (a.ny == b.ny) && (a.nytrue == b.nytrue)
           && (a.nz == b.nz) && (a.nztrue == b.nztrue) && (a.ne == b.ne)
           && (a.ne_solver == b.ne_solver) && (a.nspl == b.nspl)
           && (a.nw == b.nw) && (a.ndwdx == b.ndwdx) && (a.ndwdp == b.ndwdp)
           && (a.ndwdw == b.ndwdw) && (a.ndxdotdw == b.ndxdotdw)
           && (a.ndJydy == b.ndJydy) && (a.nnz == b.nnz) && (a.nJ == b.nJ)
           && (a.ubw == b.ubw) && (a.lbw == b.lbw);
}

void Model::initialize(
    AmiVector& x, AmiVector& dx, AmiVectorArray& sx, AmiVectorArray& /*sdx*/,
    bool computeSensitivities, std::vector<int>& roots_found
) {
    initializeStates(x);
    initializeSplines();
    if (computeSensitivities) {
        initializeStateSensitivities(sx, x);
        initializeSplineSensitivities();
    }

    fdx0(x, dx);
    if (computeSensitivities)
        fsdx0();

    if (ne)
        initEvents(x, dx, roots_found);

    // evaluate static expressions once
    auto x_pos = computeX_pos(x);
    fw(t0(), x_pos, true);
    fdwdw(t0(), x_pos, true);
    fdwdx(t0(), x_pos, true);
    if (computeSensitivities) {
        fdwdp(t0(), x_pos, true);
    }
}

void Model::reinitialize(
    realtype t, AmiVector& x, AmiVectorArray& sx, bool computeSensitivities
) {
    fx0_fixedParameters(x);

    // re-evaluate static expressions once
    auto x_pos = computeX_pos(x);
    fw(t, x_pos, true);
    fdwdw(t, x_pos, true);
    fdwdx(t, x_pos, true);
    if (computeSensitivities) {
        fsx0_fixedParameters(sx, x);
        fdwdp(t, x_pos, true);
    }
}

void Model::initializeB(
    AmiVector& xB, AmiVector& dxB, AmiVector& xQB, bool posteq
) const {
    xB.zero();
    dxB.zero();
    if (!posteq)
        xQB.zero();
}

void Model::initializeStates(AmiVector& x) {
    if (x0data_.empty()) {
        fx0(x);
    } else {
        std::vector<realtype> x0_solver(nx_solver, 0.0);
        ftotal_cl(
            state_.total_cl.data(), x0data_.data(),
            state_.unscaledParameters.data(), state_.fixedParameters.data()
        );
        fx_solver(x0_solver.data(), x0data_.data());
        std::copy(x0_solver.cbegin(), x0_solver.cend(), x.data());
    }

    checkFinite(x.getVector(), ModelQuantity::x0);
}

void Model::initializeSplines() {
    splines_ = fcreate_splines(
        state_.unscaledParameters.data(), state_.fixedParameters.data()
    );
    state_.spl_.resize(splines_.size(), 0.0);
    for (auto& spline : splines_) {
        spline.compute_coefficients();
    }
}

void Model::initializeSplineSensitivities() {
    derived_state_.sspl_ = SUNMatrixWrapper(splines_.size(), np());
    int allnodes = 0;
    for (auto const& spline : splines_) {
        allnodes += spline.n_nodes();
    }

    std::vector<realtype> dspline_valuesdp(allnodes * nplist(), 0.0);
    std::vector<realtype> dspline_slopesdp(allnodes * nplist(), 0.0);
    std::vector<realtype> tmp_dvalues(allnodes, 0.0);
    std::vector<realtype> tmp_dslopes(allnodes, 0.0);
    for (int ip = 0; ip < nplist(); ip++) {
        std::fill(tmp_dvalues.begin(), tmp_dvalues.end(), 0.0);
        std::fill(tmp_dslopes.begin(), tmp_dslopes.end(), 0.0);
        fdspline_valuesdp(
            tmp_dvalues.data(), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), plist(ip)
        );
        fdspline_slopesdp(
            tmp_dslopes.data(), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), plist(ip)
        );
        /* NB dspline_valuesdp/dspline_slopesdp must be filled
         * using the following order for the indices
         * (from slower to faster): spline, node, parameter.
         * That is what the current spline implementation expects.
         */
        int k = 0;
        int offset = ip;
        for (auto const& spline : splines_) {
            for (int n = 0; n < spline.n_nodes(); n++) {
                dspline_valuesdp[offset] = tmp_dvalues[k];
                dspline_slopesdp[offset] = tmp_dslopes[k];
                offset += nplist();
                k += 1;
            }
        }
        assert(k == allnodes);
    }

    int spline_offset = 0;
    for (auto& spline : splines_) {
        spline.compute_coefficients_sensi(
            nplist(), spline_offset, dspline_valuesdp, dspline_slopesdp
        );
        spline_offset += spline.n_nodes() * nplist();
    }
}

void Model::initializeStateSensitivities(
    AmiVectorArray& sx, AmiVector const& x
) {
    if (sx0data_.empty()) {
        fsx0(sx, x);
    } else {
        realtype* stcl = nullptr;
        std::vector<realtype> sx0_solver_slice(nx_solver, 0.0);
        for (int ip = 0; ip < nplist(); ip++) {
            if (ncl() > 0)
                stcl = &state_.stotal_cl.at(plist(ip) * ncl());
            fstotal_cl(
                stcl, &sx0data_.at(ip * nx_rdata), plist(ip),
                derived_state_.x_rdata_.data(),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.total_cl.data()
            );
            fsx_solver(sx0_solver_slice.data(), &sx0data_.at(ip * nx_rdata));
            for (int ix = 0; ix < nx_solver; ix++) {
                sx.at(ix, ip) = sx0_solver_slice.at(ix);
            }
        }
    }
}

void Model::initEvents(
    AmiVector const& x, AmiVector const& dx, std::vector<int>& roots_found
) {
    std::vector<realtype> rootvals(ne, 0.0);
    froot(simulation_parameters_.tstart_, x, dx, rootvals);
    std::fill(roots_found.begin(), roots_found.end(), 0);
    for (int ie = 0; ie < ne; ie++) {
        if (rootvals.at(ie) < 0) {
            state_.h.at(ie) = 0.0;
        } else {
            state_.h.at(ie) = 1.0;
            if (!root_initial_values_.at(ie)) // only false->true triggers event
                roots_found.at(ie) = 1;
        }
    }
}

int Model::nplist() const { return gsl::narrow<int>(state_.plist.size()); }

int Model::np() const {
    return gsl::narrow<int>(static_cast<ModelDimensions const&>(*this).np);
}

int Model::nk() const {
    return gsl::narrow<int>(state_.fixedParameters.size());
}

int Model::ncl() const { return nx_rdata - nx_solver; }

int Model::nx_reinit() const { return nx_solver_reinit; }

double const* Model::k() const { return state_.fixedParameters.data(); }

int Model::nMaxEvent() const { return nmaxevent_; }

void Model::setNMaxEvent(int nmaxevent) { nmaxevent_ = nmaxevent; }

int Model::nt() const {
    return gsl::narrow<int>(simulation_parameters_.ts_.size());
}

std::vector<ParameterScaling> const& Model::getParameterScale() const {
    return simulation_parameters_.pscale;
}

void Model::setParameterScale(ParameterScaling pscale) {
    simulation_parameters_.pscale.assign(
        simulation_parameters_.pscale.size(), pscale
    );
    scaleParameters(
        state_.unscaledParameters, simulation_parameters_.pscale,
        simulation_parameters_.parameters
    );
    sx0data_.clear();
}

void Model::setParameterScale(std::vector<ParameterScaling> const& pscaleVec) {
    if (pscaleVec.size() != simulation_parameters_.parameters.size())
        throw AmiException("Dimension mismatch. Size of parameter scaling does "
                           "not match number of model parameters.");
    simulation_parameters_.pscale = pscaleVec;
    scaleParameters(
        state_.unscaledParameters, simulation_parameters_.pscale,
        simulation_parameters_.parameters
    );
    sx0data_.clear();
}

std::vector<realtype> const& Model::getUnscaledParameters() const {
    return state_.unscaledParameters;
}

std::vector<realtype> const& Model::getParameters() const {
    return simulation_parameters_.parameters;
}

realtype Model::getParameterById(std::string const& par_id) const {
    if (!hasParameterIds())
        throw AmiException(
            "Could not access parameters by id as they are not set"
        );
    return getValueById(
        getParameterIds(), simulation_parameters_.parameters, par_id,
        "parameters", "id"
    );
}

realtype Model::getParameterByName(std::string const& par_name) const {
    if (!hasParameterNames())
        throw AmiException(
            "Could not access parameters by name as they are not set"
        );
    return getValueById(
        getParameterNames(), simulation_parameters_.parameters, par_name,
        "parameters", "name"
    );
}

void Model::setParameters(std::vector<realtype> const& p) {
    if (p.size() != (unsigned)np())
        throw AmiException("Dimension mismatch. Size of parameters does not "
                           "match number of model parameters.");
    simulation_parameters_.parameters = p;
    state_.unscaledParameters.resize(simulation_parameters_.parameters.size());
    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
}

void Model::setParameterById(
    std::map<std::string, realtype> const& p, bool ignoreErrors
) {
    for (auto& kv : p) {
        try {
            setParameterById(kv.first, kv.second);
        } catch (AmiException const&) {
            if (!ignoreErrors)
                throw;
        }
    }
}

void Model::setParameterById(std::string const& par_id, realtype value) {
    if (!hasParameterIds())
        throw AmiException(
            "Could not access parameters by id as they are not set"
        );

    setValueById(
        getParameterIds(), simulation_parameters_.parameters, value, par_id,
        "parameter", "id"
    );
    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
}

int Model::setParametersByIdRegex(
    std::string const& par_id_regex, realtype value
) {
    if (!hasParameterIds())
        throw AmiException(
            "Could not access parameters by id as they are not set"
        );
    int n_found = setValueByIdRegex(
        getParameterIds(), simulation_parameters_.parameters, value,
        par_id_regex, "parameter", "id"
    );
    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
    return n_found;
}

void Model::setParameterByName(std::string const& par_name, realtype value) {
    if (!hasParameterNames())
        throw AmiException(
            "Could not access parameters by name as they are not set"
        );

    setValueById(
        getParameterNames(), simulation_parameters_.parameters, value, par_name,
        "parameter", "name"
    );
    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
}

void Model::setParameterByName(
    std::map<std::string, realtype> const& p, bool ignoreErrors
) {
    for (auto& kv : p) {
        try {
            setParameterByName(kv.first, kv.second);
        } catch (AmiException const&) {
            if (!ignoreErrors)
                throw;
        }
    }
}

int Model::setParametersByNameRegex(
    std::string const& par_name_regex, realtype value
) {
    if (!hasParameterNames())
        throw AmiException(
            "Could not access parameters by name as they are not set"
        );

    int n_found = setValueByIdRegex(
        getParameterNames(), simulation_parameters_.parameters, value,
        par_name_regex, "parameter", "name"
    );

    unscaleParameters(
        simulation_parameters_.parameters, simulation_parameters_.pscale,
        state_.unscaledParameters
    );
    return n_found;
}

std::vector<realtype> const& Model::getFixedParameters() const {
    return state_.fixedParameters;
}

realtype Model::getFixedParameterById(std::string const& par_id) const {
    if (!hasFixedParameterIds())
        throw AmiException(
            "Could not access fixed parameters by id as they are not set"
        );

    return getValueById(
        getFixedParameterIds(), state_.fixedParameters, par_id,
        "fixedParameters", "id"
    );
}

realtype Model::getFixedParameterByName(std::string const& par_name) const {
    if (!hasFixedParameterNames())
        throw AmiException(
            "Could not access fixed parameters by name as they are not set"
        );

    return getValueById(
        getFixedParameterNames(), state_.fixedParameters, par_name,
        "fixedParameters", "name"
    );
}

void Model::setFixedParameters(std::vector<realtype> const& k) {
    if (k.size() != (unsigned)nk())
        throw AmiException("Dimension mismatch. Size of fixedParameters does "
                           "not match number of fixed model parameters.");
    state_.fixedParameters = k;
}

void Model::setFixedParameterById(std::string const& par_id, realtype value) {
    if (!hasFixedParameterIds())
        throw AmiException(
            "Could not access fixed parameters by id as they are not set"
        );

    setValueById(
        getFixedParameterIds(), state_.fixedParameters, value, par_id,
        "fixedParameters", "id"
    );
}

int Model::setFixedParametersByIdRegex(
    std::string const& par_id_regex, realtype value
) {
    if (!hasFixedParameterIds())
        throw AmiException(
            "Could not access fixed parameters by id as they are not set"
        );

    return setValueByIdRegex(
        getFixedParameterIds(), state_.fixedParameters, value, par_id_regex,
        "fixedParameters", "id"
    );
}

void Model::setFixedParameterByName(
    std::string const& par_name, realtype value
) {
    if (!hasFixedParameterNames())
        throw AmiException(
            "Could not access fixed parameters by name as they are not set"
        );

    setValueById(
        getFixedParameterNames(), state_.fixedParameters, value, par_name,
        "fixedParameters", "name"
    );
}

int Model::setFixedParametersByNameRegex(
    std::string const& par_name_regex, realtype value
) {
    if (!hasFixedParameterNames())
        throw AmiException(
            "Could not access fixed parameters by name as they are not set"
        );

    return setValueByIdRegex(
        getFixedParameterIds(), state_.fixedParameters, value, par_name_regex,
        "fixedParameters", "name"
    );
}

std::string Model::getName() const { return ""; }

bool Model::hasParameterNames() const {
    return np() == 0 || !getParameterNames().empty();
}

std::vector<std::string> Model::getParameterNames() const {
    return std::vector<std::string>();
}

bool Model::hasStateNames() const {
    return nx_rdata == 0 || !getStateNames().empty();
}

std::vector<std::string> Model::getStateNames() const {
    return std::vector<std::string>();
}

std::vector<std::string> Model::getStateNamesSolver() const {
    return std::vector<std::string>();
}

bool Model::hasFixedParameterNames() const {
    return nk() == 0 || !getFixedParameterNames().empty();
}

std::vector<std::string> Model::getFixedParameterNames() const {
    return std::vector<std::string>();
}

bool Model::hasObservableNames() const {
    return ny == 0 || !getObservableNames().empty();
}

std::vector<std::string> Model::getObservableNames() const {
    return std::vector<std::string>();
}

bool Model::hasExpressionNames() const {
    return ny == 0 || !getExpressionNames().empty();
}

std::vector<std::string> Model::getExpressionNames() const {
    return std::vector<std::string>();
}

bool Model::hasParameterIds() const {
    return np() == 0 || !getParameterIds().empty();
}

std::vector<std::string> Model::getParameterIds() const {
    return std::vector<std::string>();
}

bool Model::hasStateIds() const {
    return nx_rdata == 0 || !getStateIds().empty();
}

std::vector<std::string> Model::getStateIds() const {
    return std::vector<std::string>();
}

std::vector<std::string> Model::getStateIdsSolver() const {
    return std::vector<std::string>();
}

bool Model::hasFixedParameterIds() const {
    return nk() == 0 || !getFixedParameterIds().empty();
}

std::vector<std::string> Model::getFixedParameterIds() const {
    return std::vector<std::string>();
}

bool Model::hasObservableIds() const {
    return ny == 0 || !getObservableIds().empty();
}

std::vector<std::string> Model::getObservableIds() const {
    return std::vector<std::string>();
}

bool Model::hasExpressionIds() const {
    return ny == 0 || !getExpressionIds().empty();
}

std::vector<std::string> Model::getExpressionIds() const {
    return std::vector<std::string>();
}

bool Model::hasQuadraticLLH() const { return true; }

std::vector<realtype> const& Model::getTimepoints() const {
    return simulation_parameters_.ts_;
}

double Model::getTimepoint(int const it) const {
    return simulation_parameters_.ts_.at(it);
}

void Model::setTimepoints(std::vector<realtype> const& ts) {
    if (!std::is_sorted(ts.begin(), ts.end()))
        throw AmiException("Encountered non-monotonic timepoints, please order"
                           " timepoints such that they are monotonically"
                           " increasing!");
    simulation_parameters_.ts_ = ts;
}

double Model::t0() const { return simulation_parameters_.tstart_; }

void Model::setT0(double t0) { simulation_parameters_.tstart_ = t0; }

std::vector<bool> const& Model::getStateIsNonNegative() const {
    return state_is_non_negative_;
}

void Model::setStateIsNonNegative(std::vector<bool> const& nonNegative) {
    auto any_state_non_negative
        = std::any_of(nonNegative.begin(), nonNegative.end(), [](bool x) {
              return x;
          });
    if (nx_solver != nx_rdata) {
        if (any_state_non_negative)
            throw AmiException("Non-negative states are not supported with"
                               " conservation laws enabled.");
        // nothing to do, as `state_is_non_negative_` will always be all-false
        // in case of conservation laws
        return;
    }
    if (state_is_non_negative_.size() != gsl::narrow<unsigned long>(nx_rdata)) {
        throw AmiException(
            "Dimension of input stateIsNonNegative (%u) does "
            "not agree with number of state variables (%d)",
            state_is_non_negative_.size(), nx_rdata
        );
    }
    state_is_non_negative_ = nonNegative;
    any_state_non_negative_ = any_state_non_negative;
}

void Model::setAllStatesNonNegative() {
    setStateIsNonNegative(std::vector<bool>(nx_solver, true));
}

std::vector<int> const& Model::getParameterList() const { return state_.plist; }

int Model::plist(int pos) const { return state_.plist.at(pos); }

void Model::setParameterList(std::vector<int> const& plist) {
    int np = this->np(); // cannot capture 'this' in lambda expression
    if (std::any_of(plist.begin(), plist.end(), [&np](int idx) {
            return idx < 0 || idx >= np;
        })) {
        throw AmiException("Indices in plist must be in [0..np]");
    }
    state_.plist = plist;

    initializeVectors();
}

std::vector<realtype> Model::getInitialStates() {
    if (!x0data_.empty()) {
        return x0data_;
    }

    /* Initial states have not been set explicitly on this instance, so we
     * compute it, but don't save it, as this would have to be invalidated upon
     * changing parameters etc.
     */
    std::vector<realtype> x0(nx_rdata, 0.0);
    fx0(x0.data(), simulation_parameters_.tstart_,
        state_.unscaledParameters.data(), state_.fixedParameters.data());
    return x0;
}

void Model::setInitialStates(std::vector<realtype> const& x0) {
    if (x0.size() != (unsigned)nx_rdata && !x0.empty())
        throw AmiException("Dimension mismatch. Size of x0 does not match "
                           "number of model states.");

    if (x0.empty()) {
        x0data_.clear();
        return;
    }

    x0data_ = x0;
}

bool Model::hasCustomInitialStates() const { return !x0data_.empty(); }

std::vector<realtype> Model::getInitialStateSensitivities() {
    if (!sx0data_.empty()) {
        return sx0data_;
    }

    /* Initial state sensitivities have not been set explicitly on this
     * instance, so we compute it, but don't save it, as this would have to be
     * invalidated upon changing parameters etc.
     */
    std::vector<realtype> sx0(nx_rdata * nplist(), 0.0);
    auto x0 = getInitialStates();
    for (int ip = 0; ip < nplist(); ip++) {
        fsx0(
            sx0.data(), simulation_parameters_.tstart_, x0.data(),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            plist(ip)
        );
    }
    return sx0;
}

void Model::setInitialStateSensitivities(std::vector<realtype> const& sx0) {
    if (sx0.size() != (unsigned)nx_rdata * nplist() && !sx0.empty())
        throw AmiException("Dimension mismatch. Size of sx0 does not match "
                           "number of model states * number of parameter "
                           "selected for sensitivities.");

    if (sx0.empty()) {
        sx0data_.clear();
        return;
    }

    realtype chainrulefactor = 1.0;
    std::vector<realtype> sx0_rdata(nx_rdata * nplist(), 0.0);
    for (int ip = 0; ip < nplist(); ip++) {

        // revert chainrule
        switch (simulation_parameters_.pscale.at(plist(ip))) {
        case ParameterScaling::log10:
            chainrulefactor = state_.unscaledParameters.at(plist(ip)) * log(10);
            break;
        case ParameterScaling::ln:
            chainrulefactor = state_.unscaledParameters.at(plist(ip));
            break;
        case ParameterScaling::none:
            chainrulefactor = 1.0;
            break;
        }

        for (int ix = 0; ix < nx_rdata; ++ix) {
            sx0_rdata.at(ip * nx_rdata + ix)
                = sx0.at(ip * nx_rdata + ix) / chainrulefactor;
        }
    }
    setUnscaledInitialStateSensitivities(sx0_rdata);
}

bool Model::hasCustomInitialStateSensitivities() const {
    return !sx0data_.empty();
}

void Model::setUnscaledInitialStateSensitivities(
    std::vector<realtype> const& sx0
) {
    if (sx0.size() != (unsigned)nx_rdata * nplist() && !sx0.empty())
        throw AmiException("Dimension mismatch. Size of sx0 does not match "
                           "number of model states * number of parameter "
                           "selected for sensitivities.");

    if (sx0.empty()) {
        sx0data_.clear();
        return;
    }

    sx0data_ = sx0;
}

void Model::setSteadyStateComputationMode(SteadyStateComputationMode const mode
) {
    steadystate_computation_mode_ = mode;
}

SteadyStateComputationMode Model::getSteadyStateComputationMode() const {
    return steadystate_computation_mode_;
}

void Model::setSteadyStateSensitivityMode(SteadyStateSensitivityMode const mode
) {
    steadystate_sensitivity_mode_ = mode;
}

SteadyStateSensitivityMode Model::getSteadyStateSensitivityMode() const {
    return steadystate_sensitivity_mode_;
}

void Model::setReinitializeFixedParameterInitialStates(bool flag) {
    if (flag && !isFixedParameterStateReinitializationAllowed())
        throw AmiException(
            "State reinitialization cannot be enabled for this model "
            "as this feature was disabled at compile time. Most likely,"
            " this was because some initial states depending on "
            "fixedParameters also depended on parameters."
        );
    simulation_parameters_.reinitializeFixedParameterInitialStates = flag;

    if (flag) {
        simulation_parameters_
            .reinitializeAllFixedParameterDependentInitialStatesForSimulation(
                nx_rdata
            );
    } else {
        simulation_parameters_.reinitialization_state_idxs_sim.clear();
    }
}

bool Model::getReinitializeFixedParameterInitialStates() const {
    return simulation_parameters_.reinitializeFixedParameterInitialStates
           || !simulation_parameters_.reinitialization_state_idxs_sim.empty();
}

void Model::requireSensitivitiesForAllParameters() {
    state_.plist.resize(np());
    std::iota(state_.plist.begin(), state_.plist.end(), 0);
    initializeVectors();
}

void Model::getExpression(
    gsl::span<realtype> w, realtype const t, AmiVector const& x
) {
    fw(t, computeX_pos(x), false);
    writeSlice(derived_state_.w_, w);
}

void Model::getObservable(
    gsl::span<realtype> y, realtype const t, AmiVector const& x
) {
    fy(t, x);
    writeSlice(derived_state_.y_, y);
}

ObservableScaling Model::getObservableScaling(int /*iy*/) const {
    return ObservableScaling::lin;
}

void Model::getObservableSensitivity(
    gsl::span<realtype> sy, realtype const t, AmiVector const& x,
    AmiVectorArray const& sx
) {
    if (!ny)
        return;

    fdydx(t, x);
    fsspl(t);
    fdydp(t, x);

    derived_state_.sx_.resize(nx_solver * nplist());
    sx.flatten_to_vector(derived_state_.sx_);

    // compute sy = 1.0*dydx*sx + 1.0*dydp
    // dydx A[ny,nx_solver] * sx B[nx_solver,nplist] = sy C[ny,nplist]
    //        M  K                 K  N                     M  N
    //        lda                  ldb                      ldc

    if (nx_solver) {
        setNaNtoZero(derived_state_.dydx_);
        setNaNtoZero(derived_state_.sx_);
        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, ny, nplist(), nx_solver, 1.0,
            derived_state_.dydx_.data(), ny, derived_state_.sx_.data(),
            nx_solver, 1.0, derived_state_.dydp_.data(), ny
        );
    }
    writeSlice(derived_state_.dydp_, sy);

    if (always_check_finite_)
        checkFinite(sy, ModelQuantity::sy, nplist());
}

void Model::getObservableSigma(
    gsl::span<realtype> sigmay, int const it, ExpData const* edata
) {
    fsigmay(it, edata);
    writeSlice(derived_state_.sigmay_, sigmay);
}

void Model::getObservableSigmaSensitivity(
    gsl::span<realtype> ssigmay, gsl::span<realtype const> sy, int const it,
    ExpData const* edata
) {
    fdsigmaydp(it, edata);
    writeSlice(derived_state_.dsigmaydp_, ssigmay);

    if (pythonGenerated) {
        // ssigmay = dsigmaydy*(dydx_solver*sx+dydp)+dsigmaydp
        //         = dsigmaydy*sy+dsigmaydp

        fdsigmaydy(it, edata);

        // compute ssigmay = 1.0 * dsigmaydp + 1.0 * dsigmaydy * sy
        // dsigmaydp C[ny,nplist] += dsigmaydy A[ny,ny] * sy B[ny,nplist]
        //             M  N                      M  K          K  N
        //             ldc                       lda           ldb
        setNaNtoZero(derived_state_.dsigmaydy_);
        derived_state_.sy_.assign(sy.begin(), sy.end());
        setNaNtoZero(derived_state_.sy_);
        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, ny, nplist(), ny, 1.0,
            derived_state_.dsigmaydy_.data(), ny, derived_state_.sy_.data(), ny,
            1.0, ssigmay.data(), ny
        );
    }

    if (always_check_finite_)
        checkFinite(ssigmay, ModelQuantity::ssigmay, nplist());
}

void Model::addObservableObjective(
    realtype& Jy, int const it, AmiVector const& x, ExpData const& edata
) {
    fy(edata.getTimepoint(it), x);
    fsigmay(it, &edata);

    std::vector<realtype> nllh(nJ, 0.0);
    for (int iyt = 0; iyt < nytrue; iyt++) {
        if (edata.isSetObservedData(it, iyt)) {
            std::fill(nllh.begin(), nllh.end(), 0.0);
            fJy(nllh.data(), iyt, state_.unscaledParameters.data(),
                state_.fixedParameters.data(), derived_state_.y_.data(),
                derived_state_.sigmay_.data(), edata.getObservedDataPtr(it));
            Jy -= nllh.at(0);
        }
    }
}

void Model::addObservableObjectiveSensitivity(
    std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const it,
    AmiVector const& x, AmiVectorArray const& sx, ExpData const& edata
) {

    if (!ny)
        return;

    fdJydx(it, x, edata);
    fdJydp(it, x, edata);
    // Compute dJydx*sx for current 'it'
    // dJydx        rdata->nt x nJ        x nx_solver
    // sx           rdata->nt x nx_solver x nplist()
    derived_state_.sx_.resize(nx_solver * nplist());
    sx.flatten_to_vector(derived_state_.sx_);

    // C := alpha*op(A)*op(B) + beta*C,
    setNaNtoZero(derived_state_.dJydx_);
    setNaNtoZero(derived_state_.sx_);
    amici_dgemm(
        BLASLayout::colMajor, BLASTranspose::noTrans, BLASTranspose::noTrans,
        nJ, nplist(), nx_solver, 1.0, derived_state_.dJydx_.data(), nJ,
        derived_state_.sx_.data(), nx_solver, 1.0, derived_state_.dJydp_.data(),
        nJ
    );

    writeLLHSensitivitySlice(derived_state_.dJydp_, sllh, s2llh);
}

void Model::addPartialObservableObjectiveSensitivity(
    std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const it,
    AmiVector const& x, ExpData const& edata
) {
    if (!ny)
        return;

    fdJydp(it, x, edata);

    writeLLHSensitivitySlice(derived_state_.dJydp_, sllh, s2llh);
}

void Model::getAdjointStateObservableUpdate(
    gsl::span<realtype> dJydx, int const it, AmiVector const& x,
    ExpData const& edata
) {
    fdJydx(it, x, edata);
    writeSlice(derived_state_.dJydx_, dJydx);
}

void Model::getEvent(
    gsl::span<realtype> z, int const ie, realtype const t, AmiVector const& x
) {
    fz(ie, t, x);
    writeSliceEvent(derived_state_.z_, z, ie);
}

void Model::getEventSensitivity(
    gsl::span<realtype> sz, int const ie, realtype const t, AmiVector const& x,
    AmiVectorArray const& sx
) {
    if (pythonGenerated) {
        if (!nz)
            return;

        fdzdx(ie, t, x);
        fdzdp(ie, t, x);

        derived_state_.sx_.resize(nx_solver * nplist());
        sx.flatten_to_vector(derived_state_.sx_);

        // compute sz = 1.0*dzdx*sx + 1.0*dzdp
        // dzdx A[nz,nx_solver] * sx B[nx_solver,nplist] = sz C[nz,nplist]
        //        M  K                 K  N                     M  N
        //        lda                  ldb                      ldc
        setNaNtoZero(derived_state_.dzdx_);
        setNaNtoZero(derived_state_.sx_);
        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, nz, nplist(), nx_solver, 1.0,
            derived_state_.dzdx_.data(), nz, derived_state_.sx_.data(),
            nx_solver, 1.0, derived_state_.dzdp_.data(), nz
        );

        addSlice(derived_state_.dzdp_, sz);

        if (always_check_finite_)
            checkFinite(sz, ModelQuantity::sz, nplist());
    } else {
        for (int ip = 0; ip < nplist(); ip++) {
            fsz(&sz[ip * nz], ie, t, computeX_pos(x),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), sx.data(ip), plist(ip));
        }
    }
}

void Model::getUnobservedEventSensitivity(
    gsl::span<realtype> sz, int const ie
) {
    checkBufferSize(sz, nz * nplist());

    for (int iz = 0; iz < nz; ++iz)
        if (z2event_.at(iz) - 1 == ie)
            for (int ip = 0; ip < nplist(); ++ip)
                sz[ip * nz + iz] = 0.0;
}

void Model::getEventRegularization(
    gsl::span<realtype> rz, int const ie, realtype const t, AmiVector const& x
) {
    frz(ie, t, x);
    writeSliceEvent(derived_state_.rz_, rz, ie);
}

void Model::getEventRegularizationSensitivity(
    gsl::span<realtype> srz, int const ie, realtype const t, AmiVector const& x,
    AmiVectorArray const& sx
) {
    if (pythonGenerated) {
        if (!nz)
            return;

        fdrzdx(ie, t, x);
        fdrzdp(ie, t, x);

        derived_state_.sx_.resize(nx_solver * nplist());
        sx.flatten_to_vector(derived_state_.sx_);

        // compute srz = 1.0*drzdx*sx + 1.0*drzdp
        // drzdx A[nz,nx_solver] * sx B[nx_solver,nplist] = srz C[nz,nplist]
        //         M  K                 K  N                      M  N
        //         lda                  ldb                       ldc
        setNaNtoZero(derived_state_.drzdx_);
        setNaNtoZero(derived_state_.sx_);
        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, nz, nplist(), nx_solver, 1.0,
            derived_state_.drzdx_.data(), nz, derived_state_.sx_.data(),
            nx_solver, 1.0, derived_state_.drzdp_.data(), nz
        );

        addSlice(derived_state_.drzdp_, srz);

        if (always_check_finite_)
            checkFinite(srz, ModelQuantity::srz, nplist());
    } else {
        for (int ip = 0; ip < nplist(); ip++) {
            fsrz(
                &srz[ip * nz], ie, t, computeX_pos(x),
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), sx.data(ip), plist(ip)
            );
        }
    }
}

void Model::getEventSigma(
    gsl::span<realtype> sigmaz, int const ie, int const nroots,
    realtype const t, ExpData const* edata
) {
    fsigmaz(ie, nroots, t, edata);
    writeSliceEvent(derived_state_.sigmaz_, sigmaz, ie);
}

void Model::getEventSigmaSensitivity(
    gsl::span<realtype> ssigmaz, int const ie, int const nroots,
    realtype const t, ExpData const* edata
) {
    fdsigmazdp(ie, nroots, t, edata);
    writeSensitivitySliceEvent(derived_state_.dsigmazdp_, ssigmaz, ie);
}

void Model::addEventObjective(
    realtype& Jz, int const ie, int const nroots, realtype const t,
    AmiVector const& x, ExpData const& edata
) {
    fz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    std::vector<realtype> nllh(nJ, 0.0);
    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            std::fill(nllh.begin(), nllh.end(), 0.0);
            fJz(nllh.data(), iztrue, state_.unscaledParameters.data(),
                state_.fixedParameters.data(), derived_state_.z_.data(),
                derived_state_.sigmaz_.data(),
                edata.getObservedEventsPtr(nroots));
            Jz -= nllh.at(0);
        }
    }
}

void Model::addEventObjectiveRegularization(
    realtype& Jrz, int const ie, int const nroots, realtype const t,
    AmiVector const& x, ExpData const& edata
) {
    frz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    std::vector<realtype> nllh(nJ, 0.0);
    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            std::fill(nllh.begin(), nllh.end(), 0.0);
            fJrz(
                nllh.data(), iztrue, state_.unscaledParameters.data(),
                state_.fixedParameters.data(), derived_state_.rz_.data(),
                derived_state_.sigmaz_.data()
            );
            Jrz -= nllh.at(0);
        }
    }
}

void Model::addEventObjectiveSensitivity(
    std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const ie,
    int const nroots, realtype const t, AmiVector const& x,
    AmiVectorArray const& sx, ExpData const& edata
) {

    if (!nz)
        return;

    fdJzdx(ie, nroots, t, x, edata);
    fdJzdp(ie, nroots, t, x, edata);

    // sJz           nJ x nplist()
    // dJzdp         nJ x nplist()
    // dJzdx         nmaxevent x nJ        x nx_solver
    // sx            rdata->nt x nx_solver x nplist()

    // Compute dJzdx*sx for current 'ie'
    // dJzdx        rdata->nt x nJ        x nx_solver
    // sx           rdata->nt x nx_solver x nplist()
    sx.flatten_to_vector(derived_state_.sx_);

    // C := alpha*op(A)*op(B) + beta*C,
    setNaNtoZero(derived_state_.dJzdx_);
    setNaNtoZero(derived_state_.sx_);
    amici_dgemm(
        BLASLayout::colMajor, BLASTranspose::noTrans, BLASTranspose::noTrans,
        nJ, nplist(), nx_solver, 1.0, derived_state_.dJzdx_.data(), nJ,
        derived_state_.sx_.data(), nx_solver, 1.0, derived_state_.dJzdp_.data(),
        nJ
    );

    // sJy += multResult + dJydp
    writeLLHSensitivitySlice(derived_state_.dJzdp_, sllh, s2llh);
}

void Model::getAdjointStateEventUpdate(
    gsl::span<realtype> dJzdx, int const ie, int const nroots, realtype const t,
    AmiVector const& x, ExpData const& edata
) {
    fdJzdx(ie, nroots, t, x, edata);
    writeSlice(derived_state_.dJzdx_, dJzdx);
}

void Model::addPartialEventObjectiveSensitivity(
    std::vector<realtype>& sllh, std::vector<realtype>& s2llh, int const ie,
    int const nroots, realtype const t, AmiVector const& x, ExpData const& edata
) {
    if (!nz)
        return;

    fdJzdp(ie, nroots, t, x, edata);

    writeLLHSensitivitySlice(derived_state_.dJzdp_, sllh, s2llh);
}

void Model::getEventTimeSensitivity(
    std::vector<realtype>& stau, realtype const t, int const ie,
    AmiVector const& x, AmiVectorArray const& sx
) {

    std::fill(stau.begin(), stau.end(), 0.0);

    for (int ip = 0; ip < nplist(); ip++) {
        fstau(
            &stau.at(ip), t, computeX_pos(x), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.h.data(),
            state_.total_cl.data(), sx.data(ip), plist(ip), ie
        );
    }
}

void Model::addStateEventUpdate(
    AmiVector& x, int const ie, realtype const t, AmiVector const& xdot,
    AmiVector const& xdot_old
) {

    derived_state_.deltax_.assign(nx_solver, 0.0);

    std::copy_n(computeX_pos(x), nx_solver, x.data());

    // compute update
    fdeltax(
        derived_state_.deltax_.data(), t, x.data(),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), ie, xdot.data(), xdot_old.data()
    );

    if (always_check_finite_) {
        checkFinite(derived_state_.deltax_, ModelQuantity::deltax);
    }

    // update
    amici_daxpy(nx_solver, 1.0, derived_state_.deltax_.data(), 1, x.data(), 1);
}

void Model::addStateSensitivityEventUpdate(
    AmiVectorArray& sx, int const ie, realtype const t, AmiVector const& x_old,
    AmiVector const& xdot, AmiVector const& xdot_old,
    std::vector<realtype> const& stau
) {
    fw(t, x_old.data(), false);

    for (int ip = 0; ip < nplist(); ip++) {

        derived_state_.deltasx_.assign(nx_solver, 0.0);

        // compute update
        fdeltasx(
            derived_state_.deltasx_.data(), t, x_old.data(),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), derived_state_.w_.data(), plist(ip), ie,
            xdot.data(), xdot_old.data(), sx.data(ip), &stau.at(ip),
            state_.total_cl.data()
        );

        if (always_check_finite_) {
            checkFinite(
                derived_state_.deltasx_, ModelQuantity::deltasx, nplist()
            );
        }

        amici_daxpy(
            nx_solver, 1.0, derived_state_.deltasx_.data(), 1, sx.data(ip), 1
        );
    }
}

void Model::addAdjointStateEventUpdate(
    AmiVector& xB, int const ie, realtype const t, AmiVector const& x,
    AmiVector const& xdot, AmiVector const& xdot_old
) {

    derived_state_.deltaxB_.assign(nx_solver, 0.0);

    // compute update
    fdeltaxB(
        derived_state_.deltaxB_.data(), t, computeX_pos(x),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data(), ie, xdot.data(), xdot_old.data(), xB.data()
    );

    if (always_check_finite_) {
        checkFinite(derived_state_.deltaxB_, ModelQuantity::deltaxB);
    }

    // apply update
    for (int ix = 0; ix < nxtrue_solver; ++ix)
        for (int iJ = 0; iJ < nJ; ++iJ)
            xB.at(ix + iJ * nxtrue_solver)
                += derived_state_.deltaxB_.at(ix + iJ * nxtrue_solver);
}

void Model::addAdjointQuadratureEventUpdate(
    AmiVector xQB, int const ie, realtype const t, AmiVector const& x,
    AmiVector const& xB, AmiVector const& xdot, AmiVector const& xdot_old
) {
    for (int ip = 0; ip < nplist(); ip++) {
        derived_state_.deltaqB_.assign(nJ, 0.0);

        fdeltaqB(
            derived_state_.deltaqB_.data(), t, computeX_pos(x),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), plist(ip), ie, xdot.data(), xdot_old.data(),
            xB.data()
        );

        for (int iJ = 0; iJ < nJ; ++iJ)
            xQB.at(iJ) += derived_state_.deltaqB_.at(iJ);
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.deltaqB_, ModelQuantity::deltaqB, nplist());
    }
}

void Model::updateHeaviside(std::vector<int> const& rootsfound) {
    for (int ie = 0; ie < ne; ie++) {
        state_.h.at(ie) += rootsfound.at(ie);
    }
}

void Model::updateHeavisideB(int const* rootsfound) {
    for (int ie = 0; ie < ne; ie++) {
        state_.h.at(ie) -= rootsfound[ie];
    }
}

int Model::checkFinite(
    gsl::span<realtype const> array, ModelQuantity model_quantity
) const {
    auto it = std::find_if(array.begin(), array.end(), [](realtype x) {
        return !std::isfinite(x);
    });
    if (it == array.end()) {
        return AMICI_SUCCESS;
    }

    // there is some issue - produce a meaningful message
    auto flat_index = it - array.begin();

    std::string msg_id;
    std::string non_finite_type;
    if (std::isnan(array[flat_index])) {
        msg_id = "AMICI:NaN";
        non_finite_type = "NaN";
    } else if (std::isinf(array[flat_index])) {
        msg_id = "AMICI:Inf";
        non_finite_type = "Inf";
    }
    std::string element_id = std::to_string(flat_index);

    switch (model_quantity) {
    case ModelQuantity::xdot:
    case ModelQuantity::xBdot:
    case ModelQuantity::x0:
    case ModelQuantity::x:
    case ModelQuantity::x_rdata:
    case ModelQuantity::x0_rdata:
    case ModelQuantity::Jv:
    case ModelQuantity::JvB:
    case ModelQuantity::JDiag:
    case ModelQuantity::deltax:
    case ModelQuantity::deltaxB:
        if (hasStateIds()) {
            element_id = getStateIdsSolver()[flat_index];
        }
        break;
    case ModelQuantity::y:
        if (hasObservableIds()) {
            element_id = getObservableIds()[flat_index];
        }
        break;
    case ModelQuantity::w:
        if (hasExpressionIds()) {
            element_id = getExpressionIds()[flat_index];
        }
        break;
    case ModelQuantity::k:
        if (hasFixedParameterIds()) {
            element_id = getFixedParameterIds()[flat_index];
        }
        break;
    case ModelQuantity::p:
        if (hasParameterIds()) {
            element_id = getParameterIds()[flat_index];
        }
        break;
    default:
        break;
    }

    std::string model_quantity_str;
    try {
        model_quantity_str = model_quantity_to_str.at(model_quantity);
    } catch (std::out_of_range const&) {
        // Missing model quantity string - terminate if this is a debug build,
        // but show the quantity number if non-debug.
        gsl_ExpectsDebug(false);
        model_quantity_str = std::to_string(static_cast<int>(model_quantity));
    }
    if (logger)
        logger->log(
            LogSeverity::warning, msg_id,
            "AMICI encountered a %s value for %s[%i] (%s)",
            non_finite_type.c_str(), model_quantity_str.c_str(),
            gsl::narrow<int>(flat_index), element_id.c_str()
        );

    // check upstream, without infinite recursion
    if (model_quantity != ModelQuantity::k && model_quantity != ModelQuantity::p
        && model_quantity != ModelQuantity::ts) {
        checkFinite(state_.fixedParameters, ModelQuantity::k);
        checkFinite(state_.unscaledParameters, ModelQuantity::p);
        checkFinite(simulation_parameters_.ts_, ModelQuantity::ts);
        if (!always_check_finite_ && model_quantity != ModelQuantity::w) {
            // don't check twice if always_check_finite_ is true
            checkFinite(derived_state_.w_, ModelQuantity::w);
        }
    }
    return AMICI_RECOVERABLE_ERROR;
}

int Model::checkFinite(
    gsl::span<realtype const> array, ModelQuantity model_quantity,
    size_t num_cols
) const {
    auto it = std::find_if(array.begin(), array.end(), [](realtype x) {
        return !std::isfinite(x);
    });
    if (it == array.end()) {
        return AMICI_SUCCESS;
    }

    // there is some issue - produce a meaningful message
    auto flat_index = it - array.begin();
    sunindextype row, col;
    std::tie(row, col) = unravel_index(flat_index, num_cols);
    std::string msg_id;
    std::string non_finite_type;
    if (std::isnan(array[flat_index])) {
        msg_id = "AMICI:NaN";
        non_finite_type = "NaN";
    } else if (std::isinf(array[flat_index])) {
        msg_id = "AMICI:Inf";
        non_finite_type = "Inf";
    }
    std::string row_id = std::to_string(row);
    std::string col_id = std::to_string(col);

    switch (model_quantity) {
    case ModelQuantity::sy:
    case ModelQuantity::ssigmay:
    case ModelQuantity::dydp:
    case ModelQuantity::dsigmaydp:
        if (hasObservableIds())
            row_id += " " + getObservableIds()[row];
        if (hasParameterIds())
            col_id += " " + getParameterIds()[plist(gsl::narrow<int>(col))];
        break;
    case ModelQuantity::dydx:
        if (hasObservableIds())
            row_id += " " + getObservableIds()[row];
        if (hasStateIds())
            col_id += " " + getStateIdsSolver()[col];
        break;
    case ModelQuantity::deltasx:
        if (hasStateIds())
            row_id += " " + getStateIdsSolver()[row];
        if (hasParameterIds())
            col_id += " " + getParameterIds()[plist(gsl::narrow<int>(col))];
        break;
    case ModelQuantity::dJydy:
    case ModelQuantity::dJydy_matlab:
    case ModelQuantity::dJydsigma:
        if (hasObservableIds())
            col_id += " " + getObservableIds()[col];
        break;
    case ModelQuantity::dJydx:
    case ModelQuantity::dJzdx:
    case ModelQuantity::dJrzdx:
    case ModelQuantity::dzdx:
    case ModelQuantity::drzdx:
        if (hasStateIds())
            col_id += " " + getStateIdsSolver()[col];
        break;
    case ModelQuantity::deltaqB:
    case ModelQuantity::sz:
    case ModelQuantity::dzdp:
    case ModelQuantity::drzdp:
    case ModelQuantity::dsigmazdp:
        if (hasParameterIds())
            col_id += " " + getParameterIds()[plist(gsl::narrow<int>(col))];
        break;
    case ModelQuantity::dsigmaydy:
        if (hasObservableIds()) {
            auto obs_ids = getObservableIds();
            row_id += " " + obs_ids[row];
            col_id += " " + obs_ids[col];
        }
        break;
    default:
        break;
    }

    std::string model_quantity_str;
    try {
        model_quantity_str = model_quantity_to_str.at(model_quantity);
    } catch (std::out_of_range const&) {
        // Missing model quantity string - terminate if this is a debug build,
        // but show the quantity number if non-debug.
        gsl_ExpectsDebug(false);
        model_quantity_str = std::to_string(static_cast<int>(model_quantity));
    }

    if (logger)
        logger->log(
            LogSeverity::warning, msg_id,
            "AMICI encountered a %s value for %s[%i] (%s, %s)",
            non_finite_type.c_str(), model_quantity_str.c_str(),
            gsl::narrow<int>(flat_index), row_id.c_str(), col_id.c_str()
        );

    // check upstream
    checkFinite(state_.fixedParameters, ModelQuantity::k);
    checkFinite(state_.unscaledParameters, ModelQuantity::p);
    checkFinite(simulation_parameters_.ts_, ModelQuantity::ts);
    checkFinite(derived_state_.w_, ModelQuantity::w);

    return AMICI_RECOVERABLE_ERROR;
}

int Model::checkFinite(SUNMatrix m, ModelQuantity model_quantity, realtype t)
    const {
    // check flat array, to see if there are any issues
    // (faster, in particular for sparse arrays)
    auto m_flat = gsl::make_span(m);
    auto it = std::find_if(m_flat.begin(), m_flat.end(), [](realtype x) {
        return !std::isfinite(x);
    });
    if (it == m_flat.end()) {
        return AMICI_SUCCESS;
    }

    // there is some issue - produce a meaningful message
    auto flat_index = it - m_flat.begin();
    sunindextype row, col;
    std::tie(row, col) = unravel_index(flat_index, m);
    std::string msg_id;
    std::string non_finite_type;
    if (std::isnan(m_flat[flat_index])) {
        msg_id = "AMICI:NaN";
        non_finite_type = "NaN";
    } else if (std::isinf(m_flat[flat_index])) {
        msg_id = "AMICI:Inf";
        non_finite_type = "Inf";
    } else {
        throw std::runtime_error(
            "Value is not finite, but neither infinite nor NaN."
        );
    }
    std::string row_id = std::to_string(row);
    std::string col_id = std::to_string(col);

    switch (model_quantity) {
    case ModelQuantity::J:
    case ModelQuantity::JB:
        if (hasStateIds()) {
            auto state_ids = getStateIdsSolver();
            row_id += " " + state_ids[row];
            col_id += " " + state_ids[col];
        }
        break;
    case ModelQuantity::dwdx:
        if (hasExpressionIds())
            row_id += " " + getExpressionIds()[row];
        if (hasStateIds())
            col_id += " " + getStateIdsSolver()[col];
        break;
    case ModelQuantity::dwdw:
        if (hasExpressionIds()) {
            auto expr_ids = getExpressionIds();
            row_id += " " + expr_ids[row];
            col_id += " " + expr_ids[col];
        }
        break;
    case ModelQuantity::dwdp:
        if (hasExpressionIds())
            row_id += " " + getExpressionIds()[row];
        if (hasParameterIds())
            col_id += " " + getParameterIds()[col];
        break;
    default:
        break;
    }

    std::string model_quantity_str;
    try {
        model_quantity_str = model_quantity_to_str.at(model_quantity);
    } catch (std::out_of_range const&) {
        // Missing model quantity string - terminate if this is a debug build,
        // but show the quantity number if non-debug.
        gsl_ExpectsDebug(false);
        model_quantity_str = std::to_string(static_cast<int>(model_quantity));
    }

    if (logger)
        logger->log(
            LogSeverity::warning, msg_id,
            "AMICI encountered a %s value for %s[%i] (%s, %s) at t=%g",
            non_finite_type.c_str(), model_quantity_str.c_str(),
            gsl::narrow<int>(flat_index), row_id.c_str(), col_id.c_str(), t
        );

    // check upstream
    checkFinite(state_.fixedParameters, ModelQuantity::k);
    checkFinite(state_.unscaledParameters, ModelQuantity::p);
    checkFinite(simulation_parameters_.ts_, ModelQuantity::ts);
    checkFinite(derived_state_.w_, ModelQuantity::w);

    return AMICI_RECOVERABLE_ERROR;
}

void Model::setAlwaysCheckFinite(bool alwaysCheck) {
    always_check_finite_ = alwaysCheck;
}

bool Model::getAlwaysCheckFinite() const { return always_check_finite_; }

void Model::fx0(AmiVector& x) {
    std::fill(
        derived_state_.x_rdata_.begin(), derived_state_.x_rdata_.end(), 0.0
    );
    /* this function  also computes initial total abundances */
    fx0(derived_state_.x_rdata_.data(), simulation_parameters_.tstart_,
        state_.unscaledParameters.data(), state_.fixedParameters.data());
    fx_solver(x.data(), derived_state_.x_rdata_.data());
    ftotal_cl(
        state_.total_cl.data(), derived_state_.x_rdata_.data(),
        state_.unscaledParameters.data(), state_.fixedParameters.data()
    );

    checkFinite(derived_state_.x_rdata_, ModelQuantity::x0_rdata);
}

void Model::fx0_fixedParameters(AmiVector& x) {
    if (!getReinitializeFixedParameterInitialStates())
        return;

    /* we transform to the unreduced states x_rdata and then apply
     x0_fixedparameters to (i) enable updates to states that were removed from
     conservation laws and (ii) be able to correctly compute total abundances
     after updating the state variables */
    fx_rdata(
        derived_state_.x_rdata_.data(), computeX_pos(x), state_.total_cl.data(),
        state_.unscaledParameters.data(), state_.fixedParameters.data()
    );
    fx0_fixedParameters(
        derived_state_.x_rdata_.data(), simulation_parameters_.tstart_,
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        simulation_parameters_.reinitialization_state_idxs_sim
    );
    fx_solver(x.data(), derived_state_.x_rdata_.data());
    /* update total abundances */
    ftotal_cl(
        state_.total_cl.data(), derived_state_.x_rdata_.data(),
        state_.unscaledParameters.data(), state_.fixedParameters.data()
    );
}

void Model::fsx0(AmiVectorArray& sx, AmiVector const& x) {
    /* this function  also computes initial total abundance sensitivities */
    realtype* stcl = nullptr;
    for (int ip = 0; ip < nplist(); ip++) {
        if (ncl() > 0)
            stcl = &state_.stotal_cl.at(plist(ip) * ncl());
        std::fill(
            derived_state_.sx_rdata_.begin(), derived_state_.sx_rdata_.end(),
            0.0
        );
        fsx0(
            derived_state_.sx_rdata_.data(), simulation_parameters_.tstart_,
            computeX_pos(x), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), plist(ip)
        );
        fsx_solver(sx.data(ip), derived_state_.sx_rdata_.data());
        fstotal_cl(
            stcl, derived_state_.sx_rdata_.data(), plist(ip),
            derived_state_.x_rdata_.data(), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.total_cl.data()
        );
    }
}

void Model::fsx0_fixedParameters(AmiVectorArray& sx, AmiVector const& x) {
    if (!getReinitializeFixedParameterInitialStates())
        return;
    realtype* stcl = nullptr;
    for (int ip = 0; ip < nplist(); ip++) {
        if (ncl() > 0)
            stcl = &state_.stotal_cl.at(plist(ip) * ncl());
        fsx_rdata(
            derived_state_.sx_rdata_.data(), sx.data(ip), stcl,
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            x.data(), state_.total_cl.data(), plist(ip)
        );
        fsx0_fixedParameters(
            derived_state_.sx_rdata_.data(), simulation_parameters_.tstart_,
            computeX_pos(x), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), plist(ip),
            simulation_parameters_.reinitialization_state_idxs_sim
        );
        fsx_solver(sx.data(ip), derived_state_.sx_rdata_.data());
        fstotal_cl(
            stcl, derived_state_.sx_rdata_.data(), plist(ip),
            derived_state_.x_rdata_.data(), state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.total_cl.data()
        );
    }
}

void Model::fsdx0() {}

void Model::fx_rdata(AmiVector& x_rdata, AmiVector const& x) {
    fx_rdata(
        x_rdata.data(), computeX_pos(x), state_.total_cl.data(),
        state_.unscaledParameters.data(), state_.fixedParameters.data()
    );
    if (always_check_finite_)
        checkFinite(x_rdata.getVector(), ModelQuantity::x_rdata);
}

void Model::fsx_rdata(
    AmiVectorArray& sx_rdata, AmiVectorArray const& sx,
    AmiVector const& x_solver
) {
    realtype* stcl = nullptr;
    for (int ip = 0; ip < nplist(); ip++) {
        if (ncl() > 0)
            stcl = &state_.stotal_cl.at(plist(ip) * ncl());
        fsx_rdata(
            sx_rdata.data(ip), sx.data(ip), stcl,
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            x_solver.data(), state_.total_cl.data(), plist(ip)
        );
    }
}

void Model::writeSliceEvent(
    gsl::span<realtype const> slice, gsl::span<realtype> buffer, int const ie
) {
    checkBufferSize(buffer, slice.size());
    checkBufferSize(buffer, z2event_.size());
    for (unsigned izt = 0; izt < z2event_.size(); ++izt)
        if (z2event_.at(izt) - 1 == ie)
            buffer[izt] = slice[izt];
}

void Model::writeSensitivitySliceEvent(
    gsl::span<realtype const> slice, gsl::span<realtype> buffer, int const ie
) {
    checkBufferSize(buffer, slice.size());
    checkBufferSize(buffer, z2event_.size() * nplist());
    for (int ip = 0; ip < nplist(); ++ip)
        for (unsigned izt = 0; izt < z2event_.size(); ++izt)
            if (z2event_.at(izt) - 1 == ie)
                buffer[ip * nztrue + izt] = slice[ip * nztrue + izt];
}

void Model::writeLLHSensitivitySlice(
    std::vector<realtype> const& dLLhdp, std::vector<realtype>& sllh,
    std::vector<realtype>& s2llh
) {
    checkLLHBufferSize(sllh, s2llh);

    amici_daxpy(nplist(), -1.0, dLLhdp.data(), nJ, sllh.data(), 1);
    for (int iJ = 1; iJ < nJ; ++iJ)
        amici_daxpy(
            nplist(), -1.0, &dLLhdp.at(iJ), nJ, &s2llh.at(iJ - 1), nJ - 1
        );
}

void Model::checkLLHBufferSize(
    std::vector<realtype> const& sllh, std::vector<realtype> const& s2llh
) const {
    if (sllh.size() != gsl::narrow<unsigned>(nplist()))
        throw AmiException(
            "Incorrect sllh buffer size! Was %u, expected %i.", sllh.size(),
            nplist()
        );

    if (s2llh.size() != gsl::narrow<unsigned>((nJ - 1) * nplist()))
        throw AmiException(
            "Incorrect s2llh buffer size! Was %u, expected %i.", s2llh.size(),
            (nJ - 1) * nplist()
        );
}

void Model::initializeVectors() {
    sx0data_.clear();
    if (!pythonGenerated)
        derived_state_.dxdotdp = AmiVectorArray(nx_solver, nplist());
}

void Model::fy(realtype const t, AmiVector const& x) {
    if (!ny)
        return;

    auto x_pos = computeX_pos(x);

    derived_state_.y_.assign(ny, 0.0);

    fw(t, x_pos, false);
    fy(derived_state_.y_.data(), t, x_pos, state_.unscaledParameters.data(),
       state_.fixedParameters.data(), state_.h.data(),
       derived_state_.w_.data());

    if (always_check_finite_) {
        checkFinite(
            gsl::make_span(derived_state_.y_.data(), ny), ModelQuantity::y
        );
    }
}

void Model::fdydp(realtype const t, AmiVector const& x) {
    if (!ny)
        return;

    auto x_pos = computeX_pos(x);

    derived_state_.dydp_.assign(ny * nplist(), 0.0);
    fw(t, x_pos, false);
    fdwdp(t, x_pos, false);

    /* get dydp slice (ny) for current time and parameter */
    for (int ip = 0; ip < nplist(); ip++)
        if (pythonGenerated) {
            fdydp(
                &derived_state_.dydp_.at(ip * ny), t, x_pos,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), plist(ip), derived_state_.w_.data(),
                state_.total_cl.data(), state_.stotal_cl.data(),
                state_.spl_.data(), derived_state_.sspl_.data()
            );
        } else {
            fdydp(
                &derived_state_.dydp_.at(ip * ny), t, x_pos,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                state_.h.data(), plist(ip), derived_state_.w_.data(),
                derived_state_.dwdp_.data()
            );
        }

    if (always_check_finite_) {
        checkFinite(derived_state_.dydp_, ModelQuantity::dydp, nplist());
    }
}

void Model::fdydx(realtype const t, AmiVector const& x) {
    if (!ny)
        return;

    auto x_pos = computeX_pos(x);

    derived_state_.dydx_.assign(ny * nx_solver, 0.0);

    fw(t, x_pos, false);
    fdwdx(t, x_pos, false);
    fdydx(
        derived_state_.dydx_.data(), t, x_pos, state_.unscaledParameters.data(),
        state_.fixedParameters.data(), state_.h.data(),
        derived_state_.w_.data(), derived_state_.dwdx_.data()
    );

    if (always_check_finite_) {
        checkFinite(derived_state_.dydx_, ModelQuantity::dydx, ny);
    }
}

void Model::fsigmay(int const it, ExpData const* edata) {
    if (!ny)
        return;

    derived_state_.sigmay_.assign(ny, 0.0);

    fsigmay(
        derived_state_.sigmay_.data(), getTimepoint(it),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        derived_state_.y_.data()
    );

    if (edata) {
        auto sigmay_edata = edata->getObservedDataStdDevPtr(it);
        /* extract the value for the standard deviation from ExpData,
         * if the data value is NaN, use the parameter value */
        for (int iytrue = 0; iytrue < nytrue; iytrue++) {
            if (edata->isSetObservedDataStdDev(it, iytrue))
                derived_state_.sigmay_.at(iytrue) = sigmay_edata[iytrue];

            /* TODO: when moving second order code to cpp, verify
             * that this is actually what we want
             */
            for (int iJ = 1; iJ < nJ; iJ++)
                derived_state_.sigmay_.at(iytrue + iJ * nytrue) = 0;

            if (edata->isSetObservedData(it, iytrue))
                checkSigmaPositivity(
                    derived_state_.sigmay_.at(iytrue), "sigmay"
                );
        }
    }
}

void Model::fdsigmaydp(int const it, ExpData const* edata) {
    if (!ny)
        return;

    derived_state_.dsigmaydp_.assign(ny * nplist(), 0.0);

    for (int ip = 0; ip < nplist(); ip++)
        // get dsigmaydp slice (ny) for current timepoint and parameter
        fdsigmaydp(
            &derived_state_.dsigmaydp_.at(ip * ny), getTimepoint(it),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            derived_state_.y_.data(), plist(ip)
        );

    // sigmas in edata override model-sigma -> for those sigmas, set dsigmaydp
    // to zero
    if (edata) {
        for (int iy = 0; iy < nytrue; iy++) {
            if (!edata->isSetObservedDataStdDev(it, iy))
                continue;
            for (int ip = 0; ip < nplist(); ip++) {
                derived_state_.dsigmaydp_.at(ip * ny + iy) = 0.0;
            }
        }
    }

    if (always_check_finite_) {
        checkFinite(
            derived_state_.dsigmaydp_, ModelQuantity::dsigmaydp, nplist()
        );
    }
}

void Model::fdsigmaydy(int const it, ExpData const* edata) {
    if (!ny)
        return;

    derived_state_.dsigmaydy_.assign(ny * ny, 0.0);

    // get dsigmaydy slice (ny) for current timepoint
    fdsigmaydy(
        derived_state_.dsigmaydy_.data(), getTimepoint(it),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        derived_state_.y_.data()
    );

    // sigmas in edata override model-sigma -> for those sigmas, set dsigmaydy
    // to zero
    if (edata) {
        for (int isigmay = 0; isigmay < nytrue; ++isigmay) {
            if (!edata->isSetObservedDataStdDev(it, isigmay))
                continue;
            for (int iy = 0; iy < nytrue; ++iy) {
                derived_state_.dsigmaydy_.at(isigmay * ny + iy) = 0.0;
            }
        }
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.dsigmaydy_, ModelQuantity::dsigmaydy, ny);
    }
}

void Model::fdJydy(int const it, AmiVector const& x, ExpData const& edata) {
    if (!ny)
        return;

    fy(edata.getTimepoint(it), x);
    fsigmay(it, &edata);

    if (pythonGenerated) {
        fdJydsigma(it, x, edata);
        fdsigmaydy(it, &edata);
        SUNMatrixWrapper tmp_dense(nJ, ny);

        setNaNtoZero(derived_state_.dJydsigma_);
        setNaNtoZero(derived_state_.dsigmaydy_);
        for (int iyt = 0; iyt < nytrue; iyt++) {
            if (!derived_state_.dJydy_.at(iyt).capacity())
                continue;
            derived_state_.dJydy_.at(iyt).zero();
            fdJydy_colptrs(derived_state_.dJydy_.at(iyt), iyt);
            fdJydy_rowvals(derived_state_.dJydy_.at(iyt), iyt);

            if (!edata.isSetObservedData(it, iyt))
                continue;

            // get dJydy slice (ny) for current timepoint and observable
            fdJydy(
                derived_state_.dJydy_.at(iyt).data(), iyt,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.y_.data(), derived_state_.sigmay_.data(),
                edata.getObservedDataPtr(it)
            );

            // dJydy += dJydsigma * dsigmaydy
            // C(nJ,ny)  A(nJ,ny)  * B(ny,ny)
            // sparse    dense       dense
            tmp_dense.zero();
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, ny, ny, 1.0,
                &derived_state_.dJydsigma_.at(iyt * nJ * ny), nJ,
                derived_state_.dsigmaydy_.data(), ny, 1.0, tmp_dense.data(), nJ
            );

            auto tmp_sparse = SUNMatrixWrapper(tmp_dense, 0.0, CSC_MAT);
            auto ret = SUNMatScaleAdd(
                1.0, derived_state_.dJydy_.at(iyt), tmp_sparse
            );
            if (ret != SUNMAT_SUCCESS) {
                throw AmiException(
                    "SUNMatScaleAdd failed with status %d in %s", ret, __func__
                );
            }
            derived_state_.dJydy_.at(iyt).refresh();

            if (always_check_finite_) {
                checkFinite(
                    gsl::make_span(derived_state_.dJydy_.at(iyt).get()),
                    ModelQuantity::dJydy, ny
                );
            }
        }
    } else {
        std::fill(
            derived_state_.dJydy_matlab_.begin(),
            derived_state_.dJydy_matlab_.end(), 0.0
        );
        for (int iyt = 0; iyt < nytrue; iyt++) {
            if (!edata.isSetObservedData(it, iyt))
                continue;
            fdJydy(
                &derived_state_.dJydy_matlab_.at(iyt * ny * nJ), iyt,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.y_.data(), derived_state_.sigmay_.data(),
                edata.getObservedDataPtr(it)
            );
            if (always_check_finite_) {
                // get dJydy slice (ny) for current timepoint and observable
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJydy_matlab_[iyt * ny * nJ], ny * nJ
                    ),
                    ModelQuantity::dJydy, ny
                );
            }
        }
    }
}

void Model::fdJydsigma(int const it, AmiVector const& x, ExpData const& edata) {
    if (!ny)
        return;

    derived_state_.dJydsigma_.assign(nytrue * ny * nJ, 0.0);

    fy(edata.getTimepoint(it), x);
    fsigmay(it, &edata);

    for (int iyt = 0; iyt < nytrue; iyt++) {
        if (edata.isSetObservedData(it, iyt)) {
            // get dJydsigma slice (ny) for current timepoint and observable
            fdJydsigma(
                &derived_state_.dJydsigma_.at(iyt * ny * nJ), iyt,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.y_.data(), derived_state_.sigmay_.data(),
                edata.getObservedDataPtr(it)
            );
            if (always_check_finite_) {
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJydsigma_.at(iyt * ny * nJ), ny * nJ
                    ),
                    ModelQuantity::dJydsigma, ny
                );
            }
        }
    }
}

void Model::fdJydp(int const it, AmiVector const& x, ExpData const& edata) {
    // dJydy         nJ, nytrue x ny
    // dydp          nplist * ny
    // dJydp         nplist x nJ
    if (!ny)
        return;

    derived_state_.dJydp_.assign(nJ * nplist(), 0.0);

    fdJydy(it, x, edata);
    fdydp(edata.getTimepoint(it), x);

    fdJydsigma(it, x, edata);
    fdsigmaydp(it, &edata);

    setNaNtoZero(derived_state_.dJydsigma_);
    setNaNtoZero(derived_state_.dsigmaydp_);
    for (int iyt = 0; iyt < nytrue; ++iyt) {
        if (!edata.isSetObservedData(it, iyt))
            continue;

        if (pythonGenerated) {
            // dJydp = 1.0 * dJydp +  1.0 * dJydy * dydp
            for (int iplist = 0; iplist < nplist(); ++iplist) {
                derived_state_.dJydy_.at(iyt).multiply(
                    gsl::span<realtype>(
                        &derived_state_.dJydp_.at(iplist * nJ), nJ
                    ),
                    gsl::span<realtype const>(
                        &derived_state_.dydp_.at(iplist * ny), ny
                    )
                );
            }
        } else {
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nplist(), ny, 1.0,
                &derived_state_.dJydy_matlab_.at(iyt * nJ * ny), nJ,
                derived_state_.dydp_.data(), ny, 1.0,
                derived_state_.dJydp_.data(), nJ
            );
        }
        // dJydp = 1.0 * dJydp +  1.0 * dJydsigma * dsigmaydp
        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, nJ, nplist(), ny, 1.0,
            &derived_state_.dJydsigma_.at(iyt * nJ * ny), nJ,
            derived_state_.dsigmaydp_.data(), ny, 1.0,
            derived_state_.dJydp_.data(), nJ
        );
    }
}

void Model::fdJydx(int const it, AmiVector const& x, ExpData const& edata) {
    if (!ny)
        return;

    derived_state_.dJydx_.assign(nJ * nx_solver, 0.0);

    fdydx(edata.getTimepoint(it), x);
    fdJydy(it, x, edata);

    // dJydy: nJ, ny x nytrue
    // dydx :     ny x nx_solver
    // dJydx:     nJ x nx_solver x nt
    for (int iyt = 0; iyt < nytrue; ++iyt) {
        if (!edata.isSetObservedData(it, iyt))
            continue;
        // dJydy A[nyt,nJ,ny] * dydx B[ny,nx_solver] = dJydx C[it,nJ,nx_solver]
        //         slice                                       slice
        //          M  K            K  N                       M  N
        //           lda             ldb                        ldc

        if (pythonGenerated) {
            for (int ix = 0; ix < nx_solver; ++ix) {
                derived_state_.dJydy_.at(iyt).multiply(
                    gsl::span<realtype>(&derived_state_.dJydx_.at(ix * nJ), nJ),
                    gsl::span<realtype const>(
                        &derived_state_.dydx_.at(ix * ny), ny
                    )
                );
            }
        } else {
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nx_solver, ny, 1.0,
                &derived_state_.dJydy_matlab_.at(iyt * ny * nJ), nJ,
                derived_state_.dydx_.data(), ny, 1.0,
                derived_state_.dJydx_.data(), nJ
            );
        }
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.dJydx_, ModelQuantity::dJydx, nx_solver);
    }
}

void Model::fz(int const ie, realtype const t, AmiVector const& x) {

    derived_state_.z_.assign(nz, 0.0);

    fz(derived_state_.z_.data(), ie, t, computeX_pos(x),
       state_.unscaledParameters.data(), state_.fixedParameters.data(),
       state_.h.data());
}

void Model::fdzdp(int const ie, realtype const t, AmiVector const& x) {
    if (!nz)
        return;

    derived_state_.dzdp_.assign(nz * nplist(), 0.0);

    for (int ip = 0; ip < nplist(); ip++) {
        fdzdp(
            derived_state_.dzdp_.data(), ie, t, computeX_pos(x),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), plist(ip)
        );
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.dzdp_, ModelQuantity::dzdp, nplist());
    }
}

void Model::fdzdx(int const ie, realtype const t, AmiVector const& x) {
    if (!nz)
        return;

    derived_state_.dzdx_.assign(nz * nx_solver, 0.0);

    fdzdx(
        derived_state_.dzdx_.data(), ie, t, computeX_pos(x),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data()
    );

    if (always_check_finite_) {
        checkFinite(derived_state_.dzdx_, ModelQuantity::dzdx, nx_solver);
    }
}

void Model::frz(int const ie, realtype const t, AmiVector const& x) {

    derived_state_.rz_.assign(nz, 0.0);

    frz(derived_state_.rz_.data(), ie, t, computeX_pos(x),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data());
}

void Model::fdrzdp(int const ie, realtype const t, AmiVector const& x) {
    if (!nz)
        return;

    derived_state_.drzdp_.assign(nz * nplist(), 0.0);

    for (int ip = 0; ip < nplist(); ip++) {
        fdrzdp(
            derived_state_.drzdp_.data(), ie, t, computeX_pos(x),
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), plist(ip)
        );
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.drzdp_, ModelQuantity::drzdp, nplist());
    }
}

void Model::fdrzdx(int const ie, realtype const t, AmiVector const& x) {
    if (!nz)
        return;

    derived_state_.drzdx_.assign(nz * nx_solver, 0.0);

    fdrzdx(
        derived_state_.drzdx_.data(), ie, t, computeX_pos(x),
        state_.unscaledParameters.data(), state_.fixedParameters.data(),
        state_.h.data()
    );

    if (always_check_finite_) {
        checkFinite(derived_state_.drzdx_, ModelQuantity::drzdx, nx_solver);
    }
}

void Model::fsigmaz(
    int const ie, int const nroots, realtype const t, ExpData const* edata
) {
    if (!nz)
        return;

    derived_state_.sigmaz_.assign(nz, 0.0);
    fsigmaz(
        derived_state_.sigmaz_.data(), t, state_.unscaledParameters.data(),
        state_.fixedParameters.data()
    );

    if (edata) {
        for (int iztrue = 0; iztrue < nztrue; iztrue++) {
            if (z2event_.at(iztrue) - 1 == ie) {
                if (edata->isSetObservedEventsStdDev(nroots, iztrue)) {
                    auto sigmaz_edata
                        = edata->getObservedEventsStdDevPtr(nroots);
                    derived_state_.sigmaz_.at(iztrue) = sigmaz_edata[iztrue];
                }

                /* TODO: when moving second order code to cpp, verify
                 * that this is actually what we want
                 */
                for (int iJ = 1; iJ < nJ; iJ++)
                    derived_state_.sigmaz_.at(iztrue + iJ * nztrue) = 0;

                if (edata->isSetObservedEvents(nroots, iztrue))
                    checkSigmaPositivity(
                        derived_state_.sigmaz_.at(iztrue), "sigmaz"
                    );
            }
        }
    }
}

void Model::fdsigmazdp(
    int const ie, int const nroots, realtype const t, ExpData const* edata
) {
    if (!nz)
        return;

    derived_state_.dsigmazdp_.assign(nz * nplist(), 0.0);

    for (int ip = 0; ip < nplist(); ip++) {
        // get dsigmazdp slice (nz) for current event and parameter
        fdsigmazdp(
            &derived_state_.dsigmazdp_.at(ip * nz), t,
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            plist(ip)
        );
    }

    // sigmas in edata override model-sigma -> for those sigmas, set dsigmazdp
    // to zero
    if (edata) {
        for (int iz = 0; iz < nztrue; iz++) {
            if (z2event_.at(iz) - 1 == ie
                && !edata->isSetObservedEventsStdDev(nroots, iz)) {
                for (int ip = 0; ip < nplist(); ip++)
                    derived_state_.dsigmazdp_.at(iz + nz * ip) = 0;
            }
        }
    }

    if (always_check_finite_) {
        checkFinite(
            derived_state_.dsigmazdp_, ModelQuantity::dsigmazdp, nplist()
        );
    }
}

void Model::fdJzdz(
    int const ie, int const nroots, realtype const t, AmiVector const& x,
    ExpData const& edata
) {
    if (!nz)
        return;

    derived_state_.dJzdz_.assign(nztrue * nz * nJ, 0.0);

    fz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            fdJzdz(
                &derived_state_.dJzdz_.at(iztrue * nz * nJ), iztrue,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.z_.data(), derived_state_.sigmaz_.data(),
                edata.getObservedEventsPtr(nroots)
            );
            if (always_check_finite_) {
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJzdz_.at(iztrue * nz * nJ), nz * nJ
                    ),
                    ModelQuantity::dJzdz, nz
                );
            }
        }
    }
}

void Model::fdJzdsigma(
    int const ie, int const nroots, realtype const t, AmiVector const& x,
    ExpData const& edata
) {
    if (!nz)
        return;

    derived_state_.dJzdsigma_.assign(nztrue * nz * nJ, 0.0);

    fz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            fdJzdsigma(
                &derived_state_.dJzdsigma_.at(iztrue * nz * nJ), iztrue,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.z_.data(), derived_state_.sigmaz_.data(),
                edata.getObservedEventsPtr(nroots)
            );
            if (always_check_finite_) {
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJzdsigma_.at(iztrue * nz * nJ), nz * nJ
                    ),
                    ModelQuantity::dJzdsigma, nz
                );
            }
        }
    }
}

void Model::fdJzdp(
    int const ie, int const nroots, realtype t, AmiVector const& x,
    ExpData const& edata
) {
    if (!nz)
        return;
    // dJzdz         nJ x nz x nztrue
    // dJzdsigma     nJ x nz x nztrue
    // dzdp          nz x nplist()
    // dJzdp         nJ x nplist()

    derived_state_.dJzdp_.assign(nJ * nplist(), 0.0);

    fdzdp(ie, t, x);
    fdsigmazdp(ie, nroots, t, &edata);
    fdJzdz(ie, nroots, t, x, edata);
    fdJrzdz(ie, nroots, t, x, edata);
    fdJzdsigma(ie, nroots, t, x, edata);
    fdJrzdsigma(ie, nroots, t, x, edata);

    setNaNtoZero(derived_state_.dzdp_);
    setNaNtoZero(derived_state_.dsigmazdp_);
    setNaNtoZero(derived_state_.dJzdz_);
    setNaNtoZero(derived_state_.dJrzdz_);
    setNaNtoZero(derived_state_.dJzdsigma_);
    setNaNtoZero(derived_state_.dJrzdsigma_);
    for (int izt = 0; izt < nztrue; ++izt) {
        if (!edata.isSetObservedEvents(nroots, izt))
            continue;

        if (t < edata.getTimepoint(edata.nt() - 1)) {
            // with z
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nplist(), nz, 1.0,
                &derived_state_.dJzdz_.at(izt * nz * nJ), nJ,
                derived_state_.dzdp_.data(), nz, 1.0,
                derived_state_.dJzdp_.data(), nJ
            );
        } else {
            // with rz
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nplist(), nz, 1.0,
                &derived_state_.dJrzdsigma_.at(izt * nz * nJ), nJ,
                derived_state_.dsigmazdp_.data(), nz, 1.0,
                derived_state_.dJzdp_.data(), nJ
            );

            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nplist(), nz, 1.0,
                &derived_state_.dJrzdz_.at(izt * nz * nJ), nJ,
                derived_state_.dzdp_.data(), nz, 1.0,
                derived_state_.dJzdp_.data(), nJ
            );
        }

        amici_dgemm(
            BLASLayout::colMajor, BLASTranspose::noTrans,
            BLASTranspose::noTrans, nJ, nplist(), nz, 1.0,
            &derived_state_.dJzdsigma_.at(izt * nz * nJ), nJ,
            derived_state_.dsigmazdp_.data(), nz, 1.0,
            derived_state_.dJzdp_.data(), nJ
        );
    }
}

void Model::fdJzdx(
    int const ie, int const nroots, realtype const t, AmiVector const& x,
    ExpData const& edata
) {
    // dJzdz         nJ x nz        x nztrue
    // dzdx          nz x nx_solver
    // dJzdx         nJ x nx_solver x nmaxevent
    if (!nz)
        return;

    derived_state_.dJzdx_.assign(nJ * nx_solver, 0.0);

    fdJzdz(ie, nroots, t, x, edata);
    fdJrzdz(ie, nroots, t, x, edata);
    fdzdx(ie, t, x);
    fdrzdx(ie, t, x);

    setNaNtoZero(derived_state_.dJzdz_);
    setNaNtoZero(derived_state_.dJrzdz_);
    setNaNtoZero(derived_state_.dzdx_);
    setNaNtoZero(derived_state_.drzdx_);

    for (int izt = 0; izt < nztrue; ++izt) {
        if (!edata.isSetObservedEvents(nroots, izt))
            continue;

        if (t < edata.getTimepoint(edata.nt() - 1)) {
            // z
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nx_solver, nz, 1.0,
                &derived_state_.dJzdz_.at(izt * nz * nJ), nJ,
                derived_state_.dzdx_.data(), nz, 1.0,
                derived_state_.dJzdx_.data(), nJ
            );
        } else {
            // rz
            amici_dgemm(
                BLASLayout::colMajor, BLASTranspose::noTrans,
                BLASTranspose::noTrans, nJ, nx_solver, nz, 1.0,
                &derived_state_.dJrzdz_.at(izt * nz * nJ), nJ,
                derived_state_.drzdx_.data(), nz, 1.0,
                derived_state_.dJzdx_.data(), nJ
            );
        }
    }
}

void Model::fdJrzdz(
    int const ie, int const nroots, realtype const t, AmiVector const& x,
    ExpData const& edata
) {
    if (!nz)
        return;

    derived_state_.dJrzdz_.assign(nztrue * nz * nJ, 0.0);

    frz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            fdJrzdz(
                &derived_state_.dJrzdz_.at(iztrue * nz * nJ), iztrue,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.rz_.data(), derived_state_.sigmaz_.data()
            );
            if (always_check_finite_) {
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJrzdz_.at(iztrue * nz * nJ), nz * nJ
                    ),
                    ModelQuantity::dJrzdz, nz
                );
            }
        }
    }
}

void Model::fdJrzdsigma(
    int const ie, int const nroots, realtype const t, AmiVector const& x,
    ExpData const& edata
) {
    if (!nz)
        return;

    derived_state_.dJrzdsigma_.assign(nztrue * nz * nJ, 0.0);

    frz(ie, t, x);
    fsigmaz(ie, nroots, t, &edata);

    for (int iztrue = 0; iztrue < nztrue; iztrue++) {
        if (edata.isSetObservedEvents(nroots, iztrue)) {
            fdJrzdsigma(
                &derived_state_.dJrzdsigma_.at(iztrue * nz * nJ), iztrue,
                state_.unscaledParameters.data(), state_.fixedParameters.data(),
                derived_state_.rz_.data(), derived_state_.sigmaz_.data()
            );
            if (always_check_finite_) {
                checkFinite(
                    gsl::span<realtype>(
                        &derived_state_.dJrzdsigma_.at(iztrue * nz * nJ),
                        nz * nJ
                    ),
                    ModelQuantity::dJrzdsigma, nz
                );
            }
        }
    }
}

void Model::fspl(realtype const t) {
    for (int ispl = 0; ispl < nspl; ispl++)
        state_.spl_[ispl] = splines_[ispl].get_value(t);
}

void Model::fsspl(realtype const t) {
    derived_state_.sspl_.zero();
    realtype* sspl_data = derived_state_.sspl_.data();
    for (int ip = 0; ip < nplist(); ip++) {
        for (int ispl = 0; ispl < nspl; ispl++)
            sspl_data[ispl + nspl * plist(ip)]
                = splines_[ispl].get_sensitivity(t, ip, state_.spl_[ispl]);
    }
}

void Model::fw(realtype const t, realtype const* x, bool include_static) {
    if (include_static) {
        std::fill(derived_state_.w_.begin(), derived_state_.w_.end(), 0.0);
    }
    fspl(t);
    fw(derived_state_.w_.data(), t, x, state_.unscaledParameters.data(),
       state_.fixedParameters.data(), state_.h.data(), state_.total_cl.data(),
       state_.spl_.data(), include_static);

    if (always_check_finite_) {
        checkFinite(derived_state_.w_, ModelQuantity::w);
    }
}

void Model::fdwdp(realtype const t, realtype const* x, bool include_static) {
    if (!nw)
        return;

    fw(t, x, include_static);
    derived_state_.dwdp_.zero();
    if (pythonGenerated) {
        if (!ndwdp)
            return;
        fsspl(t);
        fdwdw(t, x, include_static);
        if (include_static) {
            dwdp_hierarchical_.at(0).zero();
            fdwdp_colptrs(dwdp_hierarchical_.at(0));
            fdwdp_rowvals(dwdp_hierarchical_.at(0));
        }
        fdwdp(
            dwdp_hierarchical_.at(0).data(), t, x,
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), derived_state_.w_.data(), state_.total_cl.data(),
            state_.stotal_cl.data(), state_.spl_.data(),
            derived_state_.sspl_.data(), include_static
        );

        for (int irecursion = 1; irecursion <= w_recursion_depth_;
             irecursion++) {
            dwdw_.sparse_multiply(
                dwdp_hierarchical_.at(irecursion),
                dwdp_hierarchical_.at(irecursion - 1)
            );
        }
        derived_state_.dwdp_.sparse_sum(dwdp_hierarchical_);

    } else {
        if (!derived_state_.dwdp_.capacity())
            return;
        // matlab generated
        fdwdp(
            derived_state_.dwdp_.data(), t, x, state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.h.data(),
            derived_state_.w_.data(), state_.total_cl.data(),
            state_.stotal_cl.data(), state_.spl_.data(),
            derived_state_.sspl_.data()
        );
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.dwdp_, ModelQuantity::dwdp, t);
    }
}

void Model::fdwdx(realtype const t, realtype const* x, bool include_static) {
    if (!nw)
        return;

    fw(t, x, include_static);

    derived_state_.dwdx_.zero();
    if (pythonGenerated) {
        if (!dwdx_hierarchical_.at(0).capacity())
            return;

        fdwdw(t, x, include_static);

        if (include_static) {
            dwdx_hierarchical_.at(0).zero();
            fdwdx_colptrs(dwdx_hierarchical_.at(0));
            fdwdx_rowvals(dwdx_hierarchical_.at(0));
        }
        fdwdx(
            dwdx_hierarchical_.at(0).data(), t, x,
            state_.unscaledParameters.data(), state_.fixedParameters.data(),
            state_.h.data(), derived_state_.w_.data(), state_.total_cl.data(),
            state_.spl_.data(), include_static
        );

        for (int irecursion = 1; irecursion <= w_recursion_depth_;
             irecursion++) {
            dwdw_.sparse_multiply(
                dwdx_hierarchical_.at(irecursion),
                dwdx_hierarchical_.at(irecursion - 1)
            );
        }
        derived_state_.dwdx_.sparse_sum(dwdx_hierarchical_);

    } else {
        if (!derived_state_.dwdx_.capacity())
            return;
        derived_state_.dwdx_.zero();
        fdwdx(
            derived_state_.dwdx_.data(), t, x, state_.unscaledParameters.data(),
            state_.fixedParameters.data(), state_.h.data(),
            derived_state_.w_.data(), state_.total_cl.data(), state_.spl_.data()
        );
    }

    if (always_check_finite_) {
        checkFinite(derived_state_.dwdx_, ModelQuantity::dwdx, t);
    }
}

void Model::fdwdw(realtype const t, realtype const* x, bool include_static) {
    if (!ndwdw)
        return;

    if (include_static) {
        dwdw_.zero();
        fdwdw_colptrs(dwdw_);
        fdwdw_rowvals(dwdw_);
    }

    fdwdw(
        dwdw_.data(), t, x, state_.unscaledParameters.data(),
        state_.fixedParameters.data(), state_.h.data(),
        derived_state_.w_.data(), state_.total_cl.data(), include_static
    );

    if (always_check_finite_) {
        checkFinite(dwdw_, ModelQuantity::dwdw, t);
    }
}

void Model::fx_rdata(
    realtype* x_rdata, realtype const* x_solver, realtype const* /*tcl*/,
    realtype const* /*p*/, realtype const* /*k*/
) {
    if (nx_solver != nx_rdata)
        throw AmiException(
            "A model that has differing nx_solver and nx_rdata needs "
            "to implement its own fx_rdata"
        );
    std::copy_n(x_solver, nx_solver, x_rdata);
}

void Model::fsx_rdata(
    realtype* sx_rdata, realtype const* sx_solver, realtype const* stcl,
    realtype const* p, realtype const* k, realtype const* x_solver,
    realtype const* tcl, int const ip
) {
    if (nx_solver == nx_rdata) {
        std::copy_n(sx_solver, nx_solver, sx_rdata);
        return;
    }

    // sx_rdata = dx_rdata/dx_solver * sx_solver
    //             + dx_rdata/d_tcl * stcl + dxrdata/dp

    // 1) sx_rdata(nx_rdata, 1) = dx_rdatadp
    std::fill_n(sx_rdata, nx_rdata, 0.0);
    fdx_rdatadp(sx_rdata, x_solver, tcl, p, k, ip);

    // the following could be moved to the calling function, as it's independent
    //  of `ip`

    // 2) sx_rdata(nx_rdata, 1) +=
    //          dx_rdata/dx_solver(nx_rdata,nx_solver) * sx_solver(nx_solver, 1)
    derived_state_.dx_rdatadx_solver.zero();
    fdx_rdatadx_solver(
        derived_state_.dx_rdatadx_solver.data(), x_solver, tcl, p, k
    );
    fdx_rdatadx_solver_colptrs(derived_state_.dx_rdatadx_solver);
    fdx_rdatadx_solver_rowvals(derived_state_.dx_rdatadx_solver);
    derived_state_.dx_rdatadx_solver.multiply(
        gsl::make_span(sx_rdata, nx_rdata), gsl::make_span(sx_solver, nx_solver)
    );

    // 3) sx_rdata(nx_rdata, 1) += dx_rdata/d_tcl(nx_rdata,ntcl) * stcl
    derived_state_.dx_rdatadtcl.zero();
    fdx_rdatadtcl(derived_state_.dx_rdatadtcl.data(), x_solver, tcl, p, k);
    fdx_rdatadtcl_colptrs(derived_state_.dx_rdatadtcl);
    fdx_rdatadtcl_rowvals(derived_state_.dx_rdatadtcl);
    derived_state_.dx_rdatadtcl.multiply(
        gsl::make_span(sx_rdata, nx_rdata),
        gsl::make_span(stcl, (nx_rdata - nx_solver))
    );
}

void Model::fx_solver(realtype* x_solver, realtype const* x_rdata) {
    if (nx_solver != nx_rdata)
        throw AmiException(
            "A model that has differing nx_solver and nx_rdata needs "
            "to implement its own fx_solver"
        );
    std::copy_n(x_rdata, nx_rdata, x_solver);
}

void Model::fsx_solver(realtype* sx_solver, realtype const* sx_rdata) {
    /* for the moment we do not need an implementation of fsx_solver as
     * we can simply reuse fx_solver and replace states by their
     * sensitivities */
    fx_solver(sx_solver, sx_rdata);
}

void Model::ftotal_cl(
    realtype* /*total_cl*/, realtype const* /*x_rdata*/, realtype const* /*p*/,
    realtype const* /*k*/
) {
    if (nx_solver != nx_rdata)
        throw AmiException(
            "A model that has differing nx_solver and nx_rdata needs "
            "to implement its own ftotal_cl"
        );
}

void Model::fstotal_cl(
    realtype* stotal_cl, realtype const* sx_rdata, int const ip,
    realtype const* x_rdata, realtype const* p, realtype const* k,
    realtype const* tcl
) {
    if (nx_solver == nx_rdata)
        return;

    // stotal_cl(ncl,1) =
    //            dtotal_cl/dp(ncl,1)
    //          + dtotal_cl/dx_rdata(ncl,nx_rdata) * sx_rdata(nx_rdata,1)

    // 1) stotal_cl = dtotal_cl/dp
    std::fill_n(stotal_cl, ncl(), 0.0);
    fdtotal_cldp(stotal_cl, x_rdata, p, k, ip);

    // 2) stotal_cl += dtotal_cl/dx_rdata(ncl,nx_rdata) * sx_rdata(nx_rdata,1)
    derived_state_.dtotal_cldx_rdata.zero();
    fdtotal_cldx_rdata(
        derived_state_.dtotal_cldx_rdata.data(), x_rdata, tcl, p, k
    );
    fdtotal_cldx_rdata_colptrs(derived_state_.dtotal_cldx_rdata);
    fdtotal_cldx_rdata_rowvals(derived_state_.dtotal_cldx_rdata);
    derived_state_.dtotal_cldx_rdata.multiply(
        gsl::make_span(stotal_cl, ncl()), gsl::make_span(sx_rdata, nx_rdata)
    );
}

std::vector<double> Model::get_trigger_timepoints() const {
    std::vector<double> trigger_timepoints(
        state_independent_events_.size(), 0.0
    );
    // collect keys from state_independent_events_ which are the trigger
    // timepoints
    auto it = trigger_timepoints.begin();
    for (auto const& kv : state_independent_events_) {
        *(it++) = kv.first;
    }
    std::sort(trigger_timepoints.begin(), trigger_timepoints.end());
    return trigger_timepoints;
}

const_N_Vector Model::computeX_pos(const_N_Vector x) {
    if (any_state_non_negative_) {
        for (int ix = 0; ix < derived_state_.x_pos_tmp_.getLength(); ++ix) {
            derived_state_.x_pos_tmp_.at(ix)
                = (state_is_non_negative_.at(ix) && NV_Ith_S(x, ix) < 0)
                      ? 0
                      : NV_Ith_S(x, ix);
        }
        return derived_state_.x_pos_tmp_.getNVector();
    }

    return x;
}

realtype const* Model::computeX_pos(AmiVector const& x) {
    if (any_state_non_negative_) {
        computeX_pos(x.getNVector());
        return derived_state_.x_pos_tmp_.data();
    }
    return x.data();
}

void Model::setReinitializationStateIdxs(std::vector<int> const& idxs) {
    for (auto idx : idxs) {
        if (idx < 0 || idx >= nx_rdata)
            throw AmiException("Invalid state index given: %d", idx);
    }

    simulation_parameters_.reinitialization_state_idxs_sim = idxs;
}

std::vector<int> const& Model::getReinitializationStateIdxs() const {
    return simulation_parameters_.reinitialization_state_idxs_sim;
}

AmiVectorArray const& Model::get_dxdotdp() const {
    assert(!pythonGenerated);
    return derived_state_.dxdotdp;
}

SUNMatrixWrapper const& Model::get_dxdotdp_full() const {
    assert(pythonGenerated);
    return derived_state_.dxdotdp_full;
}

} // namespace amici
