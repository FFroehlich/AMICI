#ifndef AMICI_SERIALIZATION_H
#define AMICI_SERIALIZATION_H

#include "amici/model.h"
#include "amici/rdata.h"
#include "amici/solver.h"
#include "amici/solver_cvodes.h"

#include <chrono>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

/** @file serialization.h Helper functions and forward declarations for
 * boost::serialization */
namespace boost {
namespace serialization {

/**
 * @brief Serialize a raw array to a boost archive
 * @param ar archive
 * @param p Pointer to array
 * @param size Size of p
 */
template <class Archive, typename T>
void archiveVector(Archive& ar, T** p, int size) {
    if (Archive::is_loading::value) {
        if (*p != nullptr)
            delete[] *p;
        ar & size;
        *p = size ? new T[size] : nullptr;
    } else {
        size = *p == nullptr ? 0 : size;
        ar & size;
    }
    ar& make_array<T>(*p, size);
}

#ifndef EXHALE_DOXYGEN_SHOULD_SKIP_THIS
/**
 * @brief Serialize amici::Solver to boost archive
 * @param ar Archive
 * @param s Solver instance to serialize
 */
template <class Archive>
void serialize(Archive& ar, amici::Solver& s, unsigned int const /*version*/) {
    ar & s.sensi_;
    ar & s.atol_;
    ar & s.rtol_;
    ar & s.atolB_;
    ar & s.rtolB_;
    ar & s.atol_fsa_;
    ar & s.rtol_fsa_;
    ar & s.quad_atol_;
    ar & s.quad_rtol_;
    ar & s.ss_tol_factor_;
    ar & s.ss_atol_;
    ar & s.ss_rtol_;
    ar & s.ss_tol_sensi_factor_;
    ar & s.ss_atol_sensi_;
    ar & s.ss_rtol_sensi_;
    ar & s.maxsteps_;
    ar & s.maxstepsB_;
    ar & s.newton_maxsteps_;
    ar & s.newton_damping_factor_mode_;
    ar & s.newton_damping_factor_lower_bound_;
    ar & s.ism_;
    ar & s.sensi_meth_;
    ar & s.linsol_;
    ar & s.interp_type_;
    ar & s.lmm_;
    ar & s.iter_;
    ar & s.stldet_;
    ar & s.ordering_;
    ar & s.cpu_time_;
    ar & s.cpu_timeB_;
    ar & s.newton_step_steadystate_conv_;
    ar & s.check_sensi_steadystate_conv_;
    ar & s.rdata_mode_;
    ar & s.maxtime_;
}

/**
 * @brief Serialize std::chrono::duration to boost archive
 * @param ar Archive
 * @param d Duration
 */
template <class Archive, class Period, class Rep>
void serialize(
    Archive& ar, std::chrono::duration<Period, Rep>& d,
    unsigned int const /*version*/
) {
    Period tmp_period;
    if (Archive::is_loading::value) {
        ar & tmp_period;
        d = std::chrono::duration<Period, Rep>(tmp_period);
    } else {
        tmp_period = d.count();
        ar & tmp_period;
    }
}

/**
 * @brief Serialize amici::CVodeSolver to boost archive
 * @param ar Archive
 * @param s Solver instance to serialize
 */
template <class Archive>
void serialize(
    Archive& ar, amici::CVodeSolver& s, unsigned int const /*version*/
) {
    ar& static_cast<amici::Solver&>(s);
}

/**
 * @brief Serialize amici::Model to boost archive
 * @param ar Archive
 * @param m Model instance to serialize
 */
template <class Archive>
void serialize(Archive& ar, amici::Model& m, unsigned int const /*version*/) {
    ar& dynamic_cast<amici::ModelDimensions&>(m);
    ar & m.simulation_parameters_;
    ar & m.o2mode;
    ar & m.z2event_;
    ar & m.idlist;
    ar & m.state_.h;
    ar & m.state_.unscaledParameters;
    ar & m.state_.fixedParameters;
    ar & m.state_.plist;
    ar & m.x0data_;
    ar & m.sx0data_;
    ar & m.nmaxevent_;
    ar & m.state_is_non_negative_;
    ar & m.pythonGenerated;
    ar & m.min_sigma_;
    ar & m.sigma_res_;
    ar & m.steadystate_computation_mode_;
    ar & m.steadystate_sensitivity_mode_;
    ar & m.state_independent_events_;
}

/**
 * @brief Serialize amici::SimulationParameters to boost archive
 * @param ar Archive
 * @param s amici::SimulationParameters instance to serialize
 */
template <class Archive>
void serialize(
    Archive& ar, amici::SimulationParameters& s, unsigned int const /*version*/
) {
    ar & s.fixedParameters;
    ar & s.fixedParametersPreequilibration;
    ar & s.fixedParametersPresimulation;
    ar & s.parameters;
    ar & s.x0;
    ar & s.sx0;
    ar & s.pscale;
    ar & s.plist;
    ar & s.ts_;
    ar & s.tstart_;
    ar & s.t_presim;
    ar & s.reinitializeFixedParameterInitialStates;
}

/**
 * @brief Serialize amici::ReturnData to boost archive
 * @param ar Archive
 * @param r ReturnData instance to serialize
 */

template <class Archive>
void serialize(
    Archive& ar, amici::ReturnData& r, unsigned int const /*version*/
) {
    ar& dynamic_cast<amici::ModelDimensions&>(r);
    ar & r.id;
    ar & r.nx;
    ar & r.nxtrue;
    ar & r.nplist;
    ar & r.nmaxevent;
    ar & r.nt;
    ar & r.newton_maxsteps;
    ar & r.pscale;
    ar & r.o2mode;
    ar & r.sensi;
    ar & r.sensi_meth;

    ar & r.ts;
    ar & r.xdot;
    ar & r.J;
    ar & r.w;
    ar & r.z & r.sigmaz;
    ar & r.sz & r.ssigmaz;
    ar & r.rz;
    ar & r.srz;
    ar & r.s2rz;
    ar & r.x;
    ar & r.sx;
    ar & r.y & r.sigmay;
    ar & r.sy & r.ssigmay;

    ar & r.numsteps;
    ar & r.numstepsB;
    ar & r.numrhsevals;
    ar & r.numrhsevalsB;
    ar & r.numerrtestfails;
    ar & r.numerrtestfailsB;
    ar & r.numnonlinsolvconvfails;
    ar & r.numnonlinsolvconvfailsB;
    ar & r.order;
    ar & r.cpu_time;
    ar & r.cpu_timeB;
    ar & r.cpu_time_total;
    ar & r.preeq_cpu_time;
    ar & r.preeq_cpu_timeB;
    ar & r.preeq_status;
    ar & r.preeq_numsteps;
    ar & r.preeq_wrms;
    ar & r.preeq_t;
    ar & r.posteq_cpu_time;
    ar & r.posteq_cpu_timeB;
    ar & r.posteq_status;
    ar & r.posteq_numsteps;
    ar & r.posteq_wrms;
    ar & r.posteq_t;
    ar & r.x0;
    ar & r.sx0;
    ar & r.llh;
    ar & r.chi2;
    ar & r.sllh;
    ar & r.s2llh;
    ar & r.status;
}

/**
 * @brief Serialize amici::ModelDimensions to boost archive
 * @param ar Archive
 * @param m ModelDimensions instance to serialize
 */

template <class Archive>
void serialize(
    Archive& ar, amici::ModelDimensions& m, unsigned int const /*version*/
) {
    ar & m.nx_rdata;
    ar & m.nxtrue_rdata;
    ar & m.nx_solver;
    ar & m.nxtrue_solver;
    ar & m.nx_solver_reinit;
    ar & m.np;
    ar & m.nk;
    ar & m.ny;
    ar & m.nytrue;
    ar & m.nz;
    ar & m.nztrue;
    ar & m.ne;
    ar & m.ne_solver;
    ar & m.nspl;
    ar & m.nw;
    ar & m.ndwdx;
    ar & m.ndwdp;
    ar & m.ndwdw;
    ar & m.ndxdotdw;
    ar & m.ndJydy;
    ar & m.nnz;
    ar & m.nJ;
    ar & m.ubw;
    ar & m.lbw;
}
#endif
} // namespace serialization
} // namespace boost

namespace amici {

/**
 * @brief Serialize object to char array
 *
 * @param data input object
 * @param size maximum char length
 *
 * @return The object serialized as char
 */
template <typename T> char* serializeToChar(T const& data, int* size) {

    try {
        std::string serialized;
        ::boost::iostreams::back_insert_device<std::string> inserter(serialized
        );
        ::boost::iostreams::stream<
            ::boost::iostreams::back_insert_device<std::string>>
            s(inserter);
        ::boost::archive::binary_oarchive oar(s);
        oar << data;
        s.flush();

        char* charBuffer = new char[serialized.size()];
        memcpy(charBuffer, serialized.data(), serialized.size());

        if (size)
            *size = serialized.size();

        return charBuffer;
    } catch (boost::archive::archive_exception const& e) {
        throw AmiException("Serialization to char failed: %s", e.what());
    }
}

/**
 * @brief Deserialize object that has been serialized using serializeToChar
 *
 * @param buffer serialized object
 * @param size length of buffer
 *
 * @return The deserialized object
 */

template <typename T> T deserializeFromChar(char const* buffer, int size) {
    namespace ba = ::boost::archive;
    namespace bio = ::boost::iostreams;

    bio::basic_array_source<char> device(buffer, size);
    bio::stream<bio::basic_array_source<char>> s(device);

    T data;

    try {
        // archive must be destroyed BEFORE returning
        ba::binary_iarchive iar(s);
        iar >> data;
    } catch (ba::archive_exception const& e) {
        throw AmiException("Deserialization from char failed: %s", e.what());
    }
    return data;
}

/**
 * @brief Serialize object to string
 *
 * @param data input object
 *
 * @return The object serialized as string
 */

template <typename T> std::string serializeToString(T const& data) {
    namespace ba = ::boost::archive;
    namespace bio = ::boost::iostreams;

    std::string serialized;
    bio::back_insert_device<std::string> inserter(serialized);
    bio::stream<bio::back_insert_device<std::string>> os(inserter);

    try {
        // archive must be destroyed BEFORE returning
        ba::binary_oarchive oar(os);
        oar << data;
    } catch (ba::archive_exception const& e) {
        throw AmiException("Serialization to string failed: %s", e.what());
    }

    return serialized;
}

/**
 * @brief Serialize object to std::vector<char>
 *
 * @param data input object
 *
 * @return The object serialized as std::vector<char>
 */

template <typename T> std::vector<char> serializeToStdVec(T const& data) {
    namespace ba = ::boost::archive;
    namespace bio = ::boost::iostreams;

    std::vector<char> buffer;
    bio::stream<bio::back_insert_device<std::vector<char>>> os(buffer);

    try {
        // archive must be destroyed BEFORE returning
        ba::binary_oarchive oar(os);
        oar << data;
    } catch (ba::archive_exception const& e) {
        throw AmiException("Serialization to std::vector failed: %s", e.what());
    }

    return buffer;
}

/**
 * @brief Deserialize object that has been serialized using serializeToString
 *
 * @param serialized serialized object
 *
 * @return The deserialized object
 */

template <typename T> T deserializeFromString(std::string const& serialized) {
    namespace ba = ::boost::archive;
    namespace bio = ::boost::iostreams;

    bio::basic_array_source<char> device(serialized.data(), serialized.size());
    bio::stream<bio::basic_array_source<char>> os(device);
    T deserialized;

    try {
        // archive must be destroyed BEFORE returning
        ba::binary_iarchive iar(os);
        iar >> deserialized;
    } catch (ba::archive_exception const& e) {
        throw AmiException(
            "Deserialization from std::string failed: %s", e.what()
        );
    }

    return deserialized;
}

} // namespace amici
#endif // AMICI_SERIALIZATION_H
