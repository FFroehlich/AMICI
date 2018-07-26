#ifndef AMICI_EDATA_H
#define AMICI_EDATA_H

#include "amici/defines.h"

#include <vector>

namespace amici {

class Model;
class ReturnData;

/** @brief ExpData carries all information about experimental or condition-specific data */
class ExpData {

  public:
    /** default constructor */
    ExpData();

    /**
     * @brief ExpData
     * @param nytrue
     * @param nztrue
     * @param nt
     * @param nmaxevent
     */
    ExpData(int nytrue, int nztrue, int nt, int nmaxevent);

    /**
     * @brief ExpData
     * @param nytrue
     * @param nztrue
     * @param nt
     * @param nmaxevent
     * @param my
     * @param sigmay
     * @param mz
     * @param sigmaz
     */
    ExpData(int nytrue, int nztrue, int nt, int nmaxevent,
            std::vector<realtype> const& my,
            std::vector<realtype> const& sigmay,
            std::vector<realtype> const& mz,
            std::vector<realtype> const& sigmaz);

    /**
     * constructor that initializes with Model
     *
     * @param model pointer to model specification object
     */
    ExpData(const Model &model);
    
    /**
     * constructor that initializes with returnData, adds
     *
     * @param rdata return data pointer with stored simulation results
     * @param sigma_y scalar standard deviations for all observables
     * @param sigma_z scalar standard deviations for all event observables
     */
    ExpData(const ReturnData &rdata, realtype sigma_y, realtype sigma_z);
    
    /**
     * constructor that initializes with returnData, adds
     *
     * @param rdata return data pointer with stored simulation results
     * @param sigma_y vector of standard deviations for every observable
     * @param sigma_z vector of standard deviations for every event observable
     */
    ExpData(const ReturnData &rdata, std::vector<realtype> sigma_y, std::vector<realtype> sigma_z);

    /**
     * @brief Copy constructor
     * @param other object to copy from
     */
    ExpData (const ExpData &other);

    void setObservedData(const double *observedData);
    void setObservedDataStdDev(const double *observedDataStdDev);
    void setObservedEvents(const double *observedEvents);
    void setObservedEventsStdDev(const double *observedEventsStdDev);

    ~ExpData();

    /** observed data (dimension: nt x nytrue, row-major) */
    std::vector<realtype> my;
    /** standard deviation of observed data (dimension: nt x nytrue, row-major) */
    std::vector<realtype> sigmay;

    /** observed events (dimension: nmaxevents x nztrue, row-major) */
    std::vector<realtype> mz;
    /** standard deviation of observed events/roots
     * (dimension: nmaxevents x nztrue, row-major)*/
    std::vector<realtype> sigmaz;
    
    /** number of observables */
    const int nytrue;
    /** number of event observables */
    const int nztrue;
    /** number of timepoints */
    const int nt;
    /** maximal number of event occurences */
    const int nmaxevent;

    /** condition-specific parameters of size Model::nk() or empty */
    std::vector<realtype> fixedParameters;
    /** condition-specific parameters for pre-equilibration of size Model::nk() or empty.
      * Overrides Solver::newton_preeq */
    std::vector<realtype> fixedParametersPreequilibration;
};

} // namespace amici

#endif /* AMICI_EDATA_H */
