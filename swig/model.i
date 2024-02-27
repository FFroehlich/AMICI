%module model

// Add necessary symbols to generated header
%{
#include "amici/model.h"
using namespace amici;
%}

// remove functions that use AmiVector(Array) since that class anyways cannot
// be exposed in swig
%ignore addAdjointQuadratureEventUpdate;
%ignore addAdjointStateEventUpdate;
%ignore addEventObjective;
%ignore addEventObjectiveRegularization;
%ignore addEventObjectiveSensitivity;
%ignore addObservableObjective;
%ignore addObservableObjectiveSensitivity;
%ignore addPartialEventObjectiveSensitivity;
%ignore addPartialObservableObjectiveSensitivity;
%ignore addStateEventUpdate;
%ignore addStateSensitivityEventUpdate;
%ignore fsx_rdata;
%ignore fx_rdata;
%ignore getAdjointStateEventUpdate;
%ignore getEventTimeSensitivity;
%ignore getAdjointStateObservableUpdate;
%ignore getEvent;
%ignore getEventRegularization;
%ignore getEventRegularizationSensitivity;
%ignore getEventSensitivity;
%ignore getEventTimeSensitivity;
%ignore getObservable;
%ignore getObservableSensitivity;
%ignore getExpression;
%ignore initEvents;
%ignore initialize;
%ignore initializeB;
%ignore initializeStateSensitivities;
%ignore initializeStates;
%ignore reinitialize;
%ignore ModelState;
%ignore getModelState;
%ignore setModelState;
%ignore fx0;
%ignore fx0_fixedParameters;
%ignore fsx0;
%ignore fsx0_fixedParameters;
%ignore get_dxdotdp;
%ignore get_dxdotdp_full;
%ignore checkFinite;
%ignore fJrz;
%ignore fJy;
%ignore fJz;
%ignore fdJrzdsigma;
%ignore fdJrzdz;
%ignore fdJzdsigma;
%ignore fdJzdz;
%ignore fdJydsigma;
%ignore fdeltaqB;
%ignore fdeltasx;
%ignore fdeltax;
%ignore fdeltaxB;
%ignore fdrzdp;
%ignore fdrzdx;
%ignore fdsigmaydp;
%ignore fdsigmazdp;
%ignore fdydp;
%ignore fdydx;
%ignore fdzdp;
%ignore fdzdx;
%ignore frz;
%ignore fsdx0;
%ignore fsigmay;
%ignore fsigmaz;
%ignore fsrz;
%ignore fstau;
%ignore fsz;
%ignore fw;
%ignore fy;
%ignore fz;
%ignore updateHeaviside;
%ignore updateHeavisideB;
%ignore getEventSigma;
%ignore getEventSigmaSensitivity;
%ignore getObservableSigma;
%ignore getObservableSigmaSensitivity;
%ignore getUnobservedEventSensitivity;
%ignore fdsigmaydy;
%ignore fdspline_slopesdp;
%ignore fdspline_valuesdp;
%ignore fdtotal_cldp;
%ignore fdtotal_cldx_rdata;
%ignore fdx_rdatadp;
%ignore fdx_rdatadtcl;
%ignore fdx_rdatadx_solver;
%ignore fdsigmaydy;

%newobject amici::Model::clone;

%extend amici::Model {
%pythoncode %{
def __deepcopy__(self, memo):
    return self.clone()
%}
};

%extend std::unique_ptr<amici::Model> {
%pythoncode %{
def __deepcopy__(self, memo):
    return self.clone()
%}
};

// Process symbols in header
%include "amici/model.h"
