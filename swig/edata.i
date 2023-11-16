%module edata

// Add necessary symbols to generated header
%{
#include "amici/edata.h"
using namespace amici;
%}

%ignore ConditionContext;

// ExpData.__repr__
%pythoncode %{
def _edata_repr(self: "ExpData"):
    n_data_y = sum(
        self.isSetObservedData(it, iy)
        for it in range(self.nt()) for
        iy in range(self.nytrue())
    )
    n_sigma_y = sum(
        self.isSetObservedDataStdDev(it, iy)
        for it in range(self.nt())
        for iy in range(self.nytrue())
    )
    n_data_z = sum(
        self.isSetObservedEvents(ie, iz)
        for ie in range(self.nmaxevent())
        for iz in range(self.nztrue())
    )
    n_sigma_z = sum(
        self.isSetObservedEventsStdDev(ie, iz)
        for ie in range(self.nmaxevent())
        for iz in range(self.nztrue())
    )

    custom_simulation_settings = []
    if self.pscale:
        custom_simulation_settings.append(f"parameter scales")
    if self.fixedParameters:
        custom_simulation_settings.append(f"constants")
    if self.fixedParametersPreequilibration:
        custom_simulation_settings.append(f"pre-equilibration condition")
    if self.t_presim:
        tmp = f"pre-simulation condition (t={self.t_presim})"
        if self.fixedParametersPresimulation:
            tmp += " with custom constants"
        custom_simulation_settings.append(tmp)
    if self.reinitializeFixedParameterInitialStates and self.reinitialization_state_idxs_sim:
        custom_simulation_settings.append(f"{len(self.reinitialization_state_idxs_sim)} reinitialized states (simulation)")
    if self.reinitializeFixedParameterInitialStates and self.reinitialization_state_idxs_presim:
        custom_simulation_settings.append(f"{len(self.reinitialization_state_idxs_presim)} reinitialized states (presimulation)")
    if self.parameters:
        custom_simulation_settings.append(f"parameters")
    if self.x0:
        custom_simulation_settings.append(f"initial states")
    if self.sx0:
        custom_simulation_settings.append(f"initial state sensitivities")

    if custom_simulation_settings:
        custom_simulation_settings = " with custom " + ", ".join(custom_simulation_settings)
    else:
        custom_simulation_settings = " without custom settings"

    return "\n".join([
        self.this.__repr__()[:-1],
        f"  condition '{self.id}' starting at t={self.tstart_}" + custom_simulation_settings,
        f"  {self.nt()}x{self.nytrue()} time-resolved datapoints",
        f"    ({n_data_y}/{self.nt()*self.nytrue()} measurements & {n_sigma_y}/{self.nt()*self.nytrue()} sigmas set)",
        f"  {self.nmaxevent()}x{self.nztrue()} event-resolved datapoints",
        f"    ({n_data_z}/{self.nmaxevent()*self.nztrue()} measurements & {n_sigma_z}/{self.nmaxevent()*self.nztrue()} sigmas set)",
        ">"
    ])
%}
%extend amici::ExpData {
%pythoncode %{
def __repr__(self):
    return _edata_repr(self)

def __eq__(self, other):
    return other.__class__ == self.__class__ and __eq__(self, other)

def __deepcopy__(self, memo):
    # invoke copy constructor
    return type(self)(self)

%}
};
%extend std::unique_ptr<amici::ExpData> {
%pythoncode %{
def __repr__(self):
    return _edata_repr(self)
%}
};


// Process symbols in header
%include "amici/edata.h"
