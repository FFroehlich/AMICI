#include "amici/model_state.h"

namespace amici {

ModelStateDerived::ModelStateDerived(const ModelDimensions &dim)
    : J_(dim.nx_solver, dim.nx_solver, dim.nnz, CSC_MAT),
      JB_(dim.nx_solver, dim.nx_solver, dim.nnz, CSC_MAT),
      dxdotdw_(dim.nx_solver, dim.nw, dim.ndxdotdw, CSC_MAT),
      w_(dim.nw),
      x_rdata_(dim.nx_rdata, 0.0),
      sx_rdata_(dim.nx_rdata, 0.0),
      // only required if there are conservation laws
      dx_rdatadx_solver(dim.nx_rdata - dim.nx_solver > 0
                            ? dim.nx_rdata * dim.nx_solver : 0, 0.0),
      // only required if there are conservation laws
      dx_rdatadtcl(dim.nx_rdata - dim.nx_solver > 0
                       ? dim.nx_rdata * (dim.nx_rdata - dim.nx_solver) : 0,
                   0.0),
      x_pos_tmp_(dim.nx_solver)
{}


} // namespace amici
