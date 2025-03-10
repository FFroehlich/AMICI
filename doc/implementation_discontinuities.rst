Handling of Discontinuities
===========================

This document provides guidance and rationale on the implementation of events
in AMICI. Events include any discontinuities in the right hand side of the
differential equation. There are three types of discontinuities:

- **Solution Jump Discontinuities** can be created by SBML events or delta
  functions in the right hand side.

- **Right-Hand-Side Jump Discontinuities** result in removable
  discontinuities in the solution and can be created by Piecewise,
  Heaviside functions and other logical operators in the right hand side.

- **Right-Hand-Side Removable Discontinuities** do not lead to
  discontinuities in the solution, but may lead to discontinuous higher
  order temporal derivatives and can be created by functions such as max or
  min in the right hand side.

Mathematical Considerations
---------------------------

A detailed mathematical description of the required sensitivity formulas is
provided in

* Fröhlich, F., Theis, F. J., Rädler, J. O., & Hasenauer, J. (2017).
  Parameter estimation for dynamical systems with discrete events and logical
  operations. Bioinformatics, 33(7), 1049-1056.
  doi:`10.1093/bioinformatics/btw764 <https://doi.org/10.1093/bioinformatics/btw764>`_.

Algorithmic Considerations
--------------------------

Solution Jump Discontinuities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SUNDIALS by itself does not support solution jump discontinuities. We
implement support by accessing private SUNDIALS API in
:cpp:func:`amici::Solver::resetState`,
:cpp:func:`amici::Solver::reInitPostProcess` and
:cpp:func:`amici::Solver::reInitPostProcessB`. These functions reset interval
variables to initial values to simulate a fresh integration start, but
keep/update the solution history, which is important for adjoint solutions.


Right-Hand-Side Jump Discontinuities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In principle these discontinuities do not need any special treatment, but
empirically, the solver may overstep or completely ignore the discontinuity,
leading to poor solution quality. This is particularly problematic when
step size is large and changes in step size, which can be caused by
parameter changes, inclusion of forward sensitivities or during backward
solves, may alter solutions in unexpected ways. Accordingly, finite
difference approximations, forward sensitivities as well as adjoint
sensitivities will yield poor derivative approximations.

To address these issues, we use the built-in rootfinding functionality in
SUNDIALS, which pauses the solver at the locations of discontinuities and
avoids overstepping or ignoring of discontinuities.

Another difficulty comes with the evaluation of Heaviside functions. After
or during processing of discontinuities, Heaviside functions need to be
evaluated at the left and right hand limit of discontinuities.
This is challenging as the solver may slightly over- or understep the
discontinuity timepoint by a small epsilon and limits have to be correctly
computed in both forward and backward passes.

To address this issue, AMICI uses a vector of Heaviside helper variables `h`
that keeps track of the values of the Heaviside functions that have the
respective root function as argument. These will be automatically updated
during events and take either 0 or 1 values as appropriate pre/post event
limits.

In order to fully support SBML events and Piecewise functions, AMICI uses
the SUNDIALS functionality to only track zero crossings from negative to
positive. Accordingly, two root functions are necessary to keep track of
Heaviside functions and two Heaviside function helper variables will be
created, where one corresponds to the value of `Heaviside(...)` and one
to the value of `1-Heaviside(...)`. To ensure that Heaviside functions are
correctly evaluated at the beginning of the simulation, Heaviside functions
are implement as unit steps that evaluate to `1` at `0`. The arguments of
Heaviside functions are normalized such that respective properties of
Piecewise functions are conserved for the first Heaviside function variable.
Accordingly, the value of of the second helper variable is incorrect when
simulation starts when the respective Heaviside function evaluates to zero
at initialization and should generally not be used.



Right-Hand-Side Removable Discontinuities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Removable discontinuities do not require any special treatment. Numerically,
this may be advantageous, but is currently not implemented.
