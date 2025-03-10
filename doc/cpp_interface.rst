.. _cpp_interface:

===========================
Using AMICI's C++ interface
===========================

The various import functions in of the
:ref:`Python interface <python_interface>` and
:ref:`Matlab interface <matlab_interface>` translate models defined in
different formats into C++ code. These generated model libraries, together with
the AMICI base library can be used in any C++ application for model simulation
and sensitivity analysis. This section will give a short overview over the
generated files and provide a brief introduction of how this code can be
included in other applications. Further details are available in the
:doc:`C++ API reference <_exhale_cpp_api/library_root>`.

AMICI-generated C++ model files
===============================

After importing a model using either the
:ref:`Python interface <python_interface>` or the
:ref:`Matlab interface <matlab_interface>`, the specified output directory
contains (among others) C++ code for the various model functions.

The content of a model source directory looks something like this (given
`MODEL_NAME=model_steadystate`):

.. code-block:: text

   CMakeLists.txt
   main.cpp
   deltaqB.cpp
   deltaqB.h
   [... many more files *.(cpp|h|md5|o) ]
   wrapfunctions.cpp
   wrapfunctions.h
   model_steadystate.h
   model_steadystate.cpp

These files provide the implementation of a model-specific subclass of
:cpp:class:`amici::Model`. The ``CMakeLists.txt`` file can be used to build the
model library using `CMake <https://cmake.org/>`_.
``main.cpp`` contains a simple scaffold for running a model simulation from C++.
See next section for more details on these files.


Running a model simulation
==========================

AMICI's public API is mostly available through
:ref:`amici/amici.h <file_include_amici_amici.h>`. This is the only header file
that needs to be included for basic usage. All functions there are declared within the :ref:`amici namespace <namespace_amici>`.
Additionally,
:ref:`amici/hdf5.h <file_include_amici_hdf5.h>` and :ref:`amici/serialization.h <file_include_amici_serialization.h>` may be handy for specific use cases.
The former provides some functions for reading and writing
`HDF5 <https://support.hdfgroup.org/>`_ files, latter for serialization
(requires `Boost <https://www.boost.org/>`_).
All model-specific functions are defined in the namespace ``model_$modelname``.

The main function for running an AMICI simulation is
:cpp:func:`amici::runAmiciSimulation`. This function requires

* an instance of a :cpp:class:`amici::Model` subclass as generated during model
  import. For the example `model_steadystate` the respective class is provided
  as ``Model_model_steadystate`` in ``model_steadystate.h`` in output directory
  for the given model.

* a :cpp:class:`amici::Solver` instance. This solver instance needs to match
  the requirements of the model and can be obtained from
  :cpp:func:`amici::AbstractModel::getSolver`.

* optionally an :cpp:class:`amici::ExpData` instance, which contains any
  experimental data (e.g. measurements, noise model parameters or model inputs)
  to evaluate residuals or an objective function.

This function returns a :cpp:class:`amici::ReturnData` object, which contains
all simulation results.

For running simulations for multiple experimental conditions
(multiple :cpp:class:`amici::ExpData` instances),
:cpp:func:`amici::runAmiciSimulations`
provides an alternative entry point. If AMICI (and your application)
have been compiled with OpenMP support (see installation guide), this allows
for running those simulations in parallel.

A scaffold for a standalone simulation program is automatically generated
during model import in ``main.cpp`` in the model output directory. This program
shows how to use the above-mentioned classes, how to obtain the simulation
results, and may provide a starting point for your own simulation code.

Working with multiple or anonymous models
+++++++++++++++++++++++++++++++++++++++++

AMICI model import generates a :cpp:class:`amici::Model` subclass for the
specific model, based on the name used during import. One the one hand, this
allows you to use multiple models with different names within a single
application. On the other hand, this requires you to know the name of the
model, which can be inconvenient in some cases.

When working with a single model, the ``wrapfunctions.h`` file generated during
model import can be used to avoid specifying model names explicitly. It defines
a function ``amici::generic_model::getModel()``, that returns an instance of
the model class by a generic name.

.. note::

   Including multiple ``wrapfunctions.h`` files from different
   models in a single application is not possible. When using multiple models,
   explicit names have to be used or the different model libraries need to be
   loaded dynamically at runtime.

Compiling and linking
=====================

To run AMICI simulations from within your C++ application, you need to compile
and link the following libraries:

* model library
* AMICI base library
* SUNDIALS libraries
* SuiteSparse libraries
* CBLAS-compatible BLAS
* optionally HDF5 (C, HL, and CXX components)
  set CMake option ``ENABLE_HDF5`` to ``OFF`` to build without HDF5-support
* optionally OpenMP (for parallel simulation of multiple conditions, see
  :cpp:func:`amici::runAmiciSimulations`)
* optionally boost (only when using serialization of AMICI object)

The simplest and recommended way is using the provide CMake files which take
care of all these dependencies.

Considering the simple case, that you want to simulate one specific model
in your CMake-based C++ application, you can copy or move the generated model
directory containing the ``CMakeLists.txt`` file to your application directory,
add `add_subdirectory(yourModelDirectory)` to your project's ``CMakeLists.txt``
file and build your project using CMake as usual.

Parameter estimation for AMICI models in high-performance computing environments
================================================================================

To perform parameter estimation for large or otherwise computationally
demanding AMICI models from C++ in a high-performance computing environment,
you may find the `parPE library <https://github.com/ICB-DCM/parPE/>`_ helpful.
parPE allows for the private or shared memory parallel evaluation of a cost
function requiring multiple simulations of the same model with different
inputs. It provides interfaces to different optimizers, such as Ipopt.
