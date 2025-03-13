.. _amici_cpp_installation:

Building the C++ library
========================

The following section describes building the AMICI C++ library:

.. note::

   The AMICI C++ interface only supports simulation of models imported using
   the :ref:`Python interface <python_interface>` and
   :ref:`Matlab interface <matlab_interface>`. It cannot be used for model
   import itself.

Prerequisites:

* CBLAS compatible BLAS library
* a C++20 compatible compiler
* a C compiler
* Optional:

  * HDF5 libraries
  * boost for serialization

To use AMICI from C++, run the

.. code-block:: bash

    ./scripts/buildSuiteSparse.sh
    ./scripts/buildSundials.sh
    ./scripts/buildAmici.sh

script to build the AMICI library.

.. note::

   On some systems, the CMake executable may be named something
   other than ``cmake``. In this case, set the ``CMAKE`` environment variable
   to the correct name (e.g. ``export CMAKE=cmake3``, in case you have CMake
   available as ``cmake3``).

The static library can then be linked from

.. code-block:: bash

    ./build/libamici.a

In CMake-based packages, amici can be linked via

.. code-block:: cmake

    find_package(Amici)

For further usage, consult the AMICI
:ref:`C++ interface documentation <cpp_interface>`.


Supported CBLAS libraries
-------------------------

The C++ interfaces require a system installation of a CBLAS-compatible
*Basic Linear Algebra Subprograms* (BLAS) library.
AMICI has been tested with various implementations such as Accelerate,
Intel MKL, cblas, openblas and atlas.

Optional SuperLU_MT support
---------------------------

To build AMICI with SuperLU_MT support, run

.. code-block:: bash

   ./scripts/buildSuperLUMT.sh
   ./scripts/buildSundials.sh
   cd build/
   cmake -DSUNDIALS_SUPERLUMT_ENABLE=ON ..
   make
