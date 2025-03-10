.. _amici_python_installation:

Installing the AMICI Python package
===================================

Short guide
+++++++++++

Installation of the AMICI Python package has the following prerequisites:

* Python>=3.10
* :term:`SWIG`>=3.0
* CBLAS compatible BLAS library
  (e.g., OpenBLAS, CBLAS, Atlas, Accelerate, Intel MKL)
* a C++17 compatible C++ compiler and a C compiler
  (e.g., g++>=9.1, clang>=12, Intel C++ compiler, mingw)

If these requirements are fulfilled and all relevant paths are setup properly,
AMICI can be installed using:

.. code-block:: bash

   pip3 install amici

If this worked, you can now import the Python module via::

   import amici

If this does not work for you, please follow the full instructions below.

.. note::

  To re-install a previously installed AMICI version with different
  build options or changed system libraries, pass the ``--no-cache-dir``
  option to ``pip`` to ensure a clean re-installation:

  .. code-block:: bash

     pip3 install --no-cache-dir amici


Installation on Linux
+++++++++++++++++++++

Ubuntu 22.04 / 24.04
--------------------

Install the AMICI dependencies via ``apt``
(this requires superuser privileges):

.. code-block:: bash

   sudo apt install libatlas-base-dev swig python3-dev

   # optionally for HDF5 support:
   sudo apt install libhdf5-serial-dev

   # optionally for boost support (thread-specific CPU times, extended math functions, serialization)
   sudo apt install libboost-chrono-dev libboost-math-dev libboost-serialization-dev

Install AMICI:

.. code-block:: bash

   pip3 install amici

Fedora 32
---------

Install the AMICI dependencies via ``apt``
(this requires superuser privileges):

.. code-block:: bash

   sudo dnf install blas-devel swig

Install AMICI:

.. code-block:: bash

   pip3 install amici

Arch Linux
----------

Install the AMICI dependencies via ``pacman``
(this requires superuser privileges):

.. code-block:: bash

   sudo pacman -S python swig openblas gcc hdf5 boost-libs

Export the bash variables ``BLAS_CFLAGS`` and ``BLAS_LIBS`` to point to where BLAS was installed, e.g.:

.. code-block:: bash

  export BLAS_CFLAGS="-I/usr/include/openblas/"
  export BLAS_LIBS="-lopenblas"

Install AMICI:

.. code-block:: bash

   pip3 install amici

Alternatively:

1. Check if packages are already installed with the required versions for AMICI installation.

.. code-block:: bash

   sudo pacman -Si python swig openblas gcc hdf5 boost-libs

2. Upgrade installed packages if required mininum versions are not satisfied for AMICI installation.

.. code-block:: bash

   sudo pacman -Su python swig openblas gcc hdf5 boost-libs

3. Export the bash variables ``BLAS_CFLAGS`` and ``BLAS_LIBS`` to point to where BLAS was installed, e.g.:

.. code-block:: bash

  export BLAS_CFLAGS="-I/usr/include/openblas/"
  export BLAS_LIBS="-lopenblas"

4. Install AMICI:

.. code-block:: bash

   pip3 install amici


Installation on OSX
+++++++++++++++++++

Install the AMICI dependencies using homebrew:

.. code-block:: bash

    brew install swig

    # optionally for HDF5 support:
    brew install hdf5

    # optionally for parallel simulations:
    brew install libomp
    # followed by either `brew link openmp` once,
    # or `export OpenMP_ROOT=$(brew --prefix)/opt/libomp"` where `OpenMP_ROOT` will have to be set during every re-installation of AMICI or any new model import

    # optionally for boost support (thread-specific CPU times, extended math functions, serialization)
    brew install boost && export BOOST_ROOT=$(brew --prefix)/opt/boost
    # followed by either `brew link boost` once,
    # or `export BOOST_ROOT=$(brew --prefix)/opt/boost"` where `BOOST_ROOT` will have to be set during every re-installation of AMICI or any new model import

Install AMICI:

.. code-block:: bash

    pip3 install amici


Installation on Windows
+++++++++++++++++++++++

Some general remarks:

* Consider using the `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__ and follow the instructions for
  installation on linux.
* Install all libraries in a path not containing white spaces,
  e.g. directly under C:.
* Replace the following paths according to your installation.
* Slashes can be preferable to backslashes for some environment
  variables.
* See also [#425](https://github.com/AMICI-dev/amici/issues/425) for
  further discussion.

Using the Microsoft Visual Studio
---------------------------------

We assume that Visual Studio (not to be confused with Visual Studio Code)
is already installed. Using Visual Studio Installer, the following components
need to be included:

* Microsoft Visual C++ (MSVC).
  This is part of multiple packages, including Desktop Development with C++.
* Windows Universal C Runtime.
  This is an individual component and installs some DLLs that we need.

OpenBLAS
^^^^^^^^

There are prebuilt OpenBLAS binaries available, but they did not seem to work
well here. Therefore, we recommend building OpenBLAS from scratch. This
requires an installation of CMake. CMake can be installed from
https://cmake.org/download/ (system-wide), or via ``pip install cmake``
(in the current Python environment).


To build OpenBLAS, download the following scripts from the AMICI repository:

* https://github.com/AMICI-dev/AMICI/blob/master/scripts/installOpenBLAS.ps1
* https://github.com/AMICI-dev/AMICI/blob/master/scripts/compileBLAS.cmd

The first script needs to be called in Powershell, and it needs to call
``compileBLAS.cmd``, so you will need to modify line 11:

    cmd /c "scripts\compileBLAS.cmd $version"

Additionally, in ``compileBLAS.cmd`` make sure that you point to your
Visual Studio installation on line 3.
Newer installations could be located under
``C:\Program Files\Microsoft Visual Studio\...\VC\Auxiliary\Build\vcvars64.bat``.

so that it matches your directory structure.
This will download OpenBLAS and compile it, creating
``C:\\BLAS\\OpenBLAS\\lib\\openblas.lib`` and
``C:\\BLAS\\OpenBLAS\\bin\\openblas.dll``.

You will also need to define two environment variables:

.. code-block:: text

   BLAS_LIBS="-LIBPATH:C:/BLAS/OpenBLAS/lib openblas.lib"
   BLAS_CFLAGS="-IC:/BLAS/OpenBLAS"

One way to do that is to run a PowerShell script with the following commands:

.. code-block:: text

   [System.Environment]::SetEnvironmentVariable("BLAS_LIBS", "-LIBPATH:C:/BLAS/OpenBLAS/lib openblas.lib", [System.EnvironmentVariableTarget]::User)
   [System.Environment]::SetEnvironmentVariable("BLAS_LIBS", "-LIBPATH:C:/BLAS/OpenBLAS/lib openblas.lib", [System.EnvironmentVariableTarget]::Process)
   [System.Environment]::SetEnvironmentVariable("BLAS_CFLAGS", "-IC:/BLAS/OpenBLAS/include/openblas", [System.EnvironmentVariableTarget]::User)
   [System.Environment]::SetEnvironmentVariable("BLAS_CFLAGS", "-IC:/BLAS/OpenBLAS/include/openblas", [System.EnvironmentVariableTarget]::Process)

The call ending in ``Process`` sets the environment variable in the current
process, and it is no longer in effect in the next process. The call ending in
``User`` is permanent, and takes effect the next time the user logs on.

Now you need to make sure that all required DLLs are within the scope of the
``PATH`` variable. In particular, the following directories need to be included
in ``PATH``:

* ``C:\BLAS\OpenBLAS\bin``
* ``C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64``

The first one is needed for ``openblas.dll`` and the second is needed for the
Windows Universal C Runtime.

If any DLLs are missing in the ``PATH`` variable, Python will return the
following error upon ``import amici``:

``ImportError: DLL load failed: The specified module could not be found.``

Almost all of the DLLs are standard Windows DLLs and should be included in
either Windows or Visual Studio. But, in case it is necessary to test this,
here is a list of some DLLs required by AMICI (when compiled with MSVC):

* ``openblas.dll``
* ``python37.dll``
* ``MSVCP140.dll``
* ``KERNEL32.dll``
* ``VCRUNTIME140_1.dll``
* ``VCRUNTIME140.dll``
* ``api-ms-win-crt-convert-l1-1-0.dll``
* ``api-ms-win-crt-heap-l1-1-0.dll``
* ``api-ms-win-crt-stdio-l1-1-0.dll``
* ``api-ms-win-crt-string-l1-1-0.dll``
* ``api-ms-win-crt-runtime-l1-1-0.dll``
* ``api-ms-win-crt-time-l1-1-0.dll``
* ``api-ms-win-crt-math-l1-1-0.dll``

``MSVCP140.dll``, ``VCRUNTIME140.dll``, and ``VCRUNTIME140_1.dll`` are needed
by MSVC (see Visual Studio above). ``KERNEL32.dll`` is part of Windows and in
``C:\Windows\System32``. The ``api-ms-win-crt-XXX-l1-1-0.dll`` are needed by
``openblas.dll`` and are part of the Windows Universal C Runtime.

.. note::

    Since Python 3.8, the library directory needs to be set either from Python:

    .. code-block:: python

        import os
        # directory containing `openblas.dll`
        os.add_dll_directory("C:\\BLAS\\OpenBLAS\\bin")
        import amici

    or via the environment variable ``AMICI_DLL_DIRS="C:\BLAS\OpenBLAS\bin"``.


Further topics
++++++++++++++

Installation of development versions
------------------------------------

To install development versions which have not been released to PyPI yet,
you can install AMICI with ``pip`` directly from GitHub using:

.. code-block:: bash

    pip3 install -e git+https://github.com/AMICI-dev/amici.git@develop#egg=amici\&subdirectory=python/sdist

Replace ``develop`` by the branch or commit you want to install.

Note that this will only work on Windows if you have enabled developer mode,
because symlinks are not supported by default
(`more information <https://stackoverflow.com/questions/5917249/git-symlinks-in-windows/49913019#49913019>`_).

Light installation
------------------

In case you only want to use the AMICI Python package for generating model code
for use with Matlab or C++ and don't want to bothered with any unnecessary
dependencies, you can run

.. code-block:: bash

   pip3 install --install-option --no-clibs amici

.. note::

   Following this installation, you will not be able to simulate the imported
   models in Python.

.. note::

   If you run into an error with above installation command, install all AMICI
   dependencies listed in `setup.py <https://github.com/AMICI-dev/AMICI/blob/master/python/sdist/setup.py>`_
   manually, and try again. (This is because ``pip`` ``--install-option`` is
   applied to *all* installed packages, including dependencies.)


.. _amici_python_install_env_vars:

Custom installation
-------------------

Installation of the AMICI Python package can be customized using a number of
environment variables:

+----------------------------+----------------------------------+---------------------------------+
| Variable                   | Purpose                          | Example                         |
+============================+==================================+=================================+
| ``SWIG``                   | Path to the :term:`SWIG`         | ``SWIG=$HOME/bin/swig4.0``      |
|                            | executable                       |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``CC``                     | Setting the C(++) compiler       | ``CC=/usr/bin/g++``             |
+----------------------------+----------------------------------+---------------------------------+
| ``CFLAGS``                 | Extra compiler flags used in     |                                 |
|                            | every compiler invocation        |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``BLAS_CFLAGS``            | Compiler flags for, e.g. BLAS    |                                 |
|                            |  include directories             |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``BLAS_LIBS``              | Flags for linking BLAS           |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``ENABLE_GCOV_COVERAGE``   | Set to build AMICI to generate   | ``ENABLE_GCOV_COVERAGE=TRUE``   |
|                            | code coverage information        |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``ENABLE_AMICI_DEBUGGING`` | Set to build AMICI with          | ``ENABLE_AMICI_DEBUGGING=TRUE`` |
|                            | debugging symbols                |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``AMICI_PARALLEL_COMPILE`` | Set to the number of parallel    | ``AMICI_PARALLEL_COMPILE=4``    |
|                            | processes to be used for C(++)   |                                 |
|                            | compilation (defaults to 1)      |                                 |
+----------------------------+----------------------------------+---------------------------------+
| ``AMICI_TRY_ENABLE_HDF5``  | Whether to build AMICI with      | ``AMICI_TRY_ENABLE_HDF5=OFF``   |
|                            | HDF5-support if possible.        |                                 |
|                            | Default: ``ON``                  |                                 |
+----------------------------+----------------------------------+---------------------------------+

Installation under conda
------------------------

There is no amici conda recipe available yet. However, you can install AMICI
using pip in a conda environment.

.. note::

   It is possible, but we currently don't recommend using conda for installing
   AMICI, as it commonly leads to conflicts with system installations of
   libraries and compilers.

Create a minimal conda environment via:

.. code-block:: bash

   conda create --name ENV_NAME pip python

Here, replace ``ENV_NAME`` by some name for the environment.

To activate the environment, run:

.. code-block:: bash

   source activate ENV_NAME

(and ``conda deactivate`` later to deactivate it again).

:term:`SWIG` must be installed and available in your ``PATH``, and a
CBLAS-compatible BLAS must be available. You can also use conda to
install the latter locally, using:

.. code-block:: bash

   conda install -c conda-forge openblas

To make AMICI use openblas, set the following environment variable:

.. code-block:: bash

   export BLAS_LIBS=-lopenblas

``BLAS_LIBS`` needs to be set during installation of the AMICI package, as
well as during any future model import.

To install AMICI, now run:

.. code-block:: bash

   pip install amici

The ``pip`` option ``--no-cache`` may be helpful here to make sure the
installation is done completely anew.

Now, you are ready to use AMICI in the virtual environment.

.. note::

   **conda on Mac**

   If the above installation does not work for you, try installing AMICI via:

   .. code-block:: bash

      CFLAGS="-stdlib=libc++" CC=clang CXX=clang pip3 install --verbose amici

   This will use the ``clang`` compiler.

   You will have to pass the same options when compiling any model later
   on. This can be done by inserting the following code before model import:

   .. code-block:: python

      import os
      os.environ['CC'] = 'clang'
      os.environ['CXX'] = 'clang'
      os.environ['CFLAGS'] = '-stdlib=libc++'

   (For further discussion see https://github.com/AMICI-dev/AMICI/issues/357)

Known issues:

* ``CMAKE_AR-NOTFOUND: not found``: Try ``conda install binutils``.

Optional Boost support
----------------------

`Boost <https://www.boost.org/>`_ is an optional C++ dependency only required
for special functions (including e.g. gamma derivatives) in the Python
interface. Boost can be installed via package managers via

.. code-block:: bash

    apt-get install libboost-math-dev

or

.. code-block:: bash

    brew install boost

As only headers are required, also a
`source code <https://www.boost.org/doc/libs/1_66_0/more/getting_started/unix-variants.html>`_
download suffices. The compiler must be able to find the module in the search
path.
