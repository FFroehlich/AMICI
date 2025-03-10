Debugging AMICI
===============

This document contains some information on how to debug any issues in AMICI,
in particular for C++ Python extensions.

Caveman debugging / printf-debugging
------------------------------------

The simplest approach may often be adding some print-statements to the code,
as this does not require any special tools.

Note that after each change of the C++ files, the AMICI extension *as well as
the model extension*  (if any model functions are called), need to be
recompiled.
The simplest and safest approach would be re-installation of the amici package
and re-import of the model. As this can be very time-consuming, the following
shortcut is possible, assuming you are using a development installation
(``pip install -e ...``):

.. code-block:: shell

    # rebuild the amici base extension, from within the amici root directory
    # (note that this only recompiles the amici source files, NOT third-party
    # dependencies such as sundials):
    cd python/sdist/
    python setup.py build_ext --build-lib .

    # rebuild the model, from within the model package directory:
    python setup.py build_ext --force --build-lib .

Note: Be careful when working interactively, Python may not pick up any changes
in already imported modules. The safest is to start a new Python process after
any changes.


Using a proper debugger
-----------------------

Debugging with with [gdb](https://www.sourceware.org/gdb/) is most convenient
with a minimal reproducible example that is directly invoked from gdb.
For example:

.. code-block:: shell

    # start gdb
    gdb --args python -m pytest ../tests/test_sbml_import.py::test_nosensi

    # inside gdb, set a meaningful breakpoint and launch:
    break amici::runAmiciSimulation
    run
    # ... (see one of the many gdb tutorials)

Alternative, gdb can attach to a running process by passing the ``--pid``
argument.

For many users, it may be more convenient to use gdb via some graphical user
interface as provided by various C++ IDEs.

.. note::

    For better debugging experience, but at the cost of runtime performance,
    consider building the amici and model extension with environment variable
    ``ENABLE_AMICI_DEBUGGING=TRUE``. This will include debugging symbols and
    disable compiler optimizations.
