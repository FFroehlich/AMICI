.. _amici_matlab_installation:

Installing the AMICI MATLAB toolbox
===================================

To use AMICI from MATLAB, start MATLAB and add the ``AMICI/matlab``
directory to the MATLAB path. To add all toolbox directories to the
MATLAB path, execute the matlab script:

.. code-block:: matlab

   installAMICI.m

To store the installation for further MATLAB session, the path can be
saved via:

.. code-block:: matlab

   savepath

For the compilation of ``.mex`` files, MATLAB needs to be configured with a
working C++ compiler. The C++ compiler needs to be installed and
configured via:

.. code-block:: matlab

   mex -setup c++

For a list of supported compilers we refer to the respective MathWorks
`documentation <http://mathworks.com/support/compilers/R2018b/index.html>`_.
