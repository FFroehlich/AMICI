FAQ
===

**Q**: My model fails to build.

**A**: Remove the corresponding model directory located in
AMICI/models/\ *yourmodelname* and compile again.

--------------

**Q**: It still does not compile.

**A**: Remove the directory AMICI/models/\ ``mexext`` and compile again.

--------------

**Q**: It still does not compile.

**A**: Make an `issue <https://github.com/ICB-DCM/AMICI/issues>`__ and
we will have a look.

--------------

**Q**: My Python-generated model does not compile from MATLAB.

**A**: Try building any of the available examples before. If this
succeeds, retry building the original model. Some dependencies might not
be built correctly when using only the ``compileMexFile.m`` script.

--------------

**Q**: I get an out of memory error while compiling my model on a
Windows machine.

**A**: This may be due to an old compiler version. See `issue
#161 <https://github.com/AMICI-dev/AMICI/issues/161>`__ for instructions
on how to install a new compiler.

--------------

**Q**: How are events interpreted in a DAE context?

**A**: Currently we only support impulse free events. Also sensitivities
have never been tested. Proceed with care and create an
`issue <https://github.com/AMICI-dev/AMICI/issues>`__ if any problems
arise!

--------------

**Q**: The simulation/sensitivities I get are incorrect.

**A**: There are some known issues, especially with adjoint
sensitivities, events and DAEs. If your particular problem is not
featured in the `issues <https://github.com/AMICI-dev/AMICI/issues>`__
list, please add it!
