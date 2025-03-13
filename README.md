<img src="https://raw.githubusercontent.com/AMICI-dev/AMICI/master/doc/gfx/banner.png" height="60" align="left" alt="AMICI logo">

## Advanced Multilanguage Interface for CVODES and IDAS

## About

AMICI provides a multi-language (Python, C++, Matlab) interface for the
[SUNDIALS](https://computing.llnl.gov/projects/sundials/) solvers
[CVODES](https://computing.llnl.gov/projects/sundials/cvodes)
(for ordinary differential equations) and
[IDAS](https://computing.llnl.gov/projects/sundials/idas)
(for algebraic differential equations). AMICI allows the user to read
differential equation models specified as [SBML](http://sbml.org/)
or [PySB](http://pysb.org/)
and automatically compiles such models into Python modules, C++ libraries or
Matlab `.mex` simulation files.

In contrast to the (no longer maintained)
[sundialsTB](https://computing.llnl.gov/projects/sundials/sundials-software)
Matlab interface, all necessary functions are transformed into native
C++ code, which allows for a significantly faster simulation.

Beyond forward integration, the compiled simulation file also allows for
forward sensitivity analysis, steady state sensitivity analysis and
adjoint sensitivity analysis for likelihood-based output functions.

The interface was designed to provide routines for efficient gradient
computation in parameter estimation of biochemical reaction models, but
it is also applicable to a wider range of differential equation
constrained optimization problems.

## Current build status

<a href="https://badge.fury.io/py/amici">
  <img src="https://badge.fury.io/py/amici.svg" alt="PyPI version"></a>
<a href="https://github.com/AMICI-dev/AMICI/actions/workflows/test_pypi.yml">
  <img src="https://github.com/AMICI-dev/AMICI/actions/workflows/test_pypi.yml/badge.svg" alt="PyPI installation"></a>
<a href="https://codecov.io/gh/AMICI-dev/AMICI">
  <img src="https://codecov.io/gh/AMICI-dev/AMICI/branch/master/graph/badge.svg" alt="Code coverage"></a>
<a href="https://sonarcloud.io/dashboard?id=ICB-DCM_AMICI&branch=master">
  <img src="https://sonarcloud.io/api/project_badges/measure?branch=master&project=ICB-DCM_AMICI&metric=sqale_index" alt="SonarCloud technical debt"></a>
<a href="https://zenodo.org/badge/latestdoi/43677177">
  <img src="https://zenodo.org/badge/43677177.svg" alt="Zenodo DOI"></a>
<a href="https://amici.readthedocs.io/en/latest/?badge=latest">
 <img src="https://readthedocs.org/projects/amici/badge/?version=latest" alt="ReadTheDocs status"></a>
<a href="https://bestpractices.coreinfrastructure.org/projects/3780">
  <img src="https://bestpractices.coreinfrastructure.org/projects/3780/badge" alt="coreinfrastructure bestpractices badge"></a>

## Features

* SBML import
* PySB import
* Generation of C++ code for model simulation and sensitivity
  computation
* Access to and high customizability of CVODES and IDAS solver
* Python, C++, Matlab interface
* Sensitivity analysis
  * forward
  * steady state
  * adjoint
  * first- and second-order
* Pre-equilibration and pre-simulation conditions
* Support for
  [discrete events and logical operations](https://academic.oup.com/bioinformatics/article/33/7/1049/2769435)

## Interfaces & workflow

The AMICI workflow starts with importing a model from either
[SBML](http://sbml.org/) (Matlab, Python), [PySB](http://pysb.org/) (Python),
or a Matlab definition of the model (Matlab-only). From this input,
all equations for model simulation
are derived symbolically and C++ code is generated. This code is then
compiled into a C++ library, a Python module, or a Matlab `.mex` file and
is then used for model simulation.

![AMICI workflow](https://raw.githubusercontent.com/AMICI-dev/AMICI/master/doc/gfx/amici_workflow.png)

## Getting started

The AMICI source code is available at https://github.com/AMICI-dev/AMICI/.
To install AMICI, first read the installation instructions for
[Python](https://amici.readthedocs.io/en/latest/python_installation.html),
[C++](https://amici.readthedocs.io/en/develop/cpp_installation.html) or
[Matlab](https://amici.readthedocs.io/en/develop/matlab_installation.html).
There are also instructions for using AMICI inside
[containers](https://github.com/AMICI-dev/AMICI/tree/master/container).

To get you started with Python-AMICI, the best way might be checking out this
[Jupyter notebook](https://github.com/AMICI-dev/AMICI/blob/master/doc/examples/getting_started/GettingStarted.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AMICI-dev/AMICI/develop?labpath=doc%2Fexamples%2Fgetting_started%2FGettingStarted.ipynb).

To get started with Matlab-AMICI, various examples are available
in [matlab/examples/](https://github.com/AMICI-dev/AMICI/tree/master/matlab/examples).

Comprehensive documentation is available at
[https://amici.readthedocs.io/en/latest/](https://amici.readthedocs.io/en/latest/).

Any [contributions](https://amici.readthedocs.io/en/develop/CONTRIBUTING.html)
to AMICI are welcome (code, bug reports, suggestions for improvements, ...).


## Getting help

In case of questions or problems with using AMICI, feel free to post an
[issue](https://github.com/AMICI-dev/AMICI/issues) on GitHub. We are trying to
get back to you quickly.

## Projects using AMICI

There are several tools for parameter estimation offering good integration
with AMICI:

* [pyPESTO](https://github.com/ICB-DCM/pyPESTO): Python library for
  optimization, sampling and uncertainty analysis
* [pyABC](https://github.com/ICB-DCM/pyABC): Python library for
  parallel and scalable ABC-SMC (Approximate Bayesian Computation - Sequential
  Monte Carlo)
* [parPE](https://github.com/ICB-DCM/parPE): C++ library for parameter
  estimation of ODE models offering distributed memory parallelism with focus
  on problems with many simulation conditions.

## Publications

**Citeable DOI for the latest AMICI release:**
[![DOI](https://zenodo.org/badge/43677177.svg)](https://zenodo.org/badge/latestdoi/43677177)

There is a list of [publications using AMICI](https://amici.readthedocs.io/en/latest/references.html).
If you used AMICI in your work, we are happy to include
your project, please let us know via a GitHub issue.

When using AMICI in your project, please cite
* Fröhlich, F., Weindl, D., Schälte, Y., Pathirana, D., Paszkowski, Ł., Lines, G.T., Stapor, P. and Hasenauer, J., 2021.
  AMICI: High-Performance Sensitivity Analysis for Large Ordinary Differential Equation Models. Bioinformatics, btab227,
  [DOI:10.1093/bioinformatics/btab227](https://doi.org/10.1093/bioinformatics/btab227).
```
@article{frohlich2020amici,
  title={AMICI: High-Performance Sensitivity Analysis for Large Ordinary Differential Equation Models},
  author={Fr{\"o}hlich, Fabian and Weindl, Daniel and Sch{\"a}lte, Yannik and Pathirana, Dilan and Paszkowski, {\L}ukasz and Lines, Glenn Terje and Stapor, Paul and Hasenauer, Jan},
  journal = {Bioinformatics},
  year = {2021},
  month = {04},
  issn = {1367-4803},
  doi = {10.1093/bioinformatics/btab227},
  note = {btab227},
  eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab227/36866220/btab227.pdf},
}
```

When presenting work that employs AMICI, feel free to use one of the icons in
[doc/gfx/](https://github.com/AMICI-dev/AMICI/tree/master/doc/gfx),
which are available under a
[CC0](https://github.com/AMICI-dev/AMICI/tree/master/doc/gfx/LICENSE.md)
license:

<p align="center">
  <img src="https://raw.githubusercontent.com/AMICI-dev/AMICI/master/doc/gfx/logo_text.png" height="75" alt="AMICI Logo">
</p>
