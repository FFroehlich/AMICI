.. _python_interface:

******************************
Using AMICI's Python interface
******************************

In the following we will give a detailed overview how to specify models in
Python and how to call the generated simulation files.

Model definition
================

This document provides an overview of different interfaces to import models
in AMICI. Further examples are available in the AMICI repository in the
`python/examples <https://github.com/AMICI-dev/AMICI/tree/master/python/examples>`_
directory.

SBML import
-----------

AMICI can import :term:`SBML` models via the
:py:func:`amici.sbml_import.SbmlImporter` class.

.. _amici_python_sbml_support:

Status of SBML support in Python-AMICI
++++++++++++++++++++++++++++++++++++++

Python-AMICI currently **passes 1252 out of the 1821 (~68%) test cases** from
the semantic
`SBML Test Suite <https://github.com/sbmlteam/sbml-test-suite/>`_
(`current status <https://github.com/AMICI-dev/AMICI/actions>`_).

The following SBML test suite tags are currently supported
(i.e., at least one test case with the respective test passes;
`tag descriptions <https://github.com/sbmlteam/sbml-test-suite/blob/master/docs/tags-documentation/all-tags.txt>`_):

**Component tags:**

* AlgebraicRule
* AssignmentRule
* comp
* Compartment
* CSymbolAvogadro
* CSymbolRateOf
* CSymbolTime
* Deletion
* EventNoDelay
* ExternalModelDefinition
* FunctionDefinition
* InitialAssignment
* ModelDefinition
* Parameter
* Port
* RateRule
* Reaction
* ReplacedBy
* ReplacedElement
* SBaseRef
* Species
* Submodel

**Test tags:**

* 0D-Compartment
* Amount
* AssignedConstantStoichiometry
* AssignedVariableStoichiometry
* BoolNumericSwap
* BoundaryCondition
* comp
* Concentration
* ConstantSpecies
* ConversionFactor
* ConversionFactors
* DefaultValue
* EventT0Firing
* ExtentConversionFactor
* HasOnlySubstanceUnits
* InitialValueReassigned
* L3v2MathML
* LocalParameters
* MultiCompartment
* NoMathML
* NonConstantCompartment
* NonConstantParameter
* NonUnityCompartment
* NonUnityStoichiometry
* ReversibleReaction
* SpeciesReferenceInMath
* SubmodelOutput
* TimeConversionFactor
* UncommonMathML
* VolumeConcentrationRates

Additional support may be added in the future. However, the following features are
unlikely to be supported:

- `factorial()`, `ceil()`, `floor()`, due to incompatibility with
  symbolic sensitivity computations
- `delay()` due to missing :term:`SUNDIALS` solver support
- events with delays, events with non-persistent triggers

Tutorials
+++++++++

A basic tutorial on how to import and simulate SBML models is available in the
`Getting Started notebook <GettingStarted.ipynb>`_, while a more detailed example
including customized import and sensitivity computation is available in the
`Example Steadystate notebook <ExampleSteadystate.ipynb>`_.

PySB import
-----------

AMICI can import :term:`PySB` models via
:py:func:`amici.pysb_import.pysb2amici`.

BNGL import
-----------

AMICI can import :term:`BNGL` models via
:py:func:`amici.bngl_import.bngl2amici`.

PEtab import
------------

AMICI can import :term:`PEtab`-based model definitions and run simulations for
the specified simulations conditions. For usage, see
`python/examples/example_petab/petab.ipynb <petab.ipynb>`_.

Importing plain ODEs
--------------------

The AMICI Python interface does not currently support direct import of ODEs.
However, it is straightforward to encode them as RateRules in an SBML model.
The most convenient options to do that are maybe
`Antimony <https://tellurium.readthedocs.io/en/latest/antimony.html>`_
and `yaml2sbml <https://yaml2sbml.readthedocs.io/en/latest/index.html>`_.

An example using Antimony to specify the Lotka-Volterra equations is shown below:

.. code-block:: python

    ant_model = """

    model lotka_volterra
        # see https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

        # initial conditions
        prey_density = 10;
        predator_density = 10;

        # parameters
        prey_growth_rate = 1.1;
        predator_effect_on_prey = 0.4;
        predator_death_rate = 0.4;
        prey_effect_on_predator = 0.1;

        # dx/dt
        prey_density' = prey_growth_rate * prey_density - predator_effect_on_prey * prey_density * predator_density;
        predator_density' = prey_effect_on_predator * prey_density * predator_density - predator_death_rate * predator_density;
    end
    """
    module_name = "test_antimony_example_lv"
    from amici.antimony_import import antimony2amici
    antimony2amici(
        ant_model,
        model_name=module_name,
        output_dir=module_name,
    )
    model_module = amici.import_model_module(
        module_name=module_name, module_path=outdir
    )
    amici_model = model_module.getModel()
    amici_model.setTimepoints(np.linspace(0, 100, 200))
    amici_solver = amici_model.getSolver()
    rdata = amici.runAmiciSimulation(amici_model, amici_solver)

    from amici.plotting import plot_state_trajectories
    plot_state_trajectories(rdata, model=amici_model)


The `yaml2sbml <https://yaml2sbml.readthedocs.io/en/latest/index.html>`_ package creates SBML models
from a YAML-based specification of an ODE model. Various examples are
`provided <https://yaml2sbml.readthedocs.io/en/latest/examples/examples.html>`_.
Besides the SBML model, yaml2sbml can also create
`PEtab <https://github.com/PEtab-dev/PEtab>`_ files.

SED-ML import
-------------

We also plan to implement support for the
`Simulation Experiment Description Markup Language (SED-ML) <https://sed-ml.org/>`_.

Environment variables affecting model import
============================================

In addition to the environment variables listed
:ref:`here <amici_python_install_env_vars>`, the following environment
variables control various behaviours during model import and compilation:

.. list-table:: Environment variables affecting model import
   :widths: 25 50 25
   :header-rows: 1

   * - Variable
     - Purpose
     - Example
   * - ``AMICI_EXTRACT_CSE``
     - Extract common subexpressions. May significantly reduce file size and
       compile time for large models, but makes the generated code less
       readable. Disabled by default.
     - ``AMICI_EXTRACT_CSE=1``
   * - ``AMICI_IMPORT_NPROCS``
     - Number of processes to be used for model import. Defaults to 1.
       Speeds up import of large models. Will slow down import of small models,
       benchmarking recommended.
     - ``AMICI_IMPORT_NPROCS=4``
   * - ``AMICI_EXPERIMENTAL_SBML_NONCONST_CLS``
     - Compute conservation laws for non-constant species. SBML-import only.
       See :py:func:`amici.sbml_import.SbmlImporter.sbml2amici`.
     -


Miscellaneous
=============

.. _amici_python_openmp:

OpenMP support for parallelized simulation for multiple experimental conditions
-------------------------------------------------------------------------------

AMICI can be built with OpenMP support, which allows to parallelize model
simulations for multiple experimental conditions.

On Linux and OSX this is enabled by default. This can be verified using:

.. code-block:: python

   import amici
   amici.compiledWithOpenMP()

If not already enabled by default, you can enable OpenMP support by setting
the environment variables ``AMICI_CXXFLAGS`` and ``AMICI_LDFLAGS`` to the
correct OpenMP flags of your compiler and linker, respectively. This has to be
done for both AMICI package installation *and* model compilation. When using
``gcc`` on Linux, this would be:

.. code-block:: bash

   # on your shell:
   AMICI_CXXFLAGS=-fopenmp AMICI_LDFLAGS=-fopenmp pip3 install amici

.. code-block:: python

   # in python, before model compilation:
   import os
   os.environ['AMICI_CXXFLAGS'] = '-fopenmp'
   os.environ['AMICI_LDFLAGS'] = '-fopenmp'
