AMICI developer’s guide
=======================

This document contains information for AMICI developers, not too
relevant to regular users.

Branches / releases
-------------------

AMICI roughly follows the
`GitFlow <https://nvie.com/posts/a-successful-git-branching-model/>`__.
All new contributions are merged into ``develop``. These changes are
regularly merged into ``master`` as new releases. For release versioning
we are trying to follow `semantic versioning <https://semver.org/>`__.
New releases are created on GitHub and are automatically deployed to
`Zenodo <https://doi.org/10.5281/zenodo.597928>`__ for
archiving and to obtain a digital object identifier (DOI) to make them
citable. Furthermore, our `CI pipeline <documentation/CI.md>`__ will
automatically create and deploy a new release on
`PyPI <https://pypi.org/project/amici/>`__.

We try to keep a clean git history. Therefore, feature pull requests are
squash-merged to ``develop``. Merging of release branches to master is
done via merge commits.

When starting to work on some issue
-----------------------------------

When starting to work on some Github issue, please assign yourself to
let other developers know that you are working on it to avoid duplicate
work. If the respective issue is not completely clear, it is generally a
good idea to ask for clarification before starting to work on it.

If you want to work on something new, please create a Github issue
first.

Code contributions
------------------

When making code contributions, please follow our style guide and the
process described below:

-  Check if you agree to release your contribution under the conditions
   provided in ``LICENSE``. By opening a pull requests you confirm us
   that you do agree.

-  Start a new branch from ``develop`` (on your fork, or at the main
   repository if you have access)

-  Implement your changes

-  Submit a pull request to the ``develop`` branch

-  Ensure all tests pass

-  When adding new functionality, please also provide test cases (see
   ``tests/cpp/``, ``python/tests/``,
   and `documentation/CI.md <documentation/CI.md>`__)

-  Write meaningful commit messages

-  Run all tests to ensure nothing was broken (`more
   details <documentation/CI.md>`__)

   -  Run ``scripts/buildAll.sh && scripts/run-cpp-tests.sh``.

   -  If you made changes to the Matlab or C++ code and have a Matlab
      license, please also run ``tests/cpp/wrapTestModels.m`` and
      ``tests/testModels.m``

   -  If you made changes to the Python or C++ code, run
      ``make python-tests`` in ``build``

-  When all tests are passing and you think your code is ready to merge,
   request a code review (see also our `code review
   guideline <documentation/code_review_guide.md>`__)

-  Wait for feedback. If you do not receive feedback to your pull
   request within a week, please give us a friendly reminder.

Style/compatibility guide
~~~~~~~~~~~~~~~~~~~~~~~~~

General
^^^^^^^

-  All files and functions should come with file-level and
   function-level documentation.

-  All new functionality should be covered by unit or integration tests.
   Runtime of those tests should be kept as short as possible.

Python
^^^^^^

-  In terms of Python compatibility, we follow numpy's
   `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__.

-  For the Python code we want to follow
   `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__. Although this
   is not the case for all existing code, any new contributions should
   do so. We use `black <https://black.readthedocs.io/en/stable/index.html>`__
   for code formatting.

   To run black as pre-commit hook, install the
   `pre-commit <https://pre-commit.com/>`_ package
   (e.g. ``pip install pre-commit``), and enable AMICI-hooks by running
   ``pre-commit install`` from within the AMICI directory.

-  We use Python `type
   hints <https://docs.python.org/3/library/typing.html>`__ for all
   functions (but not for class attributes, since they are not supported
   by the current Python doxygen filter). In Python code, type hints
   should be used instead of doxygen ``@type``.

   For function docstrings, follow this format:

   ::

      """One-line description.

      Possible a more detailed description

      Arguments:
          Argument1: This needs to start on the same line, otherwise the current
              doxygen filter will fail.

          Returns:
              Return value

          Raises:
              SomeError in case of some error.
      """

C++
^^^

-  We use C++17

-  We want to maintain compatibility with g++, clang, and the Intel C++
   compiler

-  For code formatting, we use ``clang-format`` and ``cmake-format``. They can
   be invoked by ``make clang-format cmake-format`` from the CMake build
   directory.

Matlab
^^^^^^

*To be defined*

Further topics
--------------

.. toctree::
   :maxdepth: 2

   Organization of the documentation <README>
   code_review_guide
   CI
   debugging
