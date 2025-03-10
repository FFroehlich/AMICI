.. _versioning_policy:

Versioning policy
=================

Versioning
----------

We use `Semantic Versioning <http://semver.org/>`_ with the modifications
described under :ref:`deprecation_policy`.

.. _deprecation_policy:

Deprecation policy
------------------

AMICI aims to provide a stable API for users. However, not all features can be
maintained indefinitely. We will deprecate features in minor releases and
where possible, issue a warning when they are used. We will keep deprecated
features for at least six months after the release that includes the
respective deprecation warning and then remove them earliest in the next minor
or major release. If a deprecated feature is the source of a major bug, we may
remove it earlier.

Python compatibility
--------------------

We follow `numpy's Python support policy <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.
