"""Shared helper modules. Importing this package itself has no torch dependency.

Members:

* ``gc.py`` -- ``GarbageCollection``: the collect/disable pair the trainer wraps
  around the phases where allocator churn would otherwise show up as time.
* ``logger_utils.py`` -- ``get_logger`` (a colour formatter that filters by rank
  at emit time) and ``get_distributed_rank``.
* ``lazy_exports.py`` -- the one implementation of the PEP 562 lazy package
  index that the package indexes here share.
* ``monitoring.py`` -- hardware probes and allocator memory snapshots; import
  this torch-backed submodule only where device monitoring is needed.

The package is a leaf on purpose. Its light-weight helpers can be imported
without a cycle -- including from ``llmtuner/__init__.py`` -- while
``monitoring`` is imported only explicitly by its users.
"""
