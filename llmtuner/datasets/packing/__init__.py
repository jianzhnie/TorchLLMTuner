"""Packing nodes: turn samples into fixed-length rows.

Three modules, split by what each part depends on:

* ``build`` -- the entry points (``build_concat_then_split_packing``,
  ``build_first_fit_packing``) and the row-length policy they share.
* ``iterators`` -- the Grain iterators that splice documents across row
  boundaries, plus the resumable state that makes a restart land mid-corpus.
* ``conversions`` -- the pure per-row conversions between a ``TextSequence``
  and the dict the packers emit.

Public builders are loaded on request; importing conversions does not load
recipe composition or source adapters.
"""

from ...utils.lazy_exports import resolve_export

_EXPORT_SOURCES = {
    "build_concat_then_split_packing": "build",
    "build_first_fit_packing": "build",
}

__all__ = sorted(_EXPORT_SOURCES)


def __getattr__(name: str):
    return resolve_export(__name__, _EXPORT_SOURCES, name)


def __dir__() -> list[str]:
    return sorted(_EXPORT_SOURCES)
