"""
Contain variables that are shared across the SAA operations

Most likely to be accessed and updated multiple times
Careful with data corruption
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional


_zero_cells: ContextVar[Optional[dict]] = ContextVar("saa_zero_cells", default=None)


@contextmanager
def zero_cell_scope():
    """Isolate zero-cell state to one top-level SAA run."""
    token = _zero_cells.set({})
    try:
        yield
    finally:
        _zero_cells.reset(token)


def update_zero_cells(att, value):
    zero_cells = dict(_zero_cells.get() or {})
    if att not in zero_cells:
        zero_cells[att] = {value}
    else:
        zero_cells[att] = set(zero_cells[att])
        zero_cells[att].add(value)
    _zero_cells.set(zero_cells)


def get_zero_cells_all():
    return _zero_cells.get() or {}


def get_zero_cells_att(att):
    return get_zero_cells_all()[att]

