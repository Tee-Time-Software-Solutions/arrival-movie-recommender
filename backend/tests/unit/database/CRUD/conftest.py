"""Shared helpers for CRUD unit tests that mock out AsyncSession."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockResult:
    """Minimal stand-in for a SQLAlchemy ``Result``.

    Supports the access patterns used by the CRUD layer: ``first()``,
    ``scalar_one()``, ``scalar()``, ``fetchall()`` and iteration.
    """

    def __init__(self, rows=None, scalar_value=None, rowcount=0):
        self._rows = list(rows) if rows else []
        self._scalar_value = scalar_value
        self.rowcount = rowcount

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one(self):
        if self._scalar_value is not None:
            return self._scalar_value
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar_value

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


def row(**kwargs) -> SimpleNamespace:
    """Build a fake SQLAlchemy ``Row`` using ``SimpleNamespace``."""
    return SimpleNamespace(**kwargs)


@pytest.fixture
def db():
    """Mock ``AsyncSession`` with async execute/commit/rollback."""
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session
