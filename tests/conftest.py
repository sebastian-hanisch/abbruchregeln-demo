"""Gemeinsame Fixtures: kleine Netze und feste Folgen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def pfad():
    """Zwei Kanten hintereinander (drei Knoten): bei p = ½ ist T = 1 mit ¼, T = 2 mit ½, T = 3 mit ¼."""
    return 3, ((0, 1), (1, 2))


@pytest.fixture(scope="session")
def dreieck():
    """Dreieck: bei p = ½ ist T = 1 mit ½ (drei oder zwei Kanten), T = 2 mit ⅜, T = 3 mit ⅛."""
    return 3, ((0, 1), (1, 2), (0, 2))
