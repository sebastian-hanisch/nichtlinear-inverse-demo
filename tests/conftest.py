"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini():
    """3 x 3-Matrix aus Stück 1 bis 8 (symmetrisch, Zeilensumme 1, Eigenwerte 1, 0,5, 0,1 zu den Vektoren (1,1,1), (1,0,-1), (1,-2,1)), Wahrheit x = (1, 2, 1), Daten b = (1,3, 1,4, 1,3)
    (rauschfrei für das lineare Modell). Handrechnung mit einem Schritt (K = 1) und Reaktion R(u) = 2u - u² (rho = 1): F(x) = (0,91, 0,84, 0,91), R'(Ax) = (-0,6, -0,8, -0,6),
    J = diag(R') A mit den Zeilen (-0,36, -0,18, -0,06), (-0,24, -0,32, -0,24), (-0,06, -0,18, -0,36); Gradient von 1/2 |F(x)|² = J^T F = (-0,5838, -0,5964, -0,5838). Zwei Schritte
    (K = 2, rho = 2, also dieselbe Reaktion R(u) = 2u - u²): F(x) = (0,987679, 0,986076, 0,987679). Tikhonov (rho = 0, lam = 0,1): x = (1,153465, 1,653465, 1,153465) = Posterior-Mittel
    aus Stück 7 mit sigma = 0,1, tau = 1 (Kovarianz mit Gamma_11 = 0,105864, Gamma_22 = 0,336634, Gamma_13 = 0,067403 und Eigenwerten 1/101, 1/26, 1/2)."""
    a = np.array([[0.6, 0.3, 0.1], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]])
    x = np.array([1.0, 2.0, 1.0])
    b = np.array([1.3, 1.4, 1.3])
    return a, x, b


@pytest.fixture
def mini_model(mini):
    """Ein Schritt, rho = 1: R(u) = 2u - u²."""
    import nli_model as m

    return m.Model(mini[0], 1, 1.0)


@pytest.fixture
def mini_model2(mini):
    """Zwei Schritte, rho = 2: dieselbe Reaktion R(u) = 2u - u² je Schritt."""
    import nli_model as m

    return m.Model(mini[0], 2, 2.0)
