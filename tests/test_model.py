"""Kernroutinen an der 3 x 3-Mini-Instanz und an einer skalaren Gleichung von Hand (DEMO-PLAYBOOK §1b); jede Einheit einzeln."""

import math

import numpy as np
import pytest

import nli_constants as C
import nli_model as m
from inverse_generator import SplitMix64

SCALAR_A = np.array([[1.0]])        # n = 1, K = 1, rho = 1: F(x) = 2x - x²; mit b = 0,75 sind 0,5 und 1,5 die Lösungen (Newton-Verfahren von Hand)


def _scalar():
    return m.Model(SCALAR_A, 1, 1.0), np.array([0.75])


# ----------------------------------------------------------------------------- Wärmeleitung

def test_heat_matrix_n2_by_hand():
    s = math.exp(-(0.1 ** 2 / 4.0) * math.pi ** 2)                      # K = 2, w = 0,1: Symbol von j = 1: exp(-(w²/2K) π²) = exp(-0,0025 π²)
    a = m.heat_matrix(2, 0.1, 2)
    assert a == pytest.approx(0.5 * np.array([[1 + s, 1 - s], [1 - s, 1 + s]]))


def test_heat_matrix_is_symmetric_mean_preserving_and_a_semigroup():
    a = m.heat_matrix(12, 0.05, 4)
    assert a == pytest.approx(a.T) and a.sum(axis=1) == pytest.approx(np.ones(12)) and np.linalg.matrix_power(a, 4) == pytest.approx(m.heat_matrix(12, 0.05, 1), abs=1e-12)
    assert np.trace(m.heat_matrix(5, 0.1, 1)) == pytest.approx(sum(math.exp(-(0.01 / 2.0) * (j * math.pi) ** 2) for j in range(5)))


def test_total_condition_by_hand_and_cap():
    assert m.total_condition(2, 0.1) == pytest.approx(math.exp(0.5 * 0.01 * math.pi ** 2))
    assert m.total_condition(1, 0.1) == 1.0
    assert m.total_condition(96, 0.05) == pytest.approx(math.exp(0.5 * 0.0025 * (95 * math.pi) ** 2)) and m.total_condition(2000, 0.5) == pytest.approx(math.exp(700.0))


# ----------------------------------------------------------------------------- Reaktion und Vorwärtsmodell

def test_reaction_and_its_derivative_by_hand(mini_model):
    assert mini_model.react(1.3) == pytest.approx(0.91) and mini_model.react(1.4) == pytest.approx(0.84) and mini_model.dreact(1.3) == pytest.approx(-0.6) and mini_model.dreact(1.4) == pytest.approx(-0.8)
    half = m.Model(np.eye(2), 2, 1.0)                                   # rho / K = 0,5
    assert half.react(0.5) == pytest.approx(0.625) and half.dreact(0.5) == pytest.approx(1.0) and half.dreact(0.0) == pytest.approx(1.5)
    assert half.react(0.0) == 0.0 and half.react(1.0) == 1.0
    assert mini_model.react(np.array([1.3, 1.4])) == pytest.approx([0.91, 0.84])


def test_forward_by_hand(mini, mini_model, mini_model2):
    a, x, b = mini
    assert mini_model.forward(x) == pytest.approx([0.91, 0.84, 0.91])
    assert mini_model2.forward(x) == pytest.approx([0.987679, 0.986076, 0.987679], abs=1e-6)
    u, states = mini_model2.forward(x, record=True)
    assert len(states) == 3 and states[0] == pytest.approx(x) and states[1] == pytest.approx([0.91, 0.84, 0.91]) and states[2] == pytest.approx(u)
    assert mini_model2.counts == {"forward": 2, "jacobian": 0, "adjoint": 0}


def test_forward_is_the_matrix_power_for_rho_zero(mini):
    a, x, b = mini
    lin = m.Model(a, 3, 0.0)
    assert lin.forward(x) == pytest.approx(np.linalg.matrix_power(a, 3) @ x)


def test_forward_rows_equals_forward_rowwise_without_counting(mini, mini_model2):
    a, x, b = mini
    rows = np.vstack([x, 0.5 * x, np.zeros(3)])
    out = mini_model2.forward_rows(rows)
    assert mini_model2.counts["forward"] == 0
    for i in range(3):
        assert out[i] == pytest.approx(mini_model2.forward(rows[i]))
    assert out[2] == pytest.approx(np.zeros(3))


def test_jacobian_by_hand(mini, mini_model, mini_model2):
    a, x, b = mini
    assert mini_model.jacobian(x) == pytest.approx(np.array([[-0.36, -0.18, -0.06], [-0.24, -0.32, -0.24], [-0.06, -0.18, -0.36]]))
    j2 = mini_model2.jacobian(x)
    assert j2[0, 0] == pytest.approx(-0.065268) and mini_model2.counts["jacobian"] == 1
    d1, d2 = np.diag([-0.6, -0.8, -0.6]), np.diag([0.222, 0.236, 0.222])
    assert j2 == pytest.approx(d2 @ a @ d1 @ a)
    # Spaltenweise Differenzenquotienten (zweite Rechenroute)
    h = 1e-6
    fd = np.column_stack([(mini_model2.forward(x + h * e) - mini_model2.forward(x - h * e)) / (2 * h) for e in np.eye(3)])
    assert j2 == pytest.approx(fd, abs=1e-8)


def test_adjoint_gradient_by_hand_and_against_the_jacobian(mini, mini_model, mini_model2):
    a, x, b = mini
    u, states = mini_model.forward(x, record=True)
    assert mini_model.adjoint(states, u) == pytest.approx([-0.5838, -0.5964, -0.5838])      # b = 0: Residuum = F(x)
    assert mini_model.counts["adjoint"] == 1
    u2, states2 = mini_model2.forward(x, record=True)
    assert mini_model2.adjoint(states2, u2 - b) == pytest.approx(mini_model2.jacobian(x).T @ (u2 - b))


def test_gradient_fd_matches_the_adjoint_and_costs_2n_forward_runs(mini, mini_model2):
    a, x, b = mini
    u, states = mini_model2.forward(x, record=True)
    g = mini_model2.adjoint(states, u - b)
    before = mini_model2.counts["forward"]
    assert m.gradient_fd(mini_model2, x, b) == pytest.approx(g, abs=1e-8)
    assert mini_model2.counts["forward"] - before == 6


# ----------------------------------------------------------------------------- Zielfunktion

def test_phi_value_and_gradient_by_hand(mini):
    a, x, b = mini
    r = np.array([0.91, 0.84, 0.91])
    assert m.phi_value(r, x, 0.1) == pytest.approx(0.8281 + 0.7056 + 0.8281 + 0.01 * 6.0)
    assert m.phi_gradient(np.array([-0.5838, -0.5964, -0.5838]), x, 0.1) == pytest.approx(2.0 * np.array([-0.5838 + 0.01, -0.5964 + 0.02, -0.5838 + 0.01]))
    assert m.phi_value(np.zeros(3), np.zeros(3), 5.0) == 0.0


# ----------------------------------------------------------------------------- Löser an der skalaren Gleichung 2x - x² = 0,75

def test_gradient_descent_first_step_is_a_unit_length_step_and_the_line_search_halves():
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.2]), "gd", 1)
    assert r.x == pytest.approx([1.2]) and r.iterations == 1 and r.status == "maxit" and not r.converged        # Schrittlänge 1: 0,2 + 1
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.9]), "gd", 1)
    assert r.x == pytest.approx([0.4]) and r.counts == {"forward": 3, "jacobian": 0, "adjoint": 2}                  # Schritt 1 -> -0,1 abgelehnt, halbiert -> 0,4


def test_gradient_descent_doubles_the_last_step():
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.2]), "gd", 2)
    g1 = 2.0 * (2.0 - 2.0 * 1.2) * (2.0 * 1.2 - 1.44 - 0.75)           # Gradient bei 1,2: -0,168
    assert g1 == pytest.approx(-0.168)
    assert r.x[0] == pytest.approx(1.2 - 2.0 / 1.248 * g1) and r.counts["forward"] == 3 and r.phi[2] < r.phi[1] < r.phi[0]       # zweite Schrittweite 2 / 1,248, ohne Halbierung angenommen


def test_gauss_newton_is_the_newton_method_for_lam_zero():
    mod, b = _scalar()
    r1 = m.solve(mod, b, 0.0, np.array([0.2]), "gn", 1)
    r2 = m.solve(mod, b, 0.0, np.array([0.2]), "gn", 2, tol=1e-30)
    assert r1.x[0] == pytest.approx(0.44375) and r2.x[0] == pytest.approx(0.497156, abs=1e-6)
    assert r1.counts == {"forward": 2, "jacobian": 1, "adjoint": 2}
    r = m.solve(mod, b, 0.0, np.array([1.2]), "gn", 50)
    assert r.converged and r.x[0] == pytest.approx(1.5, abs=1e-6)


def test_gauss_newton_line_search_halves_where_the_pure_step_overshoots():
    mod, b = _scalar()
    assert m.solve(mod, b, 0.0, np.array([0.8]), "gn_pur", 1).x == pytest.approx([0.275])
    r = m.solve(mod, b, 0.0, np.array([0.8]), "gn", 1)
    assert r.x == pytest.approx([0.5375]) and r.counts["forward"] == 3 and r.phi[1] < r.phi[0]


def test_levenberg_marquardt_damps_and_adapts_mu():
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.2]), "lm", 1)
    assert r.x[0] == pytest.approx(0.2 + 0.624 / 2.57)                  # mu = 0,01
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.8]), "lm", 1)
    assert r.x[0] == pytest.approx(0.38) and r.counts["forward"] == 3   # erster Versuch mu = 0,01 abgelehnt, mu = 0,04 angenommen
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.2]), "lm", 100)
    assert r.converged and r.x[0] == pytest.approx(0.5, abs=1e-6)


def test_pure_gauss_newton_diverges_where_the_other_solvers_converge():
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([1.0001]), "gn_pur", 50)
    assert r.status == "diverged" and r.iterations == 0 and not r.converged and r.x == pytest.approx([1.0001]) and len(r.phi) == 1
    for mode in ("gn", "lm"):
        mod, b = _scalar()
        r = m.solve(mod, b, 0.0, np.array([1.0001]), mode, 200)
        assert r.converged and min(abs(r.x[0] - 0.5), abs(r.x[0] - 1.5)) < 1e-6, mode


def test_solver_stops_at_maxit_at_convergence_and_when_the_start_is_stationary():
    mod, b = _scalar()
    r = m.solve(mod, b, 0.0, np.array([0.2]), "lm", 0)
    assert r.iterations == 0 and r.status == "maxit" and r.x == pytest.approx([0.2]) and r.phi == pytest.approx([0.39 ** 2]) and r.gnorm == pytest.approx([1.248])
    r = m.solve(mod, b, 0.0, np.array([1.0]), "lm", 10)                 # F'(1) = 0: Gradient null
    assert r.status == "converged" and r.iterations == 0 and r.converged
    r = m.solve(mod, b, 0.0, np.array([0.2]), "gn", 50, tol=1e-3)
    assert r.converged and r.gnorm[-1] <= 1e-3 * r.gnorm[0] and r.gnorm[-2] > 1e-3 * r.gnorm[0] and len(r.phi) == len(r.gnorm) == r.iterations + 1


def test_solver_reports_stalled_when_no_step_gives_descent(monkeypatch):
    mod, b = _scalar()
    monkeypatch.setattr(C, "LS_MAX", 0)
    for mode in ("gd", "gn"):
        r = m.solve(mod, b, 0.0, np.array([0.2]), mode, 10)
        assert r.status == "stalled" and r.iterations == 0 and not r.converged, mode
    monkeypatch.setattr(C, "LM_TRIES", 0)
    assert m.solve(mod, b, 0.0, np.array([0.2]), "lm", 10).status == "stalled"


def test_solver_counts_are_differences_and_phi_decreases_for_the_descent_methods(mini, mini_model):
    a, x, b = mini
    bn = mini_model.forward(x)
    mini_model.counts["forward"] += 100                                  # Zähler, der nicht bei null beginnt: das Ergebnis zählt nur die Differenz
    r = m.solve(mini_model, bn, 0.1, np.zeros(3), "lm", 40)
    assert r.counts["forward"] >= r.iterations + 1 and r.counts["forward"] < 100 and r.counts["jacobian"] == r.iterations and r.counts["adjoint"] == r.iterations + 1
    assert all(r.phi[i + 1] < r.phi[i] for i in range(r.iterations))
    for mode in ("gd", "gn", "lm"):
        res = m.solve(mini_model, bn, 0.1, np.zeros(3), mode, 25)
        assert all(res.phi[i + 1] < res.phi[i] for i in range(res.iterations)), mode


def test_linear_case_is_solved_by_one_gauss_newton_step(mini):
    a, x, b = mini
    lin = m.Model(a, 1, 0.0)
    r = m.solve(lin, b, 0.1, np.zeros(3), "gn", 5)
    assert r.iterations == 1 and r.converged and r.x == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)
    assert m.tikhonov_linear(a, b, 0.1) == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6)


def test_stationary_point_of_the_nonlinear_mini_problem(mini, mini_model):
    a, x, b = mini
    bn = mini_model.forward(x)
    r = m.solve(mini_model, bn, 0.1, np.zeros(3), "lm", 100)
    assert r.converged
    h = 1e-6

    def phi(v):
        return m.phi_value(mini_model.forward(v) - bn, v, 0.1)
    fd = np.array([(phi(r.x + h * e) - phi(r.x - h * e)) / (2 * h) for e in np.eye(3)])
    assert np.linalg.norm(fd) < 1e-6 * r.gnorm[0] + 1e-8


# ----------------------------------------------------------------------------- Start

def test_start_vector():
    assert m.start_vector(8, 0.0, 5).tolist() == [0.0] * 8
    z = np.convolve(SplitMix64(5).normal(8), np.ones(7) / 7.0, mode="same")
    assert m.start_vector(8, 1.5, 5) == pytest.approx(1.5 * (z + 1.0)) and m.start_vector(8, 3.0, 5) == pytest.approx(2.0 * m.start_vector(8, 1.5, 5))
    assert m.start_vector(8, 1.5, 5).shape == (8,) and not np.array_equal(m.start_vector(8, 1.5, 5), m.start_vector(8, 1.5, 6))
    big = m.start_vector(4000, 1.0, 9)
    assert big.mean() == pytest.approx(1.0, abs=0.02) and big.std() == pytest.approx(math.sqrt(1.0 / 7.0), rel=0.08)


# ----------------------------------------------------------------------------- Laplace-Näherung, Posterior, Sampler

def test_laplace_covariance_by_hand_in_the_linear_case(mini):
    a, x, b = mini
    cov = m.laplace_covariance(m.Model(a, 1, 0.0), np.zeros(3), 0.1, 1.0)
    assert cov[0, 0] == pytest.approx(0.105864, abs=2e-6) and cov[1, 1] == pytest.approx(0.336634, abs=2e-6) and cov[0, 2] == pytest.approx(0.067403, abs=2e-6)
    assert sorted(np.linalg.eigvalsh(cov)) == pytest.approx([1.0 / 101.0, 1.0 / 26.0, 0.5])
    mean, cov2 = m.gauss_posterior_linear(a, b, 0.1, 1.0)
    assert mean == pytest.approx([1.153465, 1.653465, 1.153465], abs=2e-6) and cov2 == pytest.approx(cov)


def test_laplace_covariance_uses_the_jacobian_at_the_map(mini, mini_model):
    a, x, b = mini
    j = mini_model.jacobian(x)
    assert m.laplace_covariance(mini_model, x, 0.1, 1.0) == pytest.approx(np.linalg.inv(j.T @ j / 0.01 + np.eye(3)))


def test_neg_log_post_by_hand(mini, mini_model):
    a, x, b = mini
    val = m.neg_log_post(mini_model, np.vstack([x, np.zeros(3)]), b, 0.1, 1.0)
    assert val[0] == pytest.approx(0.6178 / 0.02 + 3.0) and val[1] == pytest.approx(0.5 * float(b @ b) / 0.01)
    assert m.neg_log_post(mini_model, x, b, 0.1, 1.0).shape == (1,)


def test_rwm_mechanics(mini, mini_model):
    a, x, b = mini
    chol = np.linalg.cholesky(m.laplace_covariance(mini_model, x, 0.1, 1.0))
    f = np.full(3, 1.0 / 3.0)
    r = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (4, 1)), 10, 3, 0.5, chol, f)
    assert r["traces"].shape == (10, 4) and r["count"] == 8 and r["sum1"].shape == (4, 3) and r["acc"].shape == (4,) and np.all((r["acc"] >= 0) & (r["acc"] <= 1))
    r2 = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (4, 1)), 10, 3, 0.5, chol, f)
    assert np.array_equal(r["traces"], r2["traces"]) and not np.array_equal(r["traces"], m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (4, 1)), 10, 4, 0.5, chol, f)["traces"])
    assert np.all(r["sum2"] >= 0) and r["traces"][0] == pytest.approx(r["traces"][0])
    tiny = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (4, 1)), 200, 3, 1e-6, chol, f)
    huge = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (4, 1)), 200, 3, 1e3, chol, f)
    assert tiny["acc"].min() > 0.99 and huge["acc"].max() < 0.05
    assert tiny["sum1"].sum(axis=0) / (tiny["count"] * 4) == pytest.approx(x, abs=1e-3)               # winziger Schritt: bleibt am Start


def test_rwm_moments_and_burn_in_accounting(mini, mini_model):
    a, x, b = mini
    chol = np.eye(3)
    r = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (2, 1)), 100, 1, 1e-9, chol, np.ones(3))
    assert r["count"] == 80 and r["sum1"] == pytest.approx(np.tile(80 * x, (2, 1))) and r["sum2"] == pytest.approx(np.tile(80 * x ** 2, (2, 1)))
    assert r["traces"][:, 0] == pytest.approx(np.full(100, 4.0))


def test_rhat_by_hand():
    assert m.rhat(np.array([[0.0, 1.0], [0.0, 1.0]])) == pytest.approx(math.sqrt(0.5))
    assert m.rhat(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])) == pytest.approx(math.sqrt(1.0 + 1.0 / 6.0))
    assert math.isinf(m.rhat(np.array([[0.0, 0.0], [1.0, 1.0]])))


# ----------------------------------------------------------------------------- Armijo-Bedingung am Rand (Vorzeichen von c · Abstiegsrichtung)

def test_gradient_descent_rejects_a_step_without_decrease():
    mod = m.Model(SCALAR_A, 1, 1.0)
    b = np.array([0.5])
    # F(x) = 2x - x²: bei x = 0 ist r = -0,5, der Schritt der Länge 1 führt nach x = 1 mit r = +0,5, also gleichem Phi: kein Abstieg, die Armijo-Bedingung verlangt Abnahme
    r = m.solve(mod, b, 0.0, np.array([0.0]), "gd", 1)
    assert r.x == pytest.approx([0.5]) and r.counts["forward"] == 3 and r.phi[1] == pytest.approx(0.0625)


def test_gauss_newton_rejects_a_step_without_decrease():
    """b so gewählt, dass der volle Gauss-Newton-Schritt Phi nicht ändert (Bisektion auf das Verhältnis 1): die Bedingung Phi_neu <= Phi + c · alpha · g^T d (Steigung < 0) lehnt ihn ab."""
    def ratio(x0):
        mod = m.Model(SCALAR_A, 1, 1.0)
        b = np.array([0.1])
        r0 = float(mod.forward(np.array([x0]))[0] - b[0])
        j = 2.0 - 2.0 * x0
        x1 = x0 - r0 / j
        return (float(mod.forward(np.array([x1]))[0] - b[0]) ** 2) / r0 ** 2
    lo, hi = 0.5, 0.7                                                     # ratio(lo) < 1 < ratio(hi) wird unten geprüft
    assert ratio(lo) < 1.0 < ratio(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ratio(mid) < 1.0 else (lo, mid)
    x0 = 0.5 * (lo + hi)
    assert abs(ratio(x0) - 1.0) < 1e-9
    mod = m.Model(SCALAR_A, 1, 1.0)
    r = m.solve(mod, np.array([0.1]), 0.0, np.array([x0]), "gn", 1)
    j = 2.0 - 2.0 * x0
    d = -(2.0 * x0 - x0 ** 2 - 0.1) / j
    assert r.x[0] == pytest.approx(x0 + 0.5 * d) and r.counts["forward"] == 3                  # eine Halbierung, dann angenommen


def test_rwm_handles_a_step_count_that_is_not_a_multiple_of_the_block(mini, mini_model):
    a, x, b = mini
    r = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (2, 1)), 700, 3, 0.1, np.eye(3), np.ones(3) / 3.0)
    assert r["traces"].shape == (700, 2) and r["count"] == 560 and np.all(np.isfinite(r["traces"]))
    r2 = m.rwm(mini_model, b, 0.1, 1.0, np.tile(x, (2, 1)), 1200, 3, 0.1, np.eye(3), np.ones(3) / 3.0)
    assert r2["traces"].shape == (1200, 2) and r2["count"] == 960 and np.all(np.isfinite(r2["traces"]))
