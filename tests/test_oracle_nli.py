"""Orakel (DEMO-PLAYBOOK §12): adjungierter Gradient und Jacobi-Matrix gegen Differenzenquotienten, linearer Grenzfall gegen Tikhonov und die Gauß-Unschärfe des Generators,
Konvergenz des Splittings, Löser gegeneinander und gegen die Stationaritätsbedingung, Laplace-Näherung gegen den exakten Posterior (linear) und gegen MCMC, Nichtinjektivität.
Jeder Test unter etwa 10 s."""

import math

import numpy as np
import pytest

import nli_evaluation as ev
import nli_model as m
from inverse_generator import SplitMix64, grid, operator, phantom


def _model(n, w, k, rho):
    return m.Model(m.heat_matrix(n, w, k), k, rho)


# ----------------------------------------------------------------------------- Gradient und Jacobi-Matrix

@pytest.mark.parametrize("rho", [0.0, 1.0, 4.0])
@pytest.mark.parametrize("n,k", [(12, 1), (12, 4), (24, 4)])
def test_adjoint_gradient_matches_finite_differences_and_the_explicit_jacobian(rho, n, k):
    mod = _model(n, 0.03, k, rho)
    xt = phantom(grid(n))
    b = mod.forward(xt) + 0.01 * SplitMix64(3).normal(n)
    x = 0.8 * xt + 0.05
    u, states = mod.forward(x, record=True)
    g = mod.adjoint(states, u - b)
    assert np.linalg.norm(g - m.gradient_fd(mod, x, b)) < 1e-7 * np.linalg.norm(g)
    assert np.linalg.norm(g - mod.jacobian(x).T @ (u - b)) < 1e-12 * np.linalg.norm(g)


@pytest.mark.parametrize("rho", [0.5, 2.0])
def test_jacobian_matches_finite_difference_columns(rho):
    mod = _model(12, 0.03, 4, rho)
    x = phantom(grid(12)) * 0.7
    h = 1e-6
    fd = np.column_stack([(mod.forward(x + h * e) - mod.forward(x - h * e)) / (2 * h) for e in np.eye(12)])
    assert np.abs(mod.jacobian(x) - fd).max() < 1e-7


def test_adjoint_needs_one_forward_run_and_the_difference_quotient_2n():
    mod = _model(24, 0.03, 4, 1.0)
    x = phantom(grid(24))
    b = np.zeros(24)
    u, states = mod.forward(x, record=True)
    mod.adjoint(states, u - b)
    assert mod.counts["forward"] == 1
    m.gradient_fd(mod, x, b)
    assert mod.counts["forward"] == 1 + 48


# ----------------------------------------------------------------------------- Linearer Grenzfall

def test_rho_zero_is_the_matrix_power_the_cosine_representation_and_the_generators_gauss_blur():
    n, w = 96, 0.03
    a = m.heat_matrix(n, w, 4)
    j = np.arange(n)
    c = math.sqrt(2.0 / n) * np.cos(math.pi * (np.arange(n)[:, None] + 0.5) * j[None, :] / n)
    c[:, 0] = 1.0 / math.sqrt(n)
    assert c.T @ c == pytest.approx(np.eye(n), abs=1e-12)
    total = c @ np.diag(np.exp(-(w ** 2 / 2.0) * (j * math.pi) ** 2)) @ c.T
    assert np.linalg.matrix_power(a, 4) == pytest.approx(total, abs=1e-12)
    x = phantom(grid(n))
    assert m.Model(a, 4, 0.0).forward(x) == pytest.approx(total @ x, abs=1e-12)
    assert np.abs(total[30:66, :] - operator(n, w)[30:66, :]).max() < 1e-8              # Innenzeilen: reflektierende Ränder sind noch nicht zu spüren
    assert m.Model(a, 4, 0.0).forward(x).mean() == pytest.approx(x.mean())              # Mittelwerterhaltung


def test_gauss_newton_in_the_linear_case_is_tikhonov_after_one_iteration():
    inv, x, bb, sigma = ev.problem(0.0, honest=False)
    lam = 1e-2
    res = m.solve(inv, bb[0], lam, np.zeros(48), "gn", 5)
    a_tot = np.linalg.matrix_power(inv.a, 4)
    assert res.iterations == 1 and res.converged
    assert np.abs(res.x - np.linalg.solve(a_tot.T @ a_tot + lam ** 2 * np.eye(48), a_tot.T @ bb[0])).max() < 1e-9
    sv = np.linalg.svd(a_tot)                                                            # dritte Route: Tikhonov-Filter s / (s² + λ²) in der SVD
    u, s, vt = sv
    assert res.x == pytest.approx(vt.T @ ((s / (s ** 2 + lam ** 2)) * (u.T @ bb[0])), abs=1e-8)
    assert np.abs(m.solve(inv, bb[0], lam, np.zeros(48), "gn_pur", 200).x - res.x).max() < 1e-9
    lm = m.solve(inv, bb[0], lam, np.zeros(48), "lm", 200)                                # gedämpfte Schritte nähern sich von unten: Abbruch bei 1e-8 der Gradientennorm lässt 1e-4 Abstand
    assert lm.converged and np.abs(lm.x - res.x).max() < 1e-3 and lm.iterations > 1


def test_splitting_error_is_of_first_order_in_the_step_count():
    n, w = 48, 0.03
    x = phantom(grid(n))
    for rho in (1.0, 2.0):
        ref = _model(n, w, 256, rho).forward(x)
        errs = [np.linalg.norm(_model(n, w, k, rho).forward(x) - ref) for k in (2, 4, 8, 16)]
        ratios = [errs[i] / errs[i + 1] for i in range(3)]
        assert all(1.9 < r < 2.2 for r in ratios), (rho, ratios)


def test_honest_data_differ_from_the_inverse_crime_data_by_the_splitting_error():
    s, x, b_h, sigma = ev.instance(48, 0.03, 1.0, 1e-3, "honest", 5)
    s, x, b_c, sigma_c = ev.instance(48, 0.03, 1.0, 1e-3, "crime", 5)
    clean_diff = np.linalg.norm(b_h - b_c)
    assert clean_diff > 3.0 * sigma * math.sqrt(48) and ev.instance(48, 0.03, 0.0, 1e-3, "honest", 5)[2] == pytest.approx(ev.instance(48, 0.03, 0.0, 1e-3, "crime", 5)[2], abs=1e-9)


# ----------------------------------------------------------------------------- Löser gegeneinander

def test_gauss_newton_pure_with_line_search_and_levenberg_marquardt_reach_the_same_minimum():
    inv, x, bb, sigma = ev.problem(1.0)
    lam = 1e-3
    res = {mode: m.solve(inv, bb[0], lam, np.zeros(48), mode, 100) for mode in ("gn_pur", "gn", "lm")}
    assert all(r.converged for r in res.values())
    assert res["gn"].phi[-1] == pytest.approx(res["lm"].phi[-1], rel=1e-6) and res["gn_pur"].phi[-1] == pytest.approx(res["gn"].phi[-1], rel=1e-9)
    assert np.abs(res["gn"].x - res["lm"].x).max() < 1e-3
    gd = m.solve(inv, bb[0], lam, np.zeros(48), "gd", 500)
    assert not gd.converged and all(gd.phi[i + 1] <= gd.phi[i] for i in range(gd.iterations)) and gd.phi[-1] > 2.0 * res["gn"].phi[-1]
    assert res["gn"].iterations < 12 and res["lm"].iterations > res["gn"].iterations


def test_result_is_stationary_and_the_hessian_is_positive_definite():
    n = 12
    inv, x, bb, sigma = ev.problem(1.0, n=n)
    lam = 1e-2
    res = m.solve(inv, bb[0], lam, np.zeros(n), "lm", 100)
    assert res.converged

    def grad(v):
        u, states = inv.forward(v, record=True)
        return m.phi_gradient(inv.adjoint(states, u - bb[0]), v, lam)

    h = 1e-6
    fd_phi = np.array([(m.phi_value(inv.forward(res.x + h * e) - bb[0], res.x + h * e, lam) - m.phi_value(inv.forward(res.x - h * e) - bb[0], res.x - h * e, lam)) / (2 * h)
                       for e in np.eye(n)])
    assert np.linalg.norm(fd_phi) < 1e-5 * res.gnorm[0] + 1e-9 and np.linalg.norm(grad(res.x)) <= 1e-8 * res.gnorm[0] * 1.001
    hess = np.column_stack([(grad(res.x + h * e) - grad(res.x - h * e)) / (2 * h) for e in np.eye(n)])
    assert np.abs(hess - hess.T).max() < 1e-4 * np.abs(hess).max() and np.linalg.eigvalsh(0.5 * (hess + hess.T)).min() > 0.0


# ----------------------------------------------------------------------------- Laplace-Näherung

def _chain_moments(inv, b, sigma, tau, x_map, cov, chains, steps, seed):
    n = len(x_map)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
    x0 = x_map[None, :] + 2.0 * SplitMix64(seed).normal(chains * n).reshape(chains, n) @ chol.T
    r = m.rwm(inv, b, sigma, tau, x0, steps, seed + 1, 2.38 / math.sqrt(n), chol, np.full(n, 1.0 / n))
    total = r["count"] * chains
    mean = r["sum1"].sum(axis=0) / total
    return mean, np.sqrt(np.maximum(r["sum2"].sum(axis=0) / total - mean ** 2, 0.0)), r


def test_laplace_approximation_is_exact_in_the_linear_case_and_chains_reproduce_it():
    n, tau = 12, 0.3
    inv, x, bb, sigma = ev.problem(0.0, noise=1e-2, n=n)
    a_tot = np.linalg.matrix_power(inv.a, 4)
    mean, cov = m.gauss_posterior_linear(a_tot, bb[0], sigma, tau)
    x_map = m.solve(inv, bb[0], sigma / tau, np.zeros(n), "lm", 100).x
    assert x_map == pytest.approx(mean, abs=1e-8) and m.laplace_covariance(inv, x_map, sigma, tau) == pytest.approx(cov, rel=1e-8, abs=1e-14)
    cmean, csd, r = _chain_moments(inv, bb[0], sigma, tau, mean, cov, 8, 20000, 7)
    ref = np.sqrt(np.diag(cov))
    assert np.max(np.abs(cmean - mean) / ref) < 0.15 and np.max(np.abs(csd / ref - 1.0)) < 0.08 and float(r["acc"].mean()) == pytest.approx(0.25, abs=0.08)


def test_laplace_approximation_matches_the_chains_for_a_moderately_nonlinear_posterior():
    n, tau = 12, 0.3
    inv, x, bb, sigma = ev.problem(1.0, noise=1e-2, n=n)
    x_map = m.solve(inv, bb[0], sigma / tau, np.zeros(n), "lm", 100).x
    cov = m.laplace_covariance(inv, x_map, sigma, tau)
    cmean, csd, r = _chain_moments(inv, bb[0], sigma, tau, x_map, cov, 8, 20000, 11)
    ratio = np.sqrt(np.diag(cov)) / csd
    assert np.all(np.abs(ratio - 1.0) < 0.06) and np.max(np.abs(cmean - x_map) / csd) < 0.3
    assert m.rhat(r["traces"][4000:].T) < 1.05


# ----------------------------------------------------------------------------- Nichtinjektivität

def test_reaction_is_not_injective_at_u_one_and_the_map_loses_information():
    a = m.heat_matrix(8, 0.03, 1)
    mod = m.Model(a, 1, 1.0)                                    # R(u) = 2u - u²: R(0,5) = R(1,5) = 0,75, R'(1) = 0
    assert mod.forward(0.5 * np.ones(8)) == pytest.approx(mod.forward(1.5 * np.ones(8)))
    assert np.abs(mod.jacobian(np.ones(8))).max() < 1e-12
    inj = m.Model(a, 1, 0.0)
    assert np.abs(inj.forward(0.5 * np.ones(8)) - inj.forward(1.5 * np.ones(8))).max() > 0.5
