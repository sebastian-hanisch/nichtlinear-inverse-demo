"""Auswertung: Formatierung, Instanz, Analyse, Verdikte (Art und Text, Schwellen an handgesetzten Werten), Form der Experimente und ihre Reproduzierbarkeit."""

import dataclasses
import math

import numpy as np
import pytest

import nli_constants as C
import nli_evaluation as ev
import nli_model as model
from inverse_generator import SplitMix64


# ----------------------------------------------------------------------------- Formatierung

def test_de_sci_share_noise_label_rho_label_count_text():
    assert ev.de(-1.234, 2) == "−1,23" and ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.share(0.1525) == "15 %"
    assert ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich" and ev.sci(9.96) == "1,0·10¹"
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0,01 %", "0,1 %", "1 %", "3 %", "10 %"]
    assert [ev.rho_label(v) for v in C.RHO_CHOICES] == ["0", "0,5", "1", "2", "4"] and ev.rho_label(10.0) == "10"
    assert ev.count_text(1234567) == "1.234.567" and ev.count_text(5) == "5" and ev.count_text(1000) == "1.000"


def test_err_pct_digits_and_boundaries():
    assert [ev.err_pct(v) for v in (2.58, 0.1188, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,9 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]
    assert ev.err_pct(10.0) == "1,0·10¹-fach" and ev.err_pct(9.99) == "999 %" and ev.err_pct(0.995) == "100 %" and ev.err_pct(0.994) == "99,4 %"
    assert ev.err_pct(0.0095) == "0,9 %" and ev.err_pct(0.0094) == "0,94 %" and ev.err_pct(0.0001) == "0,01 %" and ev.err_pct(0.00009) == "9,0·10⁻³ %"
    assert ev.err_pct(float("inf")) == "unendlich"


def test_num_text_boundaries():
    assert ev.num_text(1000.0) == "1,0·10³" and ev.num_text(999.99) == "1000,0" and ev.num_text(0.01) == "0,01" and ev.num_text(0.0099) == "9,9·10⁻³"
    assert ev.num_text(10.0) == "10,0" and ev.num_text(9.99) == "9,99" and ev.num_text(0.0) == "0,00" and ev.num_text(float("inf")) == "unendlich"


def test_rel_err_by_hand():
    assert ev.rel_err(np.array([3.0, 4.0]), np.array([0.0, 5.0])) == pytest.approx(math.sqrt(10.0) / 5.0)
    assert ev.rel_err(np.ones(3), np.ones(3)) == 0.0


# ----------------------------------------------------------------------------- Instanz

def test_instance_is_noise_on_the_fine_or_the_inverted_model():
    s, x, b, sigma = ev.instance(24, 0.03, 1.0, 1e-3, "honest", 5)
    fine = model.Model(model.heat_matrix(24, 0.03, C.K_TRUE), C.K_TRUE, 1.0).forward(x)
    assert s.shape == x.shape == b.shape == (24,) and sigma == pytest.approx(1e-3 * float(np.sqrt(np.mean(fine ** 2))))
    assert b - fine == pytest.approx(SplitMix64(5).normal(24) * sigma)
    s, x, b2, sigma2 = ev.instance(24, 0.03, 1.0, 1e-3, "crime", 5)
    crime = model.Model(model.heat_matrix(24, 0.03, C.K_INV), C.K_INV, 1.0).forward(x)
    assert b2 - crime == pytest.approx(SplitMix64(5).normal(24) * sigma2) and not np.allclose(b, b2, atol=1e-4)
    assert np.array_equal(ev.instance(24, 0.03, 1.0, 1e-3, "honest", 5)[2], b) and not np.array_equal(ev.instance(24, 0.03, 1.0, 1e-3, "honest", 6)[2], b)


def test_problem_streams_noise_across_the_repetitions():
    inv, x, bb, sigma = ev.problem(1.0, reps=3)
    inv1, x1, b1, sigma1 = ev.problem(1.0, reps=1)
    assert bb.shape == (3, 48) and b1.shape == (1, 48) and sigma == pytest.approx(sigma1) and not np.allclose(bb[0], bb[1])
    clean = model.Model(model.heat_matrix(48, C.DEFAULT_W, C.K_TRUE), C.K_TRUE, 1.0).forward(x)
    noise = SplitMix64(C.SEED_BASE + 1).normal(144).reshape(3, 48)           # ein Strom für alle Wiederholungen (Box-Muller zieht erst alle u1, dann alle u2)
    assert (bb[1] - clean) == pytest.approx(noise[1] * sigma) and (bb[0] - clean) == pytest.approx(noise[0] * sigma)
    crime = ev.problem(1.0, honest=False)
    assert not np.allclose(crime[2][0], b1[0], atol=1e-4) and crime[0].k == C.K_INV


def test_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


# ----------------------------------------------------------------------------- Analyse

def test_analysis_fields_of_the_standard_instance():
    a = ev.analyse(ev.Settings())
    assert set(a.runs) == set(C.SOLVERS) and a.run is a.runs["lm"] and a.run.converged and a.same_min and not a.singular and a.flat == 0.0
    assert a.x0.tolist() == [0.0] * 48 and a.err == pytest.approx(a.errs["lm"]) and a.err_ref == pytest.approx(a.err, abs=1e-4) and a.err_lin > 3.0 * a.err
    assert a.resid == pytest.approx(float(np.linalg.norm(model.Model(model.heat_matrix(48, 0.03, 4), 4, 1.0).forward(a.run.x) - a.b)))
    assert a.disc == pytest.approx(a.sigma * math.sqrt(48)) and a.cond == pytest.approx(model.total_condition(48, 0.03)) and a.grad_err < 1e-8 and a.check_counts == (1, 96)
    assert a.g_adj.shape == a.g_fd.shape == a.x_lin.shape == (48,)


def test_the_reference_is_reused_only_for_a_zero_start_with_enough_iterations():
    base = dataclasses.replace(ev.Settings(), iters=300)
    a = ev.analyse(base)
    assert a.ref is a.runs["lm"]
    b = ev.analyse(dataclasses.replace(base, amp=0.5))
    assert b.ref is not b.runs["lm"] and b.ref.x == pytest.approx(a.run.x)
    c = ev.analyse(ev.Settings())
    assert c.ref is not c.runs["lm"] and c.ref.x == pytest.approx(c.run.x, abs=1e-6)


def test_linear_model_is_the_best_tikhonov_over_the_experiment_grid():
    a = ev.analyse(ev.Settings(rho=2.0))
    a_lin = np.linalg.matrix_power(model.heat_matrix(48, 0.03, 4), 4)
    errs = [ev.rel_err(model.tikhonov_linear(a_lin, a.b, lam), a.x) for lam in C.EXP_LAMS]
    assert a.err_lin == pytest.approx(min(errs)) and a.lam_lin == pytest.approx(C.EXP_LAMS[int(np.argmin(errs))]) and a.x_lin == pytest.approx(model.tikhonov_linear(a_lin, a.b, a.lam_lin))


def test_start_and_solver_choice_are_used():
    a = ev.analyse(ev.Settings(solver="gd", amp=0.5, iters=10))
    assert a.run is a.runs["gd"] and a.run.iterations == 10 and a.x0 == pytest.approx(model.start_vector(48, 0.5, 11 + 3)) and a.runs["lm"].phi[0] == pytest.approx(a.run.phi[0])


def test_singular_corner_and_flat_share():
    assert ev.analyse(ev.Settings(n=96, w=0.05)).singular and not ev.analyse(ev.Settings()).singular
    assert ev.analyse(ev.Settings(rho=4.0)).flat > 0.1 and ev.analyse(ev.Settings(rho=0.0)).flat == 0.0 and ev.analyse(ev.Settings(rho=2.0)).flat == 0.0


def test_gradient_check_agrees_for_every_reaction_strength():
    for rho in C.RHO_CHOICES:
        a = ev.analyse(ev.Settings(rho=rho, n=24))
        assert a.grad_err < 1e-7 and a.check_counts == (1, 48), rho


def test_every_preset_runs_and_every_solver_state_is_reachable():
    states = {name: ev.analyse(ev.Settings(**p)).run.status for name, p in C.PRESETS.items()}
    assert set(states.values()) == {"converged", "maxit", "diverged"}


# ----------------------------------------------------------------------------- Verdikte

def _variant(a, **changes):
    return dataclasses.replace(a, **changes)


def test_verdict_instance_success_and_each_warning_branch():
    a = ev.analyse(ev.Settings())
    kind, text = ev.verdict_instance(a)
    assert kind == "success" and "hat konvergiert und liegt im selben Minimum" in text and "kostet hier Genauigkeit" in text
    run = a.run
    kind, text = ev.verdict_instance(_variant(a, run=dataclasses.replace(run, status="diverged", converged=False)))
    assert kind == "warning" and "davongelaufen" in text
    kind, text = ev.verdict_instance(_variant(a, run=dataclasses.replace(run, status="stalled", converged=False)))
    assert kind == "warning" and "keine Schrittweite" in text
    kind, text = ev.verdict_instance(_variant(a, run=dataclasses.replace(run, status="maxit", converged=False, gnorm=[1.0, 1e-3])))
    assert kind == "warning" and "nicht konvergiert" in text and "1,0·10⁻³" in text and "nach " + str(run.iterations) + " Iterationen" in text
    kind, text = ev.verdict_instance(_variant(a, same_min=False, run=dataclasses.replace(run, phi=[1.0, 7.0]), ref=dataclasses.replace(a.ref, phi=[1.0, 2.0])))
    assert kind == "warning" and "Lokales Minimum" in text and "3,50-Fachen" in text


def test_verdict_instance_thresholds_on_hand_set_values():
    a = ev.analyse(ev.Settings())
    assert ev.verdict_instance(_variant(a, err=0.5))[0] == "success" and ev.verdict_instance(_variant(a, err=0.5001))[0] == "warning"
    assert "nicht brauchbar" in ev.verdict_instance(_variant(a, err=0.6))[1] and "flach" not in ev.verdict_instance(_variant(a, err=0.6, flat=0.0))[1]
    assert "bei 14 % der Zustände" in ev.verdict_instance(_variant(a, err=0.6, flat=0.14))[1]
    ref = _variant(a, err_ref=0.1)
    assert "ignorierbar" in ev.verdict_instance(_variant(ref, err_lin=0.1 * 1.25))[1] and "kostet hier Genauigkeit" in ev.verdict_instance(_variant(ref, err_lin=0.1 * 1.2501))[1]
    assert "das 2,0-Fache" in ev.verdict_instance(_variant(ref, err_lin=0.2))[1]
    assert "Beide Modelle versagen" in ev.verdict_instance(_variant(ref, err_ref=0.51, err_lin=0.51))[1]
    assert "Beide Modelle versagen" not in ev.verdict_instance(_variant(ref, err_ref=0.51, err_lin=0.5))[1] and "Beide Modelle versagen" not in ev.verdict_instance(_variant(ref, err_ref=0.5, err_lin=0.51))[1]


# ----------------------------------------------------------------------------- Experimente (Form und Reproduzierbarkeit)

def test_adjoint_experiment_shape_and_counts():
    res = ev.adjoint_experiment()
    assert len(res["rows"]) == 9 and [(r["rho"], r["n"]) for r in res["rows"]][:4] == [(0.0, 24), (0.0, 48), (0.0, 96), (1.0, 24)]
    for r in res["rows"]:
        assert r["nf_adj"] == 1 and r["nf_fd"] == 2 * r["n"] and r["err_fd"] < 1e-8 and r["err_jac"] < 1e-12
    assert res == ev.adjoint_experiment()
    kind, text = ev.verdict_adjoint(res)
    assert kind == "success" and "192 Vorwärtsläufe" in text and "9 Fällen" in text and "weniger als 1,0·10⁻⁹" in text


def test_laplace_experiment_shape_and_reproducibility():
    res = ev.laplace_experiment(chains=4, steps=2000)
    assert [r["rho"] for r in res["rows"]] == list(C.LAP_RHOS) and res["n"] == 24 and res["chains"] == 4
    for r in res["rows"]:
        assert 0.1 < r["acc"] < 0.5 and r["ratio_min"] <= r["ratio_median"] <= r["ratio_max"] and 0.0 <= r["cover_laplace"] <= 1.0 and r["rhat"] > 0.9
    assert res == ev.laplace_experiment(chains=4, steps=2000)
    assert ev.verdict_laplace(res)[0] == "info"


def test_verdicts_of_the_slow_experiments_on_hand_built_results():
    ignore = {"rows": [{"rho": r, "lin": 0.2 + r / 10, "nl": 0.07 + r / 100} for r in C.EXP_RHOS], "disc": {"rho": 2.0, "rms": 0.153, "best": 0.152}}
    kind, text = ev.verdict_ignore(ignore)
    assert kind == "warning" and "7,5 % / 8,0 % / 9,0 %" in text and "25,0 % / 30,0 % / 40,0 %" in text and "(11,0 % und 60,0 %)" in text and "einen Fehler von 15,3 % (RMS)" in text and "15,2 %" in text
    rows = [{"rho": 1.0, "solver": s, "iterations": i} for s, i in (("gd", 500), ("gn_pur", 6), ("gn", 6), ("lm", 12))]
    basin = [{"rho": 2.0, "amp": amp, "starts": 20, "lm": lm, "gn": 0, "gn_pur": gp} for amp, lm, gp in ((0.5, 20, 1), (1.0, 9, 4), (1.5, 3, 0))]
    kind, text = ev.verdict_solvers({"rows": rows, "basin": basin})
    assert kind == "warning" and "500 Iterationen (nicht konvergiert)" in text and "Schrittweitensuche 6" in text and "Levenberg-Marquardt 12" in text
    assert "20 von 20 Starts der Amplitude 0,5, von 3 von 20 der Amplitude 1,5" in text and "bei Amplitude 1 von 4 von 20" in text


def test_gradient_check_point_is_080_truth_plus_005():
    a = ev.analyse(ev.Settings(n=24, rho=2.0))
    inv = ev.inverse_model(24, 0.03, 2.0)
    u, states = inv.forward(0.8 * a.x + 0.05, record=True)
    assert a.g_adj == pytest.approx(inv.adjoint(states, u - a.b))
