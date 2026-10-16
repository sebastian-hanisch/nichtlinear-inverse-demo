"""Aussagen aus README, Verdikten und Hilfetexten, einzeln an den festen Strömen gemessen (Seeds ab SEED_BASE). Ändert sich eine Zahl, muss der Text mit."""

import pytest

import nli_constants as C
import nli_evaluation as ev


@pytest.fixture(scope="module")
def adjoint():
    return ev.adjoint_experiment()


@pytest.fixture(scope="module")
def ignore():
    return ev.ignore_experiment()


@pytest.fixture(scope="module")
def solvers():
    return ev.solver_experiment()


@pytest.fixture(scope="module")
def laplace():
    return ev.laplace_experiment()


# ----------------------------------------------------------------------------- Instanz und Presets

def test_standard_preset_numbers():
    a = ev.analyse(ev.Settings(**C.PRESETS["Standard"]))
    assert a.run.iterations == 7 and a.run.counts == {"forward": 8, "jacobian": 7, "adjoint": 8} and a.err == pytest.approx(0.084, abs=0.002)
    assert a.err_lin == pytest.approx(0.301, abs=0.003) and a.err_lin / a.err == pytest.approx(3.6, abs=0.1) and a.grad_err < 1e-9 and a.check_counts == (1, 96)
    assert {m: r.iterations for m, r in a.runs.items()} == {"gd": 100, "gn_pur": 6, "gn": 6, "lm": 7}
    text = ev.verdict_instance(a)[1]
    assert "7 Iterationen (8 Vorwärtsläufe, 7 Jacobi-Matrizen, 8 adjungierte Läufe)" in text and "Fehler 8,4 %" in text and "30,1 %" in text and "das 3,6-Fache" in text


def test_other_preset_numbers():
    strong = ev.analyse(ev.Settings(**C.PRESETS["Starke Reaktion (ρ = 2)"]))
    assert strong.run.iterations == 8 and strong.err == pytest.approx(0.156, abs=0.002) and strong.err_lin == pytest.approx(0.458, abs=0.003)
    gd = ev.analyse(ev.Settings(**C.PRESETS["Gradientenabstieg"]))
    assert gd.err == pytest.approx(0.121, abs=0.003) and gd.run.counts == {"forward": 199, "jacobian": 0, "adjoint": 101} and gd.runs["lm"].iterations == 11 and gd.runs["gn"].iterations == 6
    pure = ev.analyse(ev.Settings(**C.PRESETS["Gauss-Newton ohne Schrittweitensuche"]))
    assert pure.runs["gn"].iterations == 8 and pure.runs["lm"].iterations == 17 and pure.errs["gn"] == pytest.approx(0.293, abs=0.003) and pure.errs["lm"] == pytest.approx(0.293, abs=0.003)
    far = ev.analyse(ev.Settings(**C.PRESETS["Start weit weg"]))
    assert far.run.iterations == 43 and far.run.phi[-1] / far.ref.phi[-1] == pytest.approx(20.0, rel=0.05) and far.err == pytest.approx(4.17, abs=0.05) and far.err_ref == pytest.approx(0.084, abs=0.002)
    flat = ev.analyse(ev.Settings(**C.PRESETS["Flache Reaktion (ρ = 4)"]))
    assert flat.err == pytest.approx(0.629, abs=0.005) and flat.err_lin == pytest.approx(0.626, abs=0.005) and flat.flat == pytest.approx(0.135, abs=0.01)


# ----------------------------------------------------------------------------- Experiment 1

def test_adjoint_numbers(adjoint):
    rows = adjoint["rows"]
    assert len(rows) == 9 and 5e-10 < max(max(r["err_fd"], r["err_jac"]) for r in rows) < 9e-10
    assert [r["nf_fd"] for r in rows[:3]] == [48, 96, 192] and all(r["nf_adj"] == 1 for r in rows) and max(r["err_jac"] for r in rows) < 1e-15
    text = ev.verdict_adjoint(adjoint)[1]
    assert "weniger als 1,0·10⁻⁹" in text and "192 Vorwärtsläufe" in text


# ----------------------------------------------------------------------------- Experiment 2

def test_ignore_numbers(ignore):
    rows = {r["rho"]: r for r in ignore["rows"]}
    assert [round(100 * rows[r]["nl"], 1) for r in C.EXP_RHOS] == [7.2, 7.3, 8.0, 15.3, 62.5]
    assert [round(100 * rows[r]["lin"], 1) for r in C.EXP_RHOS] == [7.2, 20.0, 30.1, 45.8, 62.6]
    assert rows[0.0]["lin"] == pytest.approx(rows[0.0]["nl"], abs=1e-4) and all(rows[r]["lin_lam"] > 10.0 * rows[r]["nl_lam"] for r in (0.5, 1.0, 2.0, 4.0))
    assert rows[1.0]["lin"] / rows[1.0]["nl"] == pytest.approx(3.78, abs=0.1)
    d = ignore["disc"]
    assert d["rho"] == 2.0 and d["rms"] == pytest.approx(0.1528, abs=0.002) and d["best"] == pytest.approx(0.1526, abs=0.002) and d["lam_median"] == pytest.approx(0.01) and d["median"] == pytest.approx(0.1525, abs=0.003)
    text = ev.verdict_ignore(ignore)[1]
    assert "7,3 % / 8,0 % / 15,3 %" in text and "20,0 % / 30,1 % / 45,8 %" in text and "62,5 % und 62,6 %" in text and "einen Fehler von 15,3 % (RMS)" in text


# ----------------------------------------------------------------------------- Experiment 3

def test_solver_numbers(solvers):
    rows = {(r["rho"], r["solver"]): r for r in solvers["rows"]}
    for rho in C.SOLVER_RHOS:
        assert not rows[(rho, "gd")]["converged"] and rows[(rho, "gd")]["iterations"] == 500 and all(rows[(rho, s)]["converged"] for s in ("gn_pur", "gn", "lm"))
        assert rows[(rho, "gn")]["iterations"] == rows[(rho, "gn_pur")]["iterations"] and rows[(rho, "gn")]["err"] == pytest.approx(rows[(rho, "gn_pur")]["err"])
        assert rows[(rho, "gd")]["err"] < rows[(rho, "gn")]["err"]                     # der vorzeitige Abbruch regularisiert (Stück 5)
        assert rows[(rho, "lm")]["phi"] == pytest.approx(rows[(rho, "gn")]["phi"], rel=1e-6)
    assert [rows[(rho, "gn")]["iterations"] for rho in C.SOLVER_RHOS] == [5, 6, 8, 9] and [rows[(rho, "lm")]["iterations"] for rho in C.SOLVER_RHOS] == [12, 12, 12, 12]
    assert rows[(1.0, "gd")]["counts"]["forward"] == 998 and rows[(1.0, "gd")]["gnorm_ratio"] == pytest.approx(1.0e-4, rel=0.1) and rows[(1.0, "gn")]["counts"] == {"forward": 7, "jacobian": 6, "adjoint": 7}
    assert [round(100 * rows[(1.0, s)]["err"], 1) for s in ("gd", "gn", "lm")] == [10.2, 23.1, 23.1]


def test_basin_numbers(solvers):
    basin = {(r["rho"], r["amp"]): r for r in solvers["basin"]}
    assert [basin[(2.0, a)]["lm"] for a in C.BASIN_AMPS] == [1, 20, 20, 10, 0, 0, 0] and [basin[(2.0, a)]["gn"] for a in C.BASIN_AMPS] == [1, 20, 20, 10, 0, 0, 0]
    assert [basin[(2.0, a)]["gn_pur"] for a in C.BASIN_AMPS][:2] == [1, 20] and basin[(2.0, 0.5)]["gn_pur"] <= 2 and [basin[(2.0, a)]["gn_pur"] for a in C.BASIN_AMPS][3:] == [0, 0, 0, 0]       # bei 0,5 je nach Plattform 1 oder 2
    assert [basin[(4.0, a)]["lm"] for a in C.BASIN_AMPS] == [1, 20, 20, 5, 0, 0, 0] and [basin[(4.0, a)]["gn"] for a in C.BASIN_AMPS] == [1, 20, 20, 5, 0, 0, 0]
    assert [basin[(4.0, a)]["gn_pur"] for a in C.BASIN_AMPS] == [1, 0, 0, 0, 0, 0, 0] and basin[(2.0, 0.0)]["starts"] == 1 and basin[(2.0, 1.0)]["starts"] == 20
    assert all(basin[(rho, 1.0)]["lm"] <= 10 for rho in (2.0, 4.0))                      # „höchstens die Hälfte“ bei Amplitude 1
    text = ev.verdict_solvers(solvers)[1]
    assert "500 Iterationen (nicht konvergiert)" in text and "Schrittweitensuche 6" in text and "Levenberg-Marquardt 12" in text
    assert "20 von 20 Starts der Amplitude 0,5, von 0 von 20 der Amplitude 1,5" in text and "bei Amplitude 1 von 0 von 20" in text


# ----------------------------------------------------------------------------- Experiment 4

def test_laplace_numbers(laplace):
    rows = {r["rho"]: r for r in laplace["rows"]}
    assert [round(rows[r]["ratio_median"], 2) for r in C.LAP_RHOS] == [1.0, 1.0, 1.01, 0.9] and rows[4.0]["ratio_min"] == pytest.approx(0.62, abs=0.03) and rows[4.0]["ratio_max"] == pytest.approx(0.97, abs=0.03)
    assert rows[4.0]["ratio_min"] < 2.0 / 3.0 and all(0.97 < rows[r]["ratio_min"] and rows[r]["ratio_max"] < 1.06 for r in (0.0, 1.0, 2.0))
    assert [round(rows[r]["err_map"], 3) for r in C.LAP_RHOS] == [0.042, 0.066, 0.167, 0.651] and rows[4.0]["gap_sd"] == pytest.approx(0.49, abs=0.05) and rows[2.0]["gap_sd"] == pytest.approx(0.14, abs=0.03)
    assert [round(rows[r]["cover_laplace"], 2) for r in C.LAP_RHOS] == [0.88, 0.83, 0.75, 0.38] and all(rows[r]["cover_laplace"] == rows[r]["cover_chain"] and rows[r]["cover_laplace"] < 0.95 for r in C.LAP_RHOS)
    assert all(rows[r]["rhat"] < 1.05 and 0.2 < rows[r]["acc"] < 0.3 for r in C.LAP_RHOS) and laplace["chains"] == 20 and laplace["steps"] == 20000
    text = ev.verdict_laplace(laplace)[1]
    assert "1,00 / 1,00 / 1,01 / 0,90" in text and "von 0,62 bis 0,97" in text and "38 %" in text
