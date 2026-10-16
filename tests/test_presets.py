"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import nli_constants as C
import nli_evaluation as ev
import nli_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["rho"] in C.RHO_CHOICES and p["data"] in C.DATA_KINDS and p["solver"] in C.SOLVERS, name
        assert p["lam"] in C.LAM_CHOICES and p["iters"] in C.ITER_CHOICES and p["amp"] in C.AMP_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_the_documented_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, strong, gd, gn_pur, far, flat = (res[k] for k in C.PRESETS)
    assert std.run.converged and std.same_min and std.err == pytest.approx(0.084, abs=0.005) and std.err_lin == pytest.approx(0.301, abs=0.01) and ev.verdict_instance(std)[0] == "success"
    assert strong.run.converged and strong.err == pytest.approx(0.156, abs=0.01) and strong.err_lin == pytest.approx(0.458, abs=0.01) and ev.verdict_instance(strong)[0] == "success"
    assert gd.run.status == "maxit" and gd.run.gnorm[-1] / gd.run.gnorm[0] == pytest.approx(4.5e-4, rel=0.2) and gd.runs["lm"].converged and ev.verdict_instance(gd)[0] == "warning"
    assert gn_pur.run.status == "diverged" and gn_pur.run.iterations == 0 and gn_pur.runs["gn"].converged and gn_pur.runs["lm"].converged and ev.verdict_instance(gn_pur)[0] == "warning"
    assert far.run.converged and not far.same_min and far.err > 1.0 and far.run.phi[-1] > 10.0 * far.ref.phi[-1] and ev.verdict_instance(far)[0] == "warning"
    assert flat.run.converged and flat.err > C.FAIL_ERR and flat.err_lin > C.FAIL_ERR and flat.flat > 0.1 and "Beide Modelle versagen" in ev.verdict_instance(flat)[1]


def test_experiment_constants_are_consistent():
    assert C.GN_TOL == 1e-8 and C.ARMIJO_C == 1e-4 and C.K_TRUE == 8 * C.K_INV and C.LAP_N == 24 and C.LAP_CHAINS >= 20 and C.BASIN_STARTS == 20 and C.BASIN_AMPS[0] == 0.0
    assert C.EXP_LAMS[0] == pytest.approx(1e-4) and C.EXP_LAMS[-1] == pytest.approx(10 ** 0.5) and len(C.EXP_LAMS) == 14 and C.EXP_DISC_RHO in C.EXP_RHOS and C.MU_DOWN < 1.0 < C.MU_UP
    assert 0 < C.BURN_FRACTION < 0.5 and set(C.PRESETS["Standard"]) == set(P.PRESET_KEYS) and C.DEFAULT_RHO in C.RHO_CHOICES and C.DEFAULT_LAM in C.LAM_CHOICES and C.DEFAULT_ITER in C.ITER_CHOICES
