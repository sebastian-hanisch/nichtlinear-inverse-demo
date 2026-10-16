"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_option_of_every_select_slider_does_not_raise():
    import nli_constants as C

    for key, options in (("rho_select", C.RHO_CHOICES), ("noise_select", C.NOISE_CHOICES), ("lam_select", C.LAM_CHOICES), ("iters_select", C.ITER_CHOICES), ("amp_select", C.AMP_CHOICES)):
        for value in options:
            at = _fresh_app()
            at.select_slider(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_solver_and_data_kind_does_not_raise():
    import nli_constants as C

    for solver in C.SOLVERS:
        for data in C.DATA_KINDS:
            at = _fresh_app()
            at.selectbox(key="solver_select").set_value(solver)
            at.selectbox(key="data_select").set_value(data)
            at.select_slider(key="amp_select").set_value(1.0)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (solver, data, [str(e) for e in at.exception])


def test_the_most_singular_corner_does_not_raise_and_warns():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(96)
    at.slider(key="w_slider").set_value(0.05)
    at.select_slider(key="noise_select").set_value(1e-4)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("numerisch singulär" in w.value for w in at.warning)


def test_every_preset_is_applied_to_the_widgets():
    import nli_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert abs(at.slider(key="w_slider").value - preset["w"]) < 1e-9
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert at.select_slider(key="rho_select").value == preset["rho"]
        assert at.selectbox(key="data_select").value == preset["data"]
        assert at.selectbox(key="solver_select").value == preset["solver"]
        assert at.select_slider(key="lam_select").value == preset["lam"]
        assert at.select_slider(key="iters_select").value == preset["iters"]
        assert at.select_slider(key="amp_select").value == preset["amp"]
        assert at.number_input(key="seed_input").value == preset["seed"]


def test_the_standard_instance_shows_the_success_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("hat konvergiert und liegt im selben Minimum" in s.value for s in at.success)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_the_failing_presets_show_a_warning_and_the_diverged_note():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, text in (("Gradientenabstieg", "nicht konvergiert"), ("Gauss-Newton ohne Schrittweitensuche", "davongelaufen"), ("Start weit weg", "Lokales Minimum"),
                       ("Flache Reaktion (ρ = 4)", "Beide Modelle versagen")):
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert any(text in w.value for w in at.warning), name
    at = _fresh_app()
    at.button[labels["Gauss-Newton ohne Schrittweitensuche"]].click().run(timeout=TIMEOUT)
    assert any("davongelaufen" in i.value for i in at.info)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_a_diverged_instance_is_created():
    at = _fresh_app()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Gauss-Newton ohne Schrittweitensuche"]].click().run(timeout=TIMEOUT)
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999999"
    at.query_params["w"] = "-3"
    at.query_params["r"] = "0.5"
    at.query_params["rho"] = "7"
    at.query_params["data"] = "zufall"
    at.query_params["solver"] = "bfgs"
    at.query_params["lam"] = "0.5"
    at.query_params["iters"] = "7"
    at.query_params["amp"] = "9"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 96
    assert abs(at.slider(key="w_slider").value - 0.01) < 1e-9
    assert at.select_slider(key="noise_select").value == 1e-3
    assert at.select_slider(key="rho_select").value == 1.0
    assert at.selectbox(key="data_select").value == "honest"
    assert at.selectbox(key="solver_select").value == "lm"
    assert at.select_slider(key="lam_select").value == 1e-2
    assert at.select_slider(key="iters_select").value == 100
    assert at.select_slider(key="amp_select").value == 0.0


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="solver_select").set_value("gd").run(timeout=TIMEOUT)
    assert at.query_params["solver"] == ["gd"] or at.query_params["solver"] == "gd"
