"""Nichtlineare inverse Probleme: Gauss-Newton, Levenberg-Marquardt und der adjungierte Gradient - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 8 (Phantom, Gauß-Unschärfe, Rauschen) wird die Unschärfe zu einem nichtlinearen
Vorwärtsmodell: Wärmeleitung mit logistischer Reaktion in vier Schritten. Gesucht ist der Anfangswert. Gezeigt werden der adjungierte Gradient, Gauss-Newton und
Levenberg-Marquardt gegen den Gradientenabstieg, der Preis der Linearisierung, das Einzugsgebiet des Startpunkts und die Laplace-Näherung des Posteriors.
Kern: nli_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import nli_constants as C
from nli_evaluation import (
    Settings,
    adjoint_experiment,
    analyse,
    count_text,
    de,
    err_pct,
    ignore_experiment,
    laplace_experiment,
    noise_label,
    num_text,
    rho_label,
    sci,
    share,
    solver_experiment,
    verdict_adjoint,
    verdict_ignore,
    verdict_instance,
    verdict_laplace,
    verdict_solvers,
)
from nli_pdf_export import build_pdf
from nli_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from nli_visualization import (
    build_adjoint_chart,
    build_basin_chart,
    build_grad_chart,
    build_gradcheck_chart,
    build_ignore_chart,
    build_laplace_chart,
    build_phi_chart,
    build_recon_chart,
    build_solver_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Nichtlineare inverse Probleme – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _adjoint():
    return adjoint_experiment()


@st.cache_data(show_spinner=SPINNER)
def _ignore():
    return ignore_experiment()


@st.cache_data(show_spinner=SPINNER)
def _solvers():
    return solver_experiment()


@st.cache_data(show_spinner=SPINNER)
def _laplace():
    return laplace_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🔥 Nichtlineare inverse Probleme: Wann trägt die Linearisierung?")
st.markdown(
    """
**Die Nichtlinearität zu ignorieren kostet ein Vielfaches des Fehlers, sie zu berücksichtigen nur einen Rückwärtslauf je Gradient.** Dies ist das neunte Stück der Linie
**Inverse Probleme und Regularisierung** und baut auf **iterativer Regularisierung** und **Posterior-Sampling** auf: dasselbe Phantom, dieselbe Unschärfe, dasselbe Rauschen, aber die
Vorwärtsabbildung ist jetzt nichtlinear. Gesucht ist der **Anfangswert** einer Wärmeleitung mit logistischer Reaktion (vier Schritte, Reaktionsstärke ρ). Für ρ = 0 ist es die Unschärfe der
früheren Stücke. Die Frage ist, **wie man das Problem löst** (Gradientenabstieg, Gauss-Newton, Levenberg-Marquardt), **was der adjungierte Gradient kostet**, **was die Linearisierung
verschenkt**, **wo der Startpunkt liegen darf** und ob die **Laplace-Näherung** den Posterior noch trägt. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie man ein nichtlineares Problem löst", expanded=True):
    st.markdown(
        r"""
1. **Modell.** $F(x) = u_K$ mit $u_0 = x$ und $u_{k+1} = R(A u_k)$: erst ein Wärmeleitungsschritt $A$, dann die Reaktion $R(u) = u + \tfrac{\rho}{K}u(1-u)$. Gesucht ist $x$ aus $b \approx F(x)$.
2. **Zielfunktion.** $\Phi(x) = \|F(x) - b\|^2 + \lambda^2\|x\|^2$, wie in Stück 2, nur dass $F$ nicht mehr linear ist.
3. **Adjungierter Gradient.** Der Gradient von $\tfrac12\|F(x)-b\|^2$ kommt aus einem Vorwärtslauf und einem Rückwärtslauf über die $K$ Schritte, nicht aus $n$ Differenzenquotienten.
4. **Gauss-Newton.** Ersetze $F$ durch die Linearisierung $F(x) + J\,d$ und löse das lineare Tikhonov-Problem für den Schritt $d$. **Levenberg-Marquardt** dämpft den Schritt, wenn er $\Phi$ nicht senkt.
5. **Start.** Alle diese Verfahren sind lokal: sie finden ein Minimum in der Nähe des Starts. Im Experiment 3 liegt dieses Gebiet offen vor.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt und Messung**")
    rho = st.select_slider("Reaktionsstärke ρ", options=C.RHO_CHOICES, key="rho_select", format_func=rho_label,
                           help="0 ist das lineare Modell (Unschärfe wie in Stück 1 bis 8); je größer, desto nichtlinearer. Bei 4 verschwindet die Ableitung der Reaktion bei u = 1.")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_KINDS, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrlich: aus einem feineren Modell mit 32 Schritten (der Splitting-Fehler ist Modellfehler). Inverse crime: aus demselben Modell, das invertiert wird.")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="Wie viele Messpunkte und Unbekannte; die Jacobi-Matrix hat n × n Einträge.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite der Wärmeleitung nach vier Schritten, als Anteil der Länge 1. Je breiter, desto schlechter gestellt.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Löser**")
    solver = st.selectbox("Welcher Löser?", C.SOLVERS, key="solver_select", format_func=lambda v: C.SOLVER_LABELS[v],
                          help="Alle vier laufen immer vom selben Start; der gewählte wird hervorgehoben und liefert die Lösung oben.")
    lam = st.select_slider("Regularisierung λ", options=C.LAM_CHOICES, key="lam_select", format_func=lambda v: sci(v, 1),
                           help="Gewicht des Tikhonov-Terms in Φ(x) = |F(x) − b|² + λ²|x|²; zu klein: Rauschen wird verstärkt, zu groß: die Lösung wird zu glatt.")
    iters = st.select_slider("Iterationen (höchstens)", options=C.ITER_CHOICES, key="iters_select", help="Obergrenze für alle vier Löser; Abbruch früher, wenn die Gradientennorm auf 10⁻⁸ des Anfangswerts fällt.")
    amp = st.select_slider("Amplitude des Starts", options=C.AMP_CHOICES, key="amp_select", format_func=lambda v: de(v, 2),
                           help="0 ist der Nullstart; sonst amp · (glatte Zufallsfunktion + 1), eine Funktion der Größenordnung amp. Je größer, desto weiter vom Minimum.")
    seed = st.number_input("Zufalls-Seed des Rauschens und des Starts", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "rho_select": float(rho), "data_select": data, "solver_select": solver,
                   "lam_select": float(lam), "iters_select": int(iters), "amp_select": float(amp), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), float(rho), data, solver, float(lam), int(iters), float(amp), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.singular:
    st.warning(f"Bei dieser Wahl von n und w ist die lineare Abbildung in doppelter Genauigkeit numerisch singulär (Kondition {sci(a.cond)}): Die kleinen Singulärwerte sind "
               "Rundungsrauschen, und die Zahlen unten sind nur noch Zahlen.")
if a.run.status == "diverged":
    st.info("Der Löser ist beim ersten Schritt davongelaufen und wurde angehalten; angezeigt ist der Start.")
r = a.run
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fehler der Lösung", err_pct(a.err), delta=f"Nullstart: {err_pct(a.err_ref)}", delta_color="off", help="Abstand zur Wahrheit relativ zur Signalnorm; der Nullstart (Levenberg-Marquardt) ist der Maßstab.")
c2.metric("Linearisiertes Modell", err_pct(a.err_lin), delta=f"bester λ = {sci(a.lam_lin)}", delta_color="off",
          help="Dieselben Daten mit der Annahme ρ = 0 (Tikhonov, Stück 2); λ ist gegen die Wahrheit gewählt, das Modell steht also im besten Licht.")
c3.metric("Iterationen", f"{r.iterations}", delta=f"{r.counts['forward']} Vorwärts, {r.counts['jacobian']} Jacobi, {r.counts['adjoint']} adjungiert", delta_color="off",
          help="Aufwand des gewählten Lösers: Vorwärtsläufe (inklusive Schrittweitensuche), Jacobi-Matrizen und adjungierte Läufe.")
c4.metric("Gradientennorm", sci(r.gnorm[-1] / r.gnorm[0]), delta="Soll: unter 1,0·10⁻⁸" if not r.converged else "erreicht", delta_color="off",
          help="|∇Φ| am Ende relativ zum Start; der Abbruch liegt bei 10⁻⁸.")
st.caption(f"Φ: Start {sci(r.phi[0])}, Ende {sci(r.phi[-1])}; Residuum |F(x) − b| = {sci(a.resid)} bei Rauschniveau σ√n = {sci(a.disc)}.")

tab_rec, tab_conv, tab_grad, tab_tab = st.tabs(["Rekonstruktion", "Konvergenz", "Gradientenprüfung", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, blau die Lösung des gewählten Lösers, orange gestrichelt das linearisierte Modell beim besten λ; bei einem Start ab Amplitude 0,25 zusätzlich der Start "
               "(grau) und die Lösung des Nullstarts (schwarz punktiert). Die Achse ist fest, Lösungen außerhalb liegen außerhalb des Bildes.")
with tab_conv:
    st.plotly_chart(build_phi_chart(a), width="stretch", key="phi_chart")
    st.plotly_chart(build_grad_chart(a), width="stretch", key="grad_chart")
    st.caption("Alle vier Löser vom selben Start; der gewählte ist dick. Der Gradientenabstieg fällt langsam, weil das Problem schlecht konditioniert ist; Gauss-Newton ohne Schrittweitensuche "
               "kann beim ersten Schritt davonlaufen (dann bleibt nur der Anfangspunkt).")
    st.dataframe(
        [{"Löser": C.SOLVER_SHORT[m], "Status": {"converged": "konvergiert", "maxit": "Obergrenze erreicht", "diverged": "davongelaufen", "stalled": "keine Schrittweite"}[x.status],
          "Iterationen": x.iterations, "Vorwärtsläufe": x.counts["forward"], "Jacobi-Matrizen": x.counts["jacobian"], "adjungierte Läufe": x.counts["adjoint"], "Φ am Ende": sci(x.phi[-1]),
          "Fehler": err_pct(a.errs[m])} for m, x in a.runs.items()], hide_index=True, width="stretch")
with tab_grad:
    st.plotly_chart(build_gradcheck_chart(a), width="stretch", key="gradcheck_chart")
    st.caption(f"Am Prüfpunkt 0,8 · Wahrheit + 0,05 weicht der adjungierte Gradient um {sci(a.grad_err)} (relativ) vom zentralen Differenzenquotienten ab; der adjungierte Gradient braucht "
               f"{a.check_counts[0]} Vorwärtslauf (und einen Rückwärtslauf), der Differenzenquotient {count_text(a.check_counts[1])}.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="nichtlineare-inverse-probleme.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist die Nichtlinearität zu ignorieren?")
verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht den Fehler des Lösers mit dem des Nullstarts und dem des linearisierten Modells; ein nicht konvergierter Löser oder ein lokales Minimum steht in dem Satz.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Der adjungierte Gradient")
res1 = _adjoint()
st.caption("Gradient von ½|F(x) − b|² am Punkt 0,8 · Wahrheit + 0,05, für drei Reaktionsstärken und drei Gittergrößen: adjungiert gegen zentralen Differenzenquotienten (Schritt 10⁻⁶) und "
           "gegen Jᵀ·r mit der expliziten Jacobi-Matrix.")
st.plotly_chart(build_adjoint_chart(res1), width="stretch", key="adjoint_chart")
verdict(verdict_adjoint(res1))
st.dataframe(
    [{"ρ": rho_label(r_["rho"]), "n": r_["n"], "Abweichung gegen Differenzenquotient": sci(r_["err_fd"]), "Abweichung gegen Jᵀ·r": sci(r_["err_jac"]),
      "Vorwärtsläufe adjungiert": r_["nf_adj"], "Vorwärtsläufe Differenzen": count_text(r_["nf_fd"])} for r_ in res1["rows"]], hide_index=True, width="stretch")
st.caption("Der Differenzenquotient braucht 2n Vorwärtsläufe, der adjungierte Gradient einen Vorwärts- und einen Rückwärtslauf, unabhängig von n. Die Abweichung gegen den Differenzenquotienten "
           "ist sein Abbruch- und Rundungsfehler, die gegen Jᵀ·r liegt bei Maschinengenauigkeit.")

st.subheader("🔬 Experiment 2: Was kostet es, die Nichtlinearität zu ignorieren?")
res2 = _ignore()
st.caption(f"Ehrliche Daten, Rauschen 0,1 %, {res2['reps']} Wiederholungen je Reaktionsstärke, 14 Werte von λ zwischen 10⁻⁴ und 3: der beste RMS-Fehler über λ, linearisiertes Modell "
           "(ρ = 0 angenommen, Tikhonov) gegen das nichtlineare Modell (Levenberg-Marquardt vom Nullstart).")
st.plotly_chart(build_ignore_chart(res2), width="stretch", key="ignore_chart")
verdict(verdict_ignore(res2))
st.dataframe(
    [{"ρ": rho_label(r_["rho"]), "linearisiertes Modell": err_pct(r_["lin"]), "bester λ (linear)": sci(r_["lin_lam"]), "nichtlineares Modell": err_pct(r_["nl"]),
      "bester λ (nichtlinear)": sci(r_["nl_lam"])} for r_ in res2["rows"]], hide_index=True, width="stretch")
st.caption("Das linearisierte Modell braucht ein viel größeres λ, weil es den Modellfehler als Rauschen behandelt, und bleibt trotzdem weit zurück. Bei ρ = 0 sind beide Modelle dasselbe. "
           "Bei ρ = 4 versagen beide: die Ableitung der Reaktion verschwindet bei u = 1, den Gipfeln des Phantoms.")

st.subheader("🔬 Experiment 3: Löser und Einzugsgebiet")
res3 = _solvers()
st.caption(f"Teil 1: vier Löser vom Nullstart, λ = 10⁻³, ehrliche Daten, bis zur Gradientennorm 10⁻⁸ oder höchstens 100 Iterationen (Gradientenabstieg {C.SOLVER_GD_ITERS}). Teil 2: "
           f"Anteil von {C.BASIN_STARTS} zufälligen Starts je Amplitude, die dasselbe Minimum erreichen wie der Nullstart (Φ höchstens 0,1 % darüber).")
st.plotly_chart(build_solver_chart(res3), width="stretch", key="solver_chart")
st.caption("Schraffiert: nicht konvergiert (Obergrenze erreicht).")
verdict(verdict_solvers(res3))
st.dataframe(
    [{"ρ": rho_label(r_["rho"]), "Löser": C.SOLVER_SHORT[r_["solver"]], "Iterationen": r_["iterations"], "konvergiert": "ja" if r_["converged"] else "nein",
      "Vorwärtsläufe": count_text(r_["counts"]["forward"]), "Jacobi-Matrizen": r_["counts"]["jacobian"], "adjungierte Läufe": count_text(r_["counts"]["adjoint"]), "Φ am Ende": sci(r_["phi"]),
      "Fehler": err_pct(r_["err"])} for r_ in res3["rows"]], hide_index=True, width="stretch", height=360)
st.caption("Mit λ = 10⁻³ ist der konvergierte Fehler größer als der des vorzeitig abgebrochenen Gradientenabstiegs: wie in Stück 5 regularisiert der Abbruch. Gauss-Newton mit und ohne "
           "Schrittweitensuche stimmen vom Nullstart überein (der volle Schritt wird angenommen); sie trennen sich erst bei weiteren Starts.")
cb1, cb2 = st.columns(2)
with cb1:
    st.plotly_chart(build_basin_chart(res3, C.BASIN_RHOS[0]), width="stretch", key="basin_chart_1")
with cb2:
    st.plotly_chart(build_basin_chart(res3, C.BASIN_RHOS[1]), width="stretch", key="basin_chart_2")
st.dataframe(
    [{"ρ": rho_label(r_["rho"]), "Amplitude": de(r_["amp"], 2), "Starts": r_["starts"], "Levenberg-Marquardt": f"{r_['lm']} von {r_['starts']}", "Gauss-Newton": f"{r_['gn']} von {r_['starts']}",
      "Gauss-Newton pur": f"{r_['gn_pur']} von {r_['starts']}"} for r_ in res3["basin"]], hide_index=True, width="stretch", height=360)
st.caption("Levenberg-Marquardt und Gauss-Newton mit Schrittweitensuche liegen in beiden Bildern übereinander. "
           "Das Einzugsgebiet ist klein: schon bei Amplitude 1 landet höchstens die Hälfte der Starts im richtigen Minimum, ab 1,5 keiner. Ein Verfahren mit globaler Garantie ist das nicht; "
           "ein Nullstart oder eine grobe lineare Lösung als Start ist die praktische Regel.")

st.subheader("🔬 Experiment 4: Trägt die Laplace-Näherung?")
res4 = _laplace()
st.caption(f"Weißer Gauß-Prior (τ = {C.LAP_TAU}), ehrliche Daten bei Rauschen {noise_label(C.LAP_NOISE)}, n = {res4['n']}: Gaußsche Näherung am MAP mit der Gauss-Newton-Hesse-Matrix JᵀJ/σ² + I/τ² "
           f"gegen {res4['chains']} Ketten à {res4['steps']} Schritte (Zufallsschritt-Metropolis mit der Kovarianz der Näherung, überdispers gestartet, Verfahren aus Stück 8).")
st.plotly_chart(build_laplace_chart(res4), width="stretch", key="laplace_chart")
verdict(verdict_laplace(res4))
st.dataframe(
    [{"ρ": rho_label(r_["rho"]), "Annahme": de(r_["acc"], 2), "Fehler MAP": err_pct(r_["err_map"]), "Fehler Kettenmittel": err_pct(r_["err_mean"]), "sd-Verhältnis (Median)": de(r_["ratio_median"], 2),
      "min bis max": f"{de(r_['ratio_min'], 2)} bis {de(r_['ratio_max'], 2)}", "größter |MAP − Mittel| (sd)": de(r_["gap_sd"], 2), "Abdeckung Laplace": share(r_["cover_laplace"]),
      "Abdeckung Kette": share(r_["cover_chain"]), "R̂": de(r_["rhat"], 2)} for r_ in res4["rows"]], hide_index=True, width="stretch")
st.caption("Bis ρ = 2 liegen Näherung und Kette beieinander (Verhältnis 1,0): der Posterior ist fast gaußsch. Bei ρ = 4 wird er schief, die Näherung unterschätzt die Standardabweichung an manchen "
           "Stellen um mehr als ein Drittel. Die Abdeckung der Wahrheit bleibt unter dem Nennwert, weil die Wahrheit keine Prior-Ziehung ist (Stück 7).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
b_rows = {(r_["rho"], r_["amp"]): r_ for r_ in res3["basin"]}
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Start liegt im Einzugsgebiet** | Bei ρ = 2 erreicht Levenberg-Marquardt von Starts der Amplitude 1,5 das richtige Minimum in {b_rows[(2.0, 1.5)]['lm']} von {b_rows[(2.0, 1.5)]['starts']} Fällen; die Lösung ist dann ein lokales Minimum. | Globale Optimierung (nicht gebaut) |
| **Die Linearisierung genügt** | Bei ρ = 1 liegt der Fehler des linearisierten Modells beim {de(next(r_ for r_ in res2['rows'] if r_['rho'] == 1.0)['lin'] / next(r_ for r_ in res2['rows'] if r_['rho'] == 1.0)['nl'], 1)}-Fachen des nichtlinearen. | – |
| **Die Abbildung ist umkehrbar** | Bei ρ = 4 verschwindet R′ bei u = 1; beide Modelle liegen bei {err_pct(next(r_ for r_ in res2['rows'] if r_['rho'] == 4.0)['nl'])} Fehler. | Stück 11 (Auflösung und Versuchsplanung) |
| **Der Posterior ist gaußsch** | Die Laplace-Näherung unterschätzt bei ρ = 4 die Standardabweichung im Median um {share(1.0 - res4['rows'][-1]['ratio_median'])}. | Stück 10 (Ensemble-Kalman), 15 (Diffusions-Prior) |
| **Ein Modell, eine Welt** | Wärmeleitung und logistische Reaktion in vier Schritten (Splitting erster Ordnung), ein Phantom, ein weißer Prior; die Zahlen gelten für dieses Beispiel. | Stück 12 (blinde Entfaltung) |
| **Die Jacobi-Matrix darf explizit sein** | Bei n ≤ 96 kostet sie n Vorwärtsläufe; für große Probleme braucht man matrixfreie Verfahren (Krylov-Löser), die nicht gebaut sind. | – |
| **λ ist gegeben** | Die Löser minimieren Φ zu festem λ; das Diskrepanzprinzip (Experiment 2) wählt λ nur bei ρ = 2. | Stück 4 (Parameterwahl) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $u_0 = x$, $u_{k+1} = R(A u_k)$ für $k = 0, \dots, K-1$, $F(x) = u_K$ mit $R(u) = u + \tfrac{\rho}{K}\,u(1-u)$ und der Wärmeleitungsmatrix $A$ (reflektierende Ränder, in der Kosinusbasis
mit dem Symbol $e^{-(w^2/2K)(j\pi)^2}$): $K$ Schritte mit $\rho = 0$ ergeben genau die Gauß-Unschärfe der Breite $w$. Daten $b = F_{\mathrm{fein}}(x^\ast) + e$ mit $K = 32$ Schritten (ehrlich) oder $K = 4$.

**Jacobi-Matrix.** $J = D_K A\, D_{K-1} A \cdots D_1 A$ mit $D_k = \mathrm{diag}\,R'(A u_{k-1})$, $R'(u) = 1 + \tfrac{\rho}{K}(1 - 2u)$.

**Adjungierter Gradient.** Für $f(x) = \tfrac12\|F(x) - b\|^2$ ist $\nabla f = J^\top (F(x) - b)$. Der Rückwärtslauf $\mu_K = F(x) - b$, $\mu_k = A^\top\bigl(R'(A u_k) \odot \mu_{k+1}\bigr)$
liefert $\nabla f = \mu_0$ ohne $J$ aufzustellen.

**Gauss-Newton.** Für $\Phi(x) = \|F(x) - b\|^2 + \lambda^2\|x\|^2$ löst der Schritt $d$ das System $(J^\top J + \lambda^2 I)\,d = -(J^\top r + \lambda^2 x)$ mit $r = F(x) - b$. Mit Schrittweitensuche
(Armijo): $\Phi(x + \alpha d) \le \Phi(x) + 10^{-4}\,\alpha\,\nabla\Phi^\top d$. **Levenberg-Marquardt:** $(J^\top J + \lambda^2 I + \mu I)\,d = -(J^\top r + \lambda^2 x)$; $\mu$ sinkt auf das 0,3-Fache nach einem
Schritt, der $\Phi$ senkt, und steigt auf das Vierfache sonst. Für $\rho = 0$ ist ein Gauss-Newton-Schritt die Tikhonov-Lösung (Stück 2).

**Laplace-Näherung.** $\pi(x \mid b) \approx \mathcal N(x_{\mathrm{MAP}}, H^{-1})$ mit $H = J^\top J/\sigma^2 + I/\tau^2$ (Gauss-Newton-Hesse-Matrix am MAP, weißer Prior $\mathcal N(0, \tau^2 I)$); im linearen
Fall exakt.

**Literatur.** Das Beispiel ist erfunden; Gauss-Newton und Levenberg-Marquardt sind in J. Nocedal und S. J. Wright, *Numerical Optimization* (2. Aufl. 2006, Kapitel 10) dargestellt, die adjungierte Methode
bei R.-E. Plessix (2006) und M. Hinze u. a. (2009), die Regularisierung nichtlinearer inverser Probleme bei H. W. Engl, M. Hanke und A. Neubauer (1996) und die Laplace-Näherung für inverse Probleme bei
T. Bui-Thanh, O. Ghattas, J. Martin und G. Stadler (2013) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
