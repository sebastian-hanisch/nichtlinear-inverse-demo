"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import nli_constants as C
from nli_evaluation import Analysis, de, err_pct, num_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "ρ": "rho", "Φ": "Phi", "∇": "grad ", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "⁻": "-", "⁰": "0",
            "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    r = a.run
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Nichtlineare inverse Probleme"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Gitterpunkte n = {s.n}, Breite der Unschärfe w = {de(s.w, 3)}, Rauschen {de(100 * s.noise, 4)} % des Effektivwerts, Reaktionsstärke rho = {de(s.rho, 1)}, "
                            f"Daten: {C.DATA_SHORT[s.data]}, Löser: {C.SOLVER_SHORT[s.solver]}, lambda = {sci(s.lam)}, höchstens {s.iters} Iterationen, Startamplitude {de(s.amp, 2)}, "
                            f"Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    c = r.counts
    lines = [
        f"Status: {r.status}, {r.iterations} Iterationen; Vorwärtsläufe {c['forward']}, Jacobi-Matrizen {c['jacobian']}, adjungierte Läufe {c['adjoint']}.",
        f"Phi: Start {sci(r.phi[0])}, Ende {sci(r.phi[-1])} (Nullstart: {sci(a.ref.phi[-1])}); Gradientennorm relativ zum Start: {sci(r.gnorm[-1] / r.gnorm[0])}.",
        f"Fehler der Lösung {err_pct(a.err)}; Levenberg-Marquardt vom Nullstart {err_pct(a.err_ref)}; linearisiertes Modell (rho = 0 angenommen, bester lambda = {sci(a.lam_lin)}) {err_pct(a.err_lin)}.",
        f"Residuum |F(x) - b| = {sci(a.resid)}, Rauschniveau sigma sqrt(n) = {sci(a.disc)}; Prüfung des adjungierten Gradienten: Abweichung {sci(a.grad_err)} "
        f"({a.check_counts[0]} gegen {a.check_counts[1]} Vorwärtsläufe).",
        f"Kondition der linearen Abbildung (rho = 0): {sci(a.cond)}; wahres Rauschen sigma = {sci(a.sigma)}.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head, widths = ["Punkt", "Ort s", "Wahrheit x", "Start", "Lösung", "linear"], [16, 26, 32, 32, 36, 36]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(s.n, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.x[i], 3), de(a.x0[i], 3), num_text(float(r.x[i])), de(a.x_lin[i], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if s.n > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {s.n - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Gauss-Newton, Levenberg-Marquardt und adjungierter Gradient für nichtlineare Inverse Probleme. Quelle: nichtlinear-inverse-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
