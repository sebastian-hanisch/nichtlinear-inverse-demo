# Nichtlineare inverse Probleme: Wann trägt die Linearisierung? – Streamlit-Demo

Neuntes Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo), [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo), [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, eine Wärmeleitung (Gauß-Unschärfe) und weißes Rauschen. Neu ist, dass die Vorwärtsabbildung **nichtlinear** wird. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist; sie dient hier nur als Maßstab.

## Kernfrage

Gesucht ist der **Anfangswert** einer Wärmeleitung mit logistischer Reaktion (vier Schritte, Reaktionsstärke ρ; für ρ = 0 ist es die Unschärfe der Stücke 1 bis 8). Wie löst man so ein Problem (Gradientenabstieg, Gauss-Newton, Levenberg-Marquardt)? Was kostet der **adjungierte Gradient**? Was verschenkt man, wenn man die Nichtlinearität **ignoriert**? Wie weit darf der **Start** vom Minimum entfernt sein? Und trägt die **Laplace-Näherung** des Posteriors noch?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, Gitter, Rauschen):

- **Vorwärtsmodell:** u₀ = x, u_{k+1} = R(A u_k), F(x) = u_K mit K = 4 Schritten. A ist ein Wärmeleitungsschritt mit reflektierenden Rändern (Kosinusbasis, Symbol exp(−(w²/2K)(jπ)²); K Schritte sind genau die Unschärfe der Breite w, die Zeilensummen sind 1), R(u) = u + (ρ/K) u (1 − u) die logistische Reaktion. Für ρ = 0 ist F linear.
- **Daten:** ehrlich aus einem feineren Modell mit 32 Schritten (der Splitting-Fehler ist Modellfehler) oder aus demselben Modell (inverse crime). Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 4,6·10⁻⁴), Kondition der linearen Abbildung 1,8·10⁴.
- **Zielfunktion:** Φ(x) = ‖F(x) − b‖² + λ²‖x‖² (wie in Stück 2).
- **Löser:** Gradientenabstieg mit adjungiertem Gradienten und Armijo-Rückverfolgung, Gauss-Newton ohne und mit Schrittweitensuche, Levenberg-Marquardt mit adaptiver Dämpfung. Abbruch bei ‖∇Φ‖ ≤ 10⁻⁸ ‖∇Φ₀‖.
- **Adjungierter Gradient:** ein Vorwärts- und ein Rückwärtslauf über die K Schritte statt 2n Differenzenquotienten; die Jacobi-Matrix wird für Gauss-Newton explizit aufgestellt (n ≤ 96).

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Start:** Nullstart oder amp · (glatte Zufallsfunktion + 1) mit einer Amplitude zwischen 0,25 und 3; alle vier Löser laufen immer vom selben Start.
- **Maßstab für „dasselbe Minimum“:** Levenberg-Marquardt vom Nullstart; Φ höchstens 0,1 % darüber.
- **Linearisiertes Modell:** dieselben Daten mit der Annahme ρ = 0 (Tikhonov, Stück 2) beim λ mit dem kleinsten Fehler gegen die Wahrheit, also im besten Licht.
- **Experimente** laufen über feste Zufallsströme (SplitMix64); nie ein einzelner Seed.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03), ehrliche Daten, Rauschen 0,1 %, wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py` oder `tests/test_presets.py`.

| Frage | Befund | Test |
|---|---|---|
| Stimmt der adjungierte Gradient, und was kostet er (Experiment 1)? | Abweichung gegen den zentralen Differenzenquotienten unter **10⁻⁹** (gemessen rund 6·10⁻¹⁰, in der letzten Stelle plattformabhängig), gegen Jᵀ·r unter 10⁻¹⁵, für ρ = 0 / 1 / 4 und n = 24 / 48 / 96. **1** Vorwärtslauf (plus ein Rückwärtslauf) gegen **2n = 48 / 96 / 192**. | `test_adjoint_numbers` |
| Was kostet es, die Nichtlinearität zu ignorieren (Experiment 2, 40 Wiederholungen, bester Fehler über λ)? | ρ = 0 / 0,5 / 1 / 2 / 4: **linearisiertes Modell 7,2 / 20,0 / 30,1 / 45,8 / 62,6 %**, nichtlineares Modell (Levenberg-Marquardt) **7,2 / 7,3 / 8,0 / 15,3 / 62,5 %**. Das linearisierte Modell braucht ein mehr als zehnfach größeres λ. Bei ρ = 4 versagen beide: R′(u) = 1 + (ρ/K)(1 − 2u) verschwindet bei u = 1, den Gipfeln. | `test_ignore_numbers` |
| Trifft das Diskrepanzprinzip auch hier (ρ = 2, 100 Wiederholungen)? | RMS-Fehler **15,3 %** gegen das Optimum über λ von 15,3 %, Median-λ 0,01. | `test_ignore_numbers` |
| Welcher Löser (Experiment 3, Nullstart, λ = 10⁻³, ρ = 0,5 / 1 / 2 / 4)? | **Gradientenabstieg:** 500 Iterationen, nicht konvergiert (Gradientennorm erst beim 1,0·10⁻⁴-Fachen bei ρ = 1), 998 Vorwärtsläufe. **Gauss-Newton:** 5 / 6 / 8 / 9 Iterationen (bei ρ = 1: 7 Vorwärtsläufe, 6 Jacobi-Matrizen, 7 adjungierte Läufe). **Levenberg-Marquardt:** 12 Iterationen, dasselbe Minimum. Gauss-Newton mit und ohne Schrittweitensuche stimmen vom Nullstart überein. | `test_solver_numbers` |
| Aber der vorzeitige Abbruch des Gradientenabstiegs ist genauer? | Bei ρ = 1 und λ = 10⁻³: Gradientenabstieg nach 500 Iterationen **10,2 %**, konvergiert (Gauss-Newton, Levenberg-Marquardt) **23,1 %**: wie in Stück 5 regularisiert der Abbruch. | `test_solver_numbers` |
| Wie weit darf der Start liegen (Experiment 3, Anteil von 20 Starts im selben Minimum, ρ = 2)? | Amplitude 0,25 / 0,5 / 1 / 1,5 / 2 / 3: **Levenberg-Marquardt und Gauss-Newton 20 / 20 / 10 / 0 / 0 / 0**; Gauss-Newton ohne Schrittweitensuche 20 / **1 bis 2** (je nach Plattform) / 0 / 0 / 0 / 0. Bei ρ = 4: 20 / 20 / **5** / 0 / 0 / 0, ohne Schrittweitensuche 0 in jeder Zeile. | `test_basin_numbers` |
| Trägt die Laplace-Näherung (Experiment 4, n = 24, Rauschen 1 %, weißer Prior τ = 0,3, 20 Ketten × 20.000 Schritte)? | sd-Verhältnis (Näherung / Kette), Median: ρ = 0 / 1 / 2 / 4: **1,00 / 1,00 / 1,01 / 0,90** (bei ρ = 4 von 0,62 bis 0,97); größter \|MAP − Mittel\| 0,02 / 0,04 / 0,14 / 0,49 sd. Abdeckung der Wahrheit im 95-%-Intervall 88 / 83 / 75 / **38 %**, mit Näherung und Kette gleich. | `test_laplace_numbers` |
| Einzelstichproben (Presets) | **Standard** (ρ = 1, Levenberg-Marquardt, λ = 10⁻²): 7 Iterationen (8 Vorwärtsläufe, 7 Jacobi-Matrizen, 8 adjungierte Läufe), Fehler **8,4 %**, linearisiert **30,1 %** (das 3,6-Fache). **ρ = 2:** 15,6 % gegen 45,8 %. **Gradientenabstieg** (λ = 10⁻³): nach 100 Iterationen 4,5·10⁻⁴ der Gradientennorm. **Gauss-Newton pur** (Start 0,5, ρ = 2): schon der erste Schritt läuft davon, mit Schrittweitensuche 8, mit Dämpfung 17 Iterationen. **Start 1,5:** konvergiert nach 43 Iterationen in ein lokales Minimum mit dem 20-fachen Φ und 417 % Fehler. **ρ = 4:** 62,9 % gegen 62,6 % linearisiert. | `test_standard_preset_numbers`, `test_other_preset_numbers` |
| Mini-Instanz von Hand | 3 × 3-Matrix, ein Schritt, R(u) = 2u − u²: F(1, 2, 1) = (0,91, 0,84, 0,91), J mit den Zeilen (−0,36, −0,18, −0,06), (−0,24, −0,32, −0,24), (−0,06, −0,18, −0,36), Gradient (−0,5838, −0,5964, −0,5838); zwei Schritte F = (0,987679, 0,986076, 0,987679); skalare Gleichung 2x − x² = 0,75: Newton-Schritte 0,44375 und 0,497156. | `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Nichtlinear heißt nicht schwerer, sondern falsch, wenn man linearisiert.** Bei ρ ≤ 2 ist das nichtlineare Problem mit 8,0 und 15,3 % kaum schwerer als das lineare (7,2 %); das linearisierte Modell liegt bei 30,1 und 45,8 %. Erst bei ρ = 4 geht Information verloren (die Ableitung der Reaktion verschwindet an den Gipfeln), und dann hilft kein Löser.
- **Gauss-Newton mit Schrittweitensuche und Levenberg-Marquardt enden im selben Minimum.** Die Gliederung stellte sie als Alternativen dar; die Messung zeigt, dass der Unterschied erst bei Starts außerhalb des Einzugsgebiets zählt, und dort versagen beide. Wichtig ist die Schrittweitensuche oder Dämpfung: der reine Gauss-Newton-Schritt läuft ab Amplitude 0,5 fast immer davon (bei ρ = 2 erreicht ihn 1 von 20 Starts, bei ρ = 4 keiner).
- **Das Einzugsgebiet ist klein.** Ab Amplitude 1 erreicht höchstens die Hälfte der Starts das richtige Minimum, ab 1,5 keiner; das Verfahren ist lokal.
- **Der Wert von Φ am Ende sagt nichts über die Lösung.** In der Vormessung endeten 20 Starts im selben Φ mit Fehlern von 187 bis 647 %, weil das Problem bei λ = 10⁻³ in flachen Richtungen unterbestimmt ist; die App zeigt Φ und Fehler getrennt.
- **Ein Baustein, der die Grundannahme verletzt, fällt vor der Messung auf.** Die erste Fassung nahm für jeden Schritt die Unschärfematrix der Linie mit der Breite w/√K: bei Breiten unter einer Gitterzelle ist die Mittelpunktsregel kein Mittelungsoperator mehr, und die Hintereinanderausführung explodierte (Fehler 10⁸ %). Die Wärmeleitung in der Kosinusbasis ist exakt hintereinander ausführbar.
- **Zwei Fehler der Vormessung beim Bau korrigiert.** Das Armijo-Kriterium der Vormessung für Gauss-Newton hatte das falsche Vorzeichen (es ließ kleine Anstiege zu), und die Schrittweite des Gradientenabstiegs wurde nicht fortgeschrieben; beides ist im Bau korrigiert, die Zahlen oben sind die des Baus.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Start liegt im Einzugsgebiet** | Bei ρ = 2 erreicht Levenberg-Marquardt von Starts der Amplitude 1,5 das richtige Minimum in 0 von 20 Fällen; die Lösung ist ein lokales Minimum. | Globale Optimierung (nicht gebaut) |
| **Die Linearisierung genügt** | Bei ρ = 1 liegt der Fehler des linearisierten Modells beim 3,8-Fachen des nichtlinearen. | – |
| **Die Abbildung ist umkehrbar** | Bei ρ = 4 verschwindet R′ bei u = 1; beide Modelle liegen bei etwa 62,5 % Fehler. | Stück 11 (Auflösung und Versuchsplanung) |
| **Der Posterior ist gaußsch** | Die Laplace-Näherung unterschätzt bei ρ = 4 die Standardabweichung im Median um 10 % (an einzelnen Stellen um bis zu 38 %). | Stück 10 (Ensemble-Kalman), 15 (Diffusions-Prior) |
| **Ein Modell, eine Welt** | Wärmeleitung und logistische Reaktion in vier Schritten (Splitting erster Ordnung), ein Phantom, ein weißer Prior. | Stück 12 (blinde Entfaltung) |
| **Die Jacobi-Matrix darf explizit sein** | Bei n ≤ 96 kostet sie n Vorwärtsläufe; große Probleme brauchen matrixfreie Verfahren, die nicht gebaut sind. | – |
| **λ ist gegeben** | Die Löser minimieren Φ bei festem λ; das Diskrepanzprinzip (Experiment 2) wählt λ nur bei ρ = 2. | Stück 4 (Parameterwahl) |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo): Stück 2; ein Gauss-Newton-Schritt ist für ρ = 0 genau die Tikhonov-Lösung.
- [iterativ-demo](https://github.com/sebastian-hanisch/iterativ-demo): Stück 5; Landweber ist der Gradientenabstieg für ρ = 0, und der Abbruch regularisiert auch hier.
- [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo): Stück 8; das vorkonditionierte Zufallsschritt-Metropolis-Verfahren und die Laplace-Näherung kehren im Experiment 4 wieder.
- [newton-verfahren-demo](https://github.com/sebastian-hanisch/newton-verfahren-demo), [quasi-newton-demo](https://github.com/sebastian-hanisch/quasi-newton-demo): dort steht das Newton-Verfahren ohne Schrittweitensuche und Trust-Region; Gauss-Newton und Levenberg-Marquardt für nichtlineare kleinste Quadrate kommen in der Nichtlineare-Optimierung-Reihe nicht vor.

## Tests

Mini-Instanz von Hand (3 × 3: Wärmeleitung, Reaktion, Vorwärtsmodell, Jacobi-Matrix, adjungierter Gradient, Zielfunktion, Posterior), skalare Gleichung 2x − x² = 0,75 für jeden Löser (Newton-Schritte, Halbierung, Dämpfung, Divergenz, Abbruch, Stillstand), Orakel (adjungierter Gradient und Jacobi-Matrix gegen Differenzenquotienten, Matrixpotenz und Kosinusdarstellung gegen die Gauß-Unschärfe des Generators, Gauss-Newton gleich Tikhonov im linearen Fall, Splitting erster Ordnung, Löser gegeneinander, Stationarität und positiv definite Hesse-Matrix, Laplace-Näherung gegen den exakten linearen Posterior und gegen Ketten, Nichtinjektivität), Generator, Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Regler-Grenzen, jeder Löser und jede Datenart, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, Gitter) |
| `nli_model.py` | Wärmeleitung, Vorwärtsmodell, Jacobi-Matrix, adjungierter Gradient, Löser, Laplace-Näherung, Sampler |
| `nli_evaluation.py` | Instanz, Analyse, Experimente, Verdikte |
| `nli_visualization.py`, `nli_pdf_export.py` | Figuren, PDF |
| `nli_constants.py`, `nli_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Matrixfreie Verfahren (Krylov-Löser, inexaktes Newton):** die Jacobi-Matrix ist hier explizit, was bei n ≤ 96 reicht.
- **Trust-Region-Verfahren und Quasi-Newton für kleinste Quadrate:** Levenberg-Marquardt deckt den Dämpfungsgedanken ab.
- **Globale Optimierung und Mehrfachstarts:** das Einzugsgebiet wird gezeigt, nicht beseitigt.
- **Regularisierung durch Abbruch mit Diskrepanzprinzip im Löser:** nur als Experiment bei ρ = 2.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
