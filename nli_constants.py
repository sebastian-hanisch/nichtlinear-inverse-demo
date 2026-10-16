"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand: eine Instanz löst alle vier Löser (Jacobi-Matrix explizit, n höchstens 96), rund eine Sekunde; die Experimente nutzen feste Ströme.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1), nach K Schritten
NOISE_CHOICES = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)                   # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
RHO_CHOICES = (0.0, 0.5, 1.0, 2.0, 4.0)                          # Reaktionsstärke: 0 ist das lineare Modell der Stücke 1 bis 8
DEFAULT_RHO = 1.0
DATA_KINDS = ("honest", "crime")
DATA_LABELS = {"honest": "Ehrlich (feineres Modell mit 32 Schritten)", "crime": "Aus demselben Modell (inverse crime, 4 Schritte)"}
DATA_SHORT = {"honest": "ehrlich", "crime": "inverse crime"}
SOLVERS = ("gd", "gn_pur", "gn", "lm")
SOLVER_LABELS = {"gd": "Gradientenabstieg (adjungierter Gradient, Armijo)", "gn_pur": "Gauss-Newton ohne Schrittweitensuche",
                 "gn": "Gauss-Newton mit Schrittweitensuche (Armijo)", "lm": "Levenberg-Marquardt (adaptive Dämpfung)"}
SOLVER_SHORT = {"gd": "Gradientenabstieg", "gn_pur": "Gauss-Newton pur", "gn": "Gauss-Newton", "lm": "Levenberg-Marquardt"}
DEFAULT_SOLVER = "lm"
LAM_CHOICES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)   # Regularisierungsparameter λ in Φ = |F(x) - b|² + λ²|x|²
DEFAULT_LAM = 1e-2
ITER_CHOICES = (5, 10, 30, 100, 300, 500)
DEFAULT_ITER = 100
AMP_CHOICES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)               # Amplitude des Starts: 0 ist der Nullstart, sonst amp · (glatte Zufallsfunktion + 1)
DEFAULT_AMP = 0.0
SEED_MAX, DEFAULT_SEED = 999999, 11

# feste Modellannahmen
K_INV = 4                                                        # Schritte (Wärmeleitung, dann Reaktion) im Modell, das invertiert wird
K_TRUE = 32                                                      # Schritte im feineren Modell, das die ehrlichen Daten erzeugt (Splitting-Fehler als Modellfehler)
GN_TOL = 1e-8                                                    # Abbruch: |grad Φ| <= GN_TOL |grad Φ_0|
ARMIJO_C = 1e-4
MU_START, MU_DOWN, MU_UP, MU_MIN = 1e-2, 0.3, 4.0, 1e-12         # Levenberg-Marquardt: Dämpfung
LS_MAX = 60                                                      # Halbierungen der Schrittweitensuche höchstens
LM_TRIES = 30
DIVERGE_FACTOR = 1e12                                            # Gauss-Newton pur: Abbruch, wenn |grad Φ| auf das Billionenfache steigt
SINGULAR_COND = 1e15
SHOW_RANGE = (-1.5, 2.5)
CHECK_SHIFT, CHECK_SCALE = 0.05, 0.8                             # Prüfpunkt des Gradienten: 0,8 · Wahrheit + 0,05
FD_STEP = 1e-6

# Verdikt-Schwellen
LIN_FACTOR_OK = 1.25          # das lineare Modell ist ignorierbar, wenn sein Fehler höchstens das 1,25-Fache des nichtlinearen ist
FAIL_ERR = 0.5                # Fehler über der Hälfte der Signalnorm: nicht brauchbar
MIN_OK = 1e-3                 # Φ höchstens um dieses Verhältnis über dem Wert des Nullstarts: dasselbe Minimum

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 990000
EXP_RHOS = (0.0, 0.5, 1.0, 2.0, 4.0)                             # Experiment 2
EXP_REPS = 40
EXP_LAMS = tuple(10.0 ** (-4.0 + 4.5 * k / 13.0) for k in range(14))   # 1e-4 bis 3,2
EXP_DISC_REPS, EXP_DISC_RHO = 100, 2.0
EXP_DISC_LAMS = tuple(10.0 ** (-4.0 + 3.5 * k / 14.0) for k in range(15))   # 1e-4 bis 0,32
ADJ_RHOS, ADJ_NS = (0.0, 1.0, 4.0), (24, 48, 96)                 # Experiment 1
SOLVER_RHOS, SOLVER_LAM, SOLVER_GD_ITERS = (0.5, 1.0, 2.0, 4.0), 1e-3, 500   # Experiment 3, Teil 1
BASIN_RHOS = (2.0, 4.0)                                          # Experiment 3, Teil 2
BASIN_AMPS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
BASIN_STARTS = 20
BASIN_ITERS = 150
LAP_RHOS, LAP_N, LAP_NOISE, LAP_TAU = (0.0, 1.0, 2.0, 4.0), 24, 1e-2, 0.3   # Experiment 4
LAP_CHAINS, LAP_STEPS, BURN_FRACTION, BLOCK = 20, 20000, 0.2, 500
RWM_SCALE = 2.38              # Skalierung 2,38/sqrt(n) (Roberts, Gelman, Gilks)
Z_BAND = 1.96

COLOR_TRUTH = "#2F6B65"
COLOR_SOLUTION = "#4c78a8"
COLOR_LINEAR = "#D68A2E"
COLOR_START = "#8a8a8a"
COLOR_EXACT = "#222222"
COLOR_GREY = "#8a8a8a"
COLOR_LAPLACE = "#8e44ad"
SOLVER_COLORS = {"gd": "#8a8a8a", "gn_pur": "#c0392b", "gn": "#2F6B65", "lm": "#4c78a8"}

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 1.0, "data": "honest", "solver": "lm", "lam": 1e-2, "iters": 100, "amp": 0.0, "seed": DEFAULT_SEED},
    "Starke Reaktion (ρ = 2)": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 2.0, "data": "honest", "solver": "lm", "lam": 1e-2, "iters": 100, "amp": 0.0, "seed": DEFAULT_SEED},
    "Gradientenabstieg": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 1.0, "data": "honest", "solver": "gd", "lam": 1e-3, "iters": 100, "amp": 0.0, "seed": DEFAULT_SEED},
    "Gauss-Newton ohne Schrittweitensuche": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 2.0, "data": "honest", "solver": "gn_pur", "lam": 1e-3, "iters": 100, "amp": 0.5,
                                             "seed": DEFAULT_SEED},
    "Start weit weg": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 1.0, "data": "honest", "solver": "lm", "lam": 1e-2, "iters": 300, "amp": 1.5, "seed": DEFAULT_SEED},
    "Flache Reaktion (ρ = 4)": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "rho": 4.0, "data": "honest", "solver": "lm", "lam": 1e-2, "iters": 100, "amp": 0.0, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Mittlere Reaktion (ρ = 1), Levenberg-Marquardt vom Nullstart, ehrliche Daten: der Löser konvergiert in wenigen Schritten, das linearisierte Modell liegt weit daneben.",
    "Starke Reaktion (ρ = 2)": "Doppelte Reaktionsstärke: der Fehler des nichtlinearen Modells steigt, der des linearisierten steigt viel stärker.",
    "Gradientenabstieg": "Derselbe adjungierte Gradient, aber ohne Krümmungsinformation: nach 100 Iterationen ist die Gradientennorm erst auf das 4,5·10⁻⁴-Fache des Anfangswerts gefallen, die anderen Löser sind fertig.",
    "Gauss-Newton ohne Schrittweitensuche": "Reiner Gauss-Newton-Schritt von einem Start der Amplitude 0,5: schon der erste Schritt schießt über, mit Schrittweitensuche oder Dämpfung konvergieren die anderen Löser.",
    "Start weit weg": "Levenberg-Marquardt von einem Start der Amplitude 1,5: der Löser konvergiert, aber in ein lokales Minimum, in dem Φ weit über dem Wert des Nullstarts liegt.",
    "Flache Reaktion (ρ = 4)": "Bei ρ = 4 verschwindet die Ableitung der Reaktion bei u = 1, also an den Gipfeln: die Abbildung verliert dort Information, und kein Löser stellt sie wieder her.",
}
