"""Auswertung: Formatierung, Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import nli_constants as C
import nli_model as model
from inverse_generator import SplitMix64, grid, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num_text(x: float) -> str:
    """Zahl mit drei gültigen Stellen, ab 1000 und unter 0,01 als Zehnerpotenz."""
    if not math.isfinite(x):
        return "unendlich"
    if abs(x) >= 1000.0 or (x != 0.0 and abs(x) < 0.01):
        return sci(x, 1)
    return de(x, 1 if abs(x) >= 10 else 2)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 100 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def share(x: float) -> str:
    """Anteil (0 bis 1) als ganze Prozent."""
    return f"{de(100.0 * x, 0)} %"


def rho_label(rho: float) -> str:
    """Reaktionsstärke ohne überflüssige Nachkommastelle: 0, 0,5, 1, 2, 4."""
    return de(rho, 1).removesuffix(",0")


def count_text(k: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{k:,}".replace(",", ".")


# ----------------------------------------------------------------------------- Instanz

def rel_err(x: np.ndarray, xt: np.ndarray) -> float:
    """Fehler in der 2-Norm relativ zur Signalnorm."""
    return float(np.linalg.norm(x - xt) / np.linalg.norm(xt))


def inverse_model(n: int, w: float, rho: float) -> model.Model:
    """Das Modell, das invertiert wird: K_INV Schritte."""
    return model.Model(model.heat_matrix(n, w, C.K_INV), C.K_INV, rho)


@lru_cache(maxsize=64)
def instance(n: int, w: float, rho: float, noise: float, data: str, seed: int):
    """(s, x, b, sigma): Gitter, Wahrheit (der Anfangswert), Daten und absolutes Rauschen. `honest`: Daten aus dem feineren Modell mit K_TRUE Schritten, `crime`: aus dem invertierten."""
    s = grid(n)
    x = phantom(s)
    fine = inverse_model(n, w, rho) if data == "crime" else model.Model(model.heat_matrix(n, w, C.K_TRUE), C.K_TRUE, rho)
    clean = fine.forward(x)
    sigma = float(noise) * float(np.sqrt(np.mean(clean ** 2)))
    return s, x, clean + SplitMix64(seed).normal(n) * sigma, sigma


def problem(rho: float, noise: float = C.DEFAULT_NOISE, reps: int = 1, seed: int = C.SEED_BASE + 1, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, honest: bool = True):
    """(inverses Modell, Wahrheit, Daten (reps, n), sigma) mit ehrlichen Daten aus dem feineren Modell; das Rauschen ist ein Strom, der für alle Wiederholungen weiterläuft."""
    inv = inverse_model(n, w, rho)
    x = phantom(grid(n))
    fine = inv if not honest else model.Model(model.heat_matrix(n, w, C.K_TRUE), C.K_TRUE, rho)
    clean = fine.forward(x)
    sigma = float(noise) * float(np.sqrt(np.mean(clean ** 2)))
    return inv, x, clean[None, :] + SplitMix64(seed).normal(n * reps).reshape(reps, n) * sigma, sigma


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    rho: float = C.DEFAULT_RHO
    data: str = "honest"
    solver: str = C.DEFAULT_SOLVER
    lam: float = C.DEFAULT_LAM
    iters: int = C.DEFAULT_ITER
    amp: float = C.DEFAULT_AMP
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    x: np.ndarray
    b: np.ndarray
    sigma: float
    x0: np.ndarray
    runs: dict                # Löser -> SolveResult (alle vier vom selben Start)
    run: object               # SolveResult des gewählten Lösers
    ref: object               # Levenberg-Marquardt vom Nullstart (Maßstab für „dasselbe Minimum“)
    errs: dict                # Löser -> Fehler der Lösung
    err: float
    err_ref: float            # Fehler der Lösung des Nullstarts (Levenberg-Marquardt): der Maßstab für das nichtlineare Modell
    same_min: bool            # Phi des gewählten Lösers höchstens (1 + MIN_OK) · Phi des Nullstarts
    x_lin: np.ndarray         # Lösung des linearisierten Modells (rho = 0) beim besten lam (gegen die Wahrheit gewählt)
    lam_lin: float
    err_lin: float
    resid: float              # |F(x) - b| der Lösung
    disc: float               # sigma sqrt(n): so groß ist das Residuum, wenn nur das Rauschen übrig ist
    g_adj: np.ndarray         # adjungierter Gradient und Differenzenquotient am Prüfpunkt
    g_fd: np.ndarray
    grad_err: float
    check_counts: tuple       # Vorwärtsläufe: adjungiert, Differenzen
    cond: float
    singular: bool
    flat: float               # Anteil der Zustände der Wahrheit mit |R'| < 0,2 (die Reaktion ist dort flach)


def analyse(settings: Settings) -> Analysis:
    st = settings
    s, x, b, sigma = instance(st.n, st.w, st.rho, st.noise, st.data, st.seed)
    inv = inverse_model(st.n, st.w, st.rho)
    x0 = model.start_vector(st.n, st.amp, st.seed + 3)
    runs = {mode: model.solve(inv, b, st.lam, x0, mode, st.iters) for mode in C.SOLVERS}
    run = runs[st.solver]
    ref = runs["lm"] if st.amp == 0.0 and st.iters >= 200 else model.solve(inv, b, st.lam, np.zeros(st.n), "lm", 200)
    errs = {mode: rel_err(r.x, x) for mode, r in runs.items()}
    a_lin = np.linalg.matrix_power(inv.a, C.K_INV)
    cands = [(rel_err(model.tikhonov_linear(a_lin, b, lam), x), lam) for lam in C.EXP_LAMS]
    err_lin, lam_lin = min(cands)
    x_chk = C.CHECK_SCALE * x + C.CHECK_SHIFT
    before = inv.counts["forward"]
    u, states = inv.forward(x_chk, record=True)
    g_adj = inv.adjoint(states, u - b)
    nf_adj = inv.counts["forward"] - before
    before = inv.counts["forward"]
    g_fd = model.gradient_fd(inv, x_chk, b)
    nf_fd = inv.counts["forward"] - before
    state_x = inv.forward(x, record=True)[1]
    cond = model.total_condition(st.n, st.w)
    flat = float(np.mean([np.abs(inv.dreact(inv.a @ state_x[k])) < 0.2 for k in range(inv.k)]))
    resid = float(np.linalg.norm(inv.forward(run.x) - b))
    return Analysis(
        settings=st, s=s, x=x, b=b, sigma=sigma, x0=x0, runs=runs, run=run, ref=ref, errs=errs, err=errs[st.solver], err_ref=rel_err(ref.x, x),
        same_min=bool(math.isfinite(run.phi[-1]) and run.phi[-1] <= ref.phi[-1] * (1.0 + C.MIN_OK)), x_lin=model.tikhonov_linear(a_lin, b, lam_lin), lam_lin=float(lam_lin),
        err_lin=float(err_lin), resid=resid, disc=sigma * math.sqrt(st.n), g_adj=g_adj, g_fd=g_fd, grad_err=float(np.linalg.norm(g_adj - g_fd) / np.linalg.norm(g_fd)),
        check_counts=(nf_adj, nf_fd), cond=cond, singular=bool(cond >= C.SINGULAR_COND), flat=flat,
    )


# ----------------------------------------------------------------------------- Experiment 1: der adjungierte Gradient

def adjoint_experiment() -> dict:
    """Adjungierter Gradient gegen zentralen Differenzenquotienten und gegen J^T r, mit den Vorwärtsläufen, die beide brauchen."""
    rows = []
    for rho in C.ADJ_RHOS:
        for n in C.ADJ_NS:
            m = inverse_model(n, C.DEFAULT_W, rho)
            xt = phantom(grid(n))
            b = m.forward(xt)
            x = C.CHECK_SCALE * xt + C.CHECK_SHIFT
            m.counts["forward"] = 0
            u, states = m.forward(x, record=True)
            g_adj = m.adjoint(states, u - b)
            nf_adj = m.counts["forward"]
            m.counts["forward"] = 0
            g_fd = model.gradient_fd(m, x, b)
            nf_fd = m.counts["forward"]
            g_jac = m.jacobian(x).T @ (m.forward(x) - b)
            rows.append({"rho": rho, "n": n, "err_fd": float(np.linalg.norm(g_adj - g_fd) / np.linalg.norm(g_fd)), "err_jac": float(np.linalg.norm(g_adj - g_jac) / np.linalg.norm(g_jac)),
                         "nf_adj": nf_adj, "nf_fd": nf_fd})
    return {"rows": rows}


# ----------------------------------------------------------------------------- Experiment 2: die Linearisierung ignorieren

def ignore_experiment() -> dict:
    """Bester RMS-Fehler über lam: lineares Modell (rho = 0 angenommen) gegen nichtlineares Modell (Levenberg-Marquardt), ehrliche Daten, EXP_REPS Wiederholungen je Reaktionsstärke;
    dazu das Diskrepanzprinzip bei EXP_DISC_RHO."""
    rows = []
    for rho in C.EXP_RHOS:
        inv, x, bb, sigma = problem(rho, reps=C.EXP_REPS)
        a_lin = np.linalg.matrix_power(inverse_model(C.DEFAULT_N, C.DEFAULT_W, 0.0).a, C.K_INV)
        e_lin, e_nl = [], []
        for lam in C.EXP_LAMS:
            xs = np.linalg.solve(a_lin.T @ a_lin + lam ** 2 * np.eye(C.DEFAULT_N), a_lin.T @ bb.T).T
            e_lin.append(float(np.sqrt(np.mean([rel_err(v, x) ** 2 for v in xs]))))
            xs2 = [model.solve(inv, bb[r], lam, np.zeros(C.DEFAULT_N), "lm", 60).x for r in range(C.EXP_REPS)]
            e_nl.append(float(np.sqrt(np.mean([rel_err(v, x) ** 2 for v in xs2]))))
        i1, i2 = int(np.argmin(e_lin)), int(np.argmin(e_nl))
        rows.append({"rho": rho, "lin": e_lin[i1], "lin_lam": C.EXP_LAMS[i1], "nl": e_nl[i2], "nl_lam": C.EXP_LAMS[i2], "e_lin": e_lin, "e_nl": e_nl})
    inv, x, bb, sigma = problem(C.EXP_DISC_RHO, reps=C.EXP_DISC_REPS)
    lams = C.EXP_DISC_LAMS
    res = np.zeros((len(lams), C.EXP_DISC_REPS))
    er = np.zeros_like(res)
    for i, lam in enumerate(lams):
        for r in range(C.EXP_DISC_REPS):
            xr = model.solve(inv, bb[r], lam, np.zeros(C.DEFAULT_N), "lm", 40).x
            res[i, r] = float(np.sum((inv.forward(xr) - bb[r]) ** 2))
            er[i, r] = rel_err(xr, x)
    target = sigma ** 2 * C.DEFAULT_N
    pick = np.array([max([i for i in range(len(lams)) if res[i, r] <= target] or [0]) for r in range(C.EXP_DISC_REPS)])
    e = er[pick, np.arange(C.EXP_DISC_REPS)]
    disc = {"rho": C.EXP_DISC_RHO, "median": float(np.median(e)), "rms": float(np.sqrt(np.mean(e ** 2))), "best": float(np.sqrt(np.mean(er ** 2, axis=1)).min()),
            "lam_median": float(np.median([lams[i] for i in pick]))}
    return {"rows": rows, "disc": disc, "reps": C.EXP_REPS, "lams": list(C.EXP_LAMS)}


# ----------------------------------------------------------------------------- Experiment 3: Löser und Einzugsgebiet

def solver_experiment() -> dict:
    """Teil 1: Iterationen und Auswertungen bis zum Abbruch je Löser und rho (Nullstart); Teil 2: Anteil der Starts je Amplitude, die dasselbe Minimum wie der Nullstart erreichen."""
    rows = []
    for rho in C.SOLVER_RHOS:
        inv, x, bb, sigma = problem(rho)
        for mode in C.SOLVERS:
            inv.counts.update({"forward": 0, "jacobian": 0, "adjoint": 0})
            r = model.solve(inv, bb[0], C.SOLVER_LAM, np.zeros(C.DEFAULT_N), mode, C.SOLVER_GD_ITERS if mode == "gd" else 100)
            rows.append({"rho": rho, "solver": mode, "iterations": r.iterations, "converged": r.converged, "status": r.status, "counts": r.counts, "phi": r.phi[-1],
                         "gnorm_ratio": r.gnorm[-1] / r.gnorm[0], "err": rel_err(r.x, x)})
    basin = []
    for rho in C.BASIN_RHOS:
        inv, x, bb, sigma = problem(rho)
        ref = model.solve(inv, bb[0], C.SOLVER_LAM, np.zeros(C.DEFAULT_N), "lm", 200).phi[-1]
        for amp in C.BASIN_AMPS:
            starts = 1 if amp == 0.0 else C.BASIN_STARTS
            row = {"rho": rho, "amp": amp, "starts": starts}
            for mode in ("gn_pur", "gn", "lm"):
                ok = 0
                for k in range(starts):
                    r = model.solve(inv, bb[0], C.SOLVER_LAM, model.start_vector(C.DEFAULT_N, amp, C.SEED_BASE + 23000 + k), mode, C.BASIN_ITERS)
                    ok += int(r.status != "diverged" and math.isfinite(r.phi[-1]) and r.phi[-1] <= ref * (1.0 + C.MIN_OK))
                row[mode] = ok
            basin.append(row)
    return {"rows": rows, "basin": basin}


# ----------------------------------------------------------------------------- Experiment 4: Laplace-Näherung gegen MCMC

def laplace_experiment(chains: int = C.LAP_CHAINS, steps: int = C.LAP_STEPS) -> dict:
    """Weißer Gauß-Prior (tau), ehrliche Daten bei Rauschen LAP_NOISE, n = LAP_N: Laplace-Näherung (Gauss-Newton-Hesse-Matrix am MAP) gegen Ketten des vorkonditionierten
    Zufallsschritt-Metropolis-Verfahrens, die um die Näherung überdispers starten."""
    rows = []
    n = C.LAP_N
    for rho in C.LAP_RHOS:
        inv, x, bb, sigma = problem(rho, noise=C.LAP_NOISE, n=n)
        b = bb[0]
        tau = C.LAP_TAU
        x_map = model.solve(inv, b, sigma / tau, np.zeros(n), "lm", 100).x
        cov = model.laplace_covariance(inv, x_map, sigma, tau)
        chol = np.linalg.cholesky(cov + 1e-14 * np.eye(n))
        x0 = x_map[None, :] + 2.0 * SplitMix64(C.SEED_BASE + 5).normal(chains * n).reshape(chains, n) @ chol.T
        res = model.rwm(inv, b, sigma, tau, x0, steps, C.SEED_BASE + 3, C.RWM_SCALE / math.sqrt(n), chol, np.full(n, 1.0 / n))
        total = res["count"] * chains
        mean = res["sum1"].sum(axis=0) / total
        sd = np.sqrt(np.maximum(res["sum2"].sum(axis=0) / total - mean ** 2, 0.0))
        sd_l = np.sqrt(np.diag(cov))
        ratio = sd_l / sd
        burn = int(C.BURN_FRACTION * steps)
        rows.append({"rho": rho, "acc": float(res["acc"].mean()), "err_map": rel_err(x_map, x), "err_mean": rel_err(mean, x), "ratio_median": float(np.median(ratio)),
                     "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()), "gap_sd": float(np.max(np.abs(x_map - mean) / sd)),
                     "cover_laplace": float(np.mean(np.abs(x - x_map) <= C.Z_BAND * sd_l)), "cover_chain": float(np.mean(np.abs(x - mean) <= C.Z_BAND * sd)),
                     "rhat": model.rhat(res["traces"][burn:].T)})
    return {"rows": rows, "n": n, "chains": chains, "steps": steps}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    st = a.settings
    r = a.run
    bad = []
    if r.status == "diverged":
        bad.append("Der Löser ist davongelaufen (der Gradient wuchs über das Billionenfache oder die Werte liefen über)")
    elif r.status == "stalled":
        bad.append("Der Löser findet keine Schrittweite mit Abstieg")
    elif not r.converged:
        bad.append(f"Der Löser ist nach {r.iterations} Iterationen nicht konvergiert: die Gradientennorm ist nur auf {sci(r.gnorm[-1] / r.gnorm[0])} des Anfangswerts gefallen")
    elif not a.same_min:
        bad.append(f"Lokales Minimum: Φ liegt beim {de(r.phi[-1] / a.ref.phi[-1], 2)}-Fachen des Wertes, den der Nullstart erreicht")
    if a.err > C.FAIL_ERR:
        bad.append(f"Fehler {err_pct(a.err)}: nicht brauchbar" + (f" (die Reaktion ist bei {share(a.flat)} der Zustände der Wahrheit flach)" if a.flat > 0.0 else ""))
    c = r.counts
    lead = (f"{C.SOLVER_SHORT[st.solver]}: {r.iterations} Iterationen ({c['forward']} Vorwärtsläufe, {c['jacobian']} Jacobi-Matrizen, {c['adjoint']} adjungierte Läufe), "
            f"Fehler {err_pct(a.err)}. Maßstab: Levenberg-Marquardt vom Nullstart kommt auf {err_pct(a.err_ref)}, das linearisierte Modell (ρ = 0 angenommen, bester λ) auf {err_pct(a.err_lin)}")
    if a.err_ref > C.FAIL_ERR and a.err_lin > C.FAIL_ERR:
        lead += ": Beide Modelle versagen."
    elif a.err_lin > C.LIN_FACTOR_OK * a.err_ref:
        lead += f", das ist das {de(a.err_lin / a.err_ref, 1)}-Fache: Die Nichtlinearität zu ignorieren kostet hier Genauigkeit."
    else:
        lead += ": Die Nichtlinearität ist hier ignorierbar."
    if bad:
        return "warning", f"{lead} Zu beachten: " + "; ".join(bad) + "."
    return "success", f"{lead} Der Löser hat konvergiert und liegt im selben Minimum wie der Nullstart."


def verdict_adjoint(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    worst = max(max(r["err_fd"], r["err_jac"]) for r in rows)
    big = max(rows, key=lambda r: r["n"])
    bound = 10.0 ** math.ceil(math.log10(worst))
    return "success", (f"Der adjungierte Gradient stimmt in allen {len(rows)} Fällen mit dem Differenzenquotienten und mit J^T r auf weniger als {sci(bound)} überein. "
                       f"Er braucht {big['nf_adj']} Vorwärtslauf und einen Rückwärtslauf; der Differenzenquotient braucht bei n = {big['n']} {count_text(big['nf_fd'])} Vorwärtsläufe.")


def verdict_ignore(res: dict) -> tuple[str, str]:
    rows = {r["rho"]: r for r in res["rows"]}
    d = res["disc"]
    return "warning", (f"Beste Fehler (RMS über λ) bei ρ = {rho_label(0.5)} / {rho_label(1.0)} / {rho_label(2.0)}: nichtlineares Modell {err_pct(rows[0.5]['nl'])} / {err_pct(rows[1.0]['nl'])} / "
                       f"{err_pct(rows[2.0]['nl'])}, linearisiertes Modell {err_pct(rows[0.5]['lin'])} / {err_pct(rows[1.0]['lin'])} / {err_pct(rows[2.0]['lin'])}. "
                       f"Bei ρ = {rho_label(4.0)} versagen beide ({err_pct(rows[4.0]['nl'])} und {err_pct(rows[4.0]['lin'])}): die Ableitung der Reaktion verschwindet bei u = 1, die Abbildung verliert an den Gipfeln Information. "
                       f"Das Diskrepanzprinzip erreicht bei ρ = {rho_label(d['rho'])} einen Fehler von {err_pct(d['rms'])} (RMS), das Optimum über λ liegt bei {err_pct(d['best'])}.")


def verdict_solvers(res: dict) -> tuple[str, str]:
    rows = {(r["rho"], r["solver"]): r for r in res["rows"]}
    basin = {(r["rho"], r["amp"]): r for r in res["basin"]}
    lo, hi, one = basin[(2.0, 0.5)], basin[(2.0, 1.5)], basin[(2.0, 1.0)]
    it = lambda rho, s: rows[(rho, s)]["iterations"]  # noqa: E731
    return "warning", (f"Bei ρ = {rho_label(1.0)} vom Nullstart: Gradientenabstieg {it(1.0, 'gd')} Iterationen (nicht konvergiert), Gauss-Newton mit Schrittweitensuche {it(1.0, 'gn')}, "
                       f"Levenberg-Marquardt {it(1.0, 'lm')}. Einzugsgebiet bei ρ = {rho_label(2.0)}: Levenberg-Marquardt erreicht dasselbe Minimum von {lo['lm']} von {lo['starts']} Starts der "
                       f"Amplitude 0,5, von {hi['lm']} von {hi['starts']} der Amplitude 1,5; Gauss-Newton ohne Schrittweitensuche schon bei Amplitude 1 von {one['gn_pur']} von {one['starts']}.")


def verdict_laplace(res: dict) -> tuple[str, str]:
    rows = {r["rho"]: r for r in res["rows"]}
    return "info", (f"Standardabweichung der Laplace-Näherung geteilt durch die der Ketten, im Median: {de(rows[0.0]['ratio_median'], 2)} / {de(rows[1.0]['ratio_median'], 2)} / "
                    f"{de(rows[2.0]['ratio_median'], 2)} / {de(rows[4.0]['ratio_median'], 2)} bei ρ = 0 / 1 / 2 / 4 (bei ρ = 4 von {de(rows[4.0]['ratio_min'], 2)} bis {de(rows[4.0]['ratio_max'], 2)}). "
                    f"Das 95-%-Intervall überdeckt die Wahrheit bei ρ = 4 nur an {share(rows[4.0]['cover_laplace'])} der Punkte, mit der Näherung wie mit der Kette: der Fehler steckt im Modell (verlorene Information), nicht in der Näherung.")
