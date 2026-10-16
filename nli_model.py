"""Kern: das nichtlineare Vorwärtsmodell (Wärmeleitung und logistische Reaktion), Jacobi-Matrix, adjungierter Gradient, Löser (Gradientenabstieg, Gauss-Newton mit und ohne
Schrittweitensuche, Levenberg-Marquardt), Laplace-Näherung und ein vektorisierter Zufallsschritt-Metropolis-Sampler. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_nli.py` gegen Differenzenquotienten, die explizite Jacobi-Matrix, die Tikhonov-Lösung (linearer Grenzfall), die Konvergenz des Splittings und MCMC.

Modell (Anfangswert-Rekonstruktion): K Schritte, jeder besteht aus einem Wärmeleitungsschritt A (Matrix) und einer logistischen Reaktion R(u) = u + (rho / K) u (1 - u):
    u_0 = x,    u_{k+1} = R(A u_k),    F(x) = u_K.
Für rho = 0 ist F(x) = A^K x linear (die Unschärfe der Stücke 1 bis 8). Die Zielfunktion ist Phi(x) = |F(x) - b|^2 + lam^2 |x|^2.
Jacobi-Matrix J = D_K A D_{K-1} A ... D_1 A mit D_k = diag(R'(A u_{k-1})); Gradient von 1/2 |F(x) - b|^2 durch einen Rückwärtslauf (adjungierter Zustand):
    mu_K = F(x) - b,    mu_k = A^T (R'(A u_k) * mu_{k+1}),    Gradient = mu_0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import nli_constants as C
from inverse_generator import SplitMix64

_TINY = 1e-300


# ----------------------------------------------------------------------------- Vorwärtsmodell

def heat_matrix(n: int, w: float, k: int) -> np.ndarray:
    """Ein Wärmeleitungsschritt mit reflektierenden Rändern in der Kosinusbasis (orthonormal): Symbol exp(-(w^2 / 2k) (j pi)^2) zum Index j. k Schritte ergeben genau die Unschärfe
    der Breite w; die Zeilensummen sind 1 (Mittelwerterhaltung), die Matrix ist symmetrisch."""
    j = np.arange(n)
    c = math.sqrt(2.0 / n) * np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * j[None, :] / n)
    c[:, 0] = 1.0 / math.sqrt(n)
    return (c * np.exp(-(w ** 2 / (2.0 * k)) * (j * math.pi) ** 2)[None, :]) @ c.T


def total_condition(n: int, w: float) -> float:
    """Kondition der linearen Abbildung (rho = 0): Verhältnis von größtem und kleinstem Symbol exp(-(w^2 / 2) (j pi)^2), j = 0 .. n-1."""
    return math.exp(min(0.5 * w ** 2 * ((n - 1) * math.pi) ** 2, 700.0))


class Model:
    """F(x) = u_K mit u_{k+1} = R(A u_k); `counts` zählt Vorwärtsläufe, Jacobi-Matrizen und adjungierte Läufe."""

    def __init__(self, a: np.ndarray, k: int, rho: float):
        self.a = np.asarray(a, dtype=float)
        self.k = int(k)
        self.rho = float(rho)
        self.n = self.a.shape[0]
        self.counts = {"forward": 0, "jacobian": 0, "adjoint": 0}

    def react(self, u):
        """Logistische Reaktion R(u) = u + (rho / K) u (1 - u)."""
        return u + (self.rho / self.k) * u * (1.0 - u)

    def dreact(self, u):
        """Ableitung R'(u) = 1 + (rho / K) (1 - 2u)."""
        return 1.0 + (self.rho / self.k) * (1.0 - 2.0 * u)

    def forward(self, x, record: bool = False):
        """F(x); mit record=True zusätzlich die Zustände [u_0, ..., u_K]. Zählt als ein Vorwärtslauf."""
        self.counts["forward"] += 1
        u = np.asarray(x, dtype=float)
        states = [u]
        for _ in range(self.k):
            u = self.react(self.a @ u)
            states.append(u)
        return (u, states) if record else u

    def forward_rows(self, x_rows: np.ndarray) -> np.ndarray:
        """F auf jeder Zeile von x_rows (für die Sampler), ohne Zählung."""
        u = np.asarray(x_rows, dtype=float)
        for _ in range(self.k):
            u = self.react(u @ self.a.T)
        return u

    def jacobian(self, x) -> np.ndarray:
        """Explizite Jacobi-Matrix J = D_K A ... D_1 A. Zählt als eine Jacobi-Matrix."""
        self.counts["jacobian"] += 1
        u = np.asarray(x, dtype=float)
        j = np.eye(self.n)
        for _ in range(self.k):
            v = self.a @ u
            j = (self.dreact(v)[:, None] * self.a) @ j
            u = self.react(v)
        return j

    def adjoint(self, states: list, resid: np.ndarray) -> np.ndarray:
        """Rückwärtslauf: Gradient von 1/2 |F(x) - b|^2 aus den Zuständen des Vorwärtslaufs und dem Residuum F(x) - b. Zählt als ein adjungierter Lauf."""
        self.counts["adjoint"] += 1
        mu = np.asarray(resid, dtype=float)
        for k in range(self.k - 1, -1, -1):
            mu = self.a.T @ (self.dreact(self.a @ states[k]) * mu)
        return mu


# ----------------------------------------------------------------------------- Zielfunktion

def phi_value(resid: np.ndarray, x: np.ndarray, lam: float) -> float:
    """Phi(x) = |F(x) - b|^2 + lam^2 |x|^2 aus dem Residuum."""
    return float(resid @ resid + lam ** 2 * (x @ x))


def phi_gradient(half_grad: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:
    """Gradient von Phi aus dem Gradienten von 1/2 |F(x) - b|^2: 2 (half_grad + lam^2 x)."""
    return 2.0 * (half_grad + lam ** 2 * x)


def gradient_fd(m: Model, x: np.ndarray, b: np.ndarray, h: float = C.FD_STEP) -> np.ndarray:
    """Zentraler Differenzenquotient des Gradienten von 1/2 |F(x) - b|^2: 2n Vorwärtsläufe."""
    out = np.empty(m.n)
    for i in range(m.n):
        e = np.zeros(m.n)
        e[i] = h
        rp, rm = m.forward(x + e) - b, m.forward(x - e) - b
        out[i] = (0.5 * float(rp @ rp) - 0.5 * float(rm @ rm)) / (2.0 * h)
    return out


# ----------------------------------------------------------------------------- Löser

@dataclass
class SolveResult:
    x: np.ndarray
    iterations: int
    converged: bool
    status: str                # "converged", "maxit", "diverged" oder "stalled"
    phi: list                  # Phi nach 0, 1, ... Iterationen
    gnorm: list                # |grad Phi| nach 0, 1, ... Iterationen
    counts: dict = field(default_factory=dict)


def _trial(m: Model, x: np.ndarray, b: np.ndarray, lam: float):
    """Ein Vorwärtslauf am Kandidaten: (Phi, Residuum, Zustände)."""
    with np.errstate(all="ignore"):
        u, states = m.forward(x, record=True)
        r = u - b
        return phi_value(r, x, lam), r, states


def solve(m: Model, b: np.ndarray, lam: float, x0: np.ndarray, mode: str, maxit: int, tol: float = C.GN_TOL) -> SolveResult:
    """Minimiert Phi von x0 aus. mode `gd`: Schritt -grad Phi, Armijo-Rückverfolgung mit verdoppelter letzter Schrittweite als Anfang; `gn_pur`: Gauss-Newton-Schritt ohne
    Schrittweitensuche; `gn`: Gauss-Newton-Schritt mit Armijo-Halbierung; `lm`: Levenberg-Marquardt (Dämpfung mu: Faktor 0,3 nach Erfolg, 4 nach Misserfolg).
    Abbruch bei |grad Phi| <= tol |grad Phi_0| (converged), nach maxit Schritten (maxit), bei Überlauf oder Anstieg des Gradienten über das 1e12-Fache (diverged, der Schritt wird
    nicht angenommen) oder wenn keine Schrittweite Abstieg bringt (stalled)."""
    before = dict(m.counts)
    x = np.array(x0, dtype=float)
    f, r, states = _trial(m, x, b, lam)
    g = phi_gradient(m.adjoint(states, r), x, lam)
    g0 = float(np.linalg.norm(g))
    phis, gnorms = [f], [g0]
    mu = C.MU_START
    alpha = None
    status = "maxit"
    it = 0
    eye = np.eye(m.n)
    while True:
        if g0 == 0.0 or float(np.linalg.norm(g)) <= tol * g0:
            status = "converged"
            break
        if it >= maxit:
            break
        cand = None
        if mode == "gd":
            gg = float(g @ g)
            alpha = 1.0 / math.sqrt(gg) if alpha is None else 2.0 * alpha
            for _ in range(C.LS_MAX):
                xn = x - alpha * g
                fn, rn, sn = _trial(m, xn, b, lam)
                if fn <= f - C.ARMIJO_C * alpha * gg:
                    cand = (xn, fn, rn, sn)
                    break
                alpha *= 0.5
        else:
            j = m.jacobian(x)
            h = j.T @ j + lam ** 2 * eye
            rhs = -(j.T @ r + lam ** 2 * x)
            if mode == "gn_pur":
                xn = x + np.linalg.solve(h, rhs)
                fn, rn, sn = _trial(m, xn, b, lam)
                cand = (xn, fn, rn, sn)
            elif mode == "gn":
                d = np.linalg.solve(h, rhs)
                slope = float(g @ d)
                a = 1.0
                for _ in range(C.LS_MAX):
                    xn = x + a * d
                    fn, rn, sn = _trial(m, xn, b, lam)
                    if fn <= f + C.ARMIJO_C * a * slope:
                        cand = (xn, fn, rn, sn)
                        break
                    a *= 0.5
            else:
                for _ in range(C.LM_TRIES):
                    xn = x + np.linalg.solve(h + mu * eye, rhs)
                    fn, rn, sn = _trial(m, xn, b, lam)
                    if fn < f:
                        cand = (xn, fn, rn, sn)
                        mu = max(mu * C.MU_DOWN, C.MU_MIN)
                        break
                    mu *= C.MU_UP
        if cand is None:
            status = "stalled"
            break
        xn, fn, rn, sn = cand
        with np.errstate(all="ignore"):
            gn_ = phi_gradient(m.adjoint(sn, rn), xn, lam)
        if not (np.all(np.isfinite(xn)) and math.isfinite(fn) and np.all(np.isfinite(gn_))) or float(np.linalg.norm(gn_)) > C.DIVERGE_FACTOR * g0:
            status = "diverged"
            break
        x, f, r, g = xn, fn, rn, gn_
        it += 1
        phis.append(f)
        gnorms.append(float(np.linalg.norm(g)))
    return SolveResult(x=x, iterations=it, converged=status == "converged", status=status, phi=phis, gnorm=gnorms,
                       counts={key: m.counts[key] - before[key] for key in m.counts})


# ----------------------------------------------------------------------------- Linearer Fall und Starts

def tikhonov_linear(a_total: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Lösung von min |A x - b|^2 + lam^2 |x|^2 über die normalen Gleichungen (das linearisierte Modell, rho = 0)."""
    return np.linalg.solve(a_total.T @ a_total + lam ** 2 * np.eye(a_total.shape[1]), a_total.T @ b)


def start_vector(n: int, amp: float, seed: int) -> np.ndarray:
    """Start: Null für amp = 0, sonst amp · (glatte Zufallsfunktion + 1); die glatte Funktion ist ein gleitender Mittelwert über 7 Punkte weißen Rauschens."""
    if amp == 0.0:
        return np.zeros(n)
    z = np.convolve(SplitMix64(seed).normal(n), np.ones(7) / 7.0, mode="same")
    return amp * (z + 1.0)


# ----------------------------------------------------------------------------- Laplace-Näherung und Sampler

def laplace_covariance(m: Model, x_map: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Kovarianz der Laplace-Näherung N(x_map, H^-1) mit der Gauss-Newton-Hesse-Matrix H = J^T J / sigma^2 + I / tau^2 am MAP (weißer Gauß-Prior N(0, tau^2 I))."""
    j = m.jacobian(x_map)
    cov = np.linalg.inv(j.T @ j / sigma ** 2 + np.eye(m.n) / tau ** 2)
    return 0.5 * (cov + cov.T)


def neg_log_post(m: Model, x_rows: np.ndarray, b: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Negative Log-Posterior |F(x) - b|^2 / (2 sigma^2) + |x|^2 / (2 tau^2) für jede Zeile von x_rows."""
    x_rows = np.atleast_2d(x_rows)
    r = m.forward_rows(x_rows) - b[None, :]
    return 0.5 * np.sum(r ** 2, axis=1) / sigma ** 2 + 0.5 * np.sum(x_rows ** 2, axis=1) / tau ** 2


def rwm(m: Model, b: np.ndarray, sigma: float, tau: float, x0: np.ndarray, steps: int, seed: int, step: float, chol: np.ndarray, fvec: np.ndarray) -> dict:
    """Zufallsschritt-Metropolis, vektorisiert über die Ketten (Zeilen von x0): x' = x + step · L z, Annahme exp(U(x) - U(x')) mit der negativen Log-Posterior U.
    Gibt zurück: sum1, sum2 (Summen von x und x^2 nach dem Verwerfen, je Kette), count, acc (Annahmerate je Kette), traces (steps, chains) des Funktionals fvec @ x."""
    x = np.array(np.atleast_2d(x0), dtype=float)
    chains, n = x.shape
    rng = SplitMix64(seed)
    burn = int(C.BURN_FRACTION * steps)
    cur = -neg_log_post(m, x, b, sigma, tau)
    s1, s2 = np.zeros((chains, n)), np.zeros((chains, n))
    acc = np.zeros(chains)
    traces = np.empty((steps, chains))
    kept, done = 0, 0
    while done < steps:
        blk = min(C.BLOCK, steps - done)
        z = rng.normal(blk * chains * n).reshape(blk, chains, n)
        u = np.log(np.maximum(rng.uniform(blk * chains).reshape(blk, chains), _TINY))
        for i in range(blk):
            prop = x + step * (z[i] @ chol.T)
            new = -neg_log_post(m, prop, b, sigma, tau)
            ok = u[i] < new - cur
            x = np.where(ok[:, None], prop, x)
            cur = np.where(ok, new, cur)
            acc += ok
            traces[done + i] = x @ fvec
            if done + i >= burn:
                s1 += x
                s2 += x ** 2
                kept += 1
        done += blk
    return {"sum1": s1, "sum2": s2, "count": kept, "acc": acc / steps, "traces": traces}


def rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-Dach für Ketten der Form (chains, steps)."""
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    w = float(chains.var(axis=1, ddof=1).mean())
    bvar = n * float(chains.mean(axis=1).var(ddof=1))
    if w <= 0.0:
        return math.inf
    return math.sqrt(((n - 1) / n * w + bvar / n) / w)


def gauss_posterior_linear(a_total: np.ndarray, b: np.ndarray, sigma: float, tau: float):
    """Exakter Posterior N(m, G) des linearen Falls (rho = 0) mit weißem Prior N(0, tau^2 I): (Mittel, Kovarianz)."""
    n = a_total.shape[1]
    cov = np.linalg.inv(a_total.T @ a_total / sigma ** 2 + np.eye(n) / tau ** 2)
    cov = 0.5 * (cov + cov.T)
    return cov @ a_total.T @ b / sigma ** 2, cov
