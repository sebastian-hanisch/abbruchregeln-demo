"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird, was beim Hinsehen und Stoppen mit Intervallen zum festen Umfang passiert und was jederzeit gültige Konfidenzfolgen kosten. Alle Folgen sind exakt Bernoulli(π);
π ist die exakte Ausfallwahrscheinlichkeit eines Netzes (Schätzung) beziehungsweise der exakte p-Wert der Komponentenzahl (Monte-Carlo-Test), beide aus der Aufzählung aller Kantenmengen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import abr_algorithm as A
import abr_constants as C
from abr_scenario import named_netz, pi_values


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    return f"{x:.1e}".replace(".", ",")


def pr(x: float) -> str:
    """Wahrscheinlichkeit mit drei Nachkommastellen."""
    return "nicht bestimmbar" if np.isnan(x) else de(x, 3)


def prozent(x: float, nd: int = 0) -> str:
    return "nicht bestimmbar" if np.isnan(x) else f"{de(x * 100, nd)} %"


def sprozent(x: float, nd: int = 1) -> str:
    """Verzerrung mit Vorzeichen: +22,9 %, −3,1 %, 0,0 % (kein „−0,0“)."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if round(x * 100, nd) == 0:
        return f"{de(0, nd)} %"
    return ("+" if x > 0 else "") + f"{de(x * 100, nd)} %"


def pi_text(x: float) -> str:
    """π mit so vielen Nachkommastellen, dass zwei gültige Stellen stehen."""
    return de(x, 4 if x >= 0.001 else 5)


@dataclass(frozen=True)
class Settings:
    netz: str = C.DEFAULT_NETZ
    p: float = C.DEFAULT_P
    tobs: int = C.DEFAULT_TOBS
    eps: float = C.DEFAULT_EPS
    alpha: float = C.DEFAULT_ALPHA
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    label: str
    pi: float                              # Ausfallwahrscheinlichkeit (Schätzung)
    pi_test: float                         # exakter p-Wert P(T ≥ T_obs) (Test)
    peek: dict                             # Folge -> {"ever": Anteil irgendwann außerhalb, "at": {n: Anteil außerhalb bei n}}
    peek_n: int
    peek_n0: int
    peek_reps: int
    widths: tuple                          # je n {n, S, wald, wilson, beta, hoeffding, asym}


def label_of(settings: Settings) -> str:
    name, _, _ = named_netz(settings.netz)
    return f"{name}, Haltewahrscheinlichkeit {de(settings.p, 2)} je Kante"


# --- Experiment 1: Hinsehen bei jedem n -------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def peek_experiment(pi: float, seed: int, reps: int = C.PEEK_REPS, N: int = C.PEEK_N, n0: int = C.PEEK_N0, alpha: float = C.ALPHA_CI, n_star: int = C.PEEK_NSTAR) -> dict:
    """Anteil der Pfade, in denen die Wahrheit π bei IRGENDEINEM n zwischen n0 und N außerhalb des Intervalls liegt, und Anteil außerhalb bei n = 100, 1000, 10000, für Wald, Wilson,
    Beta-Mischungs-Folge, Hoeffding-Mischungs-Folge (auf n_star abgestimmt) und asymptotische Folge."""
    ns = np.arange(1, N + 1, dtype=float)
    sel = ns >= n0
    r_h, r_a = A.tune_r(n_star, alpha, 0.5), A.tune_r(n_star, alpha, 1.0)
    ever = {k: 0 for k in C.SEQ}
    at = {k: {n: 0 for n in (100, 1000, 10000) if n <= N} for k in C.SEQ}
    chunk = min(C.PEEK_CHUNK, reps)
    chunks = max(1, reps // chunk)
    for c in range(chunks):
        X = A.bernoulli_paths(pi, chunk, N, A.stream_seed(seed, 1, int(round(pi * 1e6)), c))
        S = np.cumsum(X, axis=1).astype(float)
        phat = S / ns
        out = {"wald": np.abs(phat - pi) > A.wald_half(phat, ns)}
        lo, hi = A.wilson(phat, ns)
        out["wilson"] = (pi < lo) | (pi > hi)
        out["beta"] = A.beta_mix_logL(S, ns, pi) >= math.log(1 / alpha)
        out["hoeffding"] = np.abs(phat - pi) > A.hoeffding_half(ns, alpha, r_h, 0.5)
        out["asym"] = np.abs(phat - pi) > A.asym_half(ns, np.sqrt(phat * (1 - phat)), alpha, r_a)
        for k, o in out.items():
            ever[k] += int(o[:, sel].any(axis=1).sum())
            for n in at[k]:
                at[k][n] += int(o[:, n - 1].sum())
    total = chunks * chunk
    return {"pi": pi, "reps": total, "N": N, "n0": n0, "rows": {k: {"ever": ever[k] / total, "at": {n: v / total for n, v in at[k].items()}} for k in C.SEQ}}


def width_table(pi: float, alpha: float = C.ALPHA_CI, n_star: int = C.PEEK_NSTAR) -> tuple:
    """Halbbreiten bei S = round(n π) Einsen: Wald, Wilson, Beta-Mischung, Hoeffding-Mischung, asymptotische Folge."""
    r_h, r_a = A.tune_r(n_star, alpha, 0.5), A.tune_r(n_star, alpha, 1.0)
    rows = []
    for n in C.WIDTH_NS:
        S = round(n * pi)
        phat = S / n
        lo, hi = A.wilson(phat, n)
        bl, bh = A.beta_cs(np.array([S]), np.array([n]), alpha)
        rows.append({"n": n, "S": S, "wald": float(A.wald_half(phat, n)), "wilson": float((hi - lo) / 2), "beta": float((bh[0] - bl[0]) / 2),
                     "hoeffding": float(A.hoeffding_half(n, alpha, r_h, 0.5)), "asym": float(A.asym_half(n, math.sqrt(phat * (1 - phat)), alpha, r_a))})
    return tuple(rows)


def analyse(settings: Settings, pi: float | None = None, label: str | None = None) -> Analysis:
    u, ptest = pi_values(settings.netz, settings.p, settings.tobs)
    pi = u if pi is None else pi
    ex = peek_experiment(pi, settings.seed)
    return Analysis(label or label_of(settings), pi, ptest, ex["rows"], ex["N"], ex["n0"], ex["reps"], width_table(pi))


def ratio(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


# --- Experiment 3: Abbruchregeln ---------------------------------------------------------------------------------------------------------------

RULES = ("rel_wald", "rel_wilson", "rel_beta", "abs_wald")
RULE_LABELS = {"rel_wald": "relativ, Wald", "rel_wilson": "relativ, Wilson", "rel_beta": "relativ, Konfidenzfolge", "abs_wald": "absolut, Wald (ohne Schutz)"}


@lru_cache(maxsize=256)
def stop_experiment(pi: float, eps: float, seed: int, reps: int = C.STOP_REPS) -> dict:
    """Abbruchregeln auf dem Prüfgitter (erster Blick bei 30, Faktor 1,03): relativ (Halbbreite ≤ ε·p̂, p̂ > 0) mit Wald, Wilson und Beta-Konfidenzfolge; absolut (Wald-Halbbreite ≤ H_ABS, ohne Schutz gegen p̂ = 0).
    Je Regel: mittleres n, n/n*, Abdeckung des Endintervalls, relative Verzerrung von p̂, Anteil der Läufe, die mit null Ereignissen stoppen, Anteil ohne Stopp bis nmax."""
    nstar = A.Z95**2 * (1 - pi) / (eps**2 * pi)
    nmax = int(min(C.STOP_NMAX_MAX, max(C.STOP_NMAX_MIN, 6 * nstar)))
    G = A.check_grid(C.STOP_N0, nmax, C.STOP_FAC)
    rng = A.rng_for(seed, 3, int(round(pi * 1e6)), int(round(eps * 100)))
    S = A.binomial_paths_on_grid(pi, G, reps, rng)
    n = G[None, :].astype(float)
    phat = S / n
    wh = A.wald_half(phat, n)
    lo, hi = A.wilson(phat, n)
    bl, bh = A.beta_cs(S.ravel(), np.broadcast_to(n, S.shape).ravel(), C.ALPHA_CI)
    bl, bh = bl.reshape(S.shape), bh.reshape(S.shape)
    pos = S > 0
    spec = {"rel_wald": (pos & (wh <= eps * phat), phat - wh, phat + wh), "rel_wilson": (pos & ((hi - lo) / 2 <= eps * phat), lo, hi),
            "rel_beta": (pos & ((bh - bl) / 2 <= eps * phat), bl, bh), "abs_wald": (wh <= C.H_ABS, phat - wh, phat + wh)}
    r = np.arange(reps)
    out = {}
    for name, (cond, l, h) in spec.items():
        stopped = cond.any(axis=1)
        idx = np.where(stopped, cond.argmax(axis=1), len(G) - 1)
        ns = G[idx].astype(float)
        cov = ((l[r, idx] <= pi) & (pi <= h[r, idx]))
        covw = ((lo[r, idx] <= pi) & (pi <= hi[r, idx]))
        ref = nstar if name != "abs_wald" else A.Z95**2 * pi * (1 - pi) / C.H_ABS**2
        out[name] = {"n": float(ns[stopped].mean()) if stopped.any() else float("nan"), "n_ref": ref, "cov": float(cov[stopped].mean()) if stopped.any() else float("nan"),
                     "cov_wilson": float(covw[stopped].mean()) if stopped.any() else float("nan"), "bias": float(phat[r, idx][stopped].mean() / pi - 1) if stopped.any() else float("nan"),
                     "zero": float(((S[r, idx] == 0) & stopped).mean()), "unstopped": float(1 - stopped.mean())}
    return {"pi": pi, "eps": eps, "nstar": nstar, "nmax": nmax, "reps": reps, "rows": out}


# --- Experiment 4: Stopp nach k Ereignissen ----------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def inverse_experiment(pi: float, seed: int, reps: int = C.INV_REPS) -> dict:
    """Stopp beim k-ten Ereignis (k = 5, 10, 30, 100): Verzerrung von k/n und von (k − 1)/(n − 1), Abdeckung des Wald-Intervalls um k/n, mittleres n."""
    rows = []
    for k in C.INV_KS:
        n = A.negbin_n(pi, k, reps, A.stream_seed(seed, 4, int(round(pi * 1e6)), k)).astype(float)
        naive, unb = k / n, (k - 1) / (n - 1)
        half = A.wald_half(naive, n)
        rows.append({"k": k, "bias_naive": float(naive.mean() / pi - 1), "bias_unbiased": float(unb.mean() / pi - 1), "cov": float(((naive - half <= pi) & (pi <= naive + half)).mean()), "n": float(n.mean())})
    return {"pi": pi, "rows": rows, "reps": reps}


# --- Experiment 5: Monte-Carlo-Test ------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def mctest_cells(pi: float, alpha: float, seed: int, tag: int, reps: int = C.MC_REPS) -> dict:
    """Entscheidungsrisiko (Wahrscheinlichkeit einer anderen Entscheidung als bei exaktem p-Wert: Ablehnen genau bei π ≤ α) und mittlere Zahl der Simulationen je Verfahren."""
    exact = pi <= alpha
    rng = A.rng_for(seed, 5, tag)
    cells = {}
    for name, n in (("fix99", 99), ("fix999", 999), ("fix9999", 9999)):
        rej, ns = A.fixed_n_test(pi, n, reps, rng, alpha)
        cells[name] = {"risk": float((rej != exact).mean()), "n": float(ns.mean()), "open": 0.0}
    for name, (h, nmax) in (("bc10", (10, 999)), ("bc50", (50, 9999))):
        rej, ns = A.besag_clifford(pi, h, nmax, reps, A.stream_seed(seed, 6, tag, h), alpha)
        cells[name] = {"risk": float((rej != exact).mean()), "n": float(ns.mean()), "open": 0.0}
    rej, ns, st = A.cs_decision(pi, alpha, C.MC_DELTA, C.MC_CAP, reps, rng)
    cells["cs"] = {"risk": float(((rej != exact) & st).mean()), "n": float(ns.mean()), "open": float(1 - st.mean())}
    return {"pi": pi, "alpha": alpha, "exact_reject": exact, "cells": cells}


def mctest_experiment(settings: Settings) -> dict:
    """Zeilen für π = Vielfache von α (dazu der exakte p-Wert des gewählten Netzes mit Markierung)."""
    _, ptest = pi_values(settings.netz, settings.p, settings.tobs)
    rows = [mctest_cells(f * settings.alpha, settings.alpha, settings.seed, i) | {"factor": f, "own": False} for i, f in enumerate(C.MC_FACTORS)]
    own = mctest_cells(ptest, settings.alpha, settings.seed, 99) | {"factor": ptest / settings.alpha, "own": True}
    return {"rows": rows, "own": own, "alpha": settings.alpha, "pi_test": ptest, "reps": C.MC_REPS}


def custom_result(settings: Settings, pi: float, alpha: float) -> dict:
    """Spielwiese: eigenes π und α: Hinsehen (Experiment 1) und Entscheidungsrisiko je Verfahren (Experiment 5)."""
    a = analyse(settings, pi, f"eigenes π = {pi_text(pi)}")
    return {"analysis": a, "mctest": mctest_cells(pi, alpha, settings.seed, 77)}


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Darf man hier bei jedem n hinsehen und anhalten, wenn das Intervall passt?"""
    p = a.peek
    w = a.widths[1]
    head = (f"{a.label}: Ausfallwahrscheinlichkeit π = {pi_text(a.pi)}. Bei jedem n von {a.peek_n0} bis {de(a.peek_n, 0)} hingesehen liegt die Wahrheit in {de(a.peek_reps, 0)} Läufen irgendwann außerhalb: "
            f"Wald {pr(p['wald']['ever'])}, Wilson {pr(p['wilson']['ever'])}, Beta-Folge {pr(p['beta']['ever'])}, Hoeffding-Folge {pr(p['hoeffding']['ever'])}, asymptotische Folge {pr(p['asym']['ever'])} (nominell {de(C.ALPHA_CI, 2)}). "
            f"Zu einem festen n = 1.000 liegt Wald nur in {pr(p['wald']['at'][1000])} der Läufe außerhalb. Die Beta-Folge ist bei n = 1.000 das {dn(ratio(w['beta'], w['wald']), 2)}-Fache so breit wie Wald.")
    if p["asym"]["ever"] > 2 * C.ALPHA_CI:
        head += f" Die asymptotische Folge hält hier nicht (zu wenige Ereignisse für die geschätzte Streuung)."
    if p["wald"]["ever"] > 2 * C.ALPHA_CI:
        return "warning", head + " Intervalle zum festen Umfang taugen hier nicht zum Hinsehen."
    return "success", head + " Die Intervalle bleiben auch beim Hinsehen nahe am Niveau."


def verdict_width(a: Analysis) -> tuple[str, str]:
    ws = a.widths
    rb = [ratio(w["beta"], w["wald"]) for w in ws]
    rh = [ratio(w["hoeffding"], w["wald"]) for w in ws]
    ra = [ratio(w["asym"], w["wald"]) for w in ws]
    fin = lambda xs: [x for x in xs if np.isfinite(x)]
    if not fin(rb):
        return "info", "Bei null Ereignissen ist die Wald-Breite null; ein Verhältnis ist nicht bestimmbar."
    return "info", (f"Halbbreite gegen Wald bei n = {de(ws[0]['n'], 0)} bis {de(ws[-1]['n'], 0)}: Beta-Folge {de(min(fin(rb)), 2)}- bis {de(max(fin(rb)), 2)}-fach, asymptotische Folge {de(min(fin(ra)), 2)}- bis {de(max(fin(ra)), 2)}-fach, "
                    f"Hoeffding-Folge {de(min(fin(rh)), 2)}- bis {de(max(fin(rh)), 2)}-fach. Für gleiche Breite braucht die Beta-Folge etwa das {de(min(fin(rb)) ** 2, 1)}- bis {de(max(fin(rb)) ** 2, 1)}-Fache an Proben.")


def verdict_stop(ex: dict) -> tuple[str, str]:
    r = ex["rows"]
    text = (f"π = {pi_text(ex['pi'])}, relative Genauigkeit ε = {de(ex['eps'], 1)}, Orakelumfang n* = {de(ex['nstar'], 0)}: relativ mit Wald im Mittel n = {de(r['rel_wald']['n'], 0)} ({de(r['rel_wald']['n'] / ex['nstar'], 2)}·n*), Abdeckung {pr(r['rel_wald']['cov'])}, "
            f"Verzerrung von p̂ {sprozent(r['rel_wald']['bias'])}; mit Konfidenzfolge n = {de(r['rel_beta']['n'], 0)} ({de(r['rel_beta']['n'] / ex['nstar'], 2)}·n*), Abdeckung {pr(r['rel_beta']['cov'])}. "
            f"Absolute Regel (Wald-Halbbreite ≤ {de(C.H_ABS, 2)}): {prozent(r['abs_wald']['zero'], 0)} der Läufe stoppen mit null Ereignissen, Abdeckung des Wald-Endintervalls {pr(r['abs_wald']['cov'])} (Wilson-Endintervall {pr(r['abs_wald']['cov_wilson'])}).")
    bad = r["abs_wald"]["zero"] > 0.2 or (np.isfinite(r["abs_wald"]["cov"]) and r["abs_wald"]["cov"] < 0.85)
    return ("warning" if bad else "info"), text


def verdict_inverse(ex: dict) -> tuple[str, str]:
    rows = ex["rows"]
    return "info", (f"Stopp beim k-ten Ereignis bei π = {pi_text(ex['pi'])}: Verzerrung von k/n " + ", ".join(f"{sprozent(r['bias_naive'])} (k = {r['k']})" for r in rows)
                    + "; Verzerrung von (k − 1)/(n − 1) " + ", ".join(sprozent(r["bias_unbiased"]) for r in rows) + f". Abdeckung des Wald-Intervalls um k/n: {pr(min(r['cov'] for r in rows))} bis {pr(max(r['cov'] for r in rows))}.")


def verdict_mctest(ex: dict) -> tuple[str, str]:
    near = min((r for r in ex["rows"]), key=lambda r: abs(r["factor"] - 0.9))
    own = ex["own"]
    c, o = near["cells"], own["cells"]
    text = (f"Bei Testniveau α = {de(ex['alpha'], 2)}: bei π = {pi_text(near['pi'])} (0,9·α) ist das Entscheidungsrisiko fest n = 999 {pr(c['fix999']['risk'])}, Besag–Clifford (h = 50) {pr(c['bc50']['risk'])}, Konfidenzfolge {pr(c['cs']['risk'])} "
            f"(im Mittel {de(c['cs']['n'], 0)} Simulationen gegen 999). Der exakte p-Wert des gewählten Netzes ist {pi_text(ex['pi_test'])} ({de(own['factor'], 2)}·α): Risiko fest n = 999 {pr(o['fix999']['risk'])}, Konfidenzfolge {pr(o['cs']['risk'])} "
            f"mit im Mittel {de(o['cs']['n'], 0)} Simulationen ({prozent(o['cs']['open'], 0)} unentschieden bis {de(C.MC_CAP, 0)}).")
    return "info", text
