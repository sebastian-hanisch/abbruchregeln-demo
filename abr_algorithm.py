"""Kern der Demo (nur numpy und math).

Alle Messungen laufen auf **Bernoulli-Folgen** X_1, X_2, … mit Erfolgswahrscheinlichkeit π:
  Schätzung  X_i = 1[Netz fällt aus]            π = Ausfallwahrscheinlichkeit U = 1 − R(p) (exakt aus der Aufzählung aller Kantenmengen)
  MC-Test    X_i = 1[T_i ≥ T_obs]               π = exakter p-Wert P_H0(T ≥ T_obs) der Komponentenzahl T nach Kantenausfall (ebenfalls aus der Aufzählung)
Die Folge ist in beiden Fällen exakt Bernoulli(π); das Netz liefert nur realistische π und die Wahrheit.
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom   stream_seed, uniforms, rng_for                      SplitMix64, zählerbasiert (Binomial-Zuwächse über numpy.random.Generator mit festem, aus dem Strom abgeleiteten Seed)
  Netze          grid_edges, hantel_edges, components_table, comp_dist, unreliability, exact_pvalue
  Intervalle     wald_half, wilson, hoeffding_half, asym_half, beta_mix_logL, beta_cs   fester Umfang (Wald, Wilson) und Konfidenzfolgen (Beta-Mischung, Hoeffding-Mischung, asymptotisch)
  Regeln         check_grid, binomial_paths_on_grid, negbin_n, geometric_positions, besag_clifford
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
Z95 = 1.959963984540054


# --- Zufallsstrom ---------------------------------------------------------------------------------------------------------------------------

def mix_int(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys):
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed, count):
    """count gleichverteilte Zahlen in (0, 1), nie 0 oder 1."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def rng_for(*keys) -> np.random.Generator:
    """Erzeuger für Binomial-Zuwächse: fester Seed aus dem SplitMix64-Strom (Reproduzierbarkeit; die Binomial-Stichprobe selbst ist numpy-Implementierung)."""
    return np.random.default_rng(stream_seed(*keys))


# --- Netze: Komponentenzahl, exakte Verteilung ------------------------------------------------------------------------------------------------

def grid_edges(rows: int, cols: int):
    idx = lambda r, c: r * cols + c
    e = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                e.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                e.append((idx(r, c), idx(r + 1, c)))
    return rows * cols, tuple(e)


def hantel_edges():
    """Zwei Dreiecke (0,1,2) und (3,4,5), verbunden durch die Brücke 2–3."""
    return 6, ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3))


def components_table(n: int, edges, chunk: int = 1 << 15) -> np.ndarray:
    """Zahl der Zusammenhangskomponenten für alle 2^m Kantenmengen (Kante e = Bit e): Kleinste-Marke-Ausbreitung; die Komponenten entsprechen den Knoten, deren Marke ihr eigener Index bleibt."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    out = np.empty(1 << m, dtype=np.int16)
    base = np.arange(n, dtype=np.int16)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        lab = np.tile(base, (len(masks), 1))
        while True:
            before = int(lab.sum())
            for e, (u, v) in enumerate(edges):
                low = np.minimum(lab[:, u], lab[:, v])
                pe = present[:, e]
                lab[:, u] = np.where(pe, low, lab[:, u])
                lab[:, v] = np.where(pe, low, lab[:, v])
            if int(lab.sum()) == before:
                break
        out[start:start + len(masks)] = (lab == base[None, :]).sum(axis=1)
    return out


def popcounts(m: int) -> np.ndarray:
    k = np.zeros(1 << m, dtype=np.int64)
    masks = np.arange(1 << m, dtype=np.int64)
    for e in range(m):
        k += (masks >> e) & 1
    return k


@lru_cache(maxsize=8)
def _tables(n: int, edges: tuple):
    return components_table(n, edges), popcounts(len(edges))


def comp_dist(n: int, edges, p: float) -> dict:
    """Verteilung der Komponentenzahl T bei Haltewahrscheinlichkeit p je Kante (unabhängig): dict T -> Wahrscheinlichkeit, exakt aus der Aufzählung aller 2^m Kantenmengen."""
    comp, k = _tables(n, tuple(edges))
    m = len(edges)
    w = p**k * (1 - p) ** (m - k)
    return {int(c): float(w[comp == c].sum()) for c in np.unique(comp)}


def unreliability(n: int, edges, p: float) -> float:
    """U = 1 − R(p) = P(T ≥ 2): Wahrscheinlichkeit, dass das Netz nicht zusammenhängt."""
    d = comp_dist(n, edges, p)
    return float(sum(v for t, v in d.items() if t >= 2))


def exact_pvalue(n: int, edges, p: float, tobs: int) -> float:
    """P(T ≥ tobs) unter dem Modell (Haltewahrscheinlichkeit p je Kante)."""
    d = comp_dist(n, edges, p)
    return float(sum(v for t, v in d.items() if t >= tobs))


# --- Intervalle und Konfidenzfolgen ---------------------------------------------------------------------------------------------------------

_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def lgam(x):
    """ln Γ(x) für x ≥ ½ (Lanczos, Genauigkeit etwa 1e-14); Felder erlaubt. Ersetzt scipy zur Laufzeit."""
    x = np.asarray(x, dtype=float) - 1.0
    a = np.full(x.shape, _LANCZOS[0])
    for i in range(1, 9):
        a = a + _LANCZOS[i] / (x + i)
    t = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


def wald_half(phat, n, z=Z95):
    return z * np.sqrt(phat * (1 - phat) / n)


def wilson(phat, n, z=Z95):
    d = 1 + z * z / n
    c = (phat + z * z / (2 * n)) / d
    h = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def hoeffding_half(n, alpha, r, sigma=0.5):
    """Halbbreite der Mischungs-Konfidenzfolge für den Mittelwert: σ √(2 (1 + r n)/(r n²) · ln(√(1 + r n)/α)); σ = ½ für Bernoulli (subgaußsch)."""
    n = np.asarray(n, dtype=float)
    return sigma * np.sqrt(2 * (1 + r * n) / (r * n * n) * np.log(np.sqrt(1 + r * n) / alpha))


@lru_cache(maxsize=64)
def tune_r(n_star: int, alpha: float, sigma: float = 0.5) -> float:
    """r = τ²σ² so, dass die Mischungsfolge bei n_star die kleinste Breite hat (Gittersuche)."""
    rs = np.logspace(-6, 1, 400)
    w = hoeffding_half(n_star, alpha, rs, sigma)
    return float(rs[int(np.argmin(w))])


def asym_half(n, sigma_hat, alpha, r):
    """Asymptotische Konfidenzfolge (Waudby-Smith u. a.): dieselbe Grenze mit der geschätzten Streuung σ̂_n statt σ."""
    return hoeffding_half(n, alpha, r, 1.0) * sigma_hat


def _log_beta_ratio(S, n, a, b):
    """ln [B(S + a, F + b)/B(a, b)] = ln der Randwahrscheinlichkeit der Folge unter der Beta(a, b)-Mischung (F = n − S)."""
    F = n - S
    return lgam(S + a) + lgam(F + b) - lgam(n + a + b) + (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))


def beta_mix_logL(S, n, m, a=0.5, b=0.5):
    """ln L_n(m) der Beta(a, b)-Mischung gegen die Nullhypothese π = m (S Einsen in n Ziehungen); Felder erlaubt.
    L_n(m) = B(S + a, F + b)/B(a, b) / (m^S (1 − m)^F); als Produkt der Vorhersagen (S_i + a)/(i + a + b) geteilt durch m bzw. 1 − m entsteht dasselbe."""
    S = np.asarray(S, dtype=float)
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    return _log_beta_ratio(S, n, a, b) - (S * np.log(m) + (n - S) * np.log1p(-m))


def beta_cs(S, n, alpha, a=0.5, b=0.5, iters=55):
    """Konfidenzfolge der Beta-Mischung bei (S, n): (untere, obere) Grenze als Lösung von L_n(m) = 1/α links und rechts von p̂ (Bisektion); Felder erlaubt."""
    S = np.asarray(S, dtype=float)
    n = np.asarray(n, dtype=float)
    S, n = np.broadcast_arrays(S, n)
    phat = S / n
    F = n - S
    thr = math.log(1 / alpha)
    lb = _log_beta_ratio(S, n, a, b)                    # unabhängig von m: einmal berechnen
    f = lambda m: lb - (S * np.log(m) + F * np.log1p(-m)) - thr
    lo_a, lo_b = np.full(phat.shape, 1e-12), phat.copy()
    for _ in range(iters):
        mid = 0.5 * (lo_a + lo_b)
        big = f(mid) >= 0
        lo_a = np.where(big, mid, lo_a)
        lo_b = np.where(big, lo_b, mid)
    lower = np.where(f(np.full(phat.shape, 1e-12)) < 0, 0.0, 0.5 * (lo_a + lo_b))
    hi_a, hi_b = phat.copy(), np.full(phat.shape, 1 - 1e-12)
    for _ in range(iters):
        mid = 0.5 * (hi_a + hi_b)
        big = f(mid) >= 0
        hi_b = np.where(big, mid, hi_b)
        hi_a = np.where(big, hi_a, mid)
    upper = np.where(f(np.full(phat.shape, 1 - 1e-12)) < 0, 1.0, 0.5 * (hi_a + hi_b))
    return lower, upper


# --- Pfade, Prüfgitter, Regeln --------------------------------------------------------------------------------------------------------------

def bernoulli_paths(pi: float, reps: int, N: int, seed: int) -> np.ndarray:
    """reps Pfade der Länge N aus festen Strömen: (reps, N) bool."""
    return uniforms(seed, reps * N).reshape(reps, N) < pi


def check_grid(n0: int, nmax: float, fac: float = 1.03) -> np.ndarray:
    """Geometrisches Prüfgitter n0, n0·fac, … ≤ nmax (ganze Zahlen, eindeutig, aufsteigend)."""
    g, n = [], float(n0)
    while n <= nmax:
        g.append(int(round(n)))
        n *= fac
    return np.array(sorted(set(g)), dtype=np.int64)


def binomial_paths_on_grid(pi: float, G: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Zahl der Einsen S nach G[j] Ziehungen für reps Pfade, exakt aus binomialen Zuwächsen: (reps, len(G)) float."""
    inc = np.diff(np.concatenate([[0], G]))
    return np.cumsum(rng.binomial(inc[None, :], pi, size=(reps, len(G))), axis=1).astype(float)


def negbin_n(pi: float, k: int, reps: int, seed: int) -> np.ndarray:
    """Zahl der Ziehungen bis zum k-ten Ereignis (Summe von k geometrischen Wartezeiten, Inversionsverfahren aus festen Strömen): reps ganze Zahlen ≥ k."""
    u = uniforms(seed, reps * k).reshape(reps, k)
    g = np.maximum(np.ceil(np.log(u) / math.log1p(-pi)), 1)
    return g.sum(axis=1).astype(np.int64)


def geometric_positions(pi: float, h: int, reps: int, seed: int) -> np.ndarray:
    """Positionen der ersten h Ereignisse (kumulierte geometrische Wartezeiten): (reps, h)."""
    u = uniforms(seed, reps * h).reshape(reps, h)
    g = np.maximum(np.ceil(np.log(u) / math.log1p(-pi)), 1)
    return np.cumsum(g, axis=1).astype(np.int64)


def besag_clifford(pi: float, h: int, nmax: int, reps: int, seed: int, alpha: float):
    """Besag–Clifford: ziehe bis zum h-ten Ausschlag oder bis nmax; p̂ = h/n beim Stopp nach h Ausschlägen, sonst (S + 1)/(nmax + 1). Rückgabe: (Ablehnen: p̂ ≤ α, Zahl der Simulationen)."""
    pos = geometric_positions(pi, h, reps, seed)
    nh = pos[:, -1]
    stop_h = nh <= nmax
    s_cap = (pos <= nmax).sum(axis=1)
    phat = np.where(stop_h, h / nh, (s_cap + 1) / (nmax + 1))
    return phat <= alpha, np.where(stop_h, nh, nmax).astype(float)


def fixed_n_test(pi: float, n: int, reps: int, rng: np.random.Generator, alpha: float):
    """Fester Umfang: p̂ = (1 + S)/(1 + n), Ablehnen bei p̂ ≤ α. Rückgabe: (Ablehnen, Simulationen)."""
    S = rng.binomial(n, pi, size=reps)
    return (1 + S) / (1 + n) <= alpha, np.full(reps, float(n))


def cs_decision(pi: float, alpha: float, delta: float, cap: int, reps: int, rng: np.random.Generator, n0: int = 30, fac: float = 1.03):
    """Konfidenzfolgen-Entscheidung: ziehe, bis α nicht mehr in der Beta-Mischungs-Folge zum Niveau δ liegt (Prüfgitter), höchstens `cap`; entscheide Ablehnen, wenn p̂ < α.
    Rückgabe: (Ablehnen, Simulationen, entschieden)."""
    G = check_grid(n0, cap, fac)
    S = binomial_paths_on_grid(pi, G, reps, rng)
    cond = beta_mix_logL(S, G[None, :], alpha) >= math.log(1 / delta)
    stopped = cond.any(axis=1)
    idx = np.where(stopped, cond.argmax(axis=1), len(G) - 1)
    r = np.arange(reps)
    return (S[r, idx] / G[idx]) < alpha, G[idx].astype(float), stopped
