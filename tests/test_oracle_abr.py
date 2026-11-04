"""Unabhängige Gegenproben (scipy, networkx): Aufzählung der Komponentenzahl, Beta-Mischung, Ville-Schranke, Binomial- und negative Binomialverteilung; ORACLE-PLAYBOOK §5."""

import itertools
import math

import networkx as nx
import numpy as np
import pytest
from scipy import optimize, special, stats

import abr_algorithm as A


def _nx_dist(n, edges, p):
    """Verteilung der Komponentenzahl per Aufzählung mit networkx (unabhängig vom Code unter Test)."""
    d = {}
    m = len(edges)
    for mask in itertools.product((0, 1), repeat=m):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(e for e, b in zip(edges, mask) if b)
        w = p ** sum(mask) * (1 - p) ** (m - sum(mask))
        t = nx.number_connected_components(G)
        d[t] = d.get(t, 0.0) + w
    return d


@pytest.mark.parametrize("name,p", [("pfad", 0.3), ("dreieck", 0.7), ("hantel", 0.7), ("hantel", 0.9), ("gitter22", 0.5)])
def test_komponentenverteilung_gegen_networkx(name, p, pfad, dreieck):
    n, edges = {"pfad": pfad, "dreieck": dreieck, "hantel": A.hantel_edges(), "gitter22": A.grid_edges(2, 2)}[name]
    ref, got = _nx_dist(n, edges, p), A.comp_dist(n, edges, p)
    assert set(ref) == set(got)
    for t in ref:
        assert got[t] == pytest.approx(ref[t], abs=1e-12)
    assert sum(got.values()) == pytest.approx(1.0)


def test_gitter_3x3_gegen_bekannten_wert_der_reihe():
    # Zuverlässigkeit des 3×3-Gitters bei p = 0,7 (aus Stück 1, per Aufzählung und Frontier bestätigt): R = 0,5110, also U = 0,4890
    assert A.unreliability(*A.grid_edges(3, 3), 0.7) == pytest.approx(0.48901, abs=2e-5)
    assert A.exact_pvalue(*A.grid_edges(3, 3), 0.7, 4) == pytest.approx(0.04935, abs=2e-5)


def test_ausfallwahrscheinlichkeit_waechst_mit_sinkendem_p():
    n, e = A.grid_edges(3, 3)
    us = [A.unreliability(n, e, p) for p in (0.95, 0.9, 0.8, 0.7, 0.5)]
    assert us == sorted(us)


@pytest.mark.parametrize("S,n", [(0, 1), (1, 2), (3, 10), (17, 100), (250, 1000), (0, 500)])
def test_beta_mischung_gegen_betaln(S, n):
    m = 0.2
    ref = special.betaln(S + 0.5, n - S + 0.5) - special.betaln(0.5, 0.5) - (S * math.log(m) + (n - S) * math.log1p(-m))
    assert float(A.beta_mix_logL(S, n, m)) == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize("S,n", [(10, 1000), (1, 100), (0, 200), (500, 1000), (7, 30)])
def test_enden_der_beta_folge_loesen_l_gleich_eins_durch_alpha(S, n):
    lo, hi = A.beta_cs(np.array([S]), np.array([n]), 0.05)
    thr = math.log(20)
    for edge in (float(lo[0]), float(hi[0])):
        if 0 < edge < 1:
            assert float(A.beta_mix_logL(S, n, edge)) == pytest.approx(thr, abs=1e-6)
    assert lo[0] <= S / n <= hi[0]
    # im Inneren ist L kleiner als 1/α, außerhalb größer
    inner = 0.5 * (lo[0] + hi[0])
    assert float(A.beta_mix_logL(S, n, inner)) < thr


def test_ville_schranke_der_beta_folge_per_monte_carlo():
    # P(∃n: L_n(π) ≥ 1/α) ≤ α: 4000 Pfade der Länge 2000
    for pi in (0.3, 0.05):
        X = A.bernoulli_paths(pi, 4000, 2000, A.stream_seed(9, int(pi * 100)))
        ns = np.arange(1, 2001, dtype=float)
        S = np.cumsum(X, axis=1).astype(float)
        ever = (A.beta_mix_logL(S, ns, pi) >= math.log(20)).any(axis=1).mean()
        assert ever <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 4000)


def test_wilson_loest_die_scoregleichung():
    # Wilson-Grenzen sind die Lösungen von (p̂ − p)² = z² p (1 − p)/n (unabhängige Herleitung)
    for phat, n in ((0.1, 50), (0.01, 1000), (0.5, 20)):
        lo, hi = A.wilson(phat, n)
        for edge in (lo, hi):
            assert (phat - edge) ** 2 == pytest.approx(A.Z95**2 * edge * (1 - edge) / n, rel=1e-9)


def test_abdeckung_zum_festen_n_exakt_aus_der_binomialverteilung():
    # Wald-Abdeckung bei n = 1000, π = 0,1 exakt: Summe der Binomialwahrscheinlichkeiten über alle k mit π im Intervall
    n, pi = 1000, 0.1
    k = np.arange(n + 1)
    phat = k / n
    inside = np.abs(phat - pi) <= A.wald_half(phat, n)
    cov = float(stats.binom.pmf(k, n, pi)[inside].sum())
    assert 0.93 < cov < 0.96
    # Simulation stimmt
    S = A.rng_for(1).binomial(n, pi, size=200000)
    sim = float((np.abs(S / n - pi) <= A.wald_half(S / n, n)).mean())
    assert sim == pytest.approx(cov, abs=0.004)


@pytest.mark.parametrize("pi,k", [(0.1, 5), (0.01, 10), (0.3, 3)])
def test_wartezeit_gegen_negative_binomialverteilung(pi, k):
    n = A.negbin_n(pi, k, 200000, 11) - k                    # diskret: Vergleich der Zähldichte (Kolmogorov–Smirnov gilt nur für stetige Verteilungen)
    j = np.arange(0, int(stats.nbinom.ppf(0.99, k, pi)))
    emp = np.bincount(n, minlength=j[-1] + 1)[: len(j)] / len(n)
    assert np.abs(emp - stats.nbinom.pmf(j, k, pi)).max() < 0.004
    assert n.mean() == pytest.approx(k * (1 - pi) / pi, rel=0.02)


@pytest.mark.parametrize("pi,k", [(0.1, 5), (0.1, 30), (0.01, 10)])
def test_inverse_stichprobe_erwartungswerte_exakt(pi, k):
    # E[(k − 1)/(n − 1)] = π exakt; E[k/n] > π (Reihe über die negative Binomialverteilung)
    j = np.arange(0, 20000 if pi >= 0.1 else 200000)
    n = j + k
    pmf = stats.nbinom.pmf(j, k, pi)
    unb = float((pmf * (k - 1) / (n - 1)).sum())
    naive = float((pmf * k / n).sum())
    assert unb == pytest.approx(pi, rel=1e-6) and naive > pi
    sim = A.negbin_n(pi, k, 200000, 12).astype(float)
    assert (k / sim).mean() == pytest.approx(naive, abs=4 * (k / sim).std() / math.sqrt(len(sim)))


@pytest.mark.parametrize("pi,n,alpha", [(0.045, 999, 0.05), (0.03, 99, 0.05), (0.06, 999, 0.05), (0.2, 99, 0.1)])
def test_risiko_beim_festen_umfang_exakt_aus_der_binomialverteilung(pi, n, alpha):
    # Ablehnen ⇔ (1 + S)/(1 + n) ≤ α ⇔ S ≤ ⌊α (1 + n)⌋ − 1; Risiko = P(Entscheidung ≠ [π ≤ α])
    reject = stats.binom.cdf(math.floor(alpha * (1 + n) + 1e-12) - 1, n, pi)
    risk = (1 - reject) if pi <= alpha else reject
    rej, _ = A.fixed_n_test(pi, n, 200000, A.rng_for(3, n), alpha)
    sim = float((rej != (pi <= alpha)).mean())
    assert sim == pytest.approx(risk, abs=0.004)


def test_besag_clifford_stoppverteilung_gegen_binomial():
    # P(h-tes Ereignis bis nmax) = P(Binomial(nmax, π) ≥ h)
    pi, h, nmax = 0.05, 10, 200
    _, ns = A.besag_clifford(pi, h, nmax, 100000, 5, 0.05)
    p_stop = float(stats.binom.sf(h - 1, nmax, pi))
    stop = np.mean(A.geometric_positions(pi, h, 100000, 5)[:, -1] <= nmax)
    assert stop == pytest.approx(p_stop, abs=0.004)


def test_konfidenzfolgen_entscheidung_haelt_das_risiko():
    for pi in (0.03, 0.07):
        rej, ns, st = A.cs_decision(pi, 0.05, 0.01, 200000, 2000, A.rng_for(8, int(pi * 100)))
        assert st.all()
        assert float((rej != (pi <= 0.05)).mean()) <= 0.02


def test_beta_folge_gegen_unabhaengige_wurzelsuche():
    S, n = 30, 400
    lo, hi = A.beta_cs(np.array([S]), np.array([n]), 0.05)
    f = lambda m: float(A.beta_mix_logL(S, n, m)) - math.log(20)
    assert lo[0] == pytest.approx(optimize.brentq(f, 1e-6, S / n), abs=1e-6) and hi[0] == pytest.approx(optimize.brentq(f, S / n, 1 - 1e-6), abs=1e-6)
