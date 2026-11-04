"""Handrechnung der Einheiten (DEMO-PLAYBOOK §1b): jede Zahl hier ist von Hand nachgerechnet, nicht aus dem Code gelesen."""

import math

import numpy as np
import pytest

import abr_algorithm as A


def test_exakte_verteilung_des_pfads_von_hand(pfad):
    # zwei Kanten, p = ½: T = 1 (beide halten) ¼, T = 2 (genau eine) ½, T = 3 (keine) ¼
    d = A.comp_dist(*pfad, 0.5)
    assert d == pytest.approx({1: 0.25, 2: 0.5, 3: 0.25})
    assert A.unreliability(*pfad, 0.5) == pytest.approx(0.75) and A.exact_pvalue(*pfad, 0.5, 3) == pytest.approx(0.25)


def test_exakte_verteilung_des_dreiecks_von_hand(dreieck):
    # Dreieck, p = ½: 3 Kanten (⅛) und 2 Kanten (⅜) halten zusammen: T = 1 mit ½; 1 Kante (⅜): T = 2; 0 Kanten (⅛): T = 3
    assert A.comp_dist(*dreieck, 0.5) == pytest.approx({1: 0.5, 2: 0.375, 3: 0.125})
    assert A.unreliability(*dreieck, 0.5) == pytest.approx(0.5)


def test_pvalue_grenzen(pfad):
    assert A.exact_pvalue(*pfad, 0.5, 1) == pytest.approx(1.0) and A.exact_pvalue(*pfad, 0.5, 4) == 0.0


def test_beta_mischung_von_hand():
    # Pfad X = (1, 0), m = ½: Vorhersage Schritt 1 ½ → Faktor ½/½ = 1; Schritt 2 (S = 1, n = 1, F = 0): (0 + ½)/(1 + 1) = ¼ → Faktor ¼/½ = ½; L_2(½) = ½
    assert float(A.beta_mix_logL(1, 1, 0.5)) == pytest.approx(0.0, abs=1e-12)
    assert float(A.beta_mix_logL(1, 2, 0.5)) == pytest.approx(math.log(0.5), abs=1e-12)


def test_beta_mischung_als_produkt_der_vorhersagen():
    x = [1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    m, S, logL = 0.3, 0, 0.0
    for i, xi in enumerate(x):
        pred = (S + 0.5) / (i + 1.0)
        logL += math.log(pred if xi else 1 - pred) - math.log(m if xi else 1 - m)
        S += xi
    assert float(A.beta_mix_logL(S, len(x), m)) == pytest.approx(logL, abs=1e-12)


def test_wald_halbbreite_von_hand():
    # p̂ = ¼, n = 4: 1,96 √(¼ · ¾ / 4) = 1,96 · 0,21651 = 0,4244
    assert float(A.wald_half(0.25, 4)) == pytest.approx(0.4244, abs=1e-4)
    assert float(A.wald_half(0.0, 100)) == 0.0                    # bei null Ereignissen ist die Breite null


def test_wilson_von_hand():
    # p̂ = ¼, n = 4: d = 1 + 3,8415/4 = 1,9604; Mitte (0,25 + 0,4802)/1,9604 = 0,3725; Halbbreite 1,96 √(0,046875 + 0,0600)/1,9604 = 0,3269
    lo, hi = A.wilson(0.25, 4)
    assert (lo + hi) / 2 == pytest.approx(0.3725, abs=1e-3) and (hi - lo) / 2 == pytest.approx(0.3269, abs=1e-3)


def test_hoeffding_halbbreite_von_hand():
    # r = 1, n = 3, α = 0,05, σ = ½: ½ √(2 · 4/9 · ln(√4/0,05)) = ½ √(0,8889 · 3,6889) = ½ · 1,8108 = 0,9054
    assert float(A.hoeffding_half(3, 0.05, 1.0, 0.5)) == pytest.approx(0.9054, abs=1e-4)


def test_orakelumfang_der_relativen_regel_von_hand():
    # π = 0,1, ε = 0,2: n* = 1,96² · 0,9/(0,04 · 0,1) = 864,4; dort ist die Wald-Halbbreite 1,96 √(0,09/864) = 0,02 = 0,2 · 0,1
    nstar = A.Z95**2 * 0.9 / (0.2**2 * 0.1)
    assert nstar == pytest.approx(864.4, abs=0.2) and float(A.wald_half(0.1, 864.36)) == pytest.approx(0.02, abs=1e-4)


class _FestesBinomial:
    def __init__(self, s):
        self.s = np.asarray(s)

    def binomial(self, n, pi, size):
        return self.s


def test_fester_umfang_von_hand():
    # n = 99, α = 0,05: p̂ = (1 + S)/100 ≤ 0,05 ⇔ S ≤ 4
    rej, ns = A.fixed_n_test(0.05, 99, 4, _FestesBinomial([3, 4, 5, 10]), 0.05)
    assert rej.tolist() == [True, True, False, False] and ns.tolist() == [99.0] * 4


def test_besag_clifford_von_hand(monkeypatch):
    # h = 2, nmax = 10: Positionen (2, 4) → Stopp bei n = 4, p̂ = 2/4 = 0,5; Positionen (3, 12) → nur ein Ausschlag bis 10 → p̂ = (1 + 1)/11 = 0,1818
    monkeypatch.setattr(A, "geometric_positions", lambda pi, h, reps, seed: np.array([[2, 4], [3, 12]]))
    rej, ns = A.besag_clifford(0.1, 2, 10, 2, 0, 0.3)
    assert ns.tolist() == [4.0, 10.0] and rej.tolist() == [False, True]          # 0,5 > 0,3; 0,1818 ≤ 0,3
    rej2, _ = A.besag_clifford(0.1, 2, 10, 2, 0, 0.15)
    assert rej2.tolist() == [False, False]


def test_pruefgitter_von_hand():
    g = A.check_grid(30, 40, 1.03)
    # 30; 30,9 → 31; 31,827 → 32; 32,78 → 33; 33,76 → 34; 34,77 → 35; 35,8 → 36; 36,9 → 37; 38,0 → 38; 39,1 → 39; 40,3 > 40
    assert g.tolist() == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    assert (np.diff(A.check_grid(30, 3000)) > 0).all()


def test_binomiale_zuwaechse_auf_dem_gitter_sind_nichtfallend():
    G = A.check_grid(30, 5000)
    S = A.binomial_paths_on_grid(0.1, G, 50, A.rng_for(1, 2))
    assert S.shape == (50, len(G)) and (np.diff(S, axis=1) >= 0).all() and (S <= G[None, :]).all()


def test_wartezeit_bis_zum_k_ten_ereignis_ist_mindestens_k():
    n = A.negbin_n(0.3, 5, 5000, 7)
    assert n.min() >= 5 and n.dtype.kind == "i"


def test_inverse_stichprobe_von_hand():
    # π = ½, k = 1: n geometrisch, E[1/n] = Σ 2^(−j)/j = ln 2 = 0,6931 (k/n-Verzerrung +38,6 %); k = 2: E[1/(n − 1)] = ½
    n1 = A.negbin_n(0.5, 1, 400000, 3).astype(float)
    assert (1 / n1).mean() == pytest.approx(math.log(2), abs=0.003)
    n2 = A.negbin_n(0.5, 2, 400000, 4).astype(float)
    assert (1 / (n2 - 1)).mean() == pytest.approx(0.5, abs=0.003)


def test_lanczos_gegen_math_lgamma():
    for x in (0.5, 1.0, 1.5, 3.7, 20.5, 1000.5, 1e5 + 0.5):
        assert float(A.lgam(x)) == pytest.approx(math.lgamma(x), rel=1e-12, abs=1e-12)


def test_strom_ist_reproduzierbar_und_verschieden():
    assert A.stream_seed(3, 4) == A.stream_seed(3, 4) and A.stream_seed(3, 4) != A.stream_seed(4, 3)
    u = A.uniforms(A.stream_seed(1), 10000)
    assert u.min() > 0 and u.max() < 1 and np.array_equal(A.uniforms(5, 100), A.uniforms(5, 100))
    assert not np.array_equal(A.rng_for(1, 2).random(5), A.rng_for(2, 1).random(5))


def test_tune_r_liefert_die_schmalste_breite_bei_n_stern():
    r = A.tune_r(1000, 0.05, 0.5)
    w = float(A.hoeffding_half(1000, 0.05, r, 0.5))
    for f in (0.1, 0.5, 2.0, 10.0):
        assert w <= float(A.hoeffding_half(1000, 0.05, r * f, 0.5)) + 1e-12
