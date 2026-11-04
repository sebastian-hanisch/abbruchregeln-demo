"""Randfälle: seltene und häufige Ereignisse, null Ereignisse, Genauigkeit unerreichbar, α an der Schwelle, Eingabeprüfung der Spielwiese, Zahlenformate."""

import math

import numpy as np
import pytest

import abr_algorithm as A
import abr_constants as C
import abr_evaluation as ev
from abr_scenario import check_custom, named_netz, pi_values


def test_netze_und_wahrscheinlichkeiten():
    for key in C.NETZE:
        name, n, edges = named_netz(key)
        u, ptest = pi_values(key, 0.9, 2)
        assert 0 < u < 1 and ptest == pytest.approx(u)                 # T_obs = 2 ist der Ausfall selbst
        assert pi_values(key, 0.9, 6)[1] < u


def test_extreme_haltewahrscheinlichkeiten_sind_endlich():
    for p in C.P_CHOICES:
        u, pt = pi_values("3x4", p, 6)
        assert 0 < pt <= u < 1


def test_ohne_ereignisse_ist_die_wald_breite_null_und_die_beta_folge_reicht_von_null_bis_zu_einer_schranke():
    assert float(A.wald_half(0.0, 50)) == 0.0
    lo, hi = A.beta_cs(np.array([0]), np.array([50]), 0.05)
    assert lo[0] == 0.0 and 0 < hi[0] < 0.2
    lo, hi = A.beta_cs(np.array([50]), np.array([50]), 0.05)
    assert hi[0] == 1.0 and 0.8 < lo[0] < 1


def test_breitentabelle_bei_null_ereignissen_hat_ein_verdikt_ohne_zahl():
    a = ev.analyse(ev.Settings(), pi=0.0004, label="eigenes π")             # bei n = 100 gilt S = 0
    assert a.widths[0]["S"] == 0 and a.widths[0]["wald"] == 0.0
    kind, text = ev.verdict_width(a)
    assert kind == "info" and "nan" not in text


def test_sehr_haeufiges_ereignis_stoppt_am_ersten_blick():
    ex = ev.stop_experiment(0.89, 0.2, 5, reps=300)
    assert ex["nstar"] < 30 and ex["rows"]["rel_wald"]["n"] < 60 and ex["rows"]["rel_wald"]["unstopped"] == 0.0


def test_genauigkeit_ausserhalb_der_hoechstzahl_bleibt_unbeendet():
    ex = ev.stop_experiment(0.0005, 0.05, 5, reps=200)                      # n* ≈ 3 · 10^6, Konfidenzfolge braucht ein Mehrfaches, Höchstzahl 6 · 10^6
    assert ex["nmax"] <= C.STOP_NMAX_MAX and ex["rows"]["rel_beta"]["unstopped"] > 0.5


def test_test_an_der_schwelle_bleibt_unentschieden():
    c = ev.mctest_cells(0.05, 0.05, 5, 1, reps=400)["cells"]
    assert c["cs"]["open"] > 0.9 and 0.4 < c["fix999"]["risk"] < 0.65


def test_test_bei_pi_gleich_alpha_hat_die_exakte_entscheidung_ablehnen():
    assert ev.mctest_cells(0.05, 0.05, 5, 1, reps=100)["exact_reject"] is True
    assert ev.mctest_cells(0.0501, 0.05, 5, 1, reps=100)["exact_reject"] is False


def test_hoechstzahl_und_pruefgitter_der_konfidenzfolgen_entscheidung():
    rej, ns, st = A.cs_decision(0.2, 0.05, 0.01, 1000, 500, A.rng_for(1))
    assert st.all() and ns.max() <= 1000 and (rej == False).all()


def test_spielwiese_eingabepruefung():
    assert check_custom(0.03, 0.05) == (0.03, 0.05)
    for args, frag in (((None, 0.05), "keine Zahl"), ((float("nan"), 0.05), "keine Zahl"), ((0.03, float("inf")), "keine Zahl"), ((0.9, 0.05), "π muss"), ((0.0001, 0.05), "π muss"), ((0.03, 0.5), "Testniveau")):
        with pytest.raises(ValueError, match=frag):
            check_custom(*args)


def test_zahlenformate():
    assert ev.de(1234.5, 1) == "1.234,5" and ev.de(-0.5, 1) == "−0,5"
    assert ev.sprozent(0.229) == "+22,9 %" and ev.sprozent(-0.031) == "−3,1 %" and ev.sprozent(-0.0001) == "0,0 %" and ev.sprozent(float("nan")) == "nicht bestimmbar"
    assert ev.pi_text(0.0530152) == "0,0530" and ev.pi_text(0.0002701) == "0,00027"
    assert ev.pr(0.6149) == "0,615" and ev.dn(float("inf")) == "unendlich" and ev.prozent(float("nan")) == "nicht bestimmbar"


def test_stichprobe_mit_wenigen_pfaden_und_kurzer_laenge_rechnet():
    ex = ev.peek_experiment(0.1, 1, reps=10, N=200)
    assert set(ex["rows"]) == set(C.SEQ) and set(ex["rows"]["wald"]["at"]) == {100}


def test_ratio_hilfsfunktion():
    assert ev.ratio(2.0, 4.0) == 0.5 and np.isnan(ev.ratio(1.0, 0.0))
