"""Verdikte (reine Funktionen in abr_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import abr_constants as C
import abr_evaluation as ev


def _peek(wald=0.6, wilson=0.5, beta=0.012, hoeff=0.0, asym=0.25):
    ever = {"wald": wald, "wilson": wilson, "beta": beta, "hoeffding": hoeff, "asym": asym}
    return {k: {"ever": v, "at": {100: v / 10, 1000: 0.057, 10000: 0.05}} for k, v in ever.items()}


def _widths(pi=0.05):
    rows = []
    for n, f in ((100, 1.7), (1000, 1.87), (10000, 2.0), (100000, 2.1)):
        w = 1.96 * (pi * (1 - pi) / n) ** 0.5
        rows.append({"n": n, "S": round(n * pi), "wald": w, "wilson": w, "beta": f * w, "hoeffding": 3.5 * w, "asym": 1.6 * w})
    return tuple(rows)


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(label="Gitter 3×3, Haltewahrscheinlichkeit 0,90 je Kante", pi=0.053, pi_test=0.053, peek=_peek(), peek_n=10000, peek_n0=30, peek_reps=1000, widths=_widths())
    return dataclasses.replace(basis, **kw)


def test_hauptverdikt_warnt_beim_hinsehen_mit_zahlen():
    kind, text = ev.verdict_main(_a())
    assert kind == "warning"
    for frag in ("Gitter 3×3, Haltewahrscheinlichkeit 0,90 je Kante: Ausfallwahrscheinlichkeit π = 0,0530", "Bei jedem n von 30 bis 10.000 hingesehen", "in 1.000 Läufen irgendwann außerhalb",
                 "Wald 0,600, Wilson 0,500, Beta-Folge 0,012, Hoeffding-Folge 0,000, asymptotische Folge 0,250 (nominell 0,05)", "Zu einem festen n = 1.000 liegt Wald nur in 0,057 der Läufe außerhalb",
                 "Die Beta-Folge ist bei n = 1.000 das 1,87-Fache so breit wie Wald", "Die asymptotische Folge hält hier nicht", "taugen hier nicht zum Hinsehen"):
        assert frag in text, frag


def test_hauptverdikt_schwellen():
    # Wald genau beim Doppelten des Niveaus: noch kein Alarm; knapp darüber: Alarm
    assert ev.verdict_main(_a(peek=_peek(wald=0.10, asym=0.05)))[0] == "success"
    assert ev.verdict_main(_a(peek=_peek(wald=0.101, asym=0.05)))[0] == "warning"
    kind, text = ev.verdict_main(_a(peek=_peek(wald=0.06, asym=0.04)))
    assert kind == "success" and "asymptotische Folge hält hier nicht" not in text
    assert "asymptotische Folge hält hier nicht" in ev.verdict_main(_a(peek=_peek(asym=0.101)))[1] and "asymptotische Folge hält hier nicht" not in ev.verdict_main(_a(peek=_peek(asym=0.10)))[1]


def test_breitenverdikt():
    kind, text = ev.verdict_width(_a())
    assert kind == "info"
    for frag in ("bei n = 100 bis 100.000", "Beta-Folge 1,70- bis 2,10-fach", "asymptotische Folge 1,60- bis 1,60-fach", "Hoeffding-Folge 3,50- bis 3,50-fach", "etwa das 2,9- bis 4,4-Fache an Proben"):
        assert frag in text, frag
    null = _a(widths=tuple({**w, "wald": 0.0} for w in _widths()))
    assert "nicht bestimmbar" in ev.verdict_width(null)[1]


def _stop(zero=0.4, cov=0.55):
    row = lambda n, ref, c, b, z=0.0: {"n": n, "n_ref": ref, "cov": c, "cov_wilson": 0.97, "bias": b, "zero": z, "unstopped": 0.0}
    return {"pi": 0.03, "eps": 0.2, "nstar": 3105.0, "nmax": 300000, "reps": 2000,
            "rows": {"rel_wald": row(3156.0, 3105.0, 0.95, 0.016), "rel_wilson": row(3178.0, 3105.0, 0.947, 0.015), "rel_beta": row(13074.0, 3105.0, 1.0, 0.004), "abs_wald": row(687.0, 1118.0, cov, 0.2, zero)}}


def test_abbruchverdikt():
    kind, text = ev.verdict_stop(_stop())
    assert kind == "warning"
    for frag in ("π = 0,0300, relative Genauigkeit ε = 0,2, Orakelumfang n* = 3.105", "relativ mit Wald im Mittel n = 3.156 (1,02·n*), Abdeckung 0,950", "Verzerrung von p̂ +1,6 %", "mit Konfidenzfolge n = 13.074 (4,21·n*), Abdeckung 1,000",
                 "40 % der Läufe stoppen mit null Ereignissen", "Abdeckung des Wald-Endintervalls 0,550 (Wilson-Endintervall 0,970)"):
        assert frag in text, frag
    assert ev.verdict_stop(_stop(zero=0.2, cov=0.9))[0] == "info" and ev.verdict_stop(_stop(zero=0.21, cov=0.9))[0] == "warning" and ev.verdict_stop(_stop(zero=0.0, cov=0.84))[0] == "warning"


def test_inverse_verdikt():
    rows = [{"k": k, "bias_naive": b, "bias_unbiased": -0.0004, "cov": 0.951, "n": 100.0} for k, b in ((5, 0.211), (10, 0.098), (30, 0.031), (100, 0.009))]
    kind, text = ev.verdict_inverse({"pi": 0.1, "rows": rows, "reps": 1000})
    assert kind == "info"
    for frag in ("bei π = 0,1000", "+21,1 % (k = 5), +9,8 % (k = 10), +3,1 % (k = 30), +0,9 % (k = 100)", "0,0 %, 0,0 %, 0,0 %, 0,0 %", "0,951 bis 0,951"):
        assert frag in text, frag


def _mc(own_factor=1.06):
    cells = lambda r, n, o=0.0: {"risk": r, "n": n, "open": o}
    mk = lambda f, r999, r50, rcs, ncs, o=0.0, own=False: {"pi": f * 0.05, "alpha": 0.05, "exact_reject": f <= 1.0, "factor": f, "own": own,
                                                          "cells": {"fix99": cells(0.4, 99), "fix999": cells(r999, 999), "fix9999": cells(0.0, 9999), "bc10": cells(0.3, 200), "bc50": cells(r50, 1000), "cs": cells(rcs, ncs, o)}}
    return {"rows": [mk(0.6, 0.0, 0.0, 0.0, 1686.0), mk(0.9, 0.229, 0.241, 0.001, 36811.0), mk(1.1, 0.2, 0.2, 0.0, 38816.0)], "own": mk(own_factor, 0.311, 0.3, 0.0, 110786.0, 0.0, True), "alpha": 0.05, "pi_test": 0.053, "reps": 2000}


def test_mctest_verdikt():
    kind, text = ev.verdict_mctest(_mc())
    assert kind == "info"
    for frag in ("Testniveau α = 0,05", "bei π = 0,0450 (0,9·α)", "fest n = 999 0,229, Besag–Clifford (h = 50) 0,241, Konfidenzfolge 0,001", "im Mittel 36.811 Simulationen gegen 999",
                 "Der exakte p-Wert des gewählten Netzes ist 0,0530 (1,06·α)", "Risiko fest n = 999 0,311, Konfidenzfolge 0,000", "im Mittel 110.786 Simulationen (0 % unentschieden bis 2.000.000)"):
        assert frag in text, frag
