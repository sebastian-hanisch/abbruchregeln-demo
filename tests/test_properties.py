"""Eigenschaften, die für jede Wahl gelten müssen (nicht nur an einem Zahlenwert): Hinsehen, Breite, Abbruchregeln, inverse Stichprobe, Monte-Carlo-Test, Reproduzierbarkeit."""

import math

import numpy as np
import pytest

import abr_algorithm as A
import abr_constants as C
import abr_evaluation as ev


def _peek(pi, reps=500, N=5000, seed=3):
    return ev.peek_experiment(pi, seed, reps=reps, N=N)["rows"]


def test_hinsehen_zerstoert_die_abdeckung_von_wald_und_wilson_aber_nicht_der_beta_folge():
    for pi in (0.3, 0.05):
        r = _peek(pi)
        assert r["wald"]["ever"] > 0.3 and r["wilson"]["ever"] > 0.3
        assert r["beta"]["ever"] <= 0.05 and r["hoeffding"]["ever"] <= 0.05


def test_zu_festem_n_liegt_wald_nahe_am_niveau():
    r = _peek(0.3, reps=1000, N=1000)
    assert r["wald"]["at"][1000] == pytest.approx(0.05, abs=0.025)


def test_asymptotische_folge_versagt_bei_seltenen_ereignissen_und_haelt_bei_haeufigen():
    assert _peek(0.01)["asym"]["ever"] > 0.25
    assert _peek(0.3)["asym"]["ever"] < 0.1


def test_breite_der_beta_folge_ist_immer_groesser_als_wald_und_etwa_doppelt():
    for pi in (0.3, 0.1, 0.03):
        for w in ev.width_table(pi):
            assert w["beta"] > w["wald"] and 1.4 < w["beta"] / w["wald"] < 2.4, (pi, w["n"])


def test_breiten_fallen_mit_n():
    ws = ev.width_table(0.1)
    for k in C.SEQ:
        v = [w[k] for w in ws]
        assert v == sorted(v, reverse=True)


def test_hoeffding_folge_ist_bei_seltenen_ereignissen_viel_breiter_als_beta():
    w = ev.width_table(0.01)[1]
    assert w["hoeffding"] > 4 * w["beta"]


def test_relative_regel_haelt_die_abdeckung_und_den_orakelumfang():
    ex = ev.stop_experiment(0.1, 0.2, 5, reps=1000)
    for k in ("rel_wald", "rel_wilson"):
        r = ex["rows"][k]
        assert r["cov"] == pytest.approx(0.95, abs=0.03) and r["n"] / ex["nstar"] == pytest.approx(1.02, abs=0.06), k
    assert ex["rows"]["rel_beta"]["cov"] > 0.99 and 3.0 < ex["rows"]["rel_beta"]["n"] / ex["nstar"] < 5.5


def test_absolute_regel_stoppt_bei_seltenen_ereignissen_mit_null_ereignissen():
    ex = ev.stop_experiment(0.01, 0.2, 5, reps=1000)
    r = ex["rows"]["abs_wald"]
    assert r["zero"] > 0.5 and r["cov"] < 0.4 and r["cov_wilson"] > 0.9
    assert ex["rows"]["rel_wald"]["zero"] == 0.0                   # die relative Regel verlangt p̂ > 0


def test_verzerrung_beim_stopp_ist_nach_oben_gerichtet_und_waechst_mit_lockerer_genauigkeit():
    b = [ev.stop_experiment(0.03, eps, 5, reps=1000)["rows"]["rel_wald"]["bias"] for eps in (0.1, 0.5)]
    assert 0 <= b[0] < b[1] and b[1] > 0.03


def test_inverse_stichprobe_naiv_verzerrt_und_korrigiert_erwartungstreu():
    ex = ev.inverse_experiment(0.1, 5, reps=50000)
    naive = [r["bias_naive"] for r in ex["rows"]]
    assert naive == sorted(naive, reverse=True) and naive[0] > 0.15 and naive[-1] < 0.03
    assert all(abs(r["bias_unbiased"]) < 0.01 for r in ex["rows"])
    assert all(0.93 < r["cov"] < 0.97 for r in ex["rows"])


def test_festes_n_hat_das_groesste_risiko_an_der_schwelle_und_es_faellt_mit_n():
    r99, r999, r9999 = (ev.mctest_cells(0.045, 0.05, 5, t)["cells"][k]["risk"] for t in (1,) for k in ("fix99", "fix999", "fix9999"))
    assert r99 > r999 > r9999
    near = ev.mctest_cells(0.049, 0.05, 5, 2)["cells"]["fix999"]["risk"]
    far = ev.mctest_cells(0.01, 0.05, 5, 3)["cells"]["fix999"]["risk"]
    assert near > 0.3 > far


def test_konfidenzfolgen_entscheidung_haelt_das_risiko_und_kostet_zur_schwelle_hin_mehr():
    cells = [ev.mctest_cells(f * 0.05, 0.05, 5, i)["cells"]["cs"] for i, f in enumerate((0.2, 0.6, 0.9))]
    assert all(c["risk"] <= 0.02 for c in cells)
    ns = [c["n"] for c in cells]
    assert ns[0] < ns[1] < ns[2]


def test_besag_clifford_spart_bei_grossem_p_proben():
    cells = ev.mctest_cells(0.5, 0.05, 5, 4)["cells"]
    assert cells["bc10"]["n"] < 30 and cells["bc50"]["n"] < 150 and cells["bc10"]["risk"] == 0.0


def test_gleiche_einstellung_gleiche_zahlen():
    assert ev.peek_experiment(0.1, 3, reps=250, N=2000) == ev.peek_experiment(0.1, 3, reps=250, N=2000)
    assert ev.stop_experiment(0.1, 0.2, 3, reps=300) == ev.stop_experiment(0.1, 0.2, 3, reps=300)
    assert ev.peek_experiment(0.1, 3, reps=250, N=2000) != ev.peek_experiment(0.1, 4, reps=250, N=2000)
