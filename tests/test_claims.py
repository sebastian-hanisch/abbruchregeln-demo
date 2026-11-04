"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung aller Kantenmengen) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen (Strom 38; 1.000 Pfade beim Hinsehen, 2.000 Läufe bei den Abbruchregeln und Tests) und sind
reproduzierbar; sie stehen hier mit Bändern, die das Rauschen (Anteile auf etwa ±0,02 genau) und kleine Plattformunterschiede der Binomialziehung abdecken."""

import pytest

import abr_constants as C
import abr_evaluation as ev
from abr_scenario import pi_values

S = ev.Settings()
CASES = {"3x3 p0.9": ("3x3", 0.9), "3x3 p0.95": ("3x3", 0.95), "hantel p0.9": ("hantel", 0.9), "3x4 p0.9": ("3x4", 0.9), "3x3 p0.7": ("3x3", 0.7)}


def _a(case):
    netz, p = CASES[case]
    return ev.analyse(ev.Settings(netz=netz, p=p))


def _near(x, soll, tol):
    return abs(x - soll) <= tol


# --- exakte Werte ------------------------------------------------------------------------------------------------------------------------------

def test_exakte_ausfallwahrscheinlichkeiten_und_p_werte():
    for case, soll in (("3x3 p0.9", 0.05302), ("3x3 p0.95", 0.01184), ("hantel p0.9", 0.14969), ("3x4 p0.9", 0.05549), ("3x3 p0.7", 0.48901)):
        netz, p = CASES[case]
        assert pi_values(netz, p, 2)[0] == pytest.approx(soll, abs=2e-5), case
    assert pi_values("3x3", 0.7, 4)[1] == pytest.approx(0.04935, abs=2e-5)
    assert pi_values("hantel", 0.8, 3)[1] == pytest.approx(0.0596, abs=5e-5)
    assert pi_values("3x3", 0.9, 2)[1] == pytest.approx(0.0530, abs=5e-5)
    assert pi_values("3x3", 0.6, 5)[1] == pytest.approx(0.0614, abs=5e-5)


# --- Hinsehen -----------------------------------------------------------------------------------------------------------------------------------

PEEK = {"3x3 p0.9": (0.615, 0.558, 0.012, 0.000, 0.255), "3x3 p0.95": (0.808, 0.503, 0.009, 0.000, 0.701), "hantel p0.9": (0.616, 0.578, 0.014, 0.000, 0.054),
        "3x4 p0.9": (0.623, 0.537, 0.019, 0.000, 0.239), "3x3 p0.7": (0.583, 0.563, 0.016, 0.039, 0.041)}


@pytest.mark.parametrize("case", list(PEEK))
def test_irgendwann_ausserhalb(case):
    a = _a(case)
    for k, soll in zip(C.SEQ, PEEK[case]):
        tol = 0.06 if soll > 0.1 else 0.025
        assert _near(a.peek[k]["ever"], soll, tol), (case, k, a.peek[k]["ever"], soll)
    assert a.peek["beta"]["ever"] <= 0.05 and a.peek["wald"]["ever"] > 0.45 and a.peek["wilson"]["ever"] > 0.4


def test_wald_bei_festem_n():
    a = _a("3x3 p0.9")
    assert _near(a.peek["wald"]["at"][100], 0.100, 0.04) and _near(a.peek["wald"]["at"][1000], 0.057, 0.03) and _near(a.peek["wald"]["at"][10000], 0.041, 0.03)
    b = _a("3x3 p0.95")
    assert _near(b.peek["wald"]["at"][100], 0.283, 0.06) and _near(b.peek["asym"]["at"][100], 0.281, 0.06)


def test_asymptotische_folge_haelt_bei_haeufigen_ereignissen_und_nicht_bei_seltenen():
    assert _a("3x3 p0.7").peek["asym"]["ever"] < 0.07 and _a("hantel p0.9").peek["asym"]["ever"] < 0.09
    assert _a("3x3 p0.95").peek["asym"]["ever"] > 0.6 and _a("3x3 p0.9").peek["asym"]["ever"] > 0.18


# --- Breite --------------------------------------------------------------------------------------------------------------------------------------

def test_breitenverhaeltnisse():
    for case, beta, hoeff in (("3x3 p0.9", (1.74, 2.16), (3.46, 4.48)), ("3x3 p0.95", (1.91, 2.16), (7.11, 9.81)), ("hantel p0.9", (1.67, 2.16), (2.17, 2.73))):
        ws = _a(case).widths
        rb = [w["beta"] / w["wald"] for w in ws]
        rh = [w["hoeffding"] / w["wald"] for w in ws]
        assert min(rb) == pytest.approx(beta[0], abs=0.06) and max(rb) == pytest.approx(beta[1], abs=0.06), case
        assert min(rh) == pytest.approx(hoeff[0], abs=0.35) and max(rh) == pytest.approx(hoeff[1], abs=0.35), case
    assert [round(w["asym"] / w["wald"], 2) for w in _a("3x3 p0.9").widths] == pytest.approx([1.95, 1.55, 1.66, 1.82], abs=0.03)


# --- Abbruchregeln ---------------------------------------------------------------------------------------------------------------------------------

def test_relative_regel_haelt_abdeckung_und_orakelumfang():
    for case, nstar in (("3x3 p0.9", 1715), ("3x3 p0.95", 8016), ("hantel p0.9", 546)):
        a = _a(case)
        ex = ev.stop_experiment(a.pi, 0.2, S.seed)
        assert ex["nstar"] == pytest.approx(nstar, rel=0.002)
        for k in ("rel_wald", "rel_wilson"):
            r = ex["rows"][k]
            assert 0.92 < r["cov"] < 0.98 and 0.98 < r["n"] / ex["nstar"] < 1.08, (case, k)
        r = ex["rows"]["rel_beta"]
        assert r["cov"] > 0.995 and 3.4 < r["n"] / ex["nstar"] < 5.0, case


def test_verzerrung_der_relativen_regel_nach_genauigkeit():
    a = _a("3x3 p0.9")
    for eps, soll in ((0.5, 0.074), (0.2, 0.011), (0.1, 0.000)):
        assert _near(ev.stop_experiment(a.pi, eps, S.seed)["rows"]["rel_wald"]["bias"], soll, 0.02), eps
    b = _a("3x3 p0.95")
    assert _near(ev.stop_experiment(b.pi, 0.5, S.seed)["rows"]["rel_wald"]["bias"], 0.067, 0.025)


def test_absolute_regel_stoppt_mit_null_ereignissen():
    for case, zero, cov in (("3x3 p0.9", 0.19, 0.767), ("3x3 p0.95", 0.70, 0.290)):
        r = ev.stop_experiment(_a(case).pi, 0.2, S.seed)["rows"]["abs_wald"]
        assert _near(r["zero"], zero, 0.05) and _near(r["cov"], cov, 0.06) and r["cov_wilson"] > 0.9, case
    h = ev.stop_experiment(_a("hantel p0.9").pi, 0.2, S.seed)["rows"]["abs_wald"]
    assert h["zero"] < 0.03 and 0.92 < h["cov"] < 0.97


# --- Stopp nach k Ereignissen ------------------------------------------------------------------------------------------------------------------------

def test_inverse_stichprobe():
    ex = ev.inverse_experiment(_a("3x3 p0.9").pi, S.seed)
    for r, soll in zip(ex["rows"], (0.229, 0.104, 0.032, 0.010)):
        assert _near(r["bias_naive"], soll, 0.01 if soll > 0.05 else 0.005) and abs(r["bias_unbiased"]) < 0.006 and 0.945 < r["cov"] < 0.96, r["k"]
    ex2 = ev.inverse_experiment(_a("3x3 p0.95").pi, S.seed)
    assert _near(ex2["rows"][0]["bias_naive"], 0.247, 0.015)


# --- Monte-Carlo-Test ------------------------------------------------------------------------------------------------------------------------------

def _row(ex, factor):
    return next(r for r in ex["rows"] if abs(r["factor"] - factor) < 1e-9)["cells"]


def test_entscheidungsrisiko_beim_festen_umfang():
    ex = ev.mctest_experiment(S)
    for f, r99, r999, r9999 in ((0.6, 0.177, 0.002, 0.0), (0.8, 0.342, 0.073, 0.0), (0.9, 0.459, 0.237, 0.007), (0.98, 0.545, 0.440, 0.328), (1.0, 0.538, 0.531, 0.502), (1.1, 0.369, 0.228, 0.015)):
        c = _row(ex, f)
        assert _near(c["fix99"]["risk"], r99, 0.04) and _near(c["fix999"]["risk"], r999, 0.04) and _near(c["fix9999"]["risk"], r9999, 0.04), f


def test_besag_clifford_und_konfidenzfolge():
    ex = ev.mctest_experiment(S)
    for f, b10, b50 in ((0.6, 0.086, 0.001), (0.8, 0.266, 0.062), (0.9, 0.418, 0.239), (1.0, 0.535, 0.512)):
        c = _row(ex, f)
        assert _near(c["bc10"]["risk"], b10, 0.045) and _near(c["bc50"]["risk"], b50, 0.04), f
    assert _near(_row(ex, 4.0)["bc10"]["n"], 49, 3) and _near(_row(ex, 10.0)["bc10"]["n"], 20, 2) and _row(ex, 4.0)["bc10"]["risk"] == 0.0
    for r in ex["rows"]:
        assert r["cells"]["cs"]["risk"] <= 0.01, r["factor"]
    cs = {f: _row(ex, f)["cs"] for f in (0.02, 0.2, 0.6, 0.8, 0.9, 4.0, 10.0)}
    assert _near(cs[0.02]["n"], 162, 15) and _near(cs[0.2]["n"], 296, 30) and 1200 < cs[0.6]["n"] < 2200 and 6000 < cs[0.8]["n"] < 10500 and 25000 < cs[0.9]["n"] < 50000
    assert _near(cs[4.0]["n"], 54, 6) and _near(cs[10.0]["n"], 30, 2)
    assert _row(ex, 1.0)["cs"]["open"] > 0.95 and 0.0 < _row(ex, 0.98)["cs"]["open"] < 0.2


def test_gewaehltes_netz_im_test():
    ex = ev.mctest_experiment(S)
    own = ex["own"]["cells"]
    assert ex["pi_test"] == pytest.approx(0.0530, abs=5e-5) and 1.05 < ex["own"]["factor"] < 1.07
    assert _near(own["fix999"]["risk"], 0.311, 0.045) and _near(own["fix9999"]["risk"], 0.090, 0.04) and own["cs"]["risk"] <= 0.01 and 60000 < own["cs"]["n"] < 190000
    k = ev.mctest_experiment(ev.Settings(netz="3x3", p=0.7, tobs=4))
    c = k["own"]["cells"]
    assert k["pi_test"] == pytest.approx(0.04935, abs=2e-5) and _near(c["fix999"]["risk"], 0.486, 0.05) and _near(c["fix9999"]["risk"], 0.383, 0.05) and 0.5 < c["cs"]["open"] < 0.9


def test_absolute_regel_bei_weiteren_netzen():
    r = ev.stop_experiment(_a("3x4 p0.9").pi, 0.2, S.seed)["rows"]["abs_wald"]
    assert _near(r["zero"], 0.19, 0.05) and _near(r["cov"], 0.760, 0.06)
    r = ev.stop_experiment(_a("3x3 p0.7").pi, 0.2, S.seed)["rows"]["abs_wald"]
    assert r["zero"] == 0.0 and 0.93 < r["cov"] < 0.98
