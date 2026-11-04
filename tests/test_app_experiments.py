"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import abr_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_stop", "Absolute Regel (Wald-Halbbreite ≤ 0,01)"),
    ("run_mctest", "Der exakte p-Wert des gewählten Netzes ist"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_breite_und_inverse_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "Bei jedem n von 30 bis 10.000 hingesehen" in t
    assert "Halbbreite gegen Wald bei n = 100 bis 100.000" in t
    assert "Stopp beim k-ten Ereignis" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_stop").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="eps_select").set_value(0.5).run(timeout=TIMEOUT)
    assert not at.exception
    assert "relative Genauigkeit ε = 0,5" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    t = _texte(at)
    assert "Ausfallwahrscheinlichkeit π = 0,0530" in t
    at.sidebar.button(key="preset_Hantel mit Brücke").click().run(timeout=TIMEOUT)
    assert "Hantel, Haltewahrscheinlichkeit 0,90 je Kante: Ausfallwahrscheinlichkeit π = 0,1497" in _texte(at)


def test_jedes_netz_und_jede_haltewahrscheinlichkeit_laeuft_durch():
    at = _app()
    for netz in C.NETZE:
        for p in (0.5, 0.95):
            at.sidebar.selectbox(key="netz_select").set_value(netz).run(timeout=TIMEOUT)
            at.sidebar.select_slider(key="p_select").set_value(p).run(timeout=TIMEOUT)
            assert not at.exception, (netz, p, [str(e) for e in at.exception])


def test_spielwiese_rechnet_mit_eigenen_werten():
    at = _app()
    assert not at.error
    at.number_input(key="eigen_pi").set_value(0.2).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "eigenes π = 0,2000" in _texte(at)
    at.number_input(key="eigen_alpha").set_value(0.1).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="eps_select").set_value(0.1).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "0.85"
    at.query_params["t"] = "99"
    at.query_params["e"] = "0.3"
    at.query_params["g"] = "9x9"              # unbekanntes Netz: verworfen
    at.query_params["a"] = "0.07"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="p_select").value in (0.8, 0.9)                      # 0,85 liegt genau zwischen 0,8 und 0,9
    assert at.sidebar.select_slider(key="tobs_select").value == 6                            # oberhalb des Größten: das Größte
    assert at.sidebar.select_slider(key="eps_select").value in (0.2, 0.5)                    # 0,3 liegt zwischen 0,2 und 0,5
    assert at.sidebar.selectbox(key="netz_select").value == C.DEFAULT_NETZ
    assert at.sidebar.select_slider(key="alpha_select").value in (0.05, 0.1)
    assert at.sidebar.number_input(key="seed_input").value == 0
