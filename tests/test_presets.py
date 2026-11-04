"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import abr_constants as C
import abr_evaluation as ev
import abr_presets as P
from abr_scenario import pi_values

STANDARD = "Standardfall (Voreinstellung)"


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX and preset["netz"] in C.NETZE, name
        assert preset["p"] in C.P_CHOICES and preset["tobs"] in C.TOBS_CHOICES and preset["eps"] in C.EPS_CHOICES and preset["alpha"] in C.ALPHA_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    for name, preset in C.PRESETS.items():
        if name != STANDARD:
            assert preset != C.PRESETS[STANDARD], name


def test_standardfall_liegt_nah_am_testniveau():
    s = _settings(STANDARD)
    u, pt = pi_values(s.netz, s.p, s.tobs)
    assert u == pytest.approx(0.0530, abs=2e-4) and pt == pytest.approx(u) and 1.0 < u / s.alpha < 1.1
    a = ev.analyse(s)
    assert ev.verdict_main(a)[0] == "warning"


def test_preset_seltene_ausfaelle_laesst_die_asymptotische_folge_versagen():
    s = _settings("Seltene Ausfälle")
    a = ev.analyse(s)
    assert a.pi == pytest.approx(0.0118, abs=1e-4) and a.peek["asym"]["ever"] > 0.3 and "asymptotische Folge hält hier nicht" in ev.verdict_main(a)[1]
    ex = ev.stop_experiment(a.pi, s.eps, s.seed, reps=500)
    assert ex["rows"]["abs_wald"]["zero"] > 0.4


def test_preset_knapp_an_der_schwelle_hat_den_p_wert_0_0494():
    s = _settings("Knapp an der Schwelle")
    _, pt = pi_values(s.netz, s.p, s.tobs)
    assert pt == pytest.approx(0.0494, abs=2e-4) and pt < s.alpha
    c = ev.mctest_cells(pt, s.alpha, s.seed, 1, reps=500)["cells"]
    assert c["fix999"]["risk"] > 0.3 and c["cs"]["open"] > 0.0


def test_preset_hantel_hat_haeufige_ereignisse():
    s = _settings("Hantel mit Brücke")
    u, _ = pi_values(s.netz, s.p, s.tobs)
    assert u == pytest.approx(0.1497, abs=2e-4)


def test_preset_lockere_genauigkeit_ist_verzerrt():
    s = _settings("Lockere Genauigkeit")
    ex = ev.stop_experiment(pi_values(s.netz, s.p, s.tobs)[0], s.eps, s.seed, reps=1000)
    assert s.eps == 0.5 and ex["rows"]["rel_wald"]["bias"] > 0.03


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(_settings(name))
    assert 0 < a.pi < 1 and 0 < a.pi_test < 1
