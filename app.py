"""Abbruchregeln: wann man aufhören darf - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 14 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Eine Simulation läuft, bis es reicht. Intervalle zum festen Umfang verlieren beim Hinsehen und Anhalten ihre Abdeckung; jederzeit gültige
Konfidenzfolgen halten sie und kosten etwa das Drei- bis Fünffache an Proben; die Regel »bis die relative Genauigkeit erreicht ist« ist brauchbar, die absolute ohne Schutz gegen null Ereignisse nicht, wenn Ereignisse selten sind; ein
Monte-Carlo-Test kann sein Entscheidungsrisiko beschränken und bezahlt es an der Schwelle. Vehikel ist die Zuverlässigkeit der Netze der Reihe; alle Wahrheiten sind exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import dataclasses

import streamlit as st

import abr_constants as C
import abr_evaluation as ev
from abr_evaluation import Settings, de, de_e, pi_text, pr, prozent, sprozent
from abr_layout import dunkles_design, kopf_css
from abr_pdf_export import build_pdf
from abr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from abr_scenario import check_custom, named_netz
from abr_visualization import build_inverse, build_mctest, build_peek, build_stop, build_width

st.set_page_config(page_title="Abbruchregeln: wann man aufhören darf – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Hinsehen bei jedem n messen …")
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Abbruchregeln laufen lassen …")
def _stop(pi, eps, seed):
    return ev.stop_experiment(pi, eps, seed)


@st.cache_data(show_spinner="Stopp nach k Ereignissen …")
def _inverse(pi, seed):
    return ev.inverse_experiment(pi, seed)


@st.cache_data(show_spinner="Monte-Carlo-Tests laufen lassen …")
def _mctest(settings):
    return ev.mctest_experiment(settings)


@st.cache_data(show_spinner=False)
def _custom(settings, pi, alpha):
    pi, alpha = check_custom(pi, alpha)
    return ev.custom_result(settings, pi, alpha)


def _netz_label(key):
    name, n, edges = named_netz(key)
    return f"{name} ({len(edges)} Kanten)"


def _p_label(p):
    return f"{p:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in abr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🛑 Abbruchregeln: wann man aufhören darf")
st.markdown(
    """
Eine Simulation läuft, bis »es reicht«. Die **Intervalle der bisherigen Stücke** sind für einen **festen Umfang** gebaut: wer **bei jedem n hinsieht** und anhält, sobald etwas passt, verliert ihre Abdeckung (die Wahrheit liegt irgendwann außerhalb, obwohl sie bei einem einzelnen großen n etwa zu 95 % drin liegt).
**Jederzeit gültige Konfidenzfolgen** halten die Abdeckung bei beliebigem Anhalten und kosten dafür etwa das Drei- bis Fünffache an Proben; nur die nicht asymptotischen halten auch bei seltenen Ereignissen. Die Regel **»bis die relative Genauigkeit erreicht ist«** ist mit Schutz gegen null Ereignisse brauchbar, die **absolute** nicht, wenn Ereignisse selten sind.
Ein **Monte-Carlo-Test** (stimmt die Ausfallwahrscheinlichkeit mit der Spezifikation?) kann sein Entscheidungsrisiko beschränken, bezahlt es aber **an der Schwelle**.

Wer die Idee ohne Fachwörter lesen möchte, öffnet **💬 In einfachen Worten** und findet dort auch den Link auf die ausführliche Erklärseite.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/hinsehen-aufhoeren-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Eine Simulation läuft, und man hört auf, sobald das Ergebnis gut genug aussieht. Die übliche Fehlerspanne um das Ergebnis verspricht aber nur dann eine kleine Fehlerquote, wenn die Zahl der Versuche vorher feststeht. Wer nach jedem Versuch nachsieht und anhält, sobald es passt, sucht sich einen günstigen Moment aus: Der Zufall lässt den Wert schwanken, und irgendwann liegt die Wahrheit außerhalb der Fehlerspanne. Die Demo misst das an erfundenen Netzen, deren richtige Antwort genau bekannt ist. **Jederzeit gültige Konfidenzfolgen** sind breiter gebaut, sodass das Versprechen für alle Zeitpunkte zugleich gilt. Dafür brauchen sie etwa das Drei- bis Fünffache an Versuchen. Ein Haken bleibt: Wird ein Test knapp an der Grenze entschieden, wird er sehr teuer.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/hinsehen-aufhoeren-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktionieren die Regeln", expanded=True):
    st.markdown(
        """
Gesucht ist eine Wahrscheinlichkeit π: die **Ausfallwahrscheinlichkeit** U = 1 − R(p) eines Netzes, dessen Kanten unabhängig mit Wahrscheinlichkeit p halten, oder der **p-Wert** P(T ≥ T_obs) der Komponentenzahl T nach Kantenausfall (ein Test, ob eine beobachtete Zahl Komponenten unter dem Modell ungewöhnlich ist). Beides ist exakt aus der Aufzählung aller Kantenmengen bekannt; die Simulation zieht nur Einsen und Nullen, X_i = 1 mit Wahrscheinlichkeit π.

1. **Intervall zum festen Umfang:** Wald p̂ ± 1,96 √(p̂(1 − p̂)/n) oder Wilson; gültig zu **einem** vorher festgelegten n.
2. **Konfidenzfolge:** eine Folge von Intervallen, die mit Wahrscheinlichkeit 95 % **für alle n zugleich** die Wahrheit enthält (Satz von Ville). Die **Beta-Mischungs-Folge** mischt die Alternativen π über eine Beta-Verteilung und ist nicht asymptotisch; die **Hoeffding-Mischung** nutzt nur die Schranke 1/4 für die Streuung; die **asymptotische Folge** setzt die geschätzte Streuung ein und gilt nur bei genug Ereignissen.
3. **Abbruchregel:** höre auf, sobald die Halbbreite klein genug ist: **relativ** (≤ ε · p̂) oder **absolut** (≤ h).
4. **Stopp nach k Ereignissen** (inverse Stichprobe): ziehe bis zum k-ten Ausfall.
5. **Monte-Carlo-Test:** ziehe Ausschläge X_i = 1[T_i ≥ T_obs] und lehne ab, wenn der geschätzte p-Wert ≤ α ist; das **Entscheidungsrisiko** ist die Wahrscheinlichkeit einer anderen Entscheidung als beim exakten p-Wert.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Ereignis**")
    netz = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select", help="Jede Kante hält unabhängig mit Wahrscheinlichkeit p; die Ausfallwahrscheinlichkeit und die Komponentenzahl werden aus allen Kantenmengen exakt aufgezählt.")
    p = st.select_slider("Haltewahrscheinlichkeit p je Kante", options("p_select"), key="p_select", format_func=_p_label, help="Je größer p, desto seltener fällt das Netz aus.")
    tobs = st.select_slider("Beobachtete Komponentenzahl T_obs (Test)", options("tobs_select"), key="tobs_select", help="Der Test fragt: wie wahrscheinlich sind mindestens so viele Komponenten unter dem Modell? T_obs = 2 ist der Ausfall selbst.")
    st.markdown("**Regeln**")
    eps = st.select_slider("Relative Genauigkeit ε der Abbruchregel", options("eps_select"), key="eps_select", format_func=_p_label, help="Die relative Regel hält an, sobald die Halbbreite höchstens ε mal die Schätzung ist.")
    alpha = st.select_slider("Testniveau α", options("alpha_select"), key="alpha_select", format_func=_p_label, help="Der Monte-Carlo-Test lehnt ab, wenn der geschätzte p-Wert höchstens α ist.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"netz_select": netz, "p_select": float(p), "tobs_select": int(tobs), "eps_select": float(eps), "alpha_select": float(alpha), "seed_input": int(seed)})

settings = Settings(str(netz), float(p), int(tobs), float(eps), float(alpha), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"{a.label}; {de(a.peek_reps, 0)} Läufe, Hinsehen bei jedem n von {a.peek_n0} bis {de(a.peek_n, 0)}; Intervalle zum Niveau 95 %.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ausfallwahrscheinlichkeit π", pi_text(a.pi), help="Exakt aus der Aufzählung aller Kantenmengen: Wahrscheinlichkeit, dass das Netz nicht zusammenhängt.")
m2.metric("Wald, irgendwann außerhalb", pr(a.peek["wald"]["ever"]), delta=f"bei festem n = 1.000: {pr(a.peek['wald']['at'][1000])}", delta_color="off", help="Anteil der Läufe, in denen die Wahrheit bei irgendeinem n außerhalb des Wald-Intervalls liegt (nominell 0,05).")
m3.metric("Beta-Folge, irgendwann außerhalb", pr(a.peek["beta"]["ever"]), delta="nominell höchstens 0,05", delta_color="off", help="Dasselbe für die jederzeit gültige Beta-Mischungs-Folge.")
m4.metric("Exakter p-Wert (Test)", pi_text(a.pi_test), delta=f"P(T ≥ {settings.tobs})", delta_color="off", help="Wahrscheinlichkeit von mindestens T_obs Komponenten unter dem Modell (Experiment 5).")
st.plotly_chart(build_peek(a), width="stretch", key="main_chart")
st.caption("Anteil der Läufe, in denen die Wahrheit bei irgendeinem n zwischen 30 und 10.000 außerhalb des Intervalls liegt. Bei einem einzelnen großen n wären es etwa 5 %; wer bei jedem n hinsieht, trifft viel öfter einen Fehlgriff.")

if st.button("📄 PDF der Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="abbruchregeln.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Darf man bei jedem n hinsehen und anhalten, sobald das Intervall passt?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Hinsehen bei jedem n")
st.caption("Fünf Intervalle beziehungsweise Folgen, die Wahrheit π bekannt: der Anteil der Läufe, in denen sie bei irgendeinem n außerhalb liegt, und der Anteil bei den festen n = 100, 1.000 und 10.000.")
st.dataframe({"Folge": [C.SEQ_LABELS[k] for k in C.SEQ], "irgendwann außerhalb": [pr(a.peek[k]["ever"]) for k in C.SEQ],
              **{f"außerhalb bei n = {de(n, 0)}": [pr(a.peek[k]["at"][n]) for k in C.SEQ] for n in (100, 1000, 10000)}}, hide_index=True, width="stretch", key="peek_table")

st.markdown("---")
st.subheader("🔬 Experiment 2: Was die Zeit-Gleichmäßigkeit kostet")
st.caption("Halbbreite der fünf Intervalle bei S = round(n·π) Ereignissen: je enger, desto weniger Proben. Die Beta-Folge ist etwa doppelt so breit wie Wald, die Hoeffding-Folge bei seltenen Ereignissen vielfach, weil sie nur die Streuungsschranke 1/4 kennt.")
st.plotly_chart(build_width(a), width="stretch", key="width_chart")
st.dataframe({"n": [de(w["n"], 0) for w in a.widths], "Ereignisse S": [w["S"] for w in a.widths], **{C.SEQ_LABELS[k]: [de_e(w[k]) for w in a.widths] for k in C.SEQ},
              "Beta / Wald": [ev.dn(ev.ratio(w["beta"], w["wald"]), 2) for w in a.widths]}, hide_index=True, width="stretch", key="width_table")
kind, text = ev.verdict_width(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Abbruchregeln")
st.caption(f"Vier Regeln auf einem Prüfgitter (erster Blick bei {C.STOP_N0}, Faktor {de(C.STOP_FAC, 2)}), {de(C.STOP_REPS, 0)} Läufe: relativ (Halbbreite ≤ ε·p̂ mit p̂ > 0) mit Wald, Wilson und Konfidenzfolge sowie absolut (Wald-Halbbreite ≤ {de(C.H_ABS, 2)}, ohne Schutz gegen null Ereignisse).")
if st.button("▶ Messung starten", key="run_stop"):
    st.session_state["stop_done"] = True
if st.session_state.get("stop_done"):
    ex3 = _stop(a.pi, settings.eps, settings.seed)
    st.plotly_chart(build_stop(ex3), width="stretch", key="stop_chart")
    from abr_evaluation import RULE_LABELS, RULES
    st.dataframe({"Regel": [RULE_LABELS[k] for k in RULES], "mittleres n": [de(ex3["rows"][k]["n"], 0) for k in RULES], "Vergleichsumfang": [de(ex3["rows"][k]["n_ref"], 0) for k in RULES],
                  "Abdeckung Endintervall": [pr(ex3["rows"][k]["cov"]) for k in RULES], "Abdeckung Wilson-Endintervall": [pr(ex3["rows"][k]["cov_wilson"]) for k in RULES],
                  "Verzerrung von p̂": [sprozent(ex3["rows"][k]["bias"]) for k in RULES], "Stopp mit null Ereignissen": [prozent(ex3["rows"][k]["zero"], 0) for k in RULES],
                  "ohne Stopp": [prozent(ex3["rows"][k]["unstopped"], 0) for k in RULES]}, hide_index=True, width="stretch", key="stop_table")
    kind, text = ev.verdict_stop(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Stopp nach k Ereignissen")
st.caption(f"Man zieht, bis das k-te Ausfallereignis eintritt (negative Binomialverteilung, {de(C.INV_REPS, 0)} Läufe je k). Die naive Schätzung k/n ist nach oben verzerrt, (k − 1)/(n − 1) nicht.")
ex4 = _inverse(a.pi, settings.seed)
st.plotly_chart(build_inverse(ex4), width="stretch", key="inverse_chart")
st.dataframe({"k": [r["k"] for r in ex4["rows"]], "mittleres n": [de(r["n"], 0) for r in ex4["rows"]], "Verzerrung k/n": [sprozent(r["bias_naive"]) for r in ex4["rows"]],
              "Verzerrung (k−1)/(n−1)": [sprozent(r["bias_unbiased"]) for r in ex4["rows"]], "Abdeckung Wald um k/n": [pr(r["cov"]) for r in ex4["rows"]]}, hide_index=True, width="stretch", key="inverse_table")
kind, text = ev.verdict_inverse(ex4)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Monte-Carlo-Test")
st.caption(f"Entscheidungsrisiko (andere Entscheidung als beim exakten p-Wert, Ablehnen genau bei π ≤ α) und Zahl der Simulationen für sechs Verfahren, bei π als Vielfachem des Testniveaus α = {de(settings.alpha, 2)} und beim exakten p-Wert des gewählten Netzes; {de(C.MC_REPS, 0)} Läufe je Zelle. Die Konfidenzfolge (δ = {de(C.MC_DELTA, 2)}) hört auf, sobald α nicht mehr in der Folge liegt, höchstens bei {de(C.MC_CAP, 0)}.")
if st.button("▶ Messung starten", key="run_mctest"):
    st.session_state["mctest_done"] = True
if st.session_state.get("mctest_done"):
    ex5 = _mctest(settings)
    left, right = build_mctest(ex5)
    c1, c2 = st.columns(2)
    c1.plotly_chart(left, width="stretch", key="mctest_risk")
    c2.plotly_chart(right, width="stretch", key="mctest_cost")
    rows5 = sorted(ex5["rows"] + [ex5["own"]], key=lambda r: r["factor"])
    st.dataframe({"π": [pi_text(r["pi"]) + (" (gewähltes Netz)" if r["own"] else "") for r in rows5], "π / α": [de(r["factor"], 2) for r in rows5], "exakte Entscheidung": ["ablehnen" if r["exact_reject"] else "nicht ablehnen" for r in rows5],
                  **{f"Risiko, {C.MC_LABELS[k]}": [pr(r["cells"][k]["risk"]) for r in rows5] for k in C.MC_DESIGNS}, **{f"Simulationen, {C.MC_LABELS[k]}": [de(r["cells"][k]["n"], 0) for r in rows5] for k in C.MC_DESIGNS}},
                 hide_index=True, width="stretch", key="mctest_table")
    kind, text = ev.verdict_mctest(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: eine eigene Wahrscheinlichkeit"):
    st.caption("Eigenes π (Ausfallwahrscheinlichkeit beziehungsweise p-Wert) und eigenes Testniveau α. Es zeigt das Hinsehen (Experiment 1) und das Entscheidungsrisiko der sechs Test-Verfahren (Experiment 5) für dieses π.")
    s1, s2 = st.columns(2)
    pi_in = s1.number_input("π", C.CUSTOM_PI_MIN, C.CUSTOM_PI_MAX, 0.03, 0.005, key="eigen_pi", format="%.4f")
    al_in = s2.number_input("Testniveau α", C.CUSTOM_ALPHA_MIN, C.CUSTOM_ALPHA_MAX, 0.05, 0.005, key="eigen_alpha", format="%.3f")
    try:
        cres = _custom(settings, float(pi_in), float(al_in))
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        st.plotly_chart(build_peek(ca), width="stretch", key="custom_peek")
        kind, text = ev.verdict_main(ca)
        getattr(st, kind)(text)
        mc = cres["mctest"]
        st.dataframe({"Verfahren": [C.MC_LABELS[k] for k in C.MC_DESIGNS], "Entscheidungsrisiko": [pr(mc["cells"][k]["risk"]) for k in C.MC_DESIGNS], "mittlere Simulationen": [de(mc["cells"][k]["n"], 0) for k in C.MC_DESIGNS],
                      "unentschieden": [prozent(mc["cells"][k]["open"], 0) for k in C.MC_DESIGNS]}, hide_index=True, width="stretch", key="custom_table")
        st.caption(f"Exakte Entscheidung bei π = {pi_text(float(pi_in))} und α = {de(float(al_in), 3)}: {'ablehnen (π ≤ α)' if mc['exact_reject'] else 'nicht ablehnen (π > α)'}.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Umfang steht vorher fest** | Wer bei jedem n hinsieht und anhält, hat mit Wald oder Wilson die Wahrheit in etwa 46 bis 82 % der Läufe irgendwann außerhalb (Experiment 1, über alle wählbaren Netze und p). | Experimente 4 (A/B-Tests) |
| **Das Intervall ist nicht asymptotisch** | Die asymptotische Konfidenzfolge versagt bei seltenen Ereignissen (Preset »Seltene Ausfälle«); nur die Beta-Mischung hält bei jedem n. | – |
| **Zeit-Gleichmäßigkeit ist kostenlos** | Sie kostet 1,6- bis 2,2-fache Breite, also drei- bis fünfmal so viele Proben (Experiment 2). | – |
| **Die Abbruchregel ist relativ** | Die absolute Halbbreite stoppt bei seltenen Ereignissen mit null Ereignissen (Experiment 3); die relative Regel ist leicht nach oben verzerrt, bei lockerer Genauigkeit spürbar (+7 bis +10 %; Preset »Lockere Genauigkeit«). | Stück 1 |
| **Stopp nach k Ereignissen mit k/n** | k/n ist nach oben verzerrt; (k − 1)/(n − 1) nicht (Experiment 4). | – |
| **Das Testrisiko ist beschränkt** | Es lässt sich beschränken, aber die Kosten wachsen wie 1/(π − α)², und bei π = α entscheidet kein Verfahren zuverlässig (Experiment 5, Preset »Knapp an der Schwelle«). | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Folgen sind **exakt Bernoulli(π)**; die Netze (Hantel, Gitter) sind **erfunden**, und die Ausfallwahrscheinlichkeit wie der p-Wert der Komponentenzahl sind exakt aus der Aufzählung aller Kantenmengen bekannt. Die Zufallszahlen kommen aus festen Strömen (SplitMix64);
die Binomial-Zuwächse auf dem Prüfgitter ziehen mit `numpy.random.Generator` bei festem, aus dem Strom abgeleitetem Seed (die Bänder der Tests decken Versionsunterschiede ab).

**Grenzen.** Gemessen werden nur **Bernoulli-Folgen**, keine Mittelwerte stetiger Größen mit geschätzter Streuung (dort gelten die Aussagen zur asymptotischen Folge). Es gibt eine Beta-Mischung (Beta(½, ½)) und eine Abstimmung (n* = 1.000) für die Hoeffding-Mischung und die asymptotische Folge; die Wetten-Folgen von Waudby-Smith und Ramdas
und der Test von Fischer und Ramdas sind **nicht gebaut**, ebenso Gandys Algorithmus; die Konfidenzfolgen-Entscheidung in Experiment 5 ist ein eigenes, verwandtes Verfahren. Beim Stopp wird auf einem geometrischen Prüfgitter geprüft (gültig, aber gröber als bei jedem n).

**Bekannte Probleme und Unsicherheiten.** Die Wiederholungen (1.000 bis 2.000 je Zelle) lassen Anteile auf etwa ±0,02 genau; die Risiken im Experiment 5 rauschen entsprechend. Die Kosten der Konfidenzfolgen-Entscheidung sind bei π = α unbegrenzt und bei π nahe α nur bis zur Höchstzahl gemessen.
Die Literaturangaben (Waudby-Smith u. a. 2024, Gandy 2009, Fischer und Ramdas 2024) sind nur im Abstract eingesehen; die Beta-Mischung ist eine Ville-Folge aus dem Gedächtnis (Robbins-Mischungsargument), die Schranke wird per Simulation geprüft. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/hinsehen-aufhoeren-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine Folge von Ja-Nein-Ausgängen $X_1, X_2, \dots$ mit unbekannter Wahrscheinlichkeit $\pi$. Gesucht ist ein Bereich für $\pi$, der die Wahrheit mit Wahrscheinlichkeit mindestens $1-\alpha$ enthält, und zwar auch dann, wenn man nach jeder Probe nachsieht und beliebig anhält.

| Symbol | Bedeutung |
|---|---|
| $X_i$ | Ausgang der $i$-ten Probe (1 = Ereignis, 0 = kein Ereignis) |
| $S_n$, $\hat p_n$ | Zahl der Ereignisse nach $n$ Proben und Anteil $S_n/n$ |
| $\pi$ | wahre Wahrscheinlichkeit des Ereignisses (bzw. der exakte p-Wert beim Test) |
| $z$ | Quantil der Normalverteilung zum gewählten Niveau |
| $\alpha$ | zugesagte Fehlerwahrscheinlichkeit (bzw. Testniveau) |
| $L_n(m)$ | Verhältnis der Mischwahrscheinlichkeit zur Wahrscheinlichkeit unter $\pi=m$ |
| $B$ | Beta-Funktion |
| $\varepsilon$, $n^*$ | geforderte relative Genauigkeit, Orakelumfang (Umfang, der bei bekanntem $\pi$ nötig wäre) |
| $k$, $n_k$ | Zahl der Ereignisse, bei der man anhält, und Zahl der Proben bis dahin |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/hinsehen-aufhoeren-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Folge.** $X_1,X_2,\dots$ unabhängig Bernoulli($\pi$); $S_n=\sum_{i\le n}X_i$, $\hat p_n=S_n/n$. Schätzung: $\pi=U=\mathbb P(T\ge2)$, $T$ die Komponentenzahl nach Kantenausfall. Test: $\pi=\mathbb P_{H_0}(T\ge T_{\rm obs})$, $X_i=\mathbf 1[T_i\ge T_{\rm obs}]$.
Exakt: $\mathbb P(T=t)=\sum_{E:\,T(E)=t}p^{|E|}(1-p)^{m-|E|}$ über alle $2^m$ Kantenmengen $E$.

**Intervalle zum festen $n$.** Wald: $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$, $z=1{,}96$. Wilson: Mitte $\frac{\hat p+z^2/2n}{1+z^2/n}$, Halbbreite $\frac{z\sqrt{\hat p(1-\hat p)/n+z^2/4n^2}}{1+z^2/n}$.

**Konfidenzfolge (Ville).** Ist $L_n$ ein nichtnegatives Supermartingal mit $L_0=1$, gilt $\mathbb P(\exists n:\,L_n\ge1/\alpha)\le\alpha$. Beta-Mischung: $L_n(m)=\dfrac{B(S_n+\frac12,\,n-S_n+\frac12)/B(\frac12,\frac12)}{m^{S_n}(1-m)^{n-S_n}}$ ist das Verhältnis der Mischwahrscheinlichkeit zur Wahrscheinlichkeit unter $\pi=m$; für $m=\pi$ ein Martingal. Die Folge ist $\{m:\,L_n(m)<1/\alpha\}$, **für alle $n$ zugleich** gültig und nicht asymptotisch.
Hoeffding-Mischung: Mischung über $\lambda\sim N(0,\tau^2)$ der subgaußschen Exponentialmartingale ($\sigma=\tfrac12$): $|S_n-n\pi|\le\sigma\sqrt{\tfrac{2(1+rn)}{r}\ln\tfrac{\sqrt{1+rn}}{\alpha}}$, $r=\tau^2\sigma^2$, $r$ auf $n^*$ abgestimmt. Asymptotische Folge: dieselbe Grenze mit $\hat\sigma_n$ statt $\sigma$ (nur asymptotisch gültig).

**Abbruchregeln.** Relativ: halte bei $\hat p>0$ und $h_n\le\varepsilon\hat p_n$; Orakelumfang $n^*=z^2(1-\pi)/(\varepsilon^2\pi)$. Absolut: halte bei $h_n\le h$; bei $S_n=0$ ist die Wald-Halbbreite null. Inverse Stichprobe: $n_k$ negativ binomial; $\mathbb E[(k-1)/(n_k-1)]=\pi$ (exakt), $\mathbb E[k/n_k]>\pi$.

**Monte-Carlo-Test.** $\hat p=(1+S_n)/(1+n)$, Ablehnen bei $\hat p\le\alpha$. **Entscheidungsrisiko** $=\mathbb P(\text{Entscheidung}\ne\mathbf 1[\pi\le\alpha])$; beim festen $n$ aus der Binomialverteilung exakt. Besag–Clifford: halte beim $h$-ten Ausschlag (dann $\hat p=h/n$) oder bei $n_{\max}$ (dann $(S+1)/(n_{\max}+1)$).
Konfidenzfolgen-Entscheidung: halte, sobald $\alpha\notin$ Folge zum Niveau $\delta$; das Risiko ist durch $\delta$ beschränkt, die erwartete Zahl der Schritte wächst wie $1/(\pi-\alpha)^2$.

**Literatur (aus dem Gedächtnis, wenn nicht anders vermerkt).** Waudby-Smith, Arbour, Sinha, Kennedy und Ramdas 2024, „Time-uniform central limit theory and asymptotic confidence sequences“ (Annals of Statistics; arXiv 2103.06476, nur Abstract eingesehen); Gandy 2009, „Sequential implementation of Monte Carlo tests with uniformly bounded resampling risk“ (JASA; arXiv math/0612488, nur Abstract eingesehen);
Fischer und Ramdas 2024, „Sequential Monte-Carlo testing by betting“ (arXiv 2401.07365, nur Abstract eingesehen); Besag und Clifford 1991, „Sequential Monte Carlo p-values“ (Biometrika); Robbins 1970 (Mischungsfolgen); Howard, Ramdas, McAuliffe und Sekhon 2021 (Konfidenzfolgen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
