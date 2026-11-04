# Abbruchregeln: wann man aufhören darf – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-abbruchregeln-demo.streamlit.app/)** -->

Vierzehntes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [geschachtelte-erwartungen-demo](https://github.com/sebastian-hanisch/geschachtelte-erwartungen-demo) (Stück 13) und Weiterführung von [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo) (Stück 1, Intervalle zum festen Umfang): Hier geht es um das **Aufhören**. Eine Simulation läuft, bis „es reicht“; die Demo misst, was dabei mit den Intervallen und Tests passiert. Vehikel ist die **Zuverlässigkeit der Netze der Reihe** (Hantel, Gitter 3×3 und 3×4); Ausfallwahrscheinlichkeit und p-Wert sind exakt aus der Aufzählung aller Kantenmengen bekannt.

## Warum dieses Problem

Die Intervalle der Stücke 1 bis 13 sind für einen **festen** Umfang gebaut. Wer bei jedem n hinsieht und anhält, sobald etwas passt („das Intervall ist klein genug“, „die Spezifikation ist ausgeschlossen“), bricht diese Annahme. **Jederzeit gültige Konfidenzfolgen** (Waudby-Smith, Arbour, Sinha, Kennedy und Ramdas, Annals of Statistics 2024, arXiv 2103.06476; nur Abstract gelesen) halten die Abdeckung bei beliebigem Anhalten (die Folgen der Quelle asymptotisch, die hier gebaute Beta-Mischung nicht asymptotisch; die Zeitschriftenangabe stammt aus dem Gedächtnis). Für Tests auf Simulationen: Besag und Clifford (1991, aus dem Gedächtnis), Gandy (JASA 2009, arXiv math/0612488: offener sequentieller Algorithmus mit gleichmäßig beschränktem Resampling-Risiko) und **Fischer und Ramdas** (arXiv 2401.07365, jederzeit gültiger Monte-Carlo-Test, der Besag–Clifford verallgemeinert), alle nur im Abstract gelesen.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **„Festes ‚bis das Intervall klein genug ist‘ ist verzerrt“ ist zu grob.** Die **relative** Regel (Halbbreite ≤ ε·p̂, mit Schutz gegen p̂ = 0) hält mit Wald und Wilson die Abdeckung (0,93 bis 0,96), braucht bei großem n* etwa das 1,02-Fache des Orakelumfangs n* = z²(1 − π)/(ε²π) (bei sehr kleinem n* bestimmt der erste Blick bei n = 30 den Umfang) und ist bei ε = 0,2 nur leicht nach oben verzerrt (+1,1 %), bei ε = 0,5 spürbar (+7,4 %, bei der Hantel +9,2 % mit Wald und +10,1 % mit Wilson). Die **absolute** Regel ohne Schutz versagt: bei Gitter 3×3, p = 0,95 (π = 0,0118) stoppen **70 %** der Läufe schon am Start mit null Ereignissen, weil die Wald-Halbbreite dann null ist; die Abdeckung des Endintervalls fällt auf **0,29** (bei p = 0,9: 19 % und 0,77).
- **Hinsehen bei jedem n zerstört die Abdeckung von Wald und Wilson:** bei einem **einzelnen** großen n liegt die Wahrheit zu etwa 5 % außerhalb, bei jedem n von 30 bis 10.000 hingesehen **irgendwann** in 50 bis 81 % der Läufe (Gitter 3×3, p = 0,9: Wald 0,62, Wilson 0,56). Die Beta-Mischungs-Konfidenzfolge kommt auf 0,01 bis 0,02.
- **Nicht jede „anytime“-Folge hält:** die **asymptotische** Folge versagt bei seltenen Ereignissen (irgendwann außerhalb 0,70 bei π = 0,0118, 0,26 bei π = 0,053), hält aber bei π von etwa 0,15 bis 0,5 (0,04 bei π = 0,49, 0,054 bei der Hantel mit π = 0,15); bei der Hantel mit p = 0,95 (π = 0,064) liegt sie bei 0,221, beim Gitter 3×3 mit p = 0,5 (π = 0,89) bei 0,097.
- **Was die Zeit-Gleichmäßigkeit kostet:** die Beta-Folge ist **1,65- bis 2,16-mal** so breit wie Wald (also etwa 3- bis 5-mal so viele Proben für dieselbe Breite); als Abbruchregel braucht sie das **3- bis 5-Fache** von n* (3,7 bis 4,9 bei vier Netzen, 3,15 und 3,60 beim Gitter 3×3 mit p = 0,7; ε = 0,2 und 0,1) bei einer Abdeckung von 0,999 bis 1,000. Die Hoeffding-Mischung (Streuungsschranke 1/4) ist bei seltenen Ereignissen bis **9,8-mal** so breit.
- **Stopp nach k Ereignissen:** k/n ist nach oben verzerrt (+22,9 % bei k = 5, +10,4 % bei k = 10, +3,2 % bei k = 30, +1,0 % bei k = 100), (k − 1)/(n − 1) ist erwartungstreu (Verzerrung höchstens ±0,5 %).
- **Monte-Carlo-Test:** das **Entscheidungsrisiko** (andere Entscheidung als beim exakten p-Wert) ist bei festem Umfang an der Schwelle groß und beschränkt die Besag–Clifford-Variante mit kleinem h nicht; die **Konfidenzfolgen-Entscheidung** (Niveau δ = 0,01) hatte in allen gemessenen Zellen ein Risiko von höchstens 0,002 (garantiert ist nur δ = 0,01), kostet aber in der Nähe der Schwelle ≈ 1/(π − α)² Simulationen (weiter weg weniger) und bei π = α unendlich viele. Der exakte p-Wert von „mindestens vier Komponenten“ im Gitter 3×3 bei p = 0,7 ist **0,0494**, knapp unter α = 0,05.
- **Kein scipy zur Laufzeit:** die Gammafunktion der Beta-Mischung läuft über eine Lanczos-Näherung (gegen `math.lgamma` und `scipy` bis 1e-12 geprüft), die Binomial-Zuwächse über `numpy.random.Generator` bei festem, aus dem SplitMix64-Strom abgeleitetem Seed.

## Modell

Nachweis einer Wahrscheinlichkeit π durch eine Bernoulli-Folge X_1, X_2, … (Folge exakt Bernoulli(π), das Netz liefert nur π): **Schätzung** der Ausfallwahrscheinlichkeit U = 1 − R(p) = P(T ≥ 2) (X_i = 1[Netz fällt aus]) und **Test** „ist die beobachtete Komponentenzahl T_obs unter dem Modell ungewöhnlich?“ mit p-Wert π = P_H0(T ≥ T_obs) (X_i = 1[T_i ≥ T_obs]); jede Kante hält unabhängig mit Wahrscheinlichkeit p, T ist die Zahl der Zusammenhangskomponenten. Die exakten Werte stammen aus der Aufzählung aller 2^m Kantenmengen: Gitter 3×3, p = 0,9: U = 0,0530; p = 0,95: 0,0118; Hantel, p = 0,9: 0,1497; Gitter 3×4, p = 0,9: 0,0555; Gitter 3×3, p = 0,7: 0,4890.

## Methodik

**Intervalle zum festen n:** Wald p̂ ± z√(p̂(1 − p̂)/n) und Wilson. **Konfidenzfolgen** (Niveau 95 %, für alle n zugleich): **Beta-Mischung** L_n(m) = B(S + ½, F + ½)/B(½, ½) / (m^S (1 − m)^F), Folge {m : L_n(m) < 1/α} (Ville, nicht asymptotisch); **Hoeffding-Mischung** (σ = ½) und **asymptotische Folge** (geschätztes σ̂_n), beide auf n* = 1.000 abgestimmt. **Experiment 1:** 1.000 Pfade bis n = 10.000, Blick bei jedem n ab 30. **Experiment 3:** relative Regel (Wald, Wilson, Konfidenzfolge) und absolute Regel (Wald-Halbbreite ≤ 0,01, ohne Schutz) auf einem geometrischen Prüfgitter (erster Blick bei 30, Faktor 1,03), 2.000 Läufe. **Experiment 4:** Zeit bis zum k-ten Ereignis (negative Binomialverteilung aus geometrischen Wartezeiten), 100.000 Läufe je k.
**Experiment 5:** Entscheidungsrisiko gegen die exakte Entscheidung „ablehnen genau bei π ≤ α“ für festen Umfang (n = 99, 999, 9999), Besag–Clifford (h = 10, nmax = 999 und h = 50, nmax = 9999) und die Konfidenzfolgen-Entscheidung (Beta-Folge zum Niveau δ = 0,01, Höchstzahl 2 · 10⁶ Simulationen), 2.000 Läufe je Zelle, π als Vielfaches von α und der exakte p-Wert des gewählten Netzes.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Strom 38, Testniveau α = 0,05, `tests/test_claims.py` rechnet sie nach (in Bändern von ±0,02 bis ±0,09).

| Netz, p | π | Wald | Wilson | Beta-Folge | asymptotische Folge | absolute Regel: Stopp ohne Ereignis, Abdeckung |
|---|---|---|---|---|---|---|
| Gitter 3×3, 0,9 | 0,0530 | 0,615 | 0,558 | 0,012 | 0,255 | 19 %, 0,767 |
| Gitter 3×3, 0,95 | 0,0118 | 0,808 | 0,503 | 0,009 | 0,701 | 70 %, 0,290 |
| Hantel, 0,9 | 0,1497 | 0,616 | 0,578 | 0,014 | 0,054 | 1 %, 0,941 |
| Gitter 3×4, 0,9 | 0,0555 | 0,623 | 0,537 | 0,019 | 0,239 | 19 %, 0,760 |
| Gitter 3×3, 0,7 | 0,4890 | 0,583 | 0,563 | 0,016 | 0,041 | 0 %, 0,956 |

(Spalten 3 bis 6: Anteil der 1.000 Läufe, in denen die Wahrheit bei **irgendeinem** n zwischen 30 und 10.000 außerhalb liegt; Hoeffding-Folge 0,000 bis 0,039. Absolute Regel mit ε = 0,2 als relativer Genauigkeit der anderen Regeln.)

- **Wald bei festem n** (Gitter 3×3, p = 0,9): außerhalb 0,100 bei n = 100, 0,057 bei n = 1.000, 0,041 bei n = 10.000; bei p = 0,95 und n = 100: 0,283 (asymptotische Folge 0,281). Das sind Simulationswerte aus 1.000 Pfaden (Standardfehler 0,007 bis 0,014); exakt aus der Binomialsumme sind es 0,102 / 0,052 / 0,050 und 0,305.
- **Relative Regel** (Gitter 3×3, p = 0,9, ε = 0,2, n* = 1.715): Wald 1,02·n*, Abdeckung 0,957; Wilson 1,03·n*, 0,949; Konfidenzfolge 4,04·n*, 1,000; Verzerrung von p̂ +1,1 %. Bei ε = 0,5 (n* = 274): +7,4 %; bei ε = 0,1: 0,0 %.
- **Halbbreiten** (Beta-Folge gegen Wald): 1,74 / 1,87 / 2,02 / 2,16 bei n = 10² bis 10⁵ (p = 0,9); Hoeffding 3,46 bis 4,48 (p = 0,9) und 7,11 bis 9,81 (p = 0,95); asymptotisch 1,95 / 1,55 / 1,66 / 1,82.
- **Test, Gitter 3×3 mit p = 0,9, T_obs = 2 (π = 0,0530, 1,06·α):** Risiko fest n = 99 / 999 / 9999: 0,400 / 0,311 / 0,090; Besag–Clifford h = 10: 0,375, h = 50: 0,320; Konfidenzfolge 0,000 mit im Mittel 110.786 Simulationen.
- **Test, Risiko gegen π/α** (n = 99 / 999 / 9999 | Besag–Clifford h = 10 / h = 50): 0,6: 0,177 / 0,002 / 0,000 | 0,086 / 0,001; 0,8: 0,342 / 0,073 / 0,000 | 0,266 / 0,062; 0,9: 0,459 / 0,237 / 0,007 | 0,418 / 0,239; 0,98: 0,545 / 0,440 / 0,328 | 0,498 / 0,474; 1,0: 0,538 / 0,531 / 0,502 | 0,535 / 0,512.
- **Konfidenzfolgen-Entscheidung:** mittlere Simulationen 162 (π = 0,02·α), 296 (0,2·α), 1.672 (0,6·α), 8.049 (0,8·α), 36.447 (0,9·α), 1,1 · 10⁶ (0,98·α, 5 % offen), bei π = α alle Läufe offen (2 · 10⁶); bei klaren Fällen (4·α, 10·α) 54 und 30.
- **Besag–Clifford (h = 10):** 49 Simulationen bei π = 4·α und 20 bei 10·α, Risiko 0.
- **Knapp an der Schwelle** (Gitter 3×3, p = 0,7, T ≥ 4: π = 0,04935): Risiko fest n = 999: 0,486, n = 9999: 0,383; Konfidenzfolge: 72 % der Läufe bis 2 · 10⁶ Simulationen unentschieden.

## Ehrliche Grenzen

- **Nur Bernoulli-Folgen,** keine Mittelwerte stetiger Größen mit geschätzter Streuung (für diese gelten die Aussagen zur asymptotischen Folge besonders).
- **Eine Beta-Mischung** (Beta(½, ½)) und eine Abstimmung (n* = 1.000) für die Hoeffding-Mischung und die asymptotische Folge; **die Wetten-Folgen von Waudby-Smith und Ramdas, der Test von Fischer und Ramdas und Gandys Algorithmus sind nicht gebaut;** die Konfidenzfolgen-Entscheidung ist ein eigenes, verwandtes Verfahren.
- **Prüfgitter statt jedes n beim Stopp** (gültig, aber gröber), Konfidenzfolge zum Niveau δ = 0,01 im Test.
- **Die Binomial-Zuwächse** kommen von `numpy.random.Generator` mit festem Seed; über numpy-Versionen hinweg können die Zufallszahlen abweichen (die Bänder der Tests decken das ab).
- **Die Wiederholungen** (1.000 bis 2.000 je Zelle) lassen Anteile auf etwa ±0,02 genau; die Risiken im Test rauschen entsprechend.
- **Quellen nur als Abstract gelesen;** die Ville-Eigenschaft der Beta-Mischung wird per Simulation und gegen `scipy.special.betaln` geprüft, nicht gegen einen Beweis aus der Quelle. Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis.
- **Abgrenzung zu `sequential-test-demo`** (Experimente, A/B-Tests mit Zwischenblicken) ist inhaltlich, nicht gerechnet.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): exakte Komponentenverteilung des Pfads (¼ / ½ / ¼) und des Dreiecks (½ / ⅜ / ⅛) von Hand; Beta-Mischung von Hand (L_2(½) = ½) und als Produkt der Vorhersagen; Wald (0,4244), Wilson (0,3725) und Hoeffding-Halbbreite (0,9054) von Hand; Orakelumfang 864; fester Umfang und Besag–Clifford an festen Zahlen von Hand (Stopp bei n = 4 mit p̂ = 0,5, sonst (1 + 1)/11); Prüfgitter; E[1/n] = ln 2 und E[1/(n − 1)] = ½ der inversen Stichprobe; Lanczos gegen `math.lgamma`.
- **Orakel** (`tests/test_oracle_abr.py`): **Komponentenverteilung gegen `networkx`-Aufzählung** (fünf Netze), Gitter 3×3 gegen den Wert der Reihe (0,48901 und 0,04935); **Beta-Mischung gegen `scipy.special.betaln`**; die Enden der Konfidenzfolge lösen L = 1/α (und gegen `scipy.optimize.brentq`); **Ville-Schranke per Monte Carlo** (4.000 Pfade); **Wilson löst die Score-Gleichung**; Abdeckung zum festen n exakt aus der Binomialverteilung; Wartezeit gegen die negative Binomialverteilung (Zähldichte); **inverse Stichprobe: E[(k − 1)/(n − 1)] = π exakt** (Reihe) und E[k/n] gegen die Reihe; **Risiko beim festen Umfang exakt aus der Binomialverteilung** und Stoppverteilung von Besag–Clifford gegen die Binomialverteilung.
- **Eigenschaften** (`tests/test_properties.py`): Hinsehen zerstört die Abdeckung von Wald und Wilson, nicht die der Beta-Folge; asymptotische Folge versagt bei seltenen Ereignissen; Breiten; relative Regel hält Abdeckung und Orakelumfang; absolute Regel stoppt ohne Ereignis; Verzerrung nach oben; inverse Stichprobe; Risiko fällt mit n; Konfidenzfolge hält das Risiko und kostet zur Schwelle hin mehr; Reproduzierbarkeit.
- **Randfälle** (`tests/test_edge_cases.py`): null Ereignisse, sehr häufiges Ereignis, Genauigkeit unerreichbar, α an der Schwelle, Höchstzahl, Eingabeprüfung der Spielwiese, Zahlenformate.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `abr_algorithm.py` | Kern: Strom, Netze und exakte Verteilung, Intervalle und Konfidenzfolgen, Prüfgitter, Regeln, Test-Verfahren |
| `abr_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `abr_evaluation.py` | Kennzahlen, Verdikt-Logik, Experimente über feste Ströme |
| `abr_layout.py` | Fester Kopf (Stylesheet) |
| `abr_pdf_export.py` | PDF der aktuellen Einstellung |
| `abr_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `abr_scenario.py` | Netze, exakte Werte, Eingabeprüfung der Spielwiese |
| `abr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und networkx |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| A/B-Tests mit Zwischenblicken (Pocock, O'Brien–Fleming, mSPRT) | [sequential-test-demo](https://github.com/sebastian-hanisch/sequential-test-demo) (Experimente) |
| Intervall zum festen Umfang bei seltenen Ereignissen | [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo) (Stück 1) |
| Permutationstest | [permutation-test-demo](https://github.com/sebastian-hanisch/permutation-test-demo) |
| Wetten-Folgen, Fischer–Ramdas-Test, Gandys Algorithmus | nicht gebaut (genannt) |
| Mittelwerte stetiger Größen, Mehrfachtests | nicht gebaut |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
