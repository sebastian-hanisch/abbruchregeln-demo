"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"hantel": ("hantel", 0, 0), "3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4)}
DEFAULT_NETZ = "3x3"
P_CHOICES = (0.5, 0.7, 0.8, 0.9, 0.95)                      # Haltewahrscheinlichkeit je Kante
P_MIN, P_MAX, DEFAULT_P = P_CHOICES[0], P_CHOICES[-1], 0.9
TOBS_CHOICES = (2, 3, 4, 5, 6)                              # beobachtete Komponentenzahl im Test (T ≥ T_obs)
TOBS_MIN, TOBS_MAX, DEFAULT_TOBS = TOBS_CHOICES[0], TOBS_CHOICES[-1], 2
EPS_CHOICES = (0.5, 0.2, 0.1)                               # relative Genauigkeit der Abbruchregel
EPS_MIN, EPS_MAX, DEFAULT_EPS = EPS_CHOICES[-1], EPS_CHOICES[0], 0.2
ALPHA_CHOICES = (0.01, 0.05, 0.1)                           # Testniveau
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = ALPHA_CHOICES[0], ALPHA_CHOICES[-1], 0.05
SEED_MAX, DEFAULT_SEED = 999999, 38
ALPHA_CI = 0.05                                             # Niveau der Intervalle und Konfidenzfolgen (95 %)

PEEK_REPS, PEEK_N, PEEK_N0, PEEK_CHUNK = 1000, 10000, 30, 250   # Experiment 1: Pfade, Länge, erster Blick
PEEK_NSTAR = 1000                                           # n*, auf das Hoeffding-Mischung und asymptotische Folge abgestimmt sind
WIDTH_NS = (100, 1000, 10000, 100000)                       # Experiment 2
STOP_REPS, STOP_N0, STOP_FAC = 2000, 30, 1.03               # Experiment 3: Läufe, erster Blick, Gitterfaktor
STOP_NMAX_MIN, STOP_NMAX_MAX = 300_000, 6_000_000
H_ABS = 0.01                                                # absolute Halbbreite der absoluten Regel
INV_KS = (5, 10, 30, 100)                                   # Experiment 4: Stopp nach k Ereignissen
INV_REPS = 100000
MC_REPS = 2000                                              # Experiment 5
MC_FACTORS = (0.02, 0.2, 0.6, 0.8, 0.9, 0.98, 1.0, 1.02, 1.1, 1.2, 1.4, 2.0, 4.0, 10.0)   # π als Vielfaches von α
MC_DELTA, MC_CAP = 0.01, 2_000_000                          # Konfidenzfolgen-Entscheidung: Niveau und Höchstzahl
MC_DESIGNS = ("fix99", "fix999", "fix9999", "bc10", "bc50", "cs")
MC_LABELS = {"fix99": "fest n = 99", "fix999": "fest n = 999", "fix9999": "fest n = 9999", "bc10": "Besag–Clifford h = 10, nmax = 999", "bc50": "Besag–Clifford h = 50, nmax = 9999", "cs": "Konfidenzfolge δ = 0,01"}

CUSTOM_PI_MIN, CUSTOM_PI_MAX = 0.0005, 0.5
CUSTOM_ALPHA_MIN, CUSTOM_ALPHA_MAX = 0.005, 0.2

SEQ = ("wald", "wilson", "beta", "hoeffding", "asym")
SEQ_LABELS = {"wald": "Wald (fester Umfang)", "wilson": "Wilson (fester Umfang)", "beta": "Beta-Mischungs-Folge", "hoeffding": "Hoeffding-Mischungs-Folge", "asym": "asymptotische Folge"}
SEQ_COLORS = {"wald": "#C0392B", "wilson": "#E08E0B", "beta": "#2F6B65", "hoeffding": "#4c78a8", "asym": "#8e6bb0"}
COLORS = {"fix99": "#C0392B", "fix999": "#E08E0B", "fix9999": "#7f8c3a", "bc10": "#8e6bb0", "bc50": "#4c78a8", "cs": "#2F6B65", "ref": "#222222", "naive": "#C0392B", "unbiased": "#2F6B65"}
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

PRESETS = {
    "Standardfall (Voreinstellung)": {"netz": DEFAULT_NETZ, "p": DEFAULT_P, "tobs": DEFAULT_TOBS, "eps": DEFAULT_EPS, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Seltene Ausfälle": {"netz": DEFAULT_NETZ, "p": 0.95, "tobs": 2, "eps": DEFAULT_EPS, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Knapp an der Schwelle": {"netz": DEFAULT_NETZ, "p": 0.7, "tobs": 4, "eps": DEFAULT_EPS, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Hantel mit Brücke": {"netz": "hantel", "p": 0.9, "tobs": 2, "eps": DEFAULT_EPS, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
    "Lockere Genauigkeit": {"netz": DEFAULT_NETZ, "p": DEFAULT_P, "tobs": DEFAULT_TOBS, "eps": 0.5, "alpha": DEFAULT_ALPHA, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×3, Haltewahrscheinlichkeit 0,9 je Kante: die Ausfallwahrscheinlichkeit liegt bei 0,053, fast genau auf dem Testniveau 0,05. Hinsehen bei jedem n verfälscht das Wald-Intervall; der Test an der Schwelle ist teuer.",
    "Seltene Ausfälle": "Bei Haltewahrscheinlichkeit 0,95 liegt die Ausfallwahrscheinlichkeit bei 0,012: die asymptotische Konfidenzfolge versagt, die absolute Abbruchregel stoppt oft mit null Ereignissen.",
    "Knapp an der Schwelle": "Gitter 3×3, p = 0,7, mindestens vier Komponenten: der exakte p-Wert ist 0,0494, knapp unter 0,05. Kein Verfahren entscheidet billig und sicher zugleich.",
    "Hantel mit Brücke": "Bei der Hantel hängt fast alles an der Brücke: die Ausfallwahrscheinlichkeit ist groß (0,15), Ereignisse sind häufig, und die Verfahren verhalten sich gutmütig.",
    "Lockere Genauigkeit": "Mit relativer Genauigkeit 0,5 stoppt die Regel früh, und die Schätzung ist nach oben verzerrt (etwa +7 %).",
}
