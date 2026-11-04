"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import abr_constants as C
from abr_evaluation import Analysis, Settings, de, dn, pi_text, pr, ratio, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "sigma", "μ": "mu", "ρ": "rho", "α": "alpha", "β": "beta", "γ": "gamma", "θ": "theta", "τ": "tau", "ε": "eps", "π": "pi", "δ": "delta", "∞": "unendlich", "Σ": "Summe", "Δ": "Delta",
            "½": "1/2", "̂": "", "ᵢ": "_i", "∫": "Integral", "⁺": "+", "′": "'", "⁻": "-", "¹": "1", "≠": "!=", "∝": "~"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Abbruchregeln: wann man aufhören darf"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{a.label}. Ausfallwahrscheinlichkeit (exakt aus der Aufzählung aller Kantenmengen) π = {pi_text(a.pi)}; exakter p-Wert der Komponentenzahl (T >= {settings.tobs}) {pi_text(a.pi_test)}. "
                            f"{de(a.peek_reps, 0)} Läufe, Hinsehen bei jedem n von {a.peek_n0} bis {de(a.peek_n, 0)}, Strom {settings.seed}. Alles simuliert, die Wahrheit ist exakt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    heads = ("Folge", "irgendwann außerhalb", "außerhalb bei n=1000", "Breite/Wald (n=1000)")
    widths = (60, 40, 40, 40)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(heads, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    w1 = a.widths[1]
    for k in C.SEQ:
        for text, w in zip((C.SEQ_LABELS[k], pr(a.peek[k]["ever"]), pr(a.peek[k]["at"][1000]), dn(ratio(w1[k], w1["wald"]), 2)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(verdict_main(a)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: abbruchregeln-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
