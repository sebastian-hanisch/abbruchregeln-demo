"""Netze und Szenario der Spielwiese (eigenes π und eigenes Testniveau)."""

from __future__ import annotations

import abr_algorithm as A
import abr_constants as C


def named_netz(key: str):
    """(Name, Knotenzahl, Kanten) eines benannten Netzes."""
    kind, r, c = C.NETZE[key]
    if kind == "hantel":
        n, edges = A.hantel_edges()
        return "Hantel", n, edges
    n, edges = A.grid_edges(r, c)
    return f"Gitter {r}×{c}", n, edges


def pi_values(key: str, p: float, tobs: int) -> tuple[float, float]:
    """(Ausfallwahrscheinlichkeit U = P(T ≥ 2), exakter p-Wert P(T ≥ T_obs)) des Netzes bei Haltewahrscheinlichkeit p je Kante."""
    _, n, edges = named_netz(key)
    return A.unreliability(n, edges, p), A.exact_pvalue(n, edges, p, tobs)


def check_custom(pi, alpha) -> tuple[float, float]:
    """Prüft die Eingaben der Spielwiese; ValueError mit lesbarer Meldung."""
    for name, x in (("π", pi), ("Testniveau α", alpha)):
        if x is None or x != x or x in (float("inf"), float("-inf")):
            raise ValueError(f"{name} ist keine Zahl.")
    if not C.CUSTOM_PI_MIN <= pi <= C.CUSTOM_PI_MAX:
        raise ValueError(f"π muss zwischen {C.CUSTOM_PI_MIN} und {C.CUSTOM_PI_MAX} liegen.".replace(".", ","))
    if not C.CUSTOM_ALPHA_MIN <= alpha <= C.CUSTOM_ALPHA_MAX:
        raise ValueError(f"Das Testniveau α muss zwischen {C.CUSTOM_ALPHA_MIN} und {C.CUSTOM_ALPHA_MAX} liegen.".replace(".", ","))
    return float(pi), float(alpha)
