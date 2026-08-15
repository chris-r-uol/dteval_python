"""Axis-label prettifying -- port of ``dte_quickText`` (``zzz.R:35``).

DTEval labels axes with markdown-ish HTML (``NO<sub>2</sub>``,
``&mu;g.m<sup>-3</sup>``) and renders them with ``ggtext::element_markdown``.
The substitutions are order-dependent -- ``PM2.5`` has to be caught before
``PM2``-style prefixes, and ``ws`` is only expanded when the whole label is
exactly ``ws`` -- so this is a literal transcription of R's sequence rather
than a tidier rewrite.
"""

from __future__ import annotations

import re

__all__ = ["quick_text"]

# (pattern, replacement) applied in order, exactly as in dte_quickText.
_SUBS: tuple[tuple[str, str], ...] = (
    ("NO2", "NO<sub>2</sub>"),
    ("no2", "NO<sub>2</sub>"),
    ("NOX", "NO<sub>x</sub>"),
    ("nox", "NO<sub>x</sub>"),
    ("NOx", "NO<sub>x</sub>"),
    ("NH3", "NH<sub>3</sub>"),
    ("nh3", "NH<sub>3</sub>"),
    ("co ", "CO "),
    ("co,", "CO,"),
    ("nmhc", "NHHC"),
)

_SUBS_TAIL: tuple[tuple[str, str], ...] = (
    ("wd", "wind dir."),
    ("rh ", "relative humidity "),
    ("PM10", "PM<sub>10</sub>"),
    ("pm10", "PM<sub>10</sub>"),
    ("pm1", "PM<sub>1</sub>"),
    ("PM1", "PM<sub>1</sub>"),
    ("PM4", "PM<sub>4</sub>"),
    ("pm4", "PM<sub>4</sub>"),
    ("PMtot", "PM<sub>total</sub>"),
    ("pmtot", "PM<sub>total</sub>"),
    ("pmc", "PM<sub>coarse</sub>"),
    ("pmcoarse", "PM<sub>coarse</sub>"),
    ("PMc", "PM<sub>coarse</sub>"),
    ("PMcoarse", "PM<sub>coarse</sub>"),
    ("pmf", "PM<sub>fine</sub>"),
    ("pmfine", "PM<sub>fine</sub>"),
    ("PMf", "PM<sub>fine</sub>"),
    ("PMfine", "PM<sub>fine</sub>"),
    ("PM2.5", "PM<sub>2.5</sub>"),
    ("pm2.5", "PM<sub>2.5</sub>"),
    ("pm25", "PM<sub>2.5</sub>"),
    ("PM2.5", "PM<sub>2.5</sub>"),
    ("PM25", "PM<sub>2.5</sub>"),
    ("pm25", "PM<sub>2.5</sub>"),
    ("O3", "O<sub>3</sub>"),
    ("o3", "O<sub>3</sub>"),
    ("ozone", "O<sub>3</sub>"),
    ("CO2", "CO<sub>2</sub>"),
    ("co2", "CO<sub>2</sub>"),
    ("SO2", "SO<sub>2</sub>"),
    ("so2", "SO<sub>2</sub>"),
    ("H2S", "H<sub>2</sub>S"),
    ("h2s", "H<sub>2</sub>S"),
    ("CH4", "CH<sub>4</sub>"),
    ("ch4", "CH<sub>4</sub>"),
    ("dgrC", "<sup>o</sup>C"),
    ("degreeC", "<sup>o</sup>C"),
    ("deg. C", "<sup>o</sup>C"),
    ("degreesC", "<sup>o</sup>C"),
    ("ug/m3", "&mu;g.m<sup>-3</sup>"),
    ("ug.m-3", "&mu;g.m<sup>-3</sup>"),
    ("ug m-3", "&mu;g.m<sup>-3</sup>"),
    ("ugm-3", "&mu;g.m<sup>-3</sup>"),
    ("mg/m3", "mg.m<sup>-3</sup>"),
    ("mg.m-3", "mg.m<sup>-3</sup>"),
    ("mg m-3", "mg.m<sup>-3</sup>"),
    ("mgm-3", "mg.m<sup>-3</sup>"),
    ("ng/m3", "ng.m<sup>-3</sup>"),
    ("ng.m-3", "ng.m<sup>-3</sup>"),
    ("ng m-3", "ng.m<sup>-3</sup>"),
    ("ngm-3", "ng.m<sup>-3</sup>"),
    ("m/s2", "m.s<sup>-2</sup>"),
    ("m/s", "m.s<sup>-1</sup>"),
    ("m.s-1", "m.s<sup>-1</sup>"),
    ("m s-1", "m.s<sup>-1</sup>"),
    ("g/km", "g.km<sup>-1</sup>"),
    ("g/s", "g.s<sup>-1</sup>"),
    ("kW/t", "kW.t<sup>-1</sup>"),
    ("g/hour", "g.hour<sup>-1</sup>"),
    ("g/hr", "g.hour<sup>-1</sup>"),
    ("g/m3", "g.m<sup>-3</sup>"),
    ("g/kg", "g.kg<sup>-1</sup>"),
    ("km/hr/s", "km.hour<sup>-1</sup>s<sup>-1</sup>"),
    ("km/hour/s", "km.hour<sup>-1</sup>s<sup>-1</sup>"),
    ("km/h/s", "km.hour<sup>-1</sup>s<sup>-1</sup>"),
    ("km/hr", "km.hour<sup>-1"),
    ("km/h", "km.hour<sup>-1"),
    ("km/hour", "km.hour<sup>-1"),
    ("r2", "R<sup>2"),
    ("R2", "R<sup>2"),
    ("\n", "<br>"),
)


def quick_text(text, auto_text: bool = True) -> str:
    """R's ``dte_quickText`` -- turn a data-series name into a display label."""
    if not auto_text:
        return text
    if text is None:
        return text
    ans = str(text)

    for pat, rep in _SUBS:
        ans = ans.replace(pat, rep)

    # R: only when the *original* label is exactly two characters and contains
    # "ws" does it become "wind spd."; otherwise the label is left alone.
    if len(str(text)) == 2 and re.search("ws", str(text)):
        ans = ans.replace("ws", "wind spd.")

    for pat, rep in _SUBS_TAIL:
        ans = ans.replace(pat, rep)
    return ans
