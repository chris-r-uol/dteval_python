"""R colour helpers -- ``hcl()`` and ``grey()``.

ggplot2's default discrete palette is
``hcl(h = seq(15, 375, length = n + 1), l = 65, c = 100)[1:n]``, and DTEval both
relies on that default and constructs it explicitly (``tube.plots.R:1210``,
``test.tube.meta.R:149``). The colours are part of the figure conventions, so
they are reproduced from R's polarLUV -> sRGB conversion rather than
approximated.
"""

from __future__ import annotations

import math

__all__ = ["grey", "hcl", "hcl_hue_palette"]

# D65 white point, as used by grDevices.
_WHITE_X, _WHITE_Y, _WHITE_Z = 95.047, 100.0, 108.883


def _gtrans(u: float) -> float:
    """Linear RGB -> sRGB companding (grDevices' gtrans with gamma 2.4)."""
    if u > 0.0031308:
        return 1.055 * (u ** (1.0 / 2.4)) - 0.055
    return 12.92 * u


def _luv_to_xyz(l: float, u: float, v: float) -> tuple[float, float, float]:
    if l <= 0 and u == 0 and v == 0:
        return 0.0, 0.0, 0.0
    y = _WHITE_Y * (((l + 16.0) / 116.0) ** 3 if l > 8.0 else l / 903.3)
    t = _WHITE_X + 15.0 * _WHITE_Y + 3.0 * _WHITE_Z
    u0 = 4.0 * _WHITE_X / t
    v0 = 9.0 * _WHITE_Y / t
    u_ = u / (13.0 * l) + u0
    v_ = v / (13.0 * l) + v0
    x = 9.0 * y * u_ / (4.0 * v_)
    z = -x / 3.0 - 5.0 * y + 3.0 * y / v_
    return x, y, z


def _xyz_to_rgb(x: float, y: float, z: float) -> tuple[float, float, float]:
    r = (3.240479 * x - 1.537150 * y - 0.498535 * z) / _WHITE_Y
    g = (-0.969256 * x + 1.875992 * y + 0.041556 * z) / _WHITE_Y
    b = (0.055648 * x - 0.204043 * y + 1.057311 * z) / _WHITE_Y
    return r, g, b


def hcl(h: float = 0.0, c: float = 35.0, l: float = 85.0, fixup: bool = True) -> str:
    """R's ``grDevices::hcl()`` -- polarLUV to a ``#RRGGBB`` string."""
    hrad = h * math.pi / 180.0
    u = c * math.cos(hrad)
    v = c * math.sin(hrad)
    x, y, z = _luv_to_xyz(l, u, v)
    r, g, b = (_gtrans(max(v_, 0.0)) for v_ in _xyz_to_rgb(x, y, z))

    def to255(value: float) -> int:
        iv = int(round(255.0 * value))
        if fixup:
            iv = min(255, max(0, iv))
        return iv

    return f"#{to255(r):02X}{to255(g):02X}{to255(b):02X}"


def hcl_hue_palette(n: int, l: float = 65.0, c: float = 100.0) -> list[str]:
    """ggplot2's default discrete colour scale for ``n`` levels."""
    if n <= 0:
        return []
    step = (375.0 - 15.0) / n  # seq(15, 375, length = n + 1)
    return [hcl(h=15.0 + step * i, c=c, l=l) for i in range(n)]


def grey(level: float) -> str:
    """R's ``grey()`` -- a grey level in ``[0, 1]`` as ``#RRGGBB``."""
    v = int(round(level * 255))
    v = min(255, max(0, v))
    return f"#{v:02X}{v:02X}{v:02X}"


#: R's colour names are not CSS's. Most agree, but two that DTEval uses do not,
#: and both are visible: R's `green` is pure #00FF00 where CSS/matplotlib give
#: the darker #008000, and R's `grey` is #BEBEBE against CSS #808080.
#:
#: The plot *spec* keeps R's names, because that is what R records and what the
#: parity fixtures hold. Resolution happens at the rendering boundary instead --
#: and any frontend drawing from `to_spec()` should resolve them through here
#: too, or it will quietly draw a different figure.
R_COLOUR_NAMES = {
    "green": "#00FF00",
    "grey": "#BEBEBE",
    "gray": "#BEBEBE",
}


def resolve_colour(value):
    """Resolve an R colour name to its hex value, leaving anything else alone."""
    if isinstance(value, str):
        return R_COLOUR_NAMES.get(value.lower(), value)
    return value
