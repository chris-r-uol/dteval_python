"""R-compatible number formatting.

Faithful port of the parts of R's C sources that DTEval's *values* depend on:

* ``scientific()``, ``formatReal()``  -- ``src/main/format.c``
* ``EncodeRealDrop0()``               -- ``src/main/util.c``
* ``StringFromReal()``                -- ``src/main/coerce.c`` (``as.character``)
* ``fprec()``                         -- ``src/nmath/fprec.c`` (``signif``)
* ``R_pow_di()``                      -- ``src/nmath/mlutils.c``

Why this exists
---------------
``as.character(double)`` is not cosmetic in DTEval. ``tagTubeSampleID``
(``tag.tube.data.R:571``) builds a grouping key by pasting latitude, longitude
and the two dates together and then takes ``as.numeric(factor(...))``;
``tagTubeLocation`` builds ``.location`` as ``paste("{", lat, ",", lon, "}")``.
A one-digit difference in that string changes the grouping, so every downstream
``.sample_id`` shifts. Report strings likewise embed ``signif(x, 4)`` verbatim.

R (as of 4.6) formats a double for ``as.character`` with **15 significant
digits** -- not the shortest round-trip representation Python's ``repr`` gives --
then chooses fixed vs scientific by comparing rendered widths, ties going to
fixed.

Two details matter more than they look:

* R scales by exact powers of ten from a lookup table for ``|kp| <= 22`` and by
  ``R_pow_di`` (binary exponentiation, which is *not* correctly rounded) outside
  it. Using ``10.0 ** k`` everywhere gives different last-bit results.
* ``EncodeRealDrop0`` normalises ``-0.0`` to ``0.0`` and strips trailing zeros
  after the decimal point.

This module assumes ``long double == double``, which holds on aarch64 macOS and
on any platform where R takes the non-``HAVE_LONG_DOUBLE`` branch. On x86 Linux
R uses 80-bit intermediates in ``scientific()``; see :data:`LONG_DOUBLE_NOTE`.
"""

from __future__ import annotations

import math

__all__ = [
    "KP_MAX",
    "R_AS_CHARACTER_DIGITS",
    "as_character",
    "encode_real",
    "format_real_vector",
    "r_format_numeric",
    "encode_real_drop0",
    "format_real",
    "r_pow_di",
    "scientific",
    "signif",
]

LONG_DOUBLE_NOTE = (
    "scientific() is ported from R's non-HAVE_LONG_DOUBLE branch. On platforms "
    "where R uses 80-bit long double intermediates, a handful of values may "
    "round differently in the 15th significant digit."
)

# as.character(<double>) uses 15 significant digits regardless of
# getOption("digits"). See ?as.character.
R_AS_CHARACTER_DIGITS = 15

MAX_DIGITS = 22  # fprec.c
KP_MAX = 22  # format.c, non-long-double branch

# format.c `tbl`: exact powers of ten, 1e0 .. 1e22. Built from decimal literals
# so each entry is the correctly rounded double, as in C.
_TBL: tuple[float, ...] = tuple(float(f"1e{k}") for k in range(KP_MAX + 1))

_DBL_MAX_10_EXP = 308
_R_DEC_MIN_EXPONENT = -308


def r_pow_di(x: float, n: int) -> float:
    """Port of R's ``R_pow_di`` -- binary exponentiation, *not* correctly rounded.

    R uses this rather than ``pow()`` in several places, and the accumulated
    rounding error is observable in the last significant digit, so reproducing
    the exact multiply sequence matters.
    """
    pow_ = 1.0
    if n == 0:
        return pow_
    if not math.isfinite(x):
        return x**n
    is_neg = n < 0
    if is_neg:
        n = -n
    while True:
        if n & 1:
            pow_ *= x
        n >>= 1
        if n:
            x *= x
        else:
            break
    if is_neg:
        pow_ = 1.0 / pow_
    return pow_


def _rexp10(x: float) -> float:
    """R's ``Rexp10`` -- ``pow(10.0, x)``."""
    return 10.0**x


def _nearbyint(x: float) -> float:
    """C ``nearbyint`` under the default rounding mode: half to even."""
    f = math.floor(x)
    diff = x - f
    if diff > 0.5:
        return f + 1.0
    if diff < 0.5:
        return f
    return f if math.fmod(f, 2.0) == 0.0 else f + 1.0


def scientific(x: float, digits: int = R_AS_CHARACTER_DIGITS) -> tuple[int, int, int, bool]:
    """Port of R's static ``scientific()`` (``src/main/format.c``).

    Returns ``(neg, kpower, nsig, roundingwidens)`` where ``|x| == alpha *
    10**kpower`` with ``1 <= alpha < 10``, ``nsig`` is the number of significant
    digits actually needed (at most ``digits``), and ``roundingwidens`` flags the
    case where scientific rounding would make the number *wider* than fixed
    (e.g. 9996 at 3 digits is ``1e+04`` scientific but ``9996`` fixed).
    """
    if x == 0.0:
        return 0, 0, 1, False

    if x < 0.0:
        neg = 1
        r = -x
    else:
        neg = 0
        r = x

    kp = int(math.floor(math.log10(r))) - digits + 1

    r_prec = r
    if abs(kp) <= KP_MAX:
        if kp > 0:
            r_prec /= _TBL[kp]
        elif kp < 0:
            r_prec *= _TBL[-kp]
    elif kp <= _R_DEC_MIN_EXPONENT:
        # Shifting by 303 keeps subnormal inputs representable.
        r_prec = (r_prec * 1e303) / _rexp10(float(kp + 303))
    else:
        r_prec /= _rexp10(float(kp))

    if r_prec < _TBL[digits - 1]:
        r_prec *= 10.0
        kp -= 1

    alpha = _nearbyint(r_prec)

    nsig = digits
    for _ in range(1, digits + 1):
        alpha /= 10.0
        if alpha == math.floor(alpha):
            nsig -= 1
        else:
            break
    if nsig == 0 and digits > 0:
        nsig = 1
        kp += 1

    kpower = kp + digits - 1

    rgt = digits - kpower
    rgt = 0 if rgt < 0 else (KP_MAX if rgt > KP_MAX else rgt)
    fuzz = 0.5 / _TBL[rgt]
    roundingwidens = kpower > 0 and kpower <= KP_MAX and r < _TBL[kpower] - fuzz

    return neg, kpower, nsig, roundingwidens


def format_real(
    x: float, digits: int = R_AS_CHARACTER_DIGITS, nsmall: int = 0, scipen: int = 0
) -> tuple[int, int, int]:
    """Port of ``formatReal()`` for a single value. Returns ``(w, d, e)``.

    ``e == 0`` means fixed notation with ``d`` decimals; ``e != 0`` means
    scientific with ``d`` mantissa decimals (``e == 2`` for a 3-digit exponent).
    """
    if not math.isfinite(x):
        # Matches formatReal's all-non-finite branch, then the width bumps.
        w = 0
        if x != x:  # NaN (R's NA_real_ is also NaN-valued; caller separates them)
            w = max(w, 3)
        elif x > 0:
            w = max(w, 3)
        else:
            w = max(w, 4)
        return w, 0, 0

    neg_i, kpower, nsig, roundingwidens = scientific(x, digits)

    left = kpower + 1
    if roundingwidens:
        left -= 1

    sleft = neg_i + (1 if left <= 0 else left)
    right = nsig - left
    neg = 1 if neg_i else 0

    rgt = right
    mxl = mnl = left
    mxsl = sleft
    mxns = nsig

    if digits == 0:
        rgt = 0
    if mxl < 0:
        mxsl = 1 + neg
    if rgt < 0:
        rgt = 0
    w_fixed = mxsl + rgt + (1 if rgt != 0 else 0)

    e = 2 if (mxl > 100 or mnl <= -99) else 1
    d = mxns - 1
    w = neg + (1 if d > 0 else 0) + d + 4 + e
    if w_fixed <= w + scipen:
        e = 0
        if nsmall > rgt:
            rgt = nsmall
            w_fixed = mxsl + rgt + (1 if rgt != 0 else 0)
        d = rgt
        w = w_fixed
    return w, d, e


def encode_real_drop0(x: float, w: int, d: int, e: int) -> str:
    """Port of ``EncodeRealDrop0()`` (``src/main/util.c``).

    Note the two behaviours that are easy to miss: signed zero is normalised to
    ``0.0``, and trailing zeros after the decimal point are stripped.
    """
    if x == 0.0:
        x = 0.0  # collapses -0.0
    if x != x:
        return "NaN"
    if x == math.inf:
        return "Inf"
    if x == -math.inf:
        return "-Inf"

    if e:
        buff = "%*.*e" % (min(w, 254), d, x)
    else:
        buff = "%*.*f" % (min(w, 254), d, x)
    buff = buff.strip()

    # Drop trailing zeros in the fractional part only (not in an exponent).
    dot = buff.find(".")
    if dot != -1:
        i = dot + 1
        replace = dot
        while i < len(buff) and buff[i].isdigit():
            if buff[i] != "0":
                replace = i + 1
            i += 1
        if replace != i:
            buff = buff[:replace] + buff[i:]
    return buff


R_DEFAULT_DIGITS = 7  # getOption("digits")


def format_real_vector(
    xs, digits: int = R_DEFAULT_DIGITS, nsmall: int = 0, scipen: int = 0
) -> tuple[int, int, int]:
    """``formatReal`` over a whole vector: the common ``(w, d, e)`` R picks.

    R's ``format()`` formats a numeric vector to a *shared* width and number of
    decimals -- the maxima across all elements -- which is why a longitude
    column renders as ``-1.732780`` rather than ``-1.73278``.
    """
    neg = 0
    mxl = rgt = mxsl = mxns = None
    mnl = None
    saw_na = saw_nan = saw_posinf = saw_neginf = False

    for v in xs:
        x = float(v) if v is not None else math.nan
        if x != x:
            saw_nan = True
            continue
        if x == math.inf:
            saw_posinf = True
            continue
        if x == -math.inf:
            saw_neginf = True
            continue

        neg_i, kpower, nsig, roundingwidens = scientific(x, digits)
        left = kpower + 1
        if roundingwidens:
            left -= 1
        sleft = neg_i + (1 if left <= 0 else left)
        right = nsig - left
        if neg_i:
            neg = 1
        rgt = right if rgt is None else max(rgt, right)
        mxl = left if mxl is None else max(mxl, left)
        mnl = left if mnl is None else min(mnl, left)
        mxsl = sleft if mxsl is None else max(mxsl, sleft)
        mxns = nsig if mxns is None else max(mxns, nsig)

    if mxns is None:  # every value non-finite
        w = 0
        if saw_na:
            w = max(w, 2)
        if saw_nan:
            w = max(w, 3)
        if saw_posinf:
            w = max(w, 3)
        if saw_neginf:
            w = max(w, 4)
        return w, 0, 0

    if digits == 0:
        rgt = 0
    if mxl < 0:
        mxsl = 1 + neg
    if rgt < 0:
        rgt = 0
    w_fixed = mxsl + rgt + (1 if rgt != 0 else 0)

    e = 2 if (mxl > 100 or mnl <= -99) else 1
    d = mxns - 1
    w = neg + (1 if d > 0 else 0) + d + 4 + e
    if w_fixed <= w + scipen:
        e = 0
        if nsmall > rgt:
            rgt = nsmall
            w_fixed = mxsl + rgt + (1 if rgt != 0 else 0)
        d = rgt
        w = w_fixed

    if saw_nan and w < 3:
        w = 3
    if saw_posinf and w < 3:
        w = 3
    if saw_neginf and w < 4:
        w = 4
    return w, d, e


def _fix_exponent(s: str) -> str:
    """Normalise a C ``%e`` exponent to R's shape (at least two digits).

    CPython and C agree on a two-digit minimum on the platforms we target, so
    this is usually a no-op -- but it is cheap insurance against a libc that
    emits a single digit, which would silently change every formatted value.
    """
    mant, sep, exp = s.partition("e")
    if not sep:
        return s
    sign = exp[0] if exp and exp[0] in "+-" else "+"
    body = (exp[1:] if exp and exp[0] in "+-" else exp).lstrip("0") or "0"
    if len(body) < 2:
        body = body.rjust(2, "0")
    return f"{mant}e{sign}{body}"


def encode_real(x: float, w: int, d: int, e: int) -> str:
    """``EncodeReal`` -- like :func:`encode_real_drop0` but keeps trailing zeros.

    This is what ``format()`` uses. ``as.character()`` uses the drop-zero
    variant, which is why the two disagree on ``-1.73278`` vs ``-1.732780``.
    """
    if x != x:
        return "NaN".rjust(w)
    if x == math.inf:
        return "Inf".rjust(w)
    if x == -math.inf:
        return "-Inf".rjust(w)
    if x == 0.0:
        x = 0.0  # collapse -0.0
    if e:
        return _fix_exponent("%*.*e" % (w, d, x)).rjust(w)
    return "%*.*f" % (w, d, x)


def r_format_numeric(xs, digits: int = R_DEFAULT_DIGITS, nsmall: int = 0) -> list[str]:
    """R's ``format()`` applied to a numeric vector.

    Load-bearing in ``tagTubeSampleID``: it builds the sample key with
    ``apply(test, 1, paste, collapse = "-")``, and ``apply`` coerces the frame
    to a character matrix via ``as.matrix.data.frame``, which formats numeric
    columns with ``format()`` -- *not* ``as.character()``. So the key contains
    ``-1.732780``, and using ``as.character`` here would produce different
    grouping and shift every ``.sample_id``.
    """
    vals = [math.nan if v is None else float(v) for v in xs]
    w, d, e = format_real_vector(vals, digits=digits, nsmall=nsmall)
    return [encode_real(v, w, d, e) for v in vals]


def as_character(x: float | int | bool | None) -> str:
    """R's ``as.character()`` for a single numeric scalar.

    Port of ``StringFromReal``: ``formatReal`` then ``EncodeRealDrop0``. R's
    ``NA_real_`` becomes the string ``"NA"``; a genuine ``NaN`` becomes
    ``"NaN"``. Since Python cannot distinguish the two NaN payloads reliably,
    ``None`` is the caller's way of saying ``NA``.
    """
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, int):
        return str(x)
    x = float(x)
    w, d, e = format_real(x)
    return encode_real_drop0(x, w, d, e)


def signif(x: float, digits: float = 6) -> float:
    """R's ``signif()`` -- port of ``fprec()`` (``src/nmath/fprec.c``)."""
    if x != x or digits != digits:
        return x + digits
    if not math.isfinite(x):
        return x
    if not math.isfinite(digits):
        if digits > 0:
            return x
        digits = 1.0
    if x == 0:
        return x

    dig = int(_round_half_away(digits))
    if dig > MAX_DIGITS:
        return x
    if dig < 1:
        dig = 1

    sgn = 1.0
    if x < 0.0:
        sgn = -sgn
        x = -x

    l10 = math.log10(x)
    e10 = dig - 1 - int(math.floor(l10))
    max10e = _DBL_MAX_10_EXP

    if abs(l10) < max10e - 2:
        p10 = 1.0
        if e10 > max10e:
            p10 = r_pow_di(10.0, e10 - max10e)
            e10 = max10e
        if e10 > 0:
            pow10 = r_pow_di(10.0, e10)
            return sgn * (_nearbyint((x * pow10) * p10) / pow10) / p10
        pow10 = r_pow_di(10.0, -e10)
        return sgn * (_nearbyint(x / pow10) * pow10)

    # LARGE or small
    do_round = math.log10(_DBL_MAX) - l10 >= r_pow_di(10.0, -dig)
    e2 = dig + (1 if e10 > 0 else -1) * MAX_DIGITS
    p10 = r_pow_di(10.0, e2)
    p10_big = r_pow_di(10.0, e10 - e2)
    x *= p10
    x *= p10_big
    if do_round:
        x += 0.5
    x = math.floor(x) / p10
    return sgn * x / p10_big


_DBL_MAX = 1.7976931348623157e308


def _round_half_away(x: float) -> float:
    """C's ``round()`` -- half away from zero (not banker's rounding)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
