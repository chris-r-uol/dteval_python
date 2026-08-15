"""General diffusion tube data handling -- port of ``R/misc.dt.handlers.R``.

``get_tube_x`` extracts a data-series from ``data``, or builds it by evaluating
``x`` as an R expression. ``check_tube_data`` is the multi-term wrapper that
writes any successfully evaluated terms back onto the frame so later code can
use them as plain column names.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from dteval.rcompat.reval import RExprError, r_eval

__all__ = ["DTEvalError", "check_tube_data", "get_tube_x"]


class DTEvalError(Exception):
    """Raised where the R package calls ``stop()``.

    The message keeps R's ``[functionName]>`` prefix so errors are traceable
    back to the R source they were ported from.
    """


def _blame(if_err: str, default_fun: str, default_arg: str) -> tuple[str, str]:
    """Decode R's ``"stop<<FUN>>ARG"`` convention (``misc.dt.handlers.R:80``).

    Callers pass e.g. ``if.err = "stop<<calcTubeStat>>tube"`` so the error
    reads as though it came from the calling function and names the offending
    argument.
    """
    fun_nm, x_nm = default_fun, default_arg
    if if_err.startswith("stop") and "<<" in if_err and ">>" in if_err:
        m = re.match(r"^stop<<(?P<fun>[^>]*)>>(?P<arg>.*)$", if_err)
        if m:
            if m.group("fun"):
                fun_nm = f"[{m.group('fun')}]"
            if m.group("arg"):
                x_nm = m.group("arg")
    return fun_nm, x_nm


def get_tube_x(
    data: pd.DataFrame | None,
    x: Any = None,
    *,
    test_class: str | None = None,
    if_err: str = "return.null",
):
    """Extract or build the data-series ``x`` from ``data``.

    ``x`` may be a column name or an R expression string; see
    :func:`dteval.rcompat.reval.r_eval` for the supported grammar. R falls back
    to ``data[, x]`` when the expression fails, which matters for column names
    that are not syntactically valid R -- so we do the same.

    Returns ``None`` on failure when ``if_err='return.null'`` (R's default),
    otherwise raises :class:`DTEvalError`.
    """
    fun_nm, x_nm = _blame(if_err, "[getTubeX]", "x")

    terms = [x] if isinstance(x, str) else (list(x) if x is not None else [])
    if len(terms) != 1:
        if if_err == "return.null":
            return None
        if len(terms) < 1:
            raise DTEvalError(f"{fun_nm} no {x_nm}...")
        raise DTEvalError(f"{fun_nm} Sorry, only 1 {x_nm} term allowed")

    term = terms[0]
    out = None
    failed = False
    try:
        out = r_eval(term, data)
    except (RExprError, KeyError, TypeError, ValueError, ZeroDivisionError):
        # R: out <- try(with(data, eval(parse(text=x)))); on failure it retries
        # the plain data[, x] lookup before giving up.
        if data is not None and isinstance(term, str) and term in data.columns:
            out = data[term]
        else:
            failed = True

    if failed:
        if if_err == "return.null":
            return None
        if if_err.startswith("stop"):
            raise DTEvalError(f"{fun_nm} Sorry, can't find/build {x_nm} term '{term}'\n")
        return None

    if test_class is not None and not _is_class(out, test_class):
        if if_err == "return.null":
            return None
        raise DTEvalError(f"{fun_nm} {x_nm} not expected {test_class} class\n")
    return out


def _is_class(value, test_class: str) -> bool:
    """R's ``is(out, test.class)`` for the classes DTEval checks."""
    s = pd.Series(value)
    if test_class == "numeric":
        # R's is(x, "numeric") is FALSE for factors and for character, which is
        # exactly the guard testTubePrecision/testTubeAccuracy rely on to reject
        # a non-numeric `tube` argument.
        if isinstance(s.dtype, pd.CategoricalDtype):
            return False
        return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
    if test_class == "character":
        return s.dtype == object or pd.api.types.is_string_dtype(s.dtype)
    if test_class == "logical":
        return pd.api.types.is_bool_dtype(s.dtype)
    if test_class in ("Date", "POSIXct"):
        return pd.api.types.is_datetime64_any_dtype(s.dtype)
    if test_class == "factor":
        return isinstance(s.dtype, pd.CategoricalDtype)
    return False


def check_tube_data(
    data: pd.DataFrame,
    x: Any = None,
    *,
    n_x: int = -1,
    if_err: str = "return.null",
    output: str = "data",
):
    """Evaluate each term in ``x`` and attach it to ``data``.

    Terms that evaluate become columns *named by the expression itself* --
    ``check_tube_data(d, "factor(y)")`` adds a column literally called
    ``factor(y)``. That looks odd but is what R does
    (``misc.dt.handlers.R:176``), and downstream code then refers to the term
    by that same string, so the naming has to match.

    With ``output='report'`` it returns the terms that resolved, instead of the
    data.
    """
    if x is None:
        return data

    fun_nm, x_nms = _blame(if_err, "[checkTubeData]", "x")
    terms = [x] if isinstance(x, str) else list(x)

    if not (n_x < 0 or len(terms) <= n_x):
        raise DTEvalError(f"{fun_nm} Sorry, only {n_x} {x_nms} term(s) allowed")

    data = data.copy()
    resolved: list[str] = []
    for term in terms:
        got = get_tube_x(data, term, if_err=if_err)
        if got is None:
            continue  # R marks it "..bad" and moves on
        series = pd.Series(got)
        if len(series) == len(data):
            # Align on position, not on the incoming index, but keep the dtype:
            # going via .to_numpy() would flatten a factor to plain strings, and
            # factor level order is what later grouping and sorting depend on.
            series = series.set_axis(data.index)
        data[term] = series
        resolved.append(term)

    if output == "data":
        return data
    if output == "report":
        return [t for t in terms if t in data.columns]
    raise DTEvalError(f"{fun_nm} unknown output; check ?checkTubeData")
