"""A restricted evaluator for the R expressions DTEval passes around as strings.

``getTubeX`` is ``with(data, eval(parse(text = x)))`` (``misc.dt.handlers.R:107``),
so throughout the package a "column name" may in fact be an arbitrary R
expression. The documented examples include ``getTubeX(data, 'factor(y)')`` and
``calcTubeStat(dt, by = "`Site Type`")``, and the package itself builds
``paste(.latitude, .longitude)`` and ``paste(a, b, sep='')`` this way.

We do **not** use Python's ``eval``. This is a small tokenizer, recursive
descent parser and evaluator over a whitelisted grammar: identifiers (including
R's dotted names), backticked names, string and numeric literals, the usual
binary and unary operators, ``%in%``, and a fixed set of functions. Anything
outside the grammar raises :class:`RExprError`, which callers turn into R's own
"can't find/build" failure path rather than guessing.

The supported function set is the one DTEval and its documentation actually
use. Extending it is a deliberate act, not an accident of ``eval``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dteval.rcompat.coerce import as_character_series, as_numeric_series
from dteval.rcompat.factor import r_factor

__all__ = ["RExprError", "SUPPORTED_FUNCTIONS", "r_eval"]


class RExprError(ValueError):
    """The expression is outside the supported grammar, or cannot be evaluated."""


# --------------------------------------------------------------------- lexer

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?L?)
  | (?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<backtick>`[^`]*`)
  | (?P<special>%in%|%%|%/%)
  | (?P<name>[A-Za-z._][A-Za-z0-9._]*)
  | (?P<op><=|>=|==|!=|&&|\|\||[-+*/^<>!&|(),=\[\]$])
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    pos: int


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if not m or m.end() == i:
            raise RExprError(f"unexpected character {src[i]!r} at position {i} in {src!r}")
        kind = m.lastgroup
        assert kind is not None
        if kind != "ws":
            toks.append(_Tok(kind, m.group(), i))
        i = m.end()
    return toks


# -------------------------------------------------------------------- parser

# (precedence, right_assoc)
_BINARY = {
    "||": (1, False), "|": (1, False),
    "&&": (2, False), "&": (2, False),
    "==": (3, False), "!=": (3, False), "<": (3, False),
    ">": (3, False), "<=": (3, False), ">=": (3, False),
    "%in%": (4, False),
    "+": (5, False), "-": (5, False),
    "*": (6, False), "/": (6, False), "%%": (6, False), "%/%": (6, False),
    "^": (8, True),
}


@dataclass
class _Node:
    kind: str
    value: Any = None
    children: tuple = ()
    names: tuple = ()


class _Parser:
    def __init__(self, toks: list[_Tok], src: str):
        self.toks = toks
        self.src = src
        self.i = 0

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, text: str | None = None) -> _Tok:
        t = self.peek()
        if t is None:
            raise RExprError(f"unexpected end of expression in {self.src!r}")
        if text is not None and t.text != text:
            raise RExprError(f"expected {text!r} but found {t.text!r} in {self.src!r}")
        self.i += 1
        return t

    def parse(self) -> _Node:
        node = self.expr(0)
        if self.peek() is not None:
            raise RExprError(f"trailing input {self.peek().text!r} in {self.src!r}")
        return node

    def expr(self, min_prec: int) -> _Node:
        left = self.unary()
        while True:
            t = self.peek()
            if t is None or t.text not in _BINARY:
                break
            prec, right_assoc = _BINARY[t.text]
            if prec < min_prec:
                break
            self.eat()
            right = self.expr(prec if right_assoc else prec + 1)
            left = _Node("binop", t.text, (left, right))
        return left

    def unary(self) -> _Node:
        t = self.peek()
        if t is not None and t.text in ("-", "+", "!"):
            self.eat()
            return _Node("unop", t.text, (self.unary(),))
        return self.postfix()

    def postfix(self) -> _Node:
        node = self.atom()
        while True:
            t = self.peek()
            if t is not None and t.text == "$":
                self.eat()
                nm = self.eat()
                node = _Node("dollar", _name_text(nm), (node,))
            else:
                break
        return node

    def atom(self) -> _Node:
        t = self.peek()
        if t is None:
            raise RExprError(f"unexpected end of expression in {self.src!r}")
        if t.text == "(":
            self.eat("(")
            node = self.expr(0)
            self.eat(")")
            return node
        if t.kind == "number":
            self.eat()
            return _Node("const", float(t.text.rstrip("L")))
        if t.kind == "string":
            self.eat()
            return _Node("const", _unquote(t.text))
        if t.kind in ("name", "backtick"):
            self.eat()
            name = _name_text(t)
            nxt = self.peek()
            if nxt is not None and nxt.text == "(" and t.kind == "name":
                return self.call(name)
            return _Node("var", name)
        raise RExprError(f"unexpected token {t.text!r} in {self.src!r}")

    def call(self, name: str) -> _Node:
        self.eat("(")
        args: list[_Node] = []
        names: list[str | None] = []
        if self.peek() is not None and self.peek().text == ")":
            self.eat(")")
            return _Node("call", name, tuple(args), tuple(names))
        while True:
            argname = None
            t = self.peek()
            if (
                t is not None
                and t.kind in ("name", "backtick")
                and self.i + 1 < len(self.toks)
                and self.toks[self.i + 1].text == "="
            ):
                argname = _name_text(t)
                self.eat()
                self.eat("=")
            args.append(self.expr(0))
            names.append(argname)
            t = self.peek()
            if t is not None and t.text == ",":
                self.eat(",")
                continue
            self.eat(")")
            break
        return _Node("call", name, tuple(args), tuple(names))


def _name_text(t: _Tok) -> str:
    return t.text[1:-1] if t.kind == "backtick" else t.text


def _unquote(s: str) -> str:
    body = s[1:-1]
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), body)


# ----------------------------------------------------------------- evaluator


def _len(v) -> int:
    if isinstance(v, pd.Series):
        return len(v)
    if isinstance(v, np.ndarray):
        return v.size
    if isinstance(v, list):
        return len(v)
    return 1


def _recycle(v, n: int, index) -> pd.Series:
    """R's recycling: shorter vectors repeat to the longest length."""
    if isinstance(v, pd.Series):
        if len(v) == n:
            return v
        if len(v) == 0:
            return v
        reps = int(math.ceil(n / len(v)))
        return pd.Series(list(v) * reps, index=index)[:n]
    return pd.Series([v] * n, index=index)


def _fn_paste(args, names, ctx, sep_default=" "):
    sep = sep_default
    collapse = None
    positional = []
    for a, nm in zip(args, names, strict=True):
        if nm == "sep":
            sep = a if isinstance(a, str) else str(a)
        elif nm == "collapse":
            collapse = None if a is None else (a if isinstance(a, str) else str(a))
        else:
            positional.append(a)
    if not positional:
        return ""
    n = max(_len(p) for p in positional)
    index = ctx.index if ctx is not None and len(ctx.index) == n else pd.RangeIndex(n)
    cols = []
    for p in positional:
        s = _recycle(p, n, index)
        # paste() converts with as.character and renders NA as the text "NA".
        chars = as_character_series(s).fillna("NA")
        cols.append(list(chars))
    joined = [sep.join(parts) for parts in zip(*cols, strict=True)]
    if collapse is not None:
        return collapse.join(joined)
    return pd.Series(joined, index=index, dtype="object")


def _fn_factor(args, names, ctx):
    levels = None
    ordered = False
    x = args[0] if args else None
    for a, nm in zip(args, names, strict=True):
        if nm == "levels":
            levels = list(a) if not isinstance(a, str) else [a]
        elif nm == "ordered":
            ordered = bool(a)
    return r_factor(x, levels=levels, ordered=ordered)


def _fn_format(args, names, ctx):
    from dteval.rcompat.dates import r_format_date

    x = args[0]
    fmt = None
    for a, nm in zip(args[1:], names[1:], strict=True):
        if nm in (None, "format"):
            fmt = a
    s = pd.Series(x)
    if pd.api.types.is_datetime64_any_dtype(s):
        return r_format_date(s, fmt or "%Y-%m-%d")
    return as_character_series(s, use_format=True)


def _numeric_reduce(fn):
    def run(args, names, ctx):
        na_rm = False
        vals = []
        for a, nm in zip(args, names, strict=True):
            if nm == "na.rm":
                na_rm = bool(a)
            else:
                vals.append(a)
        flat = np.concatenate([np.atleast_1d(as_numeric_series(v).to_numpy()) for v in vals])
        return fn(flat, na_rm)
    return run


def _agg(np_fn):
    from dteval.rcompat import stats as _st

    mapping = {
        "mean": _st.r_mean, "sum": None, "min": _st.r_min,
        "max": _st.r_max, "median": _st.r_median, "sd": _st.r_sd,
    }
    return mapping.get(np_fn)


SUPPORTED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "paste", "paste0", "factor", "as.factor", "format", "as.numeric",
        "as.character", "as.integer", "as.logical", "c", "length", "unique",
        "is.na", "nchar", "substr", "toupper", "tolower", "round", "signif",
        "abs", "sqrt", "log", "exp", "mean", "sum", "min", "max", "median",
        "sd", "ifelse", "rev", "trimws",
    }
)


def _call(name: str, args: list, names: list, ctx: pd.DataFrame | None):
    from dteval.rcompat import numfmt as _nf
    from dteval.rcompat import stats as _st

    if name not in SUPPORTED_FUNCTIONS:
        raise RExprError(
            f"function {name!r} is not in the supported R subset "
            f"({', '.join(sorted(SUPPORTED_FUNCTIONS))})"
        )
    pos = [a for a, nm in zip(args, names, strict=True) if nm is None]

    if name == "paste":
        return _fn_paste(args, names, ctx, " ")
    if name == "paste0":
        return _fn_paste(args, names, ctx, "")
    if name in ("factor", "as.factor"):
        return _fn_factor(args, names, ctx)
    if name == "format":
        return _fn_format(args, names, ctx)
    if name == "as.numeric":
        return as_numeric_series(pos[0])
    if name == "as.integer":
        return as_numeric_series(pos[0]).astype("Int64")
    if name == "as.character":
        return as_character_series(pos[0])
    if name == "as.logical":
        return pd.Series(pos[0]).astype("boolean")
    if name == "c":
        parts = [pd.Series(p) if _len(p) > 1 else pd.Series([p]) for p in pos]
        return pd.concat(parts, ignore_index=True)
    if name == "length":
        return float(_len(pos[0]))
    if name == "unique":
        from dteval.rcompat.collate import r_unique

        return pd.Series(r_unique(pos[0]))
    if name == "is.na":
        return pd.Series(pd.isna(pd.Series(pos[0])))
    if name == "nchar":
        return as_character_series(pos[0]).map(lambda v: pd.NA if pd.isna(v) else len(v))
    if name == "substr":
        s = as_character_series(pos[0])
        start, stop = int(pos[1]), int(pos[2])
        return s.map(lambda v: pd.NA if pd.isna(v) else v[start - 1 : stop])
    if name == "toupper":
        return as_character_series(pos[0]).map(lambda v: pd.NA if pd.isna(v) else v.upper())
    if name == "tolower":
        return as_character_series(pos[0]).map(lambda v: pd.NA if pd.isna(v) else v.lower())
    if name == "trimws":
        return as_character_series(pos[0]).map(lambda v: pd.NA if pd.isna(v) else v.strip())
    if name == "rev":
        return pd.Series(list(pd.Series(pos[0]))[::-1])
    if name == "round":
        nd = int(pos[1]) if len(pos) > 1 else 0
        return as_numeric_series(pos[0]).map(lambda v: _r_round(v, nd))
    if name == "signif":
        nd = int(pos[1]) if len(pos) > 1 else 6
        return as_numeric_series(pos[0]).map(lambda v: _nf.signif(v, nd))
    if name == "abs":
        return as_numeric_series(pos[0]).abs()
    if name == "sqrt":
        return as_numeric_series(pos[0]).map(math.sqrt)
    if name == "exp":
        return as_numeric_series(pos[0]).map(math.exp)
    if name == "log":
        return as_numeric_series(pos[0]).map(math.log)
    if name == "ifelse":
        cond = pd.Series(pos[0]).astype("boolean")
        n = len(cond)
        yes = _recycle(pos[1], n, cond.index)
        no = _recycle(pos[2], n, cond.index)
        return pd.Series(
            [pd.NA if pd.isna(c) else (yes.iloc[i] if c else no.iloc[i]) for i, c in enumerate(cond)],
            index=cond.index,
        )

    na_rm = any(nm == "na.rm" and bool(a) for a, nm in zip(args, names, strict=True))
    vals = np.concatenate([np.atleast_1d(as_numeric_series(p).to_numpy()) for p in pos])
    if name == "mean":
        return _st.r_mean(vals, na_rm=na_rm)
    if name == "median":
        return _st.r_median(vals, na_rm=na_rm)
    if name == "sd":
        return _st.r_sd(vals, na_rm=na_rm)
    if name == "min":
        return _st.r_min(vals, na_rm=na_rm)
    if name == "max":
        return _st.r_max(vals, na_rm=na_rm)
    if name == "sum":
        v = vals[~np.isnan(vals)] if na_rm else vals
        return float(np.cumsum(v)[-1]) if v.size else 0.0
    raise RExprError(f"function {name!r} is recognised but not implemented")


def _r_round(v: float, digits: int) -> float:
    """R's ``round()`` -- half to even."""
    if v != v:
        return v
    scale = 10.0**digits
    x = v * scale
    f = math.floor(x)
    diff = x - f
    if diff > 0.5:
        r = f + 1
    elif diff < 0.5:
        r = f
    else:
        r = f if math.fmod(f, 2.0) == 0.0 else f + 1
    return r / scale


def _binop(op: str, left, right, ctx):
    if op == "%in%":
        rvals = set(pd.Series(right).astype(object))
        return pd.Series(left).astype(object).isin(rvals)

    ln, rn = _len(left), _len(right)
    n = max(ln, rn)
    index = ctx.index if ctx is not None and len(ctx.index) == n else pd.RangeIndex(n)

    if op in ("==", "!=", "<", ">", "<=", ">="):
        ls, rs = _recycle(left, n, index), _recycle(right, n, index)
        if _looks_textual(ls) or _looks_textual(rs):
            ls, rs = as_character_series(ls), as_character_series(rs)
        ops = {
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b, ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
        }
        return ops[op](ls, rs)

    if op in ("&", "&&"):
        return _recycle(left, n, index).astype("boolean") & _recycle(right, n, index).astype("boolean")
    if op in ("|", "||"):
        return _recycle(left, n, index).astype("boolean") | _recycle(right, n, index).astype("boolean")

    ls = as_numeric_series(_recycle(left, n, index))
    rs = as_numeric_series(_recycle(right, n, index))
    if op == "+":
        return ls + rs
    if op == "-":
        return ls - rs
    if op == "*":
        return ls * rs
    if op == "/":
        return ls / rs
    if op == "^":
        return ls**rs
    if op == "%%":
        return ls % rs
    if op == "%/%":
        return (ls // rs).astype("float64")
    raise RExprError(f"operator {op!r} is not supported")


def _looks_textual(s) -> bool:
    ser = pd.Series(s)
    return (
        ser.dtype == object
        or isinstance(ser.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(ser.dtype)
    )


def _eval(node: _Node, data: pd.DataFrame | None):
    if node.kind == "const":
        return node.value
    if node.kind == "var":
        if data is not None and node.value in data.columns:
            return data[node.value]
        if node.value == "TRUE":
            return True
        if node.value == "FALSE":
            return False
        if node.value in ("NA", "NULL"):
            return None
        raise RExprError(f"object {node.value!r} not found")
    if node.kind == "dollar":
        base = _eval(node.children[0], data)
        if isinstance(base, pd.DataFrame) and node.value in base.columns:
            return base[node.value]
        raise RExprError(f"cannot extract {node.value!r}")
    if node.kind == "unop":
        v = _eval(node.children[0], data)
        if node.value == "-":
            return -as_numeric_series(v) if _len(v) > 1 else -float(v)
        if node.value == "+":
            return v
        return ~pd.Series(v).astype("boolean")
    if node.kind == "binop":
        return _binop(
            node.value, _eval(node.children[0], data), _eval(node.children[1], data), data
        )
    if node.kind == "call":
        args = [_eval(c, data) for c in node.children]
        return _call(node.value, args, list(node.names), data)
    raise RExprError(f"unsupported node {node.kind!r}")


def r_eval(expr: str, data: pd.DataFrame | None = None):
    """Evaluate an R expression string against ``data``.

    Mirrors ``with(data, eval(parse(text = expr)))`` for the supported subset.
    Raises :class:`RExprError` for anything else -- callers map that onto R's
    own failure behaviour rather than silently mis-evaluating.
    """
    if not isinstance(expr, str):
        raise RExprError(f"expected an expression string, got {type(expr).__name__}")
    src = expr.strip()
    if not src:
        raise RExprError("empty expression")

    # Fast path: a plain column name (by far the common case, and it must work
    # even for names the grammar would not otherwise accept, e.g. "site.ref").
    if data is not None and src in data.columns:
        return data[src]

    toks = _tokenize(src)
    node = _Parser(toks, src).parse()
    return _eval(node, data)
