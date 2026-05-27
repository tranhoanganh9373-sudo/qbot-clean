"""TDX 公式 parser MVP.

把 TDX (通达信) 公式字符串 -> callable Python factor function.

设计原则:
- regex tokenizer + recursive descent (lark 在中文标识符 / `#DAY` 等扩展上边界冲突太多)
- 不支持的函数 / 字段 -> `UNSUPPORTED` 标记, 公式不参与 IC 但 parser 不 crash
- 绘图函数 (STICKLINE/DRAWICON/...) -> placeholder 返回 None, 不影响其它输出列
- 输出: CompiledFactor 对象, 含 ok/skipped/unsupported_funcs 元数据

使用:
    from claude_finance.tdx_parser import compile_tdx
    cf = compile_tdx("MA5: MA(CLOSE, 5);")
    df_out = cf(df_ohlcv)  # df_ohlcv columns: close, open, high, low, vol, amount
    # df_out = {"MA5": pd.Series(...)}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from claude_finance import tdx_functions as fns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
FIELD_ALIASES: dict[str, str] = {
    "CLOSE": "close", "C": "close",
    "OPEN": "open", "O": "open",
    "HIGH": "high", "H": "high",
    "LOW": "low", "L": "low",
    "VOL": "vol", "V": "vol", "VOLUME": "vol",
    "AMOUNT": "amount", "AMO": "amount",
}

SUPPORTED_FUNCS: set[str] = {
    "MA", "EMA", "SMA", "WMA", "DMA",
    "REF", "HHV", "LLV", "SUM", "MAX", "MIN", "ABS",
    "CROSS", "EVERY", "EXIST", "COUNT", "BARSLAST",
    "IF", "IFF",
    "STD", "LOG", "SQRT",
}

DRAWING_FUNCS: set[str] = {
    "STICKLINE", "DRAWICON", "DRAWTEXT", "DRAWTEXT_FIX", "DRAWBAND",
    "DRAWNUMBER", "DRAWGBK", "DRAWLINE", "DRAWKLINE", "POLYLINE",
    "DRAWNULL", "NODRAW", "PARTLINE",
}

DECORATORS: set[str] = {
    "NODRAW", "DOTLINE", "LINETHICK0", "LINETHICK1", "LINETHICK2",
    "LINETHICK3", "LINETHICK4", "LINETHICK5", "LINETHICK6", "LINETHICK7",
    "LINETHICK8", "LINETHICK9",
    "COLORSTICK", "INVISIBLE", "CIRCLEDOT", "POINTDOT",
}

KNOWN_UNSUPPORTED_FUNCS: set[str] = {
    "FINANCE", "FINVALUE", "DYNAINFO", "BKJYVALUE", "CAPITAL", "WINNER",
    "L2_VOL", "INDEXC", "INDEXO", "INDEXH", "INDEXL", "INDEXV", "INDEXA",
    "SETCODE", "ISLASTBAR", "DATETODAY", "BARSSINCE", "SUMBARS", "CONST",
    "FILTERX", "SAR", "ROUND2", "ROUND", "INTPART", "RGB", "BLOCKSETNUM",
    "VALUEWHEN", "ZIG", "PEAK", "TROUGH", "STRCAT", "TIME", "DATE",
    "TYPE", "DAYBARPOS", "DAYBARSIZE", "MULAR", "LARGEINTRDVOL",
    "LARGEOUTTRDVOL", "BACKSET", "REFV", "REFX", "CALCSTOCK", "INSTR",
    "ATAN", "TAN", "SIN", "COS", "POW", "FRACPART", "INDEXCLOSE",
    "FORCAST", "FORECAST", "MEMA", "EXPMA", "EXPMEMA", "TR", "ATR",
}


@dataclass
class CompiledFactor:
    formula: str
    status: str  # "ok" | "skipped" | "unsupported" | "error"
    output_cols: list[str] = field(default_factory=list)
    unsupported_funcs: list[str] = field(default_factory=list)
    error: str | None = None
    _executor: Callable[[pd.DataFrame], dict[str, pd.Series]] | None = None

    def __call__(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        if self.status != "ok" or self._executor is None:
            return {}
        try:
            return self._executor(df)
        except Exception as exc:  # noqa: BLE001
            logger.debug("execution failed: %s", exc)
            return {}


# ---------------------------------------------------------------------------
TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("WS", re.compile(r"[ \t\r\n]+")),
    ("COMMENT_BRACE", re.compile(r"\{[^{}]*\}", re.DOTALL)),
    ("COMMENT_LINE", re.compile(r"//[^\n]*")),
    ("STR", re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")),
    ("NUM", re.compile(r"\d+\.\d+|\.\d+|\d+")),
    ("ASSIGN", re.compile(r":=")),
    ("EQ2", re.compile(r"==")),
    ("NEQ", re.compile(r"!=|<>")),
    ("GE", re.compile(r">=")),
    ("LE", re.compile(r"<=")),
    ("AND2", re.compile(r"&&")),
    ("OR2", re.compile(r"\|\|")),
    ("IDENT", re.compile(r"[A-Za-z_一-鿿][A-Za-z0-9_一-鿿]*")),
    ("COLON", re.compile(r":")),
    ("SEMI", re.compile(r";")),
    ("COMMA", re.compile(r",")),
    ("LPAREN", re.compile(r"\(")),
    ("RPAREN", re.compile(r"\)")),
    ("OP", re.compile(r"[+\-*/%<>=&|!^]")),
    ("HASH", re.compile(r"#[A-Za-z]+")),
]


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        matched = False
        for kind, pat in TOKEN_PATTERNS:
            m = pat.match(src, i)
            if m:
                if kind not in ("WS", "COMMENT_BRACE", "COMMENT_LINE"):
                    tokens.append(Token(kind, m.group(0), i))
                i = m.end()
                matched = True
                break
        if not matched:
            i += 1
    tokens.append(Token("EOF", "", n))
    return tokens


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.unsupported_funcs: set[str] = set()

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def at_eof(self) -> bool:
        return self.peek().kind == "EOF"

    def parse_program(self) -> list[dict[str, Any]]:
        stmts: list[dict[str, Any]] = []
        while not self.at_eof():
            if self.peek().kind == "SEMI":
                self.advance()
                continue
            try:
                stmt = self.parse_stmt()
                if stmt is not None:
                    stmts.append(stmt)
            except ParseError as exc:
                logger.debug("parse_stmt skip: %s", exc)
                self._recover_to_semi()
            if self.peek().kind == "SEMI":
                self.advance()
        return stmts

    def _recover_to_semi(self) -> None:
        depth = 0
        while not self.at_eof():
            t = self.peek()
            if t.kind == "LPAREN":
                depth += 1
            elif t.kind == "RPAREN":
                if depth > 0:
                    depth -= 1
            elif t.kind == "SEMI" and depth == 0:
                return
            self.advance()

    def parse_stmt(self) -> dict[str, Any] | None:
        first = self.peek()
        if first.kind != "IDENT":
            try:
                self.parse_expr()
            except ParseError:
                self.advance()
            self._consume_decorators()
            return None

        name = first.value
        save_pos = self.pos
        self.advance()
        nxt = self.peek()
        if nxt.kind == "ASSIGN":
            self.advance()
            expr = self.parse_expr()
            self._consume_decorators()
            return {"kind": "assign", "name": name, "expr": expr}
        if nxt.kind == "COLON":
            self.advance()
            expr = self.parse_expr()
            self._consume_decorators()
            return {"kind": "output", "name": name, "expr": expr}
        self.pos = save_pos
        try:
            self.parse_expr()
        except ParseError:
            self.advance()
        self._consume_decorators()
        return None

    def _consume_decorators(self) -> None:
        while True:
            t = self.peek()
            if t.kind == "COMMA":
                save = self.pos
                self.advance()
                nxt = self.peek()
                if nxt.kind == "IDENT":
                    name_upper = nxt.value.upper()
                    if (name_upper.startswith("COLOR") or name_upper.startswith("LINETHICK")
                            or name_upper in DECORATORS
                            or name_upper in ("RGB", "STICK", "CROSSDOT")):
                        self.advance()
                        if self.peek().kind == "LPAREN":
                            self._skip_balanced_parens()
                        continue
                self.pos = save
                break
            elif t.kind == "HASH":
                self.advance()
                continue
            else:
                break

    def _skip_balanced_parens(self) -> None:
        if self.peek().kind != "LPAREN":
            return
        depth = 0
        while not self.at_eof():
            t = self.advance()
            if t.kind == "LPAREN":
                depth += 1
            elif t.kind == "RPAREN":
                depth -= 1
                if depth == 0:
                    return

    def parse_expr(self) -> dict[str, Any]:
        return self.parse_or()

    def parse_or(self) -> dict[str, Any]:
        left = self.parse_and()
        while True:
            t = self.peek()
            if (t.kind == "IDENT" and t.value.upper() == "OR") or t.kind == "OR2":
                self.advance()
                right = self.parse_and()
                left = {"type": "binop", "op": "or", "left": left, "right": right}
            else:
                return left

    def parse_and(self) -> dict[str, Any]:
        left = self.parse_not()
        while True:
            t = self.peek()
            if (t.kind == "IDENT" and t.value.upper() == "AND") or t.kind == "AND2":
                self.advance()
                right = self.parse_not()
                left = {"type": "binop", "op": "and", "left": left, "right": right}
            else:
                return left

    def parse_not(self) -> dict[str, Any]:
        t = self.peek()
        if t.kind == "IDENT" and t.value.upper() == "NOT":
            self.advance()
            return {"type": "unop", "op": "not", "operand": self.parse_not()}
        return self.parse_cmp()

    def parse_cmp(self) -> dict[str, Any]:
        left = self.parse_add()
        while True:
            t = self.peek()
            if t.kind in ("GE", "LE", "EQ2", "NEQ"):
                op = {"GE": ">=", "LE": "<=", "EQ2": "==", "NEQ": "!="}[t.kind]
                self.advance()
                right = self.parse_add()
                left = {"type": "binop", "op": op, "left": left, "right": right}
            elif t.kind == "OP" and t.value in ("<", ">"):
                op = t.value
                self.advance()
                right = self.parse_add()
                left = {"type": "binop", "op": op, "left": left, "right": right}
            elif t.kind == "OP" and t.value == "=":
                self.advance()
                right = self.parse_add()
                left = {"type": "binop", "op": "==", "left": left, "right": right}
            else:
                return left

    def parse_add(self) -> dict[str, Any]:
        left = self.parse_mul()
        while True:
            t = self.peek()
            if t.kind == "OP" and t.value in ("+", "-"):
                op = t.value
                self.advance()
                right = self.parse_mul()
                left = {"type": "binop", "op": op, "left": left, "right": right}
            else:
                return left

    def parse_mul(self) -> dict[str, Any]:
        left = self.parse_unary()
        while True:
            t = self.peek()
            if t.kind == "OP" and t.value in ("*", "/", "%"):
                op = t.value
                self.advance()
                right = self.parse_unary()
                left = {"type": "binop", "op": op, "left": left, "right": right}
            else:
                return left

    def parse_unary(self) -> dict[str, Any]:
        t = self.peek()
        if t.kind == "OP" and t.value in ("+", "-"):
            op = t.value
            self.advance()
            operand = self.parse_unary()
            return {"type": "unop", "op": op, "operand": operand}
        return self.parse_postfix()

    def parse_postfix(self) -> dict[str, Any]:
        node = self.parse_atom()
        while self.peek().kind == "HASH":
            self.advance()
        return node

    def parse_atom(self) -> dict[str, Any]:
        t = self.peek()
        if t.kind == "NUM":
            self.advance()
            return {"type": "num", "value": float(t.value)}
        if t.kind == "STR":
            self.advance()
            return {"type": "str", "value": t.value.strip("'\"")}
        if t.kind == "LPAREN":
            self.advance()
            expr = self.parse_expr()
            if self.peek().kind == "RPAREN":
                self.advance()
            return expr
        if t.kind == "IDENT":
            name = t.value
            upper = name.upper()
            if self.peek(1).kind == "LPAREN":
                self.advance()
                self.advance()
                args: list[dict[str, Any]] = []
                if self.peek().kind != "RPAREN":
                    args.append(self.parse_expr())
                    while self.peek().kind == "COMMA":
                        self.advance()
                        args.append(self.parse_expr())
                if self.peek().kind == "RPAREN":
                    self.advance()
                if upper in DRAWING_FUNCS:
                    return {"type": "none"}
                if upper in KNOWN_UNSUPPORTED_FUNCS or upper not in SUPPORTED_FUNCS:
                    self.unsupported_funcs.add(upper)
                    return {"type": "unsupported", "name": upper}
                return {"type": "call", "name": upper, "args": args}

            self.advance()
            if upper in FIELD_ALIASES:
                return {"type": "field", "name": FIELD_ALIASES[upper]}
            if re.fullmatch(r"[NMP]\d*", upper):
                return {"type": "param", "name": upper}
            if upper == "TRUE":
                return {"type": "num", "value": 1.0}
            if upper == "FALSE":
                return {"type": "num", "value": 0.0}
            return {"type": "var", "name": name}
        self.advance()
        return {"type": "none"}


# ---------------------------------------------------------------------------
def _to_series(x: Any, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.Series(np.nan, index=index)
    return pd.Series(x, index=index)


def _eval_node(
    node: dict[str, Any],
    df: pd.DataFrame,
    env: dict[str, pd.Series],
    params: dict[str, float],
) -> Any:
    t = node["type"]
    if t == "num":
        return node["value"]
    if t == "str":
        return node["value"]
    if t == "none":
        return None
    if t == "unsupported":
        raise RuntimeError(f"unsupported func: {node.get('name')}")
    if t == "field":
        col = node["name"]
        if col not in df.columns:
            raise RuntimeError(f"missing column: {col}")
        return df[col]
    if t == "param":
        return params.get(node["name"], 1.0)
    if t == "var":
        if node["name"] in env:
            return env[node["name"]]
        return pd.Series(np.nan, index=df.index)
    if t == "binop":
        left = _eval_node(node["left"], df, env, params)
        right = _eval_node(node["right"], df, env, params)
        return _apply_binop(node["op"], left, right, df.index)
    if t == "unop":
        operand = _eval_node(node["operand"], df, env, params)
        return _apply_unop(node["op"], operand, df.index)
    if t == "call":
        name = node["name"]
        args = [_eval_node(a, df, env, params) for a in node["args"]]
        return _call_func(name, args, df.index)
    raise RuntimeError(f"unknown node type: {t}")


def _apply_binop(op: str, left: Any, right: Any, index: pd.Index) -> Any:
    if left is None or right is None:
        return None
    is_series = isinstance(left, pd.Series) or isinstance(right, pd.Series)
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if isinstance(right, pd.Series):
            return left / right.replace(0, np.nan)
        if right == 0:
            return pd.Series(np.nan, index=index) if isinstance(left, pd.Series) else np.nan
        return left / right
    if op == "%":
        return left % right
    if op in (">", "<", ">=", "<=", "==", "!="):
        op_map = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}
        meth = op_map[op]
        if is_series:
            ls = _to_series(left, index)
            rs = _to_series(right, index)
            return getattr(ls, meth)(rs).astype(float)
        return float({"gt": left > right, "lt": left < right, "ge": left >= right,
                      "le": left <= right, "eq": left == right, "ne": left != right}[meth])
    if op == "and":
        l = _to_series(left, index).astype(float)
        r = _to_series(right, index).astype(float)
        return ((l > 0) & (r > 0)).astype(float)
    if op == "or":
        l = _to_series(left, index).astype(float)
        r = _to_series(right, index).astype(float)
        return ((l > 0) | (r > 0)).astype(float)
    raise RuntimeError(f"unknown binop: {op}")


def _apply_unop(op: str, operand: Any, index: pd.Index) -> Any:
    if operand is None:
        return None
    if op == "+":
        return operand
    if op == "-":
        return -operand
    if op == "not":
        s = _to_series(operand, index)
        return (s <= 0).astype(float)
    raise RuntimeError(f"unknown unop: {op}")


def _call_func(name: str, args: list[Any], index: pd.Index) -> Any:
    func_map = {
        "MA": fns.MA, "EMA": fns.EMA, "SMA": fns.SMA, "WMA": fns.WMA, "DMA": fns.DMA,
        "REF": fns.REF, "HHV": fns.HHV, "LLV": fns.LLV, "SUM": fns.SUM,
        "MAX": fns.MAX, "MIN": fns.MIN, "ABS": fns.ABS,
        "CROSS": fns.CROSS, "EVERY": fns.EVERY, "EXIST": fns.EXIST,
        "COUNT": fns.COUNT, "BARSLAST": fns.BARSLAST,
        "IF": fns.IF, "IFF": fns.IF,
        "STD": fns.STD, "LOG": fns.LOG, "SQRT": fns.SQRT,
    }
    f = func_map.get(name)
    if f is None:
        raise RuntimeError(f"function not implemented: {name}")
    return f(*args)


# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict[str, float] = {
    "N": 20, "N1": 5, "N2": 10, "N3": 20, "N4": 40, "N5": 60, "N6": 120,
    "M": 5, "M1": 12, "M2": 26, "M3": 9,
    "P1": 5, "P2": 10, "P3": 20, "P4": 60, "P5": 120,
}


def compile_tdx(formula: str, params: dict[str, float] | None = None) -> CompiledFactor:
    """parse + compile TDX 公式 → CompiledFactor.

    Args:
        formula: TDX 公式源码 (多行字符串)
        params: 参数覆盖表 (默认 DEFAULT_PARAMS)
    Returns:
        CompiledFactor: status='ok'/'skipped'/'unsupported'/'error'
    """
    if not formula or not formula.strip():
        return CompiledFactor(formula=formula, status="skipped", error="empty")

    try:
        tokens = tokenize(formula)
        parser = Parser(tokens)
        stmts = parser.parse_program()
        unsupported = sorted(parser.unsupported_funcs)
    except Exception as exc:  # noqa: BLE001
        return CompiledFactor(formula=formula, status="error", error=str(exc))

    outputs = [s for s in stmts if s and s.get("kind") == "output"]
    assigns = [s for s in stmts if s and s.get("kind") == "assign"]

    if not outputs:
        return CompiledFactor(
            formula=formula,
            status="skipped",
            unsupported_funcs=unsupported,
            error="no output column",
        )

    var_table: dict[str, dict[str, Any]] = {}
    for s in assigns:
        var_table[s["name"]] = s["expr"]

    def _expand_unsupported(node: dict[str, Any], visited: set[str] | None = None) -> bool:
        visited = visited or set()
        if not isinstance(node, dict):
            return False
        nt = node.get("type")
        if nt == "unsupported":
            return True
        if nt == "var":
            n = node["name"]
            if n in visited or n not in var_table:
                return False
            visited2 = visited | {n}
            return _expand_unsupported(var_table[n], visited2)
        if nt == "call":
            return any(_expand_unsupported(a, visited) for a in node.get("args", []))
        if nt == "binop":
            return (_expand_unsupported(node["left"], visited)
                    or _expand_unsupported(node["right"], visited))
        if nt == "unop":
            return _expand_unsupported(node["operand"], visited)
        return False

    output_cols_raw = [s["name"] for s in outputs]
    clean_outputs = [s for s in outputs if not _expand_unsupported(s["expr"])]

    if not clean_outputs:
        return CompiledFactor(
            formula=formula,
            status="unsupported",
            output_cols=output_cols_raw,
            unsupported_funcs=unsupported,
            error="all outputs depend on unsupported funcs",
        )

    final_params = {**DEFAULT_PARAMS, **(params or {})}

    def _executor(df: pd.DataFrame) -> dict[str, pd.Series]:
        env: dict[str, pd.Series] = {}
        for s in stmts:
            if not s:
                continue
            if s["kind"] == "assign":
                try:
                    val = _eval_node(s["expr"], df, env, final_params)
                    env[s["name"]] = (_to_series(val, df.index)
                                      if val is not None
                                      else pd.Series(np.nan, index=df.index))
                except Exception:  # noqa: BLE001
                    env[s["name"]] = pd.Series(np.nan, index=df.index)
        out: dict[str, pd.Series] = {}
        for s in clean_outputs:
            try:
                val = _eval_node(s["expr"], df, env, final_params)
                out[s["name"]] = _to_series(val, df.index)
            except Exception:  # noqa: BLE001
                out[s["name"]] = pd.Series(np.nan, index=df.index)
        return out

    cf = CompiledFactor(
        formula=formula,
        status="ok",
        output_cols=[s["name"] for s in clean_outputs],
        unsupported_funcs=unsupported,
    )
    cf._executor = _executor
    return cf
