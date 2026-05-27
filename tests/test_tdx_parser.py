"""TDX parser unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from claude_finance.tdx_parser import compile_tdx, tokenize


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 60
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    return pd.DataFrame({
        "close": close,
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "vol": np.random.randint(1000, 5000, n).astype(float),
        "amount": close * 1000,
    })


# ---------- tokenizer ----------
def test_tokenize_numeric():
    toks = tokenize("3.14 + 2")
    kinds = [t.kind for t in toks if t.kind != "EOF"]
    assert kinds == ["NUM", "OP", "NUM"]


def test_tokenize_field_and_var():
    toks = [t for t in tokenize("CLOSE := X + 1") if t.kind != "EOF"]
    assert [t.kind for t in toks] == ["IDENT", "ASSIGN", "IDENT", "OP", "NUM"]


def test_tokenize_chinese_ident():
    toks = [t for t in tokenize("均线: MA(C,5)") if t.kind != "EOF"]
    assert toks[0].kind == "IDENT" and toks[0].value == "均线"


def test_tokenize_comment_braces_removed():
    toks = [t for t in tokenize("{this is a comment} A: 1") if t.kind != "EOF"]
    assert [t.kind for t in toks] == ["IDENT", "COLON", "NUM"]


def test_tokenize_comparison_ops():
    toks = [t for t in tokenize("A>=B AND C<>D") if t.kind != "EOF"]
    kinds = [t.kind for t in toks]
    assert "GE" in kinds and "NEQ" in kinds


# ---------- simple expressions ----------
def test_simple_ma(ohlcv):
    cf = compile_tdx("MA5: MA(CLOSE, 5);")
    assert cf.status == "ok"
    assert cf.output_cols == ["MA5"]
    out = cf(ohlcv)
    assert "MA5" in out
    expected = ohlcv["close"].rolling(5).mean()
    assert np.isclose(out["MA5"].iloc[10], expected.iloc[10])


def test_ema(ohlcv):
    cf = compile_tdx("E: EMA(C, 12);")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert not out["E"].iloc[20:].isna().any()


def test_ref_lag(ohlcv):
    cf = compile_tdx("L1: REF(CLOSE, 1);")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert np.isclose(out["L1"].iloc[5], ohlcv["close"].iloc[4])


# ---------- multi-line + local vars ----------
def test_macd_three_outputs():
    cf = compile_tdx("""
        DIF:=EMA(CLOSE,12)-EMA(CLOSE,26);
        DEA:=EMA(DIF,9);
        MACD:(DIF-DEA)*2;
    """)
    assert cf.status == "ok"
    assert cf.output_cols == ["MACD"]


def test_multiple_outputs():
    cf = compile_tdx("""
        MA5: MA(C, 5);
        MA10: MA(C, 10);
        MA20: MA(C, 20);
    """)
    assert cf.status == "ok"
    assert set(cf.output_cols) == {"MA5", "MA10", "MA20"}


def test_var_dependency(ohlcv):
    cf = compile_tdx("""
        X := MA(CLOSE, 5);
        Y := X * 2;
        Z : Y + 1;
    """)
    assert cf.status == "ok"
    out = cf(ohlcv)
    expected = ohlcv["close"].rolling(5).mean() * 2 + 1
    assert np.isclose(out["Z"].iloc[20], expected.iloc[20])


# ---------- comparison + logical ----------
def test_compare_and(ohlcv):
    cf = compile_tdx("BUY: C>O AND V>1000;")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert set(out["BUY"].dropna().unique()).issubset({0.0, 1.0})


def test_cross_function(ohlcv):
    cf = compile_tdx("X: CROSS(MA(C,5), MA(C,20));")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert set(out["X"].dropna().unique()).issubset({0.0, 1.0})


def test_if_function(ohlcv):
    cf = compile_tdx("Y: IF(C>REF(C,1), 1, -1);")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert set(out["Y"].dropna().unique()).issubset({1.0, -1.0})


def test_not_operator(ohlcv):
    cf = compile_tdx("X: NOT (C>O);")
    assert cf.status == "ok"


def test_double_and_pipe():
    cf = compile_tdx("X: C>O && V>1000;")
    assert cf.status == "ok"
    cf2 = compile_tdx("X: C>O || V>1000;")
    assert cf2.status == "ok"


# ---------- decorators / drawing ----------
def test_drawing_skip_pure():
    cf = compile_tdx("STICKLINE(C>O, H, L, 0, 0);")
    assert cf.status == "skipped"


def test_drawing_mixed(ohlcv):
    cf = compile_tdx("""
        MA5: MA(C, 5), COLORYELLOW;
        STICKLINE(C>O, H, L, 1, 0), COLORRED;
    """)
    assert cf.status == "ok"
    assert cf.output_cols == ["MA5"]


def test_decorator_color_linethick(ohlcv):
    cf = compile_tdx("M: MA(C,10), COLORRED, LINETHICK2, NODRAW;")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert "M" in out


# ---------- unsupported ----------
def test_unsupported_finance():
    cf = compile_tdx("PE: FINANCE(1) * C;")
    assert cf.status == "unsupported"
    assert "FINANCE" in cf.unsupported_funcs


def test_unknown_function_graceful():
    cf = compile_tdx("X: WEIRDFUNC(C, 5);")
    assert cf.status == "unsupported"
    assert "WEIRDFUNC" in cf.unsupported_funcs


def test_mixed_supported_and_unsupported():
    cf = compile_tdx("""
        BAD: FINANCE(1);
        GOOD: MA(C, 5);
    """)
    assert cf.status == "ok"
    assert cf.output_cols == ["GOOD"]


# ---------- robustness ----------
def test_empty_formula():
    cf = compile_tdx("")
    assert cf.status == "skipped"


def test_whitespace_only():
    cf = compile_tdx("   \n  \t  ")
    assert cf.status == "skipped"


def test_chinese_var_and_output():
    cf = compile_tdx("均线: MA(CLOSE, 10);")
    assert cf.status == "ok"
    assert cf.output_cols == ["均线"]


def test_garbage_does_not_crash():
    cf = compile_tdx("@@##$$ ::: ;;; ((")
    assert cf.status in ("skipped", "error", "unsupported")


def test_call_returns_empty_dict_on_failure(ohlcv):
    cf = compile_tdx("X: FINANCE(1);")
    assert cf(ohlcv) == {}


def test_param_substitution(ohlcv):
    cf = compile_tdx("X: EMA(CLOSE, P1);")
    assert cf.status == "ok"
    out = cf(ohlcv)
    expected = ohlcv["close"].ewm(span=5, adjust=False).mean()
    assert np.isclose(out["X"].iloc[-1], expected.iloc[-1])


def test_compare_eq_operator(ohlcv):
    cf = compile_tdx("X: C=O;")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert set(out["X"].dropna().unique()).issubset({0.0, 1.0})


def test_hash_suffix_graceful():
    cf = compile_tdx("X: MA(CLOSE#DAY, 5);")
    assert cf.status in ("ok", "skipped", "unsupported", "error")


def test_top_500_sample_江恩三线():
    formula = """
工作线17:EMA(CLOSE,17),COLORYELLOW;
中期线50:EMA(CLOSE,50),LINETHICK2;
长期线453:EMA(CLOSE,453),COLORRED,LINETHICK2;
DIFF:=( EMA(CLOSE,7) - EMA(CLOSE,19));
DEA:=EMA(DIFF,9);
MACD:=0.90*(DIFF-DEA);
TJ:=(DIFF>=DEA);
STICKLINE(TJ,H,L,0,0),COLORYELLOW;
"""
    cf = compile_tdx(formula)
    assert cf.status == "ok"
    assert "工作线17" in cf.output_cols
    assert "中期线50" in cf.output_cols


def test_negative_unary(ohlcv):
    cf = compile_tdx("X: -CLOSE;")
    assert cf.status == "ok"
    out = cf(ohlcv)
    assert np.isclose(out["X"].iloc[-1], -ohlcv["close"].iloc[-1])
