"""Dashboard panel — 5m K 线数据健康 + 今日集合竞价 audit.

数据源:
  - data_cache/db.duckdb view `kline_5m`(由 build_5m_hive_duckdb.py 注册)
  - data_cache/picks_today.json(当日 picks,看 first 5m bar 验证集合竞价信号)

视觉:
  - 顶部 5m schema 健康度 (rows / symbols / date 范围 / DuckDB query 耗时)
  - 今日 picks first 5m bar audit (9:30-9:35):
    sym / open / first_5m_close / first_5m_return / 集合竞价 confirmation 标记
"""
from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DUCKDB_PATH = ROOT / "data_cache" / "db.duckdb"
PICKS_TODAY = ROOT / "data_cache" / "picks_today.json"
STOCK_NAMES = ROOT / "data_cache" / "stock_names.json"


def _load_names() -> dict[str, str]:
    """读 sym → 中文名 cache. 容错: 缺失或 parse 失败返空 dict."""
    if not STOCK_NAMES.exists():
        return {}
    try:
        return json.loads(STOCK_NAMES.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _placeholder(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{message}"
        "</div>"
    )


def _connect_duckdb():
    if not DUCKDB_PATH.exists():
        return None
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return None
    try:
        return duckdb.connect(str(DUCKDB_PATH), read_only=True)
    except Exception:
        return None


def _query_schema_health(con) -> dict | None:
    try:
        t0 = time.time()
        res = con.execute("""
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT symbol) AS n_symbols,
              MIN(datetime) AS min_dt,
              MAX(datetime) AS max_dt
            FROM kline_5m
        """).fetchone()
        elapsed = time.time() - t0
        if not res or res[0] == 0:
            return None
        return {
            "rows": int(res[0]),
            "n_symbols": int(res[1]),
            "min_dt": str(res[2])[:19] if res[2] else "?",
            "max_dt": str(res[3])[:19] if res[3] else "?",
            "query_ms": elapsed * 1000,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _query_first_5m_for_today(con, syms: list[str]) -> list[dict]:
    """Per-symbol latest trading day 的 first 5m bar (9:30-9:35).
    日期不一定一致 — 每股取自己最新 trading day 的开盘 bar."""
    if not syms:
        return []
    try:
        sym_list = ",".join(f"'{s}'" for s in syms)
        rows = con.execute(f"""
            WITH per_sym_latest_date AS (
              SELECT symbol, MAX(CAST(datetime AS DATE)) AS dt
              FROM kline_5m
              WHERE symbol IN ({sym_list})
              GROUP BY symbol
            ),
            first_bars AS (
              SELECT
                k.symbol, k.datetime, k.open, k.close, k.volume,
                ROW_NUMBER() OVER (PARTITION BY k.symbol ORDER BY k.datetime) AS rn
              FROM kline_5m k
              JOIN per_sym_latest_date d
                ON k.symbol = d.symbol AND CAST(k.datetime AS DATE) = d.dt
              WHERE k.symbol IN ({sym_list})
            )
            SELECT symbol, datetime, open, close, volume,
                   (close / open - 1) * 100 AS ret_pct_first_5m
            FROM first_bars
            WHERE rn = 1
            ORDER BY symbol
        """).fetchall()
        out: list[dict] = []
        for r in rows:
            out.append({
                "sym": r[0], "datetime": str(r[1]),
                "open": float(r[2]), "close": float(r[3]),
                "volume": float(r[4]) if r[4] is not None else 0,
                "ret_first_5m_pct": float(r[5]) if r[5] is not None else 0,
            })
        return out
    except Exception:
        return []


def _load_picks() -> tuple[list[str], dict]:
    if not PICKS_TODAY.exists():
        return [], {}
    try:
        d = json.loads(PICKS_TODAY.read_text(encoding="utf-8"))
        syms = [p["sym"] for p in d.get("picks", [])[:8]]
        sym_map: dict[str, dict] = {}
        for p in d.get("picks", []):
            sym_map[p["sym"]] = {
                "score": p.get("score"),
                "z_jzf": p.get("z_jzf"),
                "jzf": p.get("jzf"),
                "final_score": p.get("final_score"),
            }
        return syms, sym_map
    except Exception:
        return [], {}


def build_kline_5m_health_section() -> str:
    con = _connect_duckdb()
    if con is None:
        return _placeholder(
            "无 DuckDB / kline_5m view — 跑 "
            "<code>python examples/build_5m_hive_duckdb.py</code> 后此 panel 自动填充."
        )

    schema = _query_schema_health(con)
    if schema is None or "error" in (schema or {}):
        err = (schema or {}).get("error", "no rows")
        try:
            con.close()
        except Exception:
            pass
        return _placeholder(f"DuckDB query 失败 / view 空: <code>{html.escape(err)}</code>")

    picks_syms, picks_map = _load_picks()
    audit_rows = _query_first_5m_for_today(con, picks_syms) if picks_syms else []
    try:
        con.close()
    except Exception:
        pass

    days_span = "n/a"
    try:
        dt_min = datetime.fromisoformat(schema["min_dt"][:19])
        dt_max = datetime.fromisoformat(schema["max_dt"][:19])
        days_span = f"{(dt_max - dt_min).days} 天 (~{(dt_max - dt_min).days/30:.1f} 月)"
    except Exception:
        pass

    schema_html = (
        '<div style="background:rgba(37,99,235,0.06); border-left:3px solid #2563eb; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>📊 5m K 线 DuckDB view 健康度</strong><br>'
        f'rows: <strong>{schema["rows"]:,}</strong> · '
        f'symbols: <strong>{schema["n_symbols"]:,}</strong> · '
        f'区间: <code>{html.escape(schema["min_dt"])}</code> → '
        f'<code>{html.escape(schema["max_dt"])}</code> ({days_span}) · '
        f'query: <strong>{schema["query_ms"]:.0f} ms</strong><br>'
        '<span style="color:var(--muted, #6b7280);">'
        'mootdx TDX server 历史上限 ~2 年 · 用法见 <code>docs/duckdb_5m_quickstart.md</code>'
        '</span></div>'
    )

    if not audit_rows:
        audit_html = (
            '<div class="placeholder-content" style="padding:14px;">'
            "今日 picks 5m audit 不可用 — picks_today.json 缺失 / "
            "或 picks symbols 在 kline_5m 上无最新数据 (post-2025 IPO?)."
            '</div>'
        )
    else:
        cols = [
            ("#", 3), ("sym", 9), ("名称", 10), ("v19.10 JZF", 10),
            ("9:30 open", 9), ("9:35 close", 9),
            ("first 5m ret", 10), ("vol", 9),
            ("confirm", 13), ("note", 18),
        ]
        assert sum(w for _, w in cols) == 100, sum(w for _, w in cols)
        colgroup = "<colgroup>" + "".join(
            f'<col style="width:{w}%;">' for _, w in cols
        ) + "</colgroup>"
        right_align_labels = {"v19.10 JZF", "9:30 open", "9:35 close",
                              "first 5m ret", "vol"}
        thead = "".join(
            f'<th style="text-align:right;">{label}</th>' if label in right_align_labels
            else f"<th>{label}</th>"
            for label, _ in cols
        )

        name_map = _load_names()
        # 计算 audit_rows 中最新日期, 判断哪些 row 是 stale (旧于最新)
        dates_in_rows = {(r["datetime"][:10] if r.get("datetime") else "") for r in audit_rows}
        latest_date_in_rows = max(d for d in dates_in_rows if d) if any(dates_in_rows) else ""

        body_rows: list[str] = []
        for i, r in enumerate(audit_rows, 1):
            sym = r["sym"]
            name = name_map.get(sym, "?")
            pm = picks_map.get(sym, {})
            jzf = pm.get("jzf")
            ret5 = r["ret_first_5m_pct"]
            jzf_str = f"{jzf:+.2f}%" if jzf is not None else "—"
            ret5_color = "#16a34a" if ret5 > 0 else "#dc2626"
            row_date = r["datetime"][:10] if r.get("datetime") else "?"
            is_stale = bool(latest_date_in_rows) and row_date != latest_date_in_rows
            stale_mark = (
                f' <span style="color:#dc2626; font-size:10px;" title="数据日期 {row_date}, '
                f'最新组日期 {latest_date_in_rows}">⚠ {row_date}</span>'
            ) if is_stale else ""
            if jzf is None:
                conf_label, conf_color, note = "no JZF", "#6b7280", "production picks 无 JZF"
            elif jzf > 0 and ret5 >= -0.5:
                conf_label, conf_color, note = "✓ 真跳空", "#16a34a", "5m 持续, 入场点合适"
            elif jzf > 0 and ret5 < -0.5:
                conf_label, conf_color, note = "⚠ 假跳空", "#f59e0b", "集合竞价后 5m 已回吐"
            elif jzf < 0 and ret5 > 0.5:
                conf_label, conf_color, note = "⚠ 反转", "#a855f7", "开低后 5m 反弹"
            else:
                conf_label, conf_color, note = "—", "#6b7280", "中性"
            if is_stale:
                note = f"⚠ 数据 stale ({row_date}, 旧于 {latest_date_in_rows}) — Hive 重建后修复"
            body_rows.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td><code>{html.escape(sym)}</code>{stale_mark}</td>"
                f"<td>{html.escape(name)}</td>"
                f"<td style='text-align:right;'>{jzf_str}</td>"
                f"<td style='text-align:right;'>{r['open']:.2f}</td>"
                f"<td style='text-align:right;'>{r['close']:.2f}</td>"
                f"<td style='color:{ret5_color}; font-weight:600; text-align:right;'>"
                f"{ret5:+.3f}%</td>"
                f"<td style='text-align:right;'>{r['volume']:,.0f}</td>"
                f"<td style='color:{conf_color}; font-weight:600;'>{conf_label}</td>"
                f"<td style='font-size:11px; color:var(--muted, #6b7280);'>{note}</td>"
                "</tr>"
            )
        audit_html = (
            '<h3 style="font-size:13px; margin:10px 0 6px 0; color:var(--muted, #6b7280);">'
            f"今日 picks 集合竞价 confirmation ({len(audit_rows)} 只 / 第一根 5m bar 9:30-9:35)"
            "</h3>"
            '<table class="data">'
            + colgroup
            + f"<thead><tr>{thead}</tr></thead>"
            + f"<tbody>{''.join(body_rows)}</tbody>"
            + "</table>"
        )

    footer = (
        '<div style="margin-top:10px; font-size:11px; color:var(--muted, #6b7280);">'
        "💡 <strong>v19.10 升级后</strong>: JZF 用 daily open 算(production 已 ship). "
        "5m enhancement layer 是 future work — 9:35 真 first 5m bar 可作信号 confirmation "
        "降低 false positive(daily JZF 高 + 5m 已回吐 = 假跳空,需排除).<br>"
        "数据源 <code>data_cache/db.duckdb</code> view <code>kline_5m</code> · "
        "mootdx 抓取 ~2 年硬上限 · paper_trade 无 inject (audit only, 不进 production scoring)"
        "</div>"
    )

    return schema_html + audit_html + footer
