"""Factor coverage health panel — 监控所有 sidecar 数据源 coverage + 新鲜度.

产生背景: 2026-05-25 发现 v19.4 margin sidecar covered=15/296 (silent NaN→0 fillna)
导致 sidecar 失效但无报警. 本面板每日把 7 个数据源的健康指标摆出来,
让此类静默 bug 立刻可见.

数据源 (全部只读, 不修改):
  1. baidu_kline  (全 A 股 universe)
  2. csi300_margin_14yr  (CSI300 universe)
  3. fund_flow_csi300    (CSI300 universe)
  4. shareholders_csi300 (CSI300 universe)
  5. industry_membership (全 A 股 universe)
  6. unlock              (全 A 股 universe)
  7. dragon_tiger        (CSI300, per-stock parquet 目录)

指标:
  - path_exists / total_rows / unique_codes
  - latest_date / days_stale (today - latest)
  - coverage_pct  (latest 日有数据的 codes / universe_size; per-stock-dir
                   类型用文件数估)
  - status emoji  (🟢 ok / 🟡 partial / 🟠 sparse / 🔴 stale / ❌ missing)

阈值:
  - 🟢 ok       cover ≥ 90% AND stale ≤ 3d
  - 🟡 partial  cover 50-90% OR stale 4-30d
  - 🟠 sparse   cover < 50%
  - 🔴 stale    stale > 30d
  - ❌ missing  path 不存在
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DC = ROOT / "data_cache"

# 阈值常量 (语义命名, 避免魔数)
COVER_OK_PCT = 90.0
COVER_PARTIAL_PCT = 50.0
STALE_OK_DAYS = 3
STALE_WARN_DAYS = 30

CSI300_UNIVERSE_SIZE = 300
FULL_A_UNIVERSE_SIZE = 5000  # 全 A 股近似值, mootdx 抓到 ~4676

# 颜色 (HTML inline style, 不依赖 css 变量, dark/light 都可读)
COLOR_OK = "#16a34a"
COLOR_WARN = "#f59e0b"
COLOR_BAD = "#dc2626"
COLOR_MUTED = "#6b7280"


def _classify_status(coverage_pct: float, days_stale: int) -> str:
    """返回 status emoji + label."""
    if days_stale > STALE_WARN_DAYS:
        return "🔴 stale"
    if coverage_pct < COVER_PARTIAL_PCT:
        return "🟠 sparse"
    if coverage_pct < COVER_OK_PCT or days_stale > STALE_OK_DAYS:
        return "🟡 partial"
    return "🟢 ok"


def _pick_date_col(df: pd.DataFrame) -> str | None:
    """选择主时间列, 按优先级."""
    for col in ("date", "datetime", "announce_date", "unlock_date",
                "data_date", "include_date"):
        if col in df.columns:
            return col
    return None


def _pick_code_col(df: pd.DataFrame) -> str | None:
    for col in ("code", "instrument", "symbol"):
        if col in df.columns:
            return col
    return None


def _check_parquet_file(
    name: str,
    path: Path,
    universe_size: int,
    coverage_mode: str = "latest_day",
) -> dict[str, Any]:
    """单个 parquet 文件健康检查.

    coverage_mode:
      'latest_day' — daily-frequency 数据 (kline/margin/fund_flow),
                     cover = latest 日 unique codes / universe.
      'any_data'   — event-frequency 数据 (shareholders/unlock 等),
                     cover = 全表 unique codes / universe.
    """
    if not path.exists():
        return {"name": name, "path_exists": False, "status": "❌ missing"}

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": name,
            "path_exists": True,
            "status": f"❌ read_error: {type(exc).__name__}",
        }

    date_col = _pick_date_col(df)
    code_col = _pick_code_col(df)

    if date_col is None or code_col is None:
        return {
            "name": name,
            "path_exists": True,
            "status": "❌ schema_unknown",
            "total_rows": len(df),
            "unique_codes": df[code_col].nunique() if code_col else 0,
        }

    # 标准化时间列
    dates = pd.to_datetime(df[date_col], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return {
            "name": name,
            "path_exists": True,
            "status": "❌ no_valid_dates",
            "total_rows": len(df),
            "unique_codes": df[code_col].nunique(),
        }

    today = pd.Timestamp.now().normalize()
    # 对于未来日期 (unlock 的 release_date 可能未来), 视为 0 stale
    days_stale = max(0, (today - latest).days)

    # coverage_mode 决定 cover 怎么算
    if coverage_mode == "any_data":
        codes_covered = int(df[code_col].nunique())
        cover_label = "any-data"
    else:
        latest_mask = dates == latest
        codes_covered = int(df.loc[latest_mask, code_col].nunique())
        cover_label = "latest-day"
    coverage_pct = round(
        codes_covered / universe_size * 100, 1
    ) if universe_size > 0 else 0.0
    # cap 100% (full-A universe 估值偏大可能 < 实际)
    coverage_pct = min(coverage_pct, 100.0)

    return {
        "name": name,
        "path_exists": True,
        "total_rows": len(df),
        "unique_codes": int(df[code_col].nunique()),
        "latest_date": latest.strftime("%Y-%m-%d"),
        "days_stale": days_stale,
        "codes_latest": codes_covered,
        "coverage_pct": coverage_pct,
        "status": _classify_status(coverage_pct, days_stale),
        "universe_size": universe_size,
        "coverage_mode": cover_label,
    }


def _check_per_stock_dir(
    name: str, directory: Path, universe_size: int
) -> dict[str, Any]:
    """per-stock parquet 目录 (e.g. dragon_tiger/000001.parquet)健康检查.

    coverage 用文件数估; freshness 从样本文件读 max 日期.
    """
    if not directory.exists() or not directory.is_dir():
        return {"name": name, "path_exists": False, "status": "❌ missing"}

    files = sorted(directory.glob("*.parquet"))
    n_files = len(files)
    if n_files == 0:
        return {
            "name": name,
            "path_exists": True,
            "status": "❌ empty_dir",
            "total_rows": 0,
            "unique_codes": 0,
        }

    # 估总 rows + freshness: 抽样最多 30 个文件以避 IO 过重
    sample_n = min(30, n_files)
    step = max(1, n_files // sample_n)
    sample_files = files[::step][:sample_n]
    total_rows_sample = 0
    latest_per_file: list[pd.Timestamp] = []
    for f in sample_files:
        try:
            sub = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            continue
        total_rows_sample += len(sub)
        date_col = _pick_date_col(sub)
        if date_col:
            d = pd.to_datetime(sub[date_col], errors="coerce").max()
            if not pd.isna(d):
                latest_per_file.append(d)

    # 估全量 rows = 样本平均 × 文件数
    if sample_files:
        avg_rows = total_rows_sample / len(sample_files)
        total_rows_est = int(avg_rows * n_files)
    else:
        total_rows_est = 0

    if latest_per_file:
        latest = max(latest_per_file)
        days_stale = max(0, (pd.Timestamp.now().normalize() - latest).days)
        latest_str = latest.strftime("%Y-%m-%d")
    else:
        days_stale = 99999
        latest_str = "?"

    coverage_pct = round(n_files / universe_size * 100, 1) if universe_size else 0.0
    coverage_pct = min(coverage_pct, 100.0)

    return {
        "name": name,
        "path_exists": True,
        "total_rows": total_rows_est,
        "unique_codes": n_files,
        "latest_date": latest_str,
        "days_stale": days_stale,
        "codes_latest": n_files,
        "coverage_pct": coverage_pct,
        "status": _classify_status(coverage_pct, days_stale),
        "universe_size": universe_size,
    }


def _check_csi300_kline_intersection(
    kline_path: Path, csi300_csv: Path
) -> dict[str, Any]:
    """专门检查 baidu_kline ∩ CSI300 (latest day) 的覆盖率.

    回答: "今天有多少 CSI300 成份股的 kline 收到了?" 这是 v19.6 sidecar 的核心 universe.
    """
    if not kline_path.exists() or not csi300_csv.exists():
        return {
            "name": "kline ∩ CSI300",
            "path_exists": False,
            "status": "❌ missing",
        }

    csi300_codes = set(pd.read_csv(csi300_csv)["code"].astype(str).tolist())
    if not csi300_codes:
        return {
            "name": "kline ∩ CSI300",
            "path_exists": True,
            "status": "❌ csi300_empty",
        }

    # 仅读 code + date 列 (大文件优化)
    kline = pd.read_parquet(kline_path, columns=["code", "date"])
    kline["code"] = kline["code"].astype(str)
    latest = kline["date"].max()
    today = pd.Timestamp.now().normalize()
    days_stale = max(0, (today - pd.Timestamp(latest)).days)

    latest_snapshot = kline[kline["date"] == latest]
    codes_latest = set(latest_snapshot["code"].tolist())
    csi300_covered = len(csi300_codes & codes_latest)
    coverage_pct = round(csi300_covered / CSI300_UNIVERSE_SIZE * 100, 1)
    coverage_pct = min(coverage_pct, 100.0)

    return {
        "name": "kline ∩ CSI300",
        "path_exists": True,
        "total_rows": len(kline),
        "unique_codes": int(kline["code"].nunique()),
        "latest_date": pd.Timestamp(latest).strftime("%Y-%m-%d"),
        "days_stale": days_stale,
        "codes_latest": csi300_covered,
        "coverage_pct": coverage_pct,
        "status": _classify_status(coverage_pct, days_stale),
        "universe_size": CSI300_UNIVERSE_SIZE,
    }


def _row_html(r: dict[str, Any]) -> str:
    name = html.escape(r.get("name", "?"))
    if not r.get("path_exists", False) or "❌" in r.get("status", ""):
        status = html.escape(r.get("status", "❌ unknown"))
        return (
            f"<tr><td>{name}</td>"
            f"<td colspan='6' style='color:{COLOR_BAD};'>{status}</td></tr>"
        )

    cov = r.get("coverage_pct", 0)
    cov_color = (
        COLOR_OK if cov >= COVER_OK_PCT
        else COLOR_WARN if cov >= COVER_PARTIAL_PCT
        else COLOR_BAD
    )
    stale = r.get("days_stale", 0)
    stale_color = (
        COLOR_OK if stale <= STALE_OK_DAYS
        else COLOR_WARN if stale <= STALE_WARN_DAYS
        else COLOR_BAD
    )
    status = html.escape(r.get("status", ""))
    universe = r.get("universe_size", 0)
    mode = r.get("coverage_mode", "latest-day")
    mode_hint = (
        f" <span style='color:{COLOR_MUTED}; font-size:10px;'>[{mode}]</span>"
    )
    cov_text = (
        f"{r.get('codes_latest', 0)}/{universe} "
        f"<span style='color:{cov_color}'>({cov}%)</span>{mode_hint}"
    )

    return f"""<tr>
  <td>{name}</td>
  <td>{status}</td>
  <td style='text-align:right'>{r.get('total_rows', 0):,}</td>
  <td style='text-align:right'>{r.get('unique_codes', 0):,}</td>
  <td>{html.escape(r.get('latest_date', '?'))}</td>
  <td style='color:{stale_color}; text-align:right'>{stale}d</td>
  <td style='text-align:right'>{cov_text}</td>
</tr>"""


def build_factor_coverage_health_section() -> str:
    """主入口 — 返回 HTML 片段, 塞到 template `{{factor_coverage_health}}`."""
    rows: list[dict[str, Any]] = []

    # 1. baidu_kline 全 universe
    rows.append(_check_parquet_file(
        "baidu_kline (full universe)",
        DC / "baidu_kline.parquet",
        FULL_A_UNIVERSE_SIZE,
    ))

    # 2. kline ∩ CSI300 (sidecar 实际 universe)
    rows.append(_check_csi300_kline_intersection(
        DC / "baidu_kline.parquet",
        DC / "csi300_constituents.csv",
    ))

    # 3. margin (CSI300)
    rows.append(_check_parquet_file(
        "margin (CSI300)",
        DC / "csi300_margin_14yr.parquet",
        CSI300_UNIVERSE_SIZE,
    ))

    # 4. fund_flow (CSI300)
    rows.append(_check_parquet_file(
        "fund_flow (CSI300)",
        DC / "fund_flow" / "fund_flow_csi300.parquet",
        CSI300_UNIVERSE_SIZE,
    ))

    # 5. shareholders (CSI300) — event-frequency, 用 any-data 覆盖率
    rows.append(_check_parquet_file(
        "shareholders (CSI300)",
        DC / "shareholders" / "shareholders_csi300.parquet",
        CSI300_UNIVERSE_SIZE,
        coverage_mode="any_data",
    ))

    # 6. industry_membership (snapshot, 行业归属表 — any-data 覆盖率)
    rows.append(_check_parquet_file(
        "industry_membership",
        DC / "industry" / "industry_membership.parquet",
        FULL_A_UNIVERSE_SIZE,
        coverage_mode="any_data",
    ))

    # 7. unlock (event-frequency — any-data 覆盖率)
    rows.append(_check_parquet_file(
        "unlock (full A-share)",
        DC / "unlock" / "unlock_detail_em.parquet",
        FULL_A_UNIVERSE_SIZE,
        coverage_mode="any_data",
    ))

    # 8. dragon_tiger (per-stock dir, CSI300)
    rows.append(_check_per_stock_dir(
        "dragon_tiger (CSI300, per-stock)",
        DC / "dragon_tiger",
        CSI300_UNIVERSE_SIZE,
    ))

    table_rows = "\n".join(_row_html(r) for r in rows)

    # 汇总: 计警告/错误数
    n_total = len(rows)
    n_ok = sum(1 for r in rows if "🟢" in r.get("status", ""))
    n_warn = sum(1 for r in rows if any(e in r.get("status", "") for e in ("🟡", "🟠")))
    n_bad = sum(1 for r in rows if any(e in r.get("status", "") for e in ("🔴", "❌")))

    summary_color = (
        COLOR_OK if n_bad == 0 and n_warn == 0
        else COLOR_WARN if n_bad == 0
        else COLOR_BAD
    )

    return f"""
<div style='margin-bottom:8px; font-size:13px;'>
  汇总: <b style='color:{summary_color}'>{n_ok}/{n_total} 🟢 ok</b>
  &middot; {n_warn} 🟡🟠 partial/sparse
  &middot; {n_bad} 🔴❌ stale/missing
</div>
<table class='data'>
<thead>
  <tr>
    <th>数据源</th>
    <th>状态</th>
    <th style='text-align:right'>总 rows</th>
    <th style='text-align:right'>unique codes</th>
    <th>latest</th>
    <th style='text-align:right'>stale</th>
    <th style='text-align:right'>cover (latest)</th>
  </tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
<div style='margin-top:10px; font-size:12px; color:{COLOR_MUTED};'>
  阈值: 🟢 ok = cover ≥ {COVER_OK_PCT:.0f}% AND stale ≤ {STALE_OK_DAYS}d
  &middot; 🟡 partial = 50-90% 或 stale 4-30d
  &middot; 🟠 sparse = cover &lt; {COVER_PARTIAL_PCT:.0f}%
  &middot; 🔴 stale = stale &gt; {STALE_WARN_DAYS}d
  &middot; ❌ missing/error
  <br>历史教训: 2026-05-25 v19.4 sidecar margin covered=15/296 (silent NaN bug) 已修;
  本面板每日自动检, 防此类隐患复现.
</div>
"""
