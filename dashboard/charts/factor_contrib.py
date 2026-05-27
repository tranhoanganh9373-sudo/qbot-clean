"""Per-stock 因子贡献 stacked bar — z(pred) baseline + sidecar 调整分解.

数据源 (read-only 重算, 不改 production):
  - data_cache/v17_dens_train24_predictions.parquet  → 月度 pred score (最新月 cross-section)
  - data_cache/baidu_kline.parquet                   → amp_imb_20d (via paper_trade_today.load_amp_imb_20d_overlay)

公式 (跟 production v19.6 一致):
  z_pred  = cross-sectional z-score of pred score
  z_amp   = cross-sectional z-score of amp_imb_20d (NaN 填 mean)
  sidecar = -SIDECAR_LAMBDA_AMP_20D × z_amp
  final   = z_pred + sidecar
  Top 8 = 按 final desc 排序前 8

输出: stacked bar HTML 片段 (<img> + table + 解读文字).

注意:
  - production paper_trade_today.py 每日实时跑 DEnsemble 模型, 用最新日 cross-section.
    dashboard 用 cache 最新月 (e.g. 2026-04-30) 是 read-only 近似.
  - 与 production 当日 picks 可能不完全重合 (cache 月度 vs production 日度),
    但 sidecar 公式与 λ 完全一致 → 因子贡献分解结论可靠.
"""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dashboard.utils import fig_to_base64
from dashboard.utils.kline_fast import get_stock_kline
from dashboard.utils.stock_names import code_with_name, name_of

TOP_N = 8

# Sparkline 显示窗口 (最近 N 个交易日)
SPARKLINE_WINDOW = 21

# Z-score 防除零兜底
_Z_EPS = 1e-12


def _placeholder_html(message: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:32px 16px;">'
        f"{message}"
        "</div>"
    )


def _load_latest_predictions(pred_path: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    """读 predictions cache, 返回 (latest_cross_section_df, latest_dt).
    返回 df 含 [instrument, score] 列.
    """
    df = pd.read_parquet(pred_path, columns=["datetime", "instrument", "score"])
    latest_dt = df["datetime"].max()
    sub = df[df["datetime"] == latest_dt].copy()
    sub = sub.dropna(subset=["score"]).reset_index(drop=True)
    return sub[["instrument", "score"]], latest_dt


def _z_score(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < _Z_EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def _compute_factor_contrib(
    pred_df: pd.DataFrame,
    amp_map: dict[str, float],
    lam_amp: float,
) -> pd.DataFrame:
    """重算 z_pred / z_amp / final_score, 跟 production 公式一致.
    pred_df: 含 [instrument, score].
    amp_map: instrument → amp_imb_20d.
    lam_amp: λ_amp_20d (production = 0.30).
    返回: pred_df 加 [amp_imb_20d, z_pred, z_amp, sidecar_contrib, final_score] 列.
    """
    out = pred_df.copy()
    out["z_pred"] = _z_score(out["score"])

    out["amp_imb_20d"] = out["instrument"].map(amp_map)
    amp_series = out["amp_imb_20d"]
    amp_mean = amp_series.mean()
    amp_std = amp_series.std(ddof=0)

    fill_value = amp_mean if pd.notna(amp_mean) else 0.0
    amp_filled = amp_series.fillna(fill_value)

    if not np.isfinite(amp_std) or amp_std < _Z_EPS:
        out["z_amp"] = 0.0
    else:
        out["z_amp"] = (amp_filled - fill_value) / amp_std

    out["sidecar_contrib"] = -lam_amp * out["z_amp"]
    out["final_score"] = out["z_pred"] + out["sidecar_contrib"]
    return out


def _build_stacked_bar_fig(top_df: pd.DataFrame, lam_amp: float) -> plt.Figure:
    """画 Top N stacked bar.
    bar 蓝色底段 = z_pred, 红/绿顶段 = sidecar_contrib (-λ·z_amp).
    """
    syms = top_df["instrument"].tolist()
    z_pred = top_df["z_pred"].to_numpy()
    sidecar = top_df["sidecar_contrib"].to_numpy()
    final = top_df["final_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(syms))
    width = 0.62

    # 底段: z_pred (蓝)
    ax.bar(
        x, z_pred, width=width,
        color="#4a90e2", alpha=0.88,
        label="z(pred) baseline",
        edgecolor="#1f4e8a", linewidth=0.6,
    )

    # 顶段: sidecar 调整 (绿正 / 红负)
    colors = ["#27ae60" if c >= 0 else "#e74c3c" for c in sidecar]
    ax.bar(
        x, sidecar, bottom=z_pred, width=width,
        color=colors, alpha=0.88,
        edgecolor="#444", linewidth=0.5,
        label=f"sidecar = -{lam_amp:.2f}·z(amp_imb_20d)",
    )

    # final_score 标注
    for i, (f, zp, sc) in enumerate(zip(final, z_pred, sidecar)):
        top = max(zp, zp + sc, 0)
        ax.text(
            i, top + 0.06, f"{f:+.2f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222",
        )

    ax.axhline(0, color="#666", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([code_with_name(s) for s in syms], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Score (z-score units)")
    ax.set_title(
        f"Top {len(syms)} 因子贡献分解  (final = z_pred − {lam_amp:.2f}·z_amp)",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    return fig


def _build_detail_table(top_df: pd.DataFrame) -> str:
    """Top N 一张 z_pred / z_amp / sidecar / final 数字表."""
    rows = []
    for _, r in top_df.iterrows():
        sym_raw = str(r["instrument"])
        sym = html.escape(sym_raw)
        nm = html.escape(name_of(sym_raw) or "")
        amp_raw = r["amp_imb_20d"]
        amp_raw_str = f"{amp_raw:+.3f}" if pd.notna(amp_raw) else "—"
        z_pred = r["z_pred"]
        z_amp = r["z_amp"]
        sidecar = r["sidecar_contrib"]
        final = r["final_score"]
        sidecar_class = "pnl-pos" if sidecar >= 0 else "pnl-neg"
        rows.append(
            f"<tr>"
            f"<td>{sym}</td>"
            f"<td>{nm}</td>"
            f"<td>{z_pred:+.3f}</td>"
            f"<td>{amp_raw_str}</td>"
            f"<td>{z_amp:+.3f}</td>"
            f'<td class="{sidecar_class}">{sidecar:+.3f}</td>'
            f"<td><b>{final:+.3f}</b></td>"
            f"</tr>"
        )
    tbody = "".join(rows)
    return (
        '<table class="data" style="margin-top:12px;">'
        "<thead><tr>"
        "<th>Code</th>"
        "<th>Name</th>"
        "<th>z(pred)</th>"
        "<th>amp_imb_20d</th>"
        "<th>z(amp)</th>"
        "<th>sidecar</th>"
        "<th>final</th>"
        "</tr></thead>"
        f"<tbody>{tbody}</tbody>"
        "</table>"
    )


def _build_explanation(
    latest_dt: pd.Timestamp,
    lam_amp: float,
    kline_status: str,
    n_with_amp: int,
    n_total: int,
) -> str:
    """图下方的解读文字."""
    sidecar_state = (
        "已应用 ✓"
        if kline_status.startswith("ok")
        else f"降级 (kline {html.escape(kline_status)}, sidecar=0, final=z_pred)"
    )
    return (
        '<div style="margin-top:14px; font-size:13px; line-height:1.6;">'
        f"<b>截面日期:</b> {latest_dt.date()} (predictions cache 最新月)<br>"
        f"<b>Sidecar 状态:</b> {sidecar_state} &middot; "
        f"amp 覆盖 {n_with_amp}/{n_total}<br>"
        "<b>读图:</b> "
        "蓝色底段 = <code>z(pred)</code> baseline (DEnsemble 纯预测), "
        '<span style="color:#27ae60;">绿色顶段</span> = sidecar 推高 '
        "(本股 amp_imb_20d 低于均值, 振幅偏弱 → 加分), "
        '<span style="color:#e74c3c;">红色顶段</span> = sidecar 压低 '
        "(本股 amp_imb_20d 高于均值, 振幅过强 → 反转减分).<br>"
        "<b>红绿段越长 = 越是真正的 sidecar pick</b> "
        "(纯 pred 排不到 Top 8, 靠 sidecar 翻盘 / 或反之被 sidecar 踢掉)."
        "</div>"
        '<div style="margin-top:6px; font-size:11px; color:var(--muted);">'
        "公式: <code>final = z(pred) − "
        f"{lam_amp:.2f}"
        " · z(amp_imb_20d)</code> — 与 production paper_trade_today.py 一致 "
        "(read-only 重算, 不改 production).<br>"
        "数据源: <code>v17_dens_train24_predictions.parquet</code> + "
        "<code>baidu_kline.parquet</code> "
        "(via paper_trade_today.load_amp_imb_20d_overlay)."
        "</div>"
    )


def _build_top8_sparkline_fig(top_picks: list[str]) -> plt.Figure | None:
    """Top picks 最近 N 个交易日 close 走势 mini chart grid (2×4).

    每股 mini line chart, color 由 21d 涨跌幅决定 (绿涨/红跌),
    title 显示 sym + 涨跌幅 %, 单个失败 (Hive 无分区) 优雅 fallback 占位.

    Returns:
        matplotlib Figure (2×4 grid) or None 如果全部 picks 都 no data.
    """
    if not top_picks:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(12, 4.5))
    fig.suptitle(
        f"Top {len(top_picks)} last-{SPARKLINE_WINDOW}-bar close",
        fontsize=11,
    )
    n_ok = 0
    for ax, sym in zip(axes.flat, top_picks):
        try:
            df = get_stock_kline(sym)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()

        if df.empty or "close" not in df.columns or "date" not in df.columns:
            ax.text(
                0.5, 0.5, f"{sym}\nno data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        last = df.tail(SPARKLINE_WINDOW)
        if len(last) < 2:
            ax.text(
                0.5, 0.5, f"{sym}\n<2 bars",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        close = last["close"]
        ret = (close.iloc[-1] / close.iloc[0] - 1) * 100 if close.iloc[0] != 0 else 0.0
        color = "#27ae60" if ret >= 0 else "#e74c3c"
        ax.plot(last["date"], close, color=color, linewidth=1.5)
        ax.fill_between(last["date"], close, alpha=0.15, color=color)
        ax.set_title(f"{code_with_name(sym)} {ret:+.1f}%", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        n_ok += 1

    # 关闭多余的 subplot (top_picks < 8 的情况)
    for ax in list(axes.flat)[len(top_picks):]:
        ax.axis("off")

    fig.tight_layout()
    return fig if n_ok > 0 else None


def _load_picks_today_json(json_path: Path) -> tuple[pd.DataFrame, dict, str] | None:
    """读 picks_today.json (production daily dump). 返 (top_df, meta_dict, source_note).

    若 JSON 不存在/schema 不匹配 → 返 None, 调用者 fallback 到 cache 重算.

    返回的 top_df cols: instrument / z_pred / z_amp / amp_imb_20d / sidecar_contrib / final_score
    跟 _compute_factor_contrib 输出的 cols 一致.
    """
    if not json_path.exists():
        return None
    try:
        import json as _json
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("production_version") != "v19.6":
            # 仅 v19.6 sidecar 对应当前 factor_contrib 公式; v19.4 后续可扩
            return None
        picks = data.get("picks", [])
        if not picks:
            return None
        lam = float(data.get("sidecar", {}).get("lambda") or 0.30)
        rows = []
        for p in picks:
            z_pred = float(p.get("z_pred", 0))
            z_amp = float(p.get("z_amp") or 0)
            sidecar_contrib = -lam * z_amp
            final_score = float(p.get("final_score", z_pred + sidecar_contrib))
            rows.append({
                "instrument": p["sym"],
                "score": float(p.get("score", 0)),
                "z_pred": z_pred,
                "amp_imb_20d": p.get("amp_imb_20d"),
                "z_amp": z_amp,
                "sidecar_contrib": sidecar_contrib,
                "final_score": final_score,
            })
        top_df = pd.DataFrame(rows)
        meta = {
            "as_of_date": data.get("as_of_date", ""),
            "production_version": data.get("production_version", ""),
            "lam": lam,
            "kline_status": data.get("kline_status", "n/a"),
            "n_with_factor": int(data.get("n_with_factor", 0)),
            "n_total": int(data.get("n_total", len(picks))),
        }
        source_note = (
            f"production picks ({meta['as_of_date']}, {meta['production_version']})"
        )
        return top_df, meta, source_note
    except Exception:
        return None


def build_factor_contrib_section(
    pred_path: Path,
    kline_path: Path,
    paper_trade_path: Path,
) -> str:
    """主入口 — 返回完整 HTML 片段 (img + table + 文字).

    优先读 data_cache/picks_today.json (production daily dump, 100% 跟 production picks 一致);
    若 JSON 不存在/v19.6 sidecar 关闭 → fallback cache 重算 (与 production 公式一致但 picks 是 cache cross-section, 不完全等同 production).

    paper_trade_path 仅用于 import load_amp_imb_20d_overlay & 取 SIDECAR_LAMBDA_AMP_20D.
    """
    # 优先 production picks_today.json
    picks_json_path = pred_path.parent / "picks_today.json"
    json_loaded = _load_picks_today_json(picks_json_path)
    if json_loaded is not None:
        top, meta, source_note = json_loaded
        lam_amp = meta["lam"]
        latest_dt = pd.Timestamp(meta["as_of_date"]) if meta["as_of_date"] else pd.Timestamp.now()
        kline_status = meta["kline_status"]
        n_with_amp = meta["n_with_factor"]
        n_total = meta["n_total"]
        try:
            fig = _build_stacked_bar_fig(top, lam_amp)
            uri = fig_to_base64(fig)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            return _placeholder_html(
                f"绘制 stacked bar 失败: <code>{type(exc).__name__}: "
                f"{html.escape(str(exc))}</code>"
            )
        img_html = (
            f'<div style="font-size:12px; color:var(--muted); margin-bottom:8px;">'
            f"数据源: <b>{html.escape(source_note)}</b>"
            f"</div>"
            f'<img src="{uri}" alt="Top {TOP_N} 因子贡献分解"/>'
        )
        top_syms = top["instrument"].tolist()
        sparkline_html = ""
        try:
            spark_fig = _build_top8_sparkline_fig(top_syms)
            if spark_fig is not None:
                spark_uri = fig_to_base64(spark_fig)
                plt.close(spark_fig)
                sparkline_html = (
                    '<div style="margin-top:14px;">'
                    f'<img src="{spark_uri}" alt="Top {len(top_syms)} 21日 kline sparkline"/>'
                    "</div>"
                )
        except Exception as exc:  # noqa: BLE001
            sparkline_html = ""
        table_html = _build_detail_table(top)
        expl_html = _build_explanation(latest_dt, lam_amp, kline_status, n_with_amp, n_total)
        return img_html + sparkline_html + table_html + expl_html

    if not pred_path.exists():
        return _placeholder_html(
            f"predictions cache 不存在: <code>{html.escape(str(pred_path))}</code>"
        )
    if not kline_path.exists():
        return _placeholder_html(
            f"kline parquet 不存在: <code>{html.escape(str(kline_path))}</code>"
        )

    # import paper_trade_today 以拿 load_amp_imb_20d_overlay + SIDECAR_LAMBDA_AMP_20D.
    # 用 importlib (避免 sys.path 污染); 注意 paper_trade_today 顶层 import qlib,
    # 但只 import 名字, qlib.init 仅在 main() 内调用 → 安全.
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location("_paper_trade_for_dashboard", paper_trade_path)
        if spec is None or spec.loader is None:
            return _placeholder_html(
                f"无法 import <code>{html.escape(str(paper_trade_path))}</code>"
            )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        load_amp_overlay = mod.load_amp_imb_20d_overlay
        lam_amp = float(mod.SIDECAR_LAMBDA_AMP_20D)
    except Exception as exc:  # noqa: BLE001
        return _placeholder_html(
            f"import paper_trade_today 失败: <code>{type(exc).__name__}: "
            f"{html.escape(str(exc))}</code>"
        )

    # 读最新 cross-section
    try:
        pred_df, latest_dt = _load_latest_predictions(pred_path)
    except Exception as exc:  # noqa: BLE001
        return _placeholder_html(
            f"读取 predictions cache 失败: <code>{type(exc).__name__}: "
            f"{html.escape(str(exc))}</code>"
        )

    if pred_df.empty:
        return _placeholder_html("predictions cache 空 cross-section.")

    # 算 amp_imb_20d (复用 production 函数, 只调用不改)
    try:
        amp_map, kline_status = load_amp_overlay(latest_dt)
    except Exception as exc:  # noqa: BLE001
        amp_map, kline_status = {}, f"error-{type(exc).__name__}"

    # 重算因子贡献
    full = _compute_factor_contrib(pred_df, amp_map, lam_amp)
    full = full.sort_values("final_score", ascending=False).reset_index(drop=True)
    top = full.head(TOP_N).copy()

    n_with_amp = int(full["amp_imb_20d"].notna().sum())
    n_total = len(full)

    # 画图
    try:
        fig = _build_stacked_bar_fig(top, lam_amp)
        uri = fig_to_base64(fig)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder_html(
            f"绘制 stacked bar 失败: <code>{type(exc).__name__}: "
            f"{html.escape(str(exc))}</code>"
        )

    img_html = f'<img src="{uri}" alt="Top {TOP_N} 因子贡献分解"/>'

    # Top N 21日 sparkline grid (Stage 2 visual 补充)
    top_syms = top["instrument"].tolist()
    sparkline_html = ""
    try:
        spark_fig = _build_top8_sparkline_fig(top_syms)
        if spark_fig is not None:
            spark_uri = fig_to_base64(spark_fig)
            plt.close(spark_fig)
            sparkline_html = (
                '<div style="margin-top:14px;">'
                f'<img src="{spark_uri}" alt="Top {len(top_syms)} 21日 kline sparkline"/>'
                "</div>"
            )
    except Exception as exc:  # noqa: BLE001
        sparkline_html = (
            '<div class="placeholder-content" style="margin-top:14px; padding:12px;">'
            f"sparkline 生成失败: <code>{type(exc).__name__}: "
            f"{html.escape(str(exc))}</code>"
            "</div>"
        )

    table_html = _build_detail_table(top)
    expl_html = _build_explanation(latest_dt, lam_amp, kline_status, n_with_amp, n_total)
    return img_html + sparkline_html + table_html + expl_html
