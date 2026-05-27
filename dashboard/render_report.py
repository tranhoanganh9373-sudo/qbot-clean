"""Dashboard Stage 1 渲染入口 — 生成 reports/daily_report_YYYYMMDD.html.

模式: Lean template.html + matplotlib base64 (self-contained, 浏览器双击打开).

Run:
  python dashboard/render_report.py            # 默认今日
  python dashboard/render_report.py --date 2026-05-26
  python dashboard/render_report.py --out reports/custom.html
"""
from __future__ import annotations

# 所有 dashboard 时间显示用北京时间 (A 股交易时区), 不参考 Mac 本机时区.
# Set TZ before datetime imports so all imported chart panels 继承.
import os as _os, time as _time
_os.environ["TZ"] = "Asia/Shanghai"
_time.tzset()

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 允许直接 `python dashboard/render_report.py` 而非 `python -m dashboard.render_report`
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.charts.ab_v19_10_vs_v19_6 import build_ab_v19_10_vs_v19_6_section
from dashboard.charts.ab_v19_6_vs_v19_4 import build_ab_section
from dashboard.charts.daily_check_status import build_daily_check_status_section
from dashboard.charts.factor_config import build_factor_config_panel
from dashboard.charts.factor_contrib import build_factor_contrib_section
from dashboard.charts.factor_coverage_health import build_factor_coverage_health_section
from dashboard.charts.factor_discovery_page import build_factor_discovery_page
from dashboard.charts.factor_ic_heatmap import build_factor_ic_heatmap_section
from dashboard.charts.factor_sandbox import build_factor_sandbox_section
from dashboard.charts.forward_oos_curve import build_forward_oos_chart
from dashboard.charts.glossary import build_glossary_section
from dashboard.charts.is_oos_gap_scatter import build_is_oos_scatter_section
from dashboard.charts.kline_5m_health import build_kline_5m_health_section
from dashboard.charts.live_5m_picks import build_live_5m_picks_section
from dashboard.charts.fetch_5m_progress import build_fetch_5m_progress_section
from dashboard.charts.pnl_attribution import build_pnl_attribution_section
from dashboard.charts.alert_center import build_alert_center_section
from dashboard.charts.market_overview import build_market_overview_section
from dashboard.charts.picks_accuracy import build_picks_accuracy_section
from dashboard.charts.server_status import build_server_status_section
from dashboard.charts.risk_metrics import build_risk_metrics_section
from dashboard.charts.risk_events import build_risk_events_section
from dashboard.charts.phase_b_history import build_phase_b_history_section
from dashboard.charts.leak_scan import build_leak_scan_section
from dashboard.charts.debate_veto_panel import build_debate_veto_section
from dashboard.charts.daily_pnl_heatmap import build_daily_pnl_heatmap_section
from dashboard.charts.data_freshness import build_data_freshness_section
from dashboard.charts.kpi_summary import build_kpi_summary_section
from dashboard.charts.model_leaderboard import build_leaderboard_section
from dashboard.charts.mt180_top_factors import build_mt180_top_factors_section
from dashboard.charts.multi_agent_debate import build_multi_agent_debate_section
from dashboard.charts.picks_rotation import build_picks_rotation_section
from dashboard.charts.picks_score_distribution import build_picks_dist_section
from dashboard.charts.portfolio_curve import build_portfolio_curve_section
from dashboard.charts.positions_pnl import build_positions_table, build_trade_input_table
from dashboard.charts.recommended_picks import build_recommended_picks_section
from dashboard.charts.shadow_paper_trade import build_shadow_paper_trade_section
from dashboard.charts.today_actions import build_today_actions_section
from dashboard.charts.sector_breakdown import build_sector_breakdown_section
from dashboard.charts.trade_history_timeline import build_trade_timeline_section
from dashboard.charts.universe_progress import build_universe_progress_section
from dashboard.charts.user_notes import build_user_notes_panel

ROOT = _ROOT
TEMPLATE_PATH = ROOT / "dashboard" / "template.html"
PORTFOLIO_XLSX = ROOT / "data_cache" / "portfolio.xlsx"
PAPER_TRADE_PY = ROOT / "examples" / "paper_trade_today.py"
PREDICTIONS_PARQUET = ROOT / "data_cache" / "v17_dens_train24_predictions.parquet"
KLINE_PARQUET = ROOT / "data_cache" / "baidu_kline.parquet"
REPORTS_DIR = ROOT / "reports"


def render(report_date: str, out_path: Path) -> Path:
    """渲染一份日报到 out_path. 返回 out_path."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"template not found: {TEMPLATE_PATH}")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    # 调每 chart / panel — 失败时 fallback 到 placeholder, 不抛
    try:
        glossary_html = build_glossary_section()
    except Exception as exc:  # noqa: BLE001
        glossary_html = (
            f'<div class="placeholder-content">glossary 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        kpi_summary_html = build_kpi_summary_section()
    except Exception as exc:  # noqa: BLE001
        kpi_summary_html = (
            f'<div class="placeholder-content">KPI summary 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        forward_html = build_forward_oos_chart(PORTFOLIO_XLSX)
    except Exception as exc:  # noqa: BLE001
        forward_html = (
            f'<div class="placeholder-content">forward_oos chart 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        positions_html = build_positions_table(PORTFOLIO_XLSX, KLINE_PARQUET)
    except Exception as exc:  # noqa: BLE001
        positions_html = (
            f'<div class="placeholder-content">positions table 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        trade_input_html = build_trade_input_table(PORTFOLIO_XLSX)
    except Exception as exc:  # noqa: BLE001
        trade_input_html = (
            f'<div class="placeholder-content">trade input 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        recommended_picks_html = build_recommended_picks_section()
    except Exception as exc:  # noqa: BLE001
        recommended_picks_html = (
            f'<div class="placeholder-content">recommended picks 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        shadow_paper_trade_html = build_shadow_paper_trade_section()
    except Exception as exc:  # noqa: BLE001
        shadow_paper_trade_html = (
            f'<div class="placeholder-content">shadow paper_trade 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        portfolio_curve_html = build_portfolio_curve_section()
    except Exception as exc:  # noqa: BLE001
        portfolio_curve_html = (
            f'<div class="placeholder-content">portfolio curve 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        kline_5m_health_html = build_kline_5m_health_section()
    except Exception as exc:  # noqa: BLE001
        kline_5m_health_html = (
            f'<div class="placeholder-content">5m kline health 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        live_5m_picks_html = build_live_5m_picks_section()
    except Exception as exc:  # noqa: BLE001
        live_5m_picks_html = (
            f'<div class="placeholder-content">live 5m picks 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        fetch_5m_progress_html = build_fetch_5m_progress_section()
    except Exception as exc:  # noqa: BLE001
        fetch_5m_progress_html = (
            f'<div class="placeholder-content">fetch 5m progress 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        pnl_attribution_html = build_pnl_attribution_section()
    except Exception as exc:  # noqa: BLE001
        pnl_attribution_html = (
            f'<div class="placeholder-content">pnl attribution 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        alert_center_html = build_alert_center_section()
    except Exception as exc:  # noqa: BLE001
        alert_center_html = (
            f'<div class="placeholder-content">alert center 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        market_overview_html = build_market_overview_section()
    except Exception as exc:  # noqa: BLE001
        market_overview_html = (
            f'<div class="placeholder-content">market overview 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        picks_accuracy_html = build_picks_accuracy_section()
    except Exception as exc:  # noqa: BLE001
        picks_accuracy_html = (
            f'<div class="placeholder-content">picks accuracy 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        server_status_html = build_server_status_section()
    except Exception as exc:  # noqa: BLE001
        server_status_html = (
            f'<div class="placeholder-content">server status 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        risk_metrics_html = build_risk_metrics_section()
    except Exception as exc:  # noqa: BLE001
        risk_metrics_html = (
            f'<div class="placeholder-content">risk metrics 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        daily_pnl_heatmap_html = build_daily_pnl_heatmap_section()
    except Exception as exc:  # noqa: BLE001
        daily_pnl_heatmap_html = (
            f'<div class="placeholder-content">daily pnl heatmap 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        data_freshness_html = build_data_freshness_section()
    except Exception as exc:  # noqa: BLE001
        data_freshness_html = (
            f'<div class="placeholder-content">data freshness 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        today_actions_html = build_today_actions_section()
    except Exception as exc:  # noqa: BLE001
        today_actions_html = (
            f'<div class="placeholder-content">today actions 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        multi_agent_debate_html = build_multi_agent_debate_section()
    except Exception as exc:  # noqa: BLE001
        multi_agent_debate_html = (
            f'<div class="placeholder-content">multi-agent debate 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        factor_discovery_html = build_factor_discovery_page()
    except Exception as exc:  # noqa: BLE001
        factor_discovery_html = (
            f'<div class="placeholder-content">factor discovery page 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        mt180_top_factors_html = build_mt180_top_factors_section()
    except Exception as exc:  # noqa: BLE001
        mt180_top_factors_html = (
            f'<div class="placeholder-content">mt180 top factors 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        factor_html = build_factor_config_panel(PAPER_TRADE_PY)
    except Exception as exc:  # noqa: BLE001
        factor_html = (
            f'<div class="placeholder-content">factor config 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        contrib_html = build_factor_contrib_section(
            PREDICTIONS_PARQUET, KLINE_PARQUET, PAPER_TRADE_PY,
        )
    except Exception as exc:  # noqa: BLE001
        contrib_html = (
            f'<div class="placeholder-content">factor contrib 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        ab_v19_10_html = build_ab_v19_10_vs_v19_6_section()
    except Exception as exc:  # noqa: BLE001
        ab_v19_10_html = (
            f'<div class="placeholder-content">A/B v19.10 vs v19.6 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        ab_html = build_ab_section()
    except Exception as exc:  # noqa: BLE001
        ab_html = (
            f'<div class="placeholder-content">A/B v19.6 vs v19.4 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        leaderboard_html = build_leaderboard_section()
    except Exception as exc:  # noqa: BLE001
        leaderboard_html = (
            f'<div class="placeholder-content">model leaderboard 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        factor_ic_heatmap_html = build_factor_ic_heatmap_section()
    except Exception as exc:  # noqa: BLE001
        factor_ic_heatmap_html = (
            f'<div class="placeholder-content">factor IC heatmap 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        sector_breakdown_html = build_sector_breakdown_section()
    except Exception as exc:  # noqa: BLE001
        sector_breakdown_html = (
            f'<div class="placeholder-content">sector breakdown 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        is_oos_gap_scatter_html = build_is_oos_scatter_section()
    except Exception as exc:  # noqa: BLE001
        is_oos_gap_scatter_html = (
            f'<div class="placeholder-content">IS→OOS gap scatter 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        factor_coverage_health_html = build_factor_coverage_health_section()
    except Exception as exc:  # noqa: BLE001
        factor_coverage_health_html = (
            f'<div class="placeholder-content">factor coverage health 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        trade_history_timeline_html = build_trade_timeline_section()
    except Exception as exc:  # noqa: BLE001
        trade_history_timeline_html = (
            f'<div class="placeholder-content">trade history timeline 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        daily_check_status_html = build_daily_check_status_section()
    except Exception as exc:  # noqa: BLE001
        daily_check_status_html = (
            f'<div class="placeholder-content">daily check status 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        universe_progress_html = build_universe_progress_section()
    except Exception as exc:  # noqa: BLE001
        universe_progress_html = (
            f'<div class="placeholder-content">universe progress 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        picks_rotation_html = build_picks_rotation_section()
    except Exception as exc:  # noqa: BLE001
        picks_rotation_html = (
            f'<div class="placeholder-content">picks rotation 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        picks_distribution_html = build_picks_dist_section()
    except Exception as exc:  # noqa: BLE001
        picks_distribution_html = (
            f'<div class="placeholder-content">picks score distribution 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        user_notes_html = build_user_notes_panel(PORTFOLIO_XLSX)
    except Exception as exc:  # noqa: BLE001
        user_notes_html = (
            f'<div class="placeholder-content">user notes 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        factor_sandbox_html = build_factor_sandbox_section()
    except Exception as exc:  # noqa: BLE001
        factor_sandbox_html = (
            f'<div class="placeholder-content">factor sandbox 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        risk_events_html = build_risk_events_section()
    except Exception as exc:  # noqa: BLE001
        risk_events_html = (
            f'<div class="placeholder-content">risk events 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        phase_b_history_html = build_phase_b_history_section()
    except Exception as exc:  # noqa: BLE001
        phase_b_history_html = (
            f'<div class="placeholder-content">phase B history 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        leak_scan_html = build_leak_scan_section()
    except Exception as exc:  # noqa: BLE001
        leak_scan_html = (
            f'<div class="placeholder-content">leak scan 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    try:
        debate_veto_panel_html = build_debate_veto_section()
    except Exception as exc:  # noqa: BLE001
        debate_veto_panel_html = (
            f'<div class="placeholder-content">debate veto panel 渲染失败: '
            f"<code>{type(exc).__name__}: {exc}</code></div>"
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rendered = (
        template
        .replace("{{report_date}}", report_date)
        .replace("{{generated_at}}", generated_at)
        .replace("{{glossary}}", glossary_html)
        .replace("{{kpi_summary}}", kpi_summary_html)
        .replace("{{forward_oos_chart}}", forward_html)
        .replace("{{positions_table}}", positions_html)
        .replace("{{trade_input}}", trade_input_html)
        .replace("{{recommended_picks}}", recommended_picks_html)
        .replace("{{shadow_paper_trade}}", shadow_paper_trade_html)
        .replace("{{portfolio_curve}}", portfolio_curve_html)
        .replace("{{today_actions}}", today_actions_html)
        .replace("{{kline_5m_health}}", kline_5m_health_html)
        .replace("{{live_5m_picks}}", live_5m_picks_html)
        .replace("{{fetch_5m_progress}}", fetch_5m_progress_html)
        .replace("{{pnl_attribution}}", pnl_attribution_html)
        .replace("{{alert_center}}", alert_center_html)
        .replace("{{market_overview}}", market_overview_html)
        .replace("{{picks_accuracy}}", picks_accuracy_html)
        .replace("{{server_status}}", server_status_html)
        .replace("{{risk_metrics}}", risk_metrics_html)
        .replace("{{daily_pnl_heatmap}}", daily_pnl_heatmap_html)
        .replace("{{data_freshness}}", data_freshness_html)
        .replace("{{multi_agent_debate}}", multi_agent_debate_html)
        .replace("{{mt180_top_factors}}", mt180_top_factors_html)
        .replace("{{factor_discovery_page}}", factor_discovery_html)
        .replace("{{factor_config}}", factor_html)
        .replace("{{factor_contrib}}", contrib_html)
        .replace("{{ab_v19_6_vs_v19_4}}", ab_html)
        .replace("{{ab_v19_10_vs_v19_6}}", ab_v19_10_html)
        .replace("{{model_leaderboard}}", leaderboard_html)
        .replace("{{factor_ic_heatmap}}", factor_ic_heatmap_html)
        .replace("{{sector_breakdown}}", sector_breakdown_html)
        .replace("{{is_oos_gap_scatter}}", is_oos_gap_scatter_html)
        .replace("{{factor_coverage_health}}", factor_coverage_health_html)
        .replace("{{trade_history_timeline}}", trade_history_timeline_html)
        .replace("{{daily_check_status}}", daily_check_status_html)
        .replace("{{universe_progress}}", universe_progress_html)
        .replace("{{picks_rotation}}", picks_rotation_html)
        .replace("{{picks_distribution}}", picks_distribution_html)
        .replace("{{user_notes}}", user_notes_html)
        .replace("{{factor_sandbox}}", factor_sandbox_html)
        .replace("{{risk_events}}", risk_events_html)
        .replace("{{phase_b_history}}", phase_b_history_html)
        .replace("{{leak_scan}}", leak_scan_html)
        .replace("{{debate_veto_panel}}", debate_veto_panel_html)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def _resolve_default_out(report_date: str) -> Path:
    compact = report_date.replace("-", "")
    return REPORTS_DIR / f"daily_report_{compact}.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render claude_finance dashboard daily report (Stage 1)."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="报告日期 YYYY-MM-DD (默认今日).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=("输出 HTML 路径 (默认 reports/daily_report_YYYYMMDD.html)."),
    )
    args = parser.parse_args(argv)

    report_date = args.date or datetime.now().strftime("%Y-%m-%d")
    # 验证格式
    try:
        datetime.strptime(report_date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: --date 必须 YYYY-MM-DD, 收到 {report_date!r}", file=sys.stderr)
        return 2

    out_path = Path(args.out).resolve() if args.out else _resolve_default_out(report_date)
    written = render(report_date, out_path)
    size_kb = written.stat().st_size / 1024
    print(f"[dashboard] wrote {written}  ({size_kb:.1f} KB)")

    # 维护 reports/latest.html symlink 永远指向最新一次 render 输出,
    # 浏览器固定 bookmark `reports/latest.html` 即可,不用每天换文件名.
    # (仅当 default out path / 同目录写入时更新, 避免 --out custom.html 误改 symlink.)
    if not args.out:
        latest_link = REPORTS_DIR / "latest.html"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(written.name)
            print(f"[dashboard] latest -> {written.name}")
        except OSError as e:
            print(f"[dashboard] symlink warn: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[dashboard] open: open {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
