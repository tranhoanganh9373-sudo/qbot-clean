"""自动维护 data_cache/portfolio.xlsx — 4 sheet 实盘跟踪.

Sheet:
  Positions  持仓明细 (今日 BUY 追加 + 当前价每日更新, 公式自动算浮盈)
  Daily      每日总览 (跨日 P&L)
  Weekly     周结 (用户手填周区间/周收益, 模板)
  Training   月度训练 log (用户手填 val_loss/备注, 模板)

依赖:
  data_cache/paper_trade_log.csv  (今日 BUY/SELL 信号)
  data_cache/portfolio_state.json (当前持仓 symbols)
  data_cache/baidu_kline.parquet  (latest close)
  data_cache/universe.csv         (名称)

资金分配: score 加权
  total_capital = 50000
  daily_pool = total_capital / (K / N_DROP) = 50000 / 4 = 12500
  pick_quota = daily_pool * (score_i / sum_today_scores)
  shares = floor(pick_quota / price / 100) * 100

Run:  python examples/portfolio_excel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data_cache" / "portfolio.xlsx"
STATE = ROOT / "data_cache" / "portfolio_state.json"
LOG = ROOT / "data_cache" / "paper_trade_log.csv"
KLINE = ROOT / "data_cache" / "baidu_kline.parquet"
UNIVERSE = ROOT / "data_cache" / "universe.csv"

TOTAL_CAPITAL = 50000.0
K = 8
N_DROP = 2
DAILY_POOL = TOTAL_CAPITAL / (K / N_DROP)  # = 12500

POS_HEADERS = [
    "推荐日期", "代码", "名称", "Score", "推荐价",
    "Score权重%", "推荐金额", "推荐手数",
    "实际买入价", "实际买入数", "实际成本",
    "当前价", "当前市值", "浮盈%", "浮盈元",
    "状态", "卖出价", "卖出日期", "实现盈亏", "备注",
]
DAILY_HEADERS = ["日期", "总资产", "现金", "持仓市值",
                  "当日P&L", "累计P&L", "收益率%", "持仓数"]
WEEKLY_HEADERS = ["周次(YYYY-WW)", "周初资产", "周末资产",
                   "周P&L", "周收益%", "交易次数", "胜率%"]
TRAINING_HEADERS = ["训练日期", "训练起", "训练止",
                     "样本数", "val_loss", "best_iter", "备注"]


def name_map() -> dict[str, str]:
    df = pd.read_csv(UNIVERSE, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    out = {}
    for _, r in df.iterrows():
        prefix = "SH" if r["code"].startswith(("6", "9")) else "SZ"
        out[f"{prefix}{r['code']}"] = r["name"]
    return out


def latest_closes() -> dict[str, tuple[pd.Timestamp, float]]:
    df = pd.read_parquet(KLINE, columns=["code", "date", "close"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["sym"] = df["code"].apply(
        lambda c: ("SH" if c.startswith(("6", "9")) else "SZ") + c)
    latest = df.sort_values("date").groupby("sym").tail(1).set_index("sym")
    return {sym: (row["date"], row["close"]) for sym, row in latest.iterrows()}


def init_workbook() -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)

    for name, headers in [
        ("Positions", POS_HEADERS), ("Daily", DAILY_HEADERS),
        ("Weekly", WEEKLY_HEADERS), ("Training", TRAINING_HEADERS),
    ]:
        ws = wb.create_sheet(name)
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="305496")
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22
        ws.freeze_panes = "A2"
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 13
    return wb


def existing_positions_keys(wb: Workbook) -> set[tuple[str, str]]:
    ws = wb["Positions"]
    out = set()
    for row in ws.iter_rows(min_row=2, max_col=2, values_only=True):
        if row[0] and row[1]:
            out.add((str(row[0]), str(row[1])))
    return out


def existing_daily_dates(wb: Workbook) -> set[str]:
    ws = wb["Daily"]
    out = set()
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        if row[0]:
            out.add(str(row[0]))
    return out


def add_positions_row(ws, today: str, sym: str, name: str, score: float,
                       price: float, weight_pct: float, quota: float,
                       shares: int, note: str = "") -> None:
    r = ws.max_row + 1
    ws.cell(row=r, column=1, value=today)
    ws.cell(row=r, column=2, value=sym)
    ws.cell(row=r, column=3, value=name)
    ws.cell(row=r, column=4, value=round(score, 4))
    ws.cell(row=r, column=5, value=round(price, 2))
    ws.cell(row=r, column=6, value=round(weight_pct, 1))
    ws.cell(row=r, column=7, value=round(quota, 0))
    ws.cell(row=r, column=8, value=shares)
    ws.cell(row=r, column=11, value=f"=I{r}*J{r}")
    ws.cell(row=r, column=13, value=f"=L{r}*J{r}")
    ws.cell(row=r, column=14, value=f"=IF(I{r}>0,(L{r}-I{r})/I{r},\"\")")
    ws.cell(row=r, column=15, value=f"=IF(I{r}>0,M{r}-K{r},\"\")")
    ws.cell(row=r, column=16, value="持仓")
    ws.cell(row=r, column=19, value=f"=IF(AND(Q{r}>0,J{r}>0),(Q{r}-I{r})*J{r},\"\")")
    if note:
        ws.cell(row=r, column=20, value=note)
    ws.cell(row=r, column=14).number_format = "0.00%"


def update_current_prices(ws, closes: dict[str, tuple[pd.Timestamp, float]]) -> None:
    """每个 持仓 row 更新 L 列 当前价."""
    for r in range(2, ws.max_row + 1):
        status = ws.cell(row=r, column=16).value
        if status != "持仓":
            continue
        sym = ws.cell(row=r, column=2).value
        if sym in closes:
            _, px = closes[sym]
            ws.cell(row=r, column=12, value=round(px, 2))


def compute_daily_snapshot(ws_pos, closes) -> tuple[float, float, int]:
    """从 Positions sheet 算 持仓市值, 累计实际成本, 持仓数."""
    market_value = 0.0
    cost = 0.0
    n = 0
    for r in range(2, ws_pos.max_row + 1):
        status = ws_pos.cell(row=r, column=16).value
        if status != "持仓":
            continue
        sym = ws_pos.cell(row=r, column=2).value
        actual_price = ws_pos.cell(row=r, column=9).value or 0
        actual_shares = ws_pos.cell(row=r, column=10).value or 0
        if not actual_shares:
            continue
        cost += actual_price * actual_shares
        if sym in closes:
            _, px = closes[sym]
            market_value += px * actual_shares
        else:
            market_value += actual_price * actual_shares
        n += 1
    return market_value, cost, n


def add_daily_row(ws, today: str, market_value: float, cost: float,
                  n_holdings: int) -> None:
    cash = TOTAL_CAPITAL - cost
    total = cash + market_value
    pnl_cum = total - TOTAL_CAPITAL
    pnl_pct = pnl_cum / TOTAL_CAPITAL

    prev_total = TOTAL_CAPITAL
    if ws.max_row >= 2:
        prev = ws.cell(row=ws.max_row, column=2).value
        if isinstance(prev, (int, float)):
            prev_total = prev
    pnl_day = total - prev_total

    r = ws.max_row + 1
    ws.cell(row=r, column=1, value=today)
    ws.cell(row=r, column=2, value=round(total, 2))
    ws.cell(row=r, column=3, value=round(cash, 2))
    ws.cell(row=r, column=4, value=round(market_value, 2))
    ws.cell(row=r, column=5, value=round(pnl_day, 2))
    ws.cell(row=r, column=6, value=round(pnl_cum, 2))
    ws.cell(row=r, column=7, value=round(pnl_pct, 4))
    ws.cell(row=r, column=8, value=n_holdings)
    ws.cell(row=r, column=7).number_format = "0.00%"


def main() -> None:
    if not LOG.exists():
        print(f"找不到 {LOG}, 先跑 paper_trade_today.py")
        return

    log_df = pd.read_csv(LOG, dtype={"symbol": str})
    state = json.loads(STATE.read_text())
    today = state.get("date", str(log_df["date"].max()))
    print(f"今日: {today}")

    today_buys = log_df[(log_df["date"] == today) & (log_df["action"] == "BUY")]
    if today_buys.empty:
        print(f"  今天 ({today}) 没有 BUY 信号 — 仅更新 Daily 总览")
    else:
        print(f"  今日 BUY 信号: {len(today_buys)} 只")
        print(today_buys[["symbol", "name", "score", "price"]].to_string(index=False))

    names = name_map()
    print(f"\n加载 universe 名称表: {len(names)} 只")
    print("加载 latest closes (parquet) ...")
    closes = latest_closes()
    latest_date = max(d for d, _ in closes.values())
    print(f"  最新交易日 (parquet): {latest_date.date()}")

    if XLSX.exists():
        print(f"\n打开 existing {XLSX.name}")
        wb = load_workbook(XLSX)
    else:
        print(f"\n创建新 {XLSX.name}")
        wb = init_workbook()

    ws_pos = wb["Positions"]
    pos_keys = existing_positions_keys(wb)

    if not today_buys.empty:
        sum_score = today_buys["score"].sum()
        for _, row in today_buys.iterrows():
            sym = row["symbol"]
            key = (today, sym)
            if key in pos_keys:
                print(f"  跳过 {sym} (Positions 已有 {today})")
                continue
            weight = row["score"] / sum_score if sum_score > 0 else 0.5
            weight_pct = weight * 100
            quota = DAILY_POOL * weight
            price = float(row["price"])
            shares = int(quota // (price * 100)) * 100 if price > 0 else 0
            note = ""
            if shares == 0 and price > 0 and price * 100 <= DAILY_POOL:
                shares = 100
                quota = price * 100
                note = f"score quota买不起1手, 已上调至1手({quota:.0f}元)"
            elif shares == 0 and price * 100 > DAILY_POOL:
                note = f"1手价{price*100:.0f}元 > daily_pool{DAILY_POOL:.0f} - 跳过"
            add_positions_row(
                ws_pos, today, sym, names.get(sym, "?"),
                row["score"], price, weight_pct, quota, shares, note,
            )
            print(f"  + Positions: {sym}  weight={weight_pct:.1f}%  "
                  f"quota={quota:.0f}  推荐手数={shares}  {note}")

    update_current_prices(ws_pos, closes)

    daily_dates = existing_daily_dates(wb)
    if today in daily_dates:
        print(f"\nDaily ({today}) 已存在 — 跳过追加")
    else:
        mv, cost, n_h = compute_daily_snapshot(ws_pos, closes)
        add_daily_row(wb["Daily"], today, mv, cost, n_h)
        print(f"\n+ Daily {today}: 总资产={TOTAL_CAPITAL - cost + mv:.0f} "
              f"持仓市值={mv:.0f} 现金={TOTAL_CAPITAL - cost:.0f} 持仓数={n_h}")

    wb.save(XLSX)
    print(f"\n保存: {XLSX} ({XLSX.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
