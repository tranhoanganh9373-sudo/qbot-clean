"""Targeted retry — 只 retry 280 老板股 (000/002/300老/600/601/603),
跳过 post-2019 IPO 注册制新股 (301/605/688/001/003/305/306).
单线程 + sleep 0.5s,绕 sina 限流。
"""
from __future__ import annotations
import os, sys, time, json
import pandas as pd
from pathlib import Path
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

# 复用 Phase 1B fetcher 的核心函数
import importlib.util
spec = importlib.util.spec_from_file_location("fetcher", ROOT / "examples" / "fetch_baidu_kline_v2_akshare.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)

CODES_PATH = ROOT / "data_cache" / "retry_targeted_codes.txt"
OUT_PATH = ROOT / "data_cache" / "baidu_kline_v2_retry.parquet"
FAILED_PATH = ROOT / "data_cache" / "akshare_fetch_failed_retry_targeted.csv"
LOG_PATH = Path("/tmp/fetch_v2_retry_targeted.log")

codes = [c.strip() for c in CODES_PATH.read_text().splitlines() if c.strip()]
print(f"[targeted-retry] {len(codes)} 老板股(000/002/300老/600/601/603)", flush=True)

TDX = fetcher.TDX_SERVERS
server_idx = 0
client = fetcher.make_mootdx_client(TDX[server_idx])
session = fetcher.requests.Session(); session.trust_env = False

all_dfs = []
failed = []
t0 = time.time()
consec_fail = 0

for i, code in enumerate(codes, 1):
    # rotate server on consec fail
    if consec_fail >= 3:
        server_idx = (server_idx + 1) % len(TDX)
        try:
            client = fetcher.make_mootdx_client(TDX[server_idx])
            print(f"  [{i}] rotated → {TDX[server_idx]}", flush=True)
        except Exception:
            pass
        consec_fail = 0

    # fetch (single attempt, no retry inside — fetcher handles)
    try:
        # 调用 fetcher 内部 fetch logic
        qfq = fetcher.fetch_qfq(client, code)
        if qfq is None or len(qfq) == 0:
            failed.append((code, "qfq_empty"))
            consec_fail += 1
        else:
            hfq = fetcher.fetch_hfq_combined(client, session, code, qfq)
            if hfq is None or len(hfq) == 0:
                failed.append((code, "hfq_failed"))
                consec_fail += 1
            else:
                all_dfs.append(hfq)
                consec_fail = 0
    except Exception as e:
        failed.append((code, f"err: {type(e).__name__}"))
        consec_fail += 1

    if i % 20 == 0:
        elapsed = time.time() - t0
        ok = len(all_dfs)
        eta = (len(codes) - i) * elapsed / i if i > 0 else 0
        print(f"  [{i}/{len(codes)}] ok={ok} elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    time.sleep(0.5)

elapsed = time.time() - t0
print(f"\n[done] {elapsed:.0f}s, ok={len(all_dfs)}/{len(codes)}", flush=True)

if all_dfs:
    full = pd.concat(all_dfs, ignore_index=True)
    # 验证
    print(f"[verify] neg_close={(full['close']<0).sum()} extreme_low={(full['close']<0.5).sum()}", flush=True)
    if (full['close'] < 0).any():
        print("[FATAL] still has neg close — abort write", flush=True)
        sys.exit(1)
    full.to_parquet(OUT_PATH, index=False)
    print(f"[wrote] {OUT_PATH} ({OUT_PATH.stat().st_size/1024/1024:.1f} MB, {len(full)} rows)", flush=True)

if failed:
    pd.DataFrame(failed, columns=["code","fail_reason"]).to_csv(FAILED_PATH, index=False)
    print(f"[failed] {len(failed)} → {FAILED_PATH}", flush=True)
