"""抓取 mt180.com (指标公式评测室) 全部 marketplace 指标 + TDX 公式.

数据源: 公开 JSON API (无 auth)
  - GET /api/marketplace?page=N&pageSize=20  → list (含 metadata, formulaPreview)
  - GET /api/marketplace/{id}                 → detail (含完整 formula)

输出 (append-only, resumable):
  - data_cache/mt180/indicators_list.jsonl   每行一个 indicator metadata
  - data_cache/mt180/indicators_detail.jsonl 每行一个完整 detail (含 TDX formula)

PII 保护: 不存 authorPhone / authorId, 只留 authorNickname.

用法:
  python examples/fetch_mt180_indicators.py --stage list           # 仅 list
  python examples/fetch_mt180_indicators.py --stage detail         # 仅 detail (要先有 list)
  python examples/fetch_mt180_indicators.py --stage all            # list + detail (默认)
  python examples/fetch_mt180_indicators.py --concurrency 8        # 并发数
  python examples/fetch_mt180_indicators.py --max-pages 5          # PoC

Resumable: 再跑会跳过已抓过的 id.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

try:
    from scrapling.fetchers import AsyncFetcher  # noqa: F401
    HAS_SCRAPLING = True
except ImportError:
    HAS_SCRAPLING = False

BASE = "https://web.mt180.com"
LIST_URL = f"{BASE}/api/marketplace"
DETAIL_URL_TPL = f"{BASE}/api/marketplace/{{id}}"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_cache" / "mt180"
LIST_PATH = OUT_DIR / "indicators_list.jsonl"
DETAIL_PATH = OUT_DIR / "indicators_detail.jsonl"

PAGE_SIZE = 20
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Origin": BASE,
    "Referer": f"{BASE}/",
}

PII_FIELDS = ("authorPhone", "authorId")

RETRY = 3
RETRY_BACKOFF = (1, 3, 8)


def _strip_pii(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in PII_FIELDS}


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "id" in d:
                    ids.add(d["id"])
            except json.JSONDecodeError:
                continue
    return ids


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def _get_json(client: httpx.AsyncClient, url: str, **params) -> dict[str, Any] | None:
    last_exc: Exception | None = None
    for attempt in range(RETRY):
        try:
            resp = await client.get(url, params=params or None, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 502, 503, 504):
                await asyncio.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue
            return None
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    if last_exc:
        print(f"  GIVE UP {url}: {type(last_exc).__name__}: {last_exc}", file=sys.stderr)
    return None


async def fetch_list_page(
    client: httpx.AsyncClient, page: int
) -> tuple[int, list[dict]]:
    data = await _get_json(client, LIST_URL, page=page, pageSize=PAGE_SIZE)
    if not data or "list" not in data:
        return 0, []
    total = int(data.get("total", 0))
    return total, data.get("list", [])


async def stage_list(concurrency: int, max_pages: int | None) -> int:
    existing = _load_existing_ids(LIST_PATH)
    print(f"[list] existing on disk: {len(existing)} ids", file=sys.stderr)

    async with httpx.AsyncClient(http2=False) as client:
        total, first_list = await fetch_list_page(client, 1)
        if total == 0:
            print("[list] page 1 returned 0 — abort", file=sys.stderr)
            return 0
        pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        if max_pages is not None:
            pages = min(pages, max_pages)
        print(f"[list] total={total} pages={pages} concurrency={concurrency}", file=sys.stderr)

        written = 0
        for item in first_list:
            if item.get("id") in existing:
                continue
            _append_jsonl(LIST_PATH, _strip_pii(item))
            existing.add(item["id"])
            written += 1

        sem = asyncio.Semaphore(concurrency)
        t0 = time.time()
        completed = 1

        async def fetch_one(p: int):
            nonlocal completed, written
            async with sem:
                _t, items = await fetch_list_page(client, p)
                for item in items:
                    if item.get("id") in existing:
                        continue
                    _append_jsonl(LIST_PATH, _strip_pii(item))
                    existing.add(item["id"])
                    written += 1
                completed += 1
                if completed % 50 == 0:
                    el = time.time() - t0
                    rate = completed / el if el > 0 else 0
                    eta = (pages - completed) / rate if rate > 0 else 0
                    print(
                        f"  [list] {completed}/{pages} pages · {written} new · "
                        f"rate={rate:.1f} p/s · eta={eta/60:.1f}min",
                        file=sys.stderr,
                    )

        await asyncio.gather(*(fetch_one(p) for p in range(2, pages + 1)))
        elapsed = time.time() - t0
        print(
            f"[list] DONE  pages={pages} written={written} elapsed={elapsed/60:.1f}min",
            file=sys.stderr,
        )
        return written


def _read_list_ids() -> list[str]:
    if not LIST_PATH.exists():
        return []
    ids: list[str] = []
    with LIST_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "id" in d:
                    ids.append(d["id"])
            except json.JSONDecodeError:
                continue
    return ids


async def stage_detail(concurrency: int) -> int:
    all_ids = _read_list_ids()
    if not all_ids:
        print("[detail] no list ids found — run --stage list first", file=sys.stderr)
        return 0
    existing = _load_existing_ids(DETAIL_PATH)
    todo = [i for i in all_ids if i not in existing]
    print(
        f"[detail] total ids={len(all_ids)} already_fetched={len(existing)} "
        f"todo={len(todo)} concurrency={concurrency}",
        file=sys.stderr,
    )
    if not todo:
        return 0

    async with httpx.AsyncClient(http2=False) as client:
        sem = asyncio.Semaphore(concurrency)
        t0 = time.time()
        completed = 0
        ok = 0

        async def fetch_one(iid: str):
            nonlocal completed, ok
            async with sem:
                url = DETAIL_URL_TPL.format(id=iid)
                data = await _get_json(client, url)
                completed += 1
                if data and "indicator" in data:
                    _append_jsonl(DETAIL_PATH, _strip_pii(data["indicator"]))
                    ok += 1
                elif data and "id" in data:
                    _append_jsonl(DETAIL_PATH, _strip_pii(data))
                    ok += 1
                if completed % 200 == 0:
                    el = time.time() - t0
                    rate = completed / el if el > 0 else 0
                    eta = (len(todo) - completed) / rate if rate > 0 else 0
                    print(
                        f"  [detail] {completed}/{len(todo)} · ok={ok} · "
                        f"rate={rate:.1f}/s · eta={eta/60:.1f}min",
                        file=sys.stderr,
                    )

        await asyncio.gather(*(fetch_one(i) for i in todo))
        elapsed = time.time() - t0
        print(
            f"[detail] DONE ok={ok}/{len(todo)} elapsed={elapsed/60:.1f}min",
            file=sys.stderr,
        )
        return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("list", "detail", "all"), default="all",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="list mode PoC: 只抓前 N 页",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[mt180-fetch] scrapling={'YES' if HAS_SCRAPLING else 'NO'} "
          f"→ out_dir={OUT_DIR}", file=sys.stderr)

    async def runner():
        if args.stage in ("list", "all"):
            await stage_list(args.concurrency, args.max_pages)
        if args.stage in ("detail", "all"):
            await stage_detail(args.concurrency)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
