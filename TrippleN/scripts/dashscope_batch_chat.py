import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, Optional

import httpx
from tqdm import tqdm


def _get_api_key(explicit: Optional[str]) -> str:
    k = explicit or os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY".lower()) or ""
    if not k:
        raise SystemExit("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")
    return k


def _load_jsonl(path: str) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int,
    min_backoff_s: float,
    max_backoff_s: float,
) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(max_backoff_s, min_backoff_s * (2**attempt))
                wait = wait * (0.8 + 0.4 * random.random())
                await asyncio.sleep(wait)
                last_err = f"http_{r.status_code}: {r.text[:500]}"
                continue
            r.raise_for_status()
            return {"ok": True, "status_code": r.status_code, "json": r.json()}
        except Exception as e:
            wait = min(max_backoff_s, min_backoff_s * (2**attempt))
            wait = wait * (0.8 + 0.4 * random.random())
            await asyncio.sleep(wait)
            last_err = str(e)
    return {"ok": False, "error": last_err or "unknown_error"}


async def _run(
    items: list[Dict[str, Any]],
    out_path: str,
    api_key: str,
    base_url: str,
    concurrency: int,
    timeout_s: float,
    max_retries: int,
    min_backoff_s: float,
    max_backoff_s: float,
) -> None:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        pbar = tqdm(total=len(items), desc="batch", unit="req")

        async def one(i: int, item: Dict[str, Any]) -> None:
            custom_id = item.get("custom_id", item.get("id", i))
            body = item.get("body", item)
            async with sem:
                resp = await _post_with_retries(
                    client=client,
                    url=url,
                    headers=headers,
                    payload=body,
                    max_retries=max_retries,
                    min_backoff_s=min_backoff_s,
                    max_backoff_s=max_backoff_s,
                )
            rec = {"custom_id": custom_id, "request": body, "response": resp}
            line = json.dumps(rec, ensure_ascii=False)
            async with lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                pbar.update(1)

        tasks = [asyncio.create_task(one(i, item)) for i, item in enumerate(items)]
        for t in asyncio.as_completed(tasks):
            await t
        pbar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--min-backoff", type=float, default=1.0)
    ap.add_argument("--max-backoff", type=float, default=60.0)
    args = ap.parse_args()

    api_key = _get_api_key(args.api_key)
    items = _load_jsonl(args.input)
    if os.path.exists(args.output):
        os.remove(args.output)

    asyncio.run(
        _run(
            items=items,
            out_path=args.output,
            api_key=api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
            max_retries=args.max_retries,
            min_backoff_s=args.min_backoff,
            max_backoff_s=args.max_backoff,
        )
    )


if __name__ == "__main__":
    main()
