"""Operator CLI: ``python -m switchyard.cli keys create --name demo``.

Talks to the same Redis as the gateway (``redis.url`` from the gateway config, which honours
``REDIS_URL``). Prints JSON so it composes with ``jq`` in scripts and load tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any

from redis.asyncio import Redis

from switchyard.auth.keys import KeyStore
from switchyard.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switchyard")
    parser.add_argument("--config", help="gateway config path (default: $SWITCHYARD_CONFIG)")
    sub = parser.add_subparsers(dest="group", required=True)
    keys = sub.add_parser("keys", help="manage API keys").add_subparsers(
        dest="action", required=True
    )

    create = keys.add_parser("create", help="create a key and print it once")
    create.add_argument("--name", required=True)
    create.add_argument("--rpm", type=int, default=600, help="requests per minute")
    create.add_argument("--tpm", type=int, default=1_000_000, help="tokens per minute")

    keys.add_parser("list", help="list keys (never shows plaintext)")

    for action in ("disable", "enable"):
        cmd = keys.add_parser(action)
        cmd.add_argument("key_id")

    limits = keys.add_parser("limits", help="change a key's limits")
    limits.add_argument("key_id")
    limits.add_argument("--rpm", type=int, required=True)
    limits.add_argument("--tpm", type=int, required=True)
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    config = load_config(args.config)
    redis = Redis.from_url(config.redis.url)
    store = KeyStore(redis, config.redis.key_prefix)
    try:
        match args.action:
            case "create":
                api_key, record = await store.create(args.name, rpm=args.rpm, tpm=args.tpm)
                return {"api_key": api_key, **asdict(record)}
            case "list":
                return [asdict(r) for r in await store.list()]
            case "disable" | "enable":
                found = await store.set_disabled(args.key_id, args.action == "disable")
                return {"key_id": args.key_id, "found": found}
            case "limits":
                found = await store.update_limits(args.key_id, rpm=args.rpm, tpm=args.tpm)
                return {"key_id": args.key_id, "found": found}
            case _:  # pragma: no cover - argparse enforces choices
                raise SystemExit(2)
    finally:
        await redis.aclose()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = asyncio.run(_run(args))
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if isinstance(result, dict) and result.get("found") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
