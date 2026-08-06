
from __future__ import annotations

import argparse
import atexit
import csv
import json
import subprocess
import time
from pathlib import Path

import requests

LINEARIZATION = "data/icd11/LinearizationMiniOutput-MMS-en.txt"
DEFAULT_OUTPUT = "data/icd11/synonyms.json"
DEFAULT_API_URL = "http://localhost:8080"
SAVE_EVERY = 500
DOCKER_IMAGE = "whoicd/icd-api"
DOCKER_CONTAINER = "icd-api-synonyms"


def _fetch_entity(foundation_id: str, api_url: str) -> dict:
    resp = requests.get(
        f"{api_url}/icd/entity/{foundation_id}",
        headers={"Accept": "application/json", "Accept-Language": "en", "API-Version": "v2"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_terms(entity: dict) -> list[str]:
    terms: list[str] = []
    for key in ("synonym", "inclusion"):
        for item in entity.get(key, []):
            t = item.get("label", {}).get("@value", "").strip()
            if t:
                terms.append(t)
    return list(dict.fromkeys(terms))


def _load_category_codes(path: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("ClassKind") != "category":
                continue
            code = row["Code"].strip()
            uri = row["Foundation URI"].strip()
            if code and uri:
                result.append((code, uri.rstrip("/").split("/")[-1]))
    return result


def _start_docker(port: int) -> None:
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", DOCKER_CONTAINER,
            "-p", f"{port}:80",
            "--env", "acceptLicense=true",
            DOCKER_IMAGE,
        ],
        check=True,
    )
    atexit.register(lambda: subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER], capture_output=True))

    print(f"Waiting for container to be ready on port {port} ...")
    api_url = f"http://localhost:{port}"
    for _ in range(30):
        try:
            requests.get(f"{api_url}/icd/entity/257068234",
                         headers={"Accept": "application/json", "Accept-Language": "en", "API-Version": "v2"},
                         timeout=3)
            print("Container ready.")
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("Container did not become ready in time.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ICD-11 synonyms from a local whoicd/icd-api container.")
    parser.add_argument("--linearization", default=LINEARIZATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the local ICD-API (default: http://localhost:8080)")
    parser.add_argument("--docker", action="store_true", help="Start the whoicd/icd-api container automatically")
    parser.add_argument("--port", type=int, default=8080, help="Host port for --docker mode (default: 8080)")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay between requests in seconds")
    args = parser.parse_args()

    if args.docker:
        _start_docker(args.port)
        args.api_url = f"http://localhost:{args.port}"

    out_path = Path(args.output)
    cache: dict[str, list[str]] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Resuming from {out_path} ({len(cache):,} codes already cached).")

    codes = _load_category_codes(args.linearization)
    todo = [(c, fid) for c, fid in codes if c not in cache]
    print(f"{len(codes):,} category codes total — {len(todo):,} remaining.")

    if not todo:
        print("Nothing to fetch.")
        return

    errors = 0
    for i, (code, foundation_id) in enumerate(todo):
        try:
            entity = _fetch_entity(foundation_id, args.api_url)
            cache[code] = _extract_terms(entity)
        except requests.HTTPError as exc:
            print(f"  [WARN] {code} ({foundation_id}): {exc}")
            errors += 1

        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(todo):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            pct = (i + 1) / len(todo) * 100
            print(f"  {i+1:,}/{len(todo):,} ({pct:.0f}%) — saved.")

        if args.delay:
            time.sleep(args.delay)

    with_terms = sum(1 for v in cache.values() if v)
    print(f"\nDone. {len(cache):,} codes, {with_terms:,} with synonyms, {errors} errors.")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
