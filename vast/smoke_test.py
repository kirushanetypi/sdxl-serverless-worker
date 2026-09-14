#!/usr/bin/env python3
"""Пост-сборочная проверка образа comfyui-canon.

Что проверяет:
  1. ComfyUI вообще поднялся и отвечает на /system_stats;
  2. наш чекпойнт виден в списке ComfyUI (GET /object_info/CheckpointLoaderSimple);
  3. файл чекпойнта на месте, нужного размера и с нужным sha256
     (--checkpoint-file + --expected-*);
  4. файл — валидный SDXL single-file чекпойнт (заголовок safetensors содержит
     ключи unet/conditioner/first_stage_model);
  5. по флагу --load — ComfyUI реально грузит чекпойнт (POST /prompt
     с одним CheckpointLoaderSimple). На CPU это тяжело, на GPU — быстро.

Где гоняется:
  - в CI после сборки: `docker exec comfy-smoke python3 /tmp/smoke_test.py ...`
    (в GitHub Actions GPU нет, ComfyUI стартует с --cpu);
  - на живом инстансе Vast: `python3 vast/smoke_test.py --base-url http://<ip>:8188`.

Код возврата 0 — всё хорошо, 1 — конкретная проверка упала (причина печатается).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

CKPT_FILENAME = "waiNSFWIllustrious_v150.safetensors"
CKPT_SHA256 = "befc694a296f75e996488ebf9f9db8a1493bd059b6e704b975829e87d5aeb4fa"
CKPT_BYTES = 6938040682

#: Ключи, которые обязаны быть в single-file SDXL чекпойнте.
SDXL_KEY_PREFIXES = ("model.diffusion_model.", "conditioner.", "first_stage_model.")


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body or b"{}")


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def wait_ready(base_url: str, timeout: float) -> dict | None:
    """Ждёт /system_stats. Возвращает ответ или None по таймауту."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            stats = http_json(f"{base_url}/system_stats", timeout=10)
        except Exception as exc:  # noqa: BLE001 — ждём и показываем причину
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
            continue
        return stats
    print(f"  последняя ошибка опроса: {last}")
    return None


def check_visible(base_url: str, filename: str) -> tuple[bool, list[str]]:
    data = http_json(f"{base_url}/object_info/CheckpointLoaderSimple", timeout=30)
    node = (data.get("CheckpointLoaderSimple") or {}).get("input", {})
    try:
        names = list(node["required"]["ckpt_name"][0])
    except (KeyError, TypeError, IndexError):
        return False, []
    return filename in names, names


def check_file(path: str, expected_bytes: int, expected_sha: str) -> int:
    import os

    size = os.path.getsize(path)
    print(f"  файл {path}: {size} байт")
    if size != expected_bytes:
        return fail(f"размер {size} != ожидаемого {expected_bytes}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha:
        return fail(f"sha256 {digest.hexdigest()} != ожидаемого {expected_sha}")
    print(f"  sha256 совпал: {expected_sha}")
    return 0


def check_safetensors(path: str) -> int:
    """Читает заголовок safetensors и проверяет, что это SDXL-чекпойнт."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("  safetensors нет — проверку заголовка пропускаю")
        return 0
    try:
        with safe_open(path, framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
    except Exception as exc:  # noqa: BLE001 — битый файл важнее трейсбека
        return fail(f"safetensors не открывается: {type(exc).__name__}: {exc}")
    print(f"  тензоров в чекпойнте: {len(keys)}")
    missing = [p for p in SDXL_KEY_PREFIXES if not any(k.startswith(p) for k in keys)]
    if missing:
        return fail(f"в чекпойнте нет ключей {missing} — это не single-file SDXL")
    print("  ключи SDXL на месте: " + ", ".join(SDXL_KEY_PREFIXES))
    return 0


def load_checkpoint(base_url: str, filename: str, timeout: float) -> int:
    """Реальная загрузка чекпойнта ComfyUI (проверяет, что файл читается)."""
    workflow = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": filename}}}
    started = time.time()
    try:
        resp = http_json(f"{base_url}/prompt", {"prompt": workflow, "client_id": "smoke"}, timeout=60)
    except urllib.error.HTTPError as exc:
        return fail(f"POST /prompt -> HTTP {exc.code}: {exc.read()[:400]!r}")
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        return fail(f"нет prompt_id в ответе: {resp}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        entry = (http_json(f"{base_url}/history/{prompt_id}", timeout=30) or {}).get(prompt_id)
        if entry:
            status = (entry.get("status") or {}).get("status_str")
            if status == "success":
                print(f"  чекпойнт загружен за {time.time() - started:.0f}с")
                return 0
            return fail(f"загрузка чекпойнта: {json.dumps(entry.get('status'))[:600]}")
        time.sleep(3)
    return fail(f"загрузка чекпойнта не завершилась за {timeout:.0f}с")


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка образа ComfyUI с запечённым каноном")
    ap.add_argument("--base-url", default="http://127.0.0.1:8188")
    ap.add_argument("--checkpoint-file", default="", help="путь к файлу чекпойнта (внутри контейнера)")
    ap.add_argument("--expected-bytes", type=int, default=CKPT_BYTES)
    ap.add_argument("--expected-sha256", default=CKPT_SHA256)
    ap.add_argument("--filename", default=CKPT_FILENAME, help="имя чекпойнта, как его видит ComfyUI")
    ap.add_argument("--timeout", type=float, default=300.0, help="сколько ждать готовности ComfyUI")
    ap.add_argument("--load", action="store_true", help="ещё и загрузить чекпойнт через /prompt")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    print(f"[1/4] жду ComfyUI на {base} (до {args.timeout:.0f}с)")
    stats = wait_ready(base, args.timeout)
    if stats is None:
        return fail("ComfyUI не поднялся")
    device = (stats.get("devices") or [{}])[0]
    system = stats.get("system") or {}
    print(
        "  поднялся: ComfyUI %s, python %s, устройство %s"
        % (system.get("comfyui_version"), system.get("python_version"), device.get("name"))
    )

    print(f"[2/4] чекпойнт виден в ComfyUI: {args.filename}")
    visible, names = check_visible(base, args.filename)
    print(f"  чекпойнтов в списке: {names}")
    if not visible:
        return fail(f"{args.filename} не виден в /object_info/CheckpointLoaderSimple")

    if args.checkpoint_file:
        print(f"[3/4] проверяю файл {args.checkpoint_file}")
        rc = check_file(args.checkpoint_file, args.expected_bytes, args.expected_sha256)
        if rc:
            return rc
        rc = check_safetensors(args.checkpoint_file)
        if rc:
            return rc
    else:
        print("[3/4] путь к файлу не задан — проверку файла пропускаю")

    if args.load:
        print("[4/4] гружу чекпойнт в ComfyUI")
        rc = load_checkpoint(base, args.filename, timeout=max(args.timeout, 600.0))
        if rc:
            return rc
    else:
        print("[4/4] загрузку чекпойнта не просили (--load)")

    print("OK: образ собран, ComfyUI поднимается, канон на месте")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
