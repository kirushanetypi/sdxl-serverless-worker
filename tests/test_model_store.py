"""Unit tests for the network-volume model store (no network, no torch).

Downloads are injected, so the whole cache-on-first-use logic is covered
offline: first call downloads, second call is a cache hit, a broken download is
reported instead of cached.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_store import (  # noqa: E402
    classify_ref,
    ensure,
    ensure_dirs,
    is_cached,
    load_registry,
    local_path_for,
    resolve_root,
    save_registry,
    slug_for,
)


# --------------------------------------------------------------------------- #
# reference classification
# --------------------------------------------------------------------------- #
def test_classify_hf_repo():
    assert classify_ref("John6666/wai-nsfw-illustrious-sdxl-v150-sdxl") == {
        "kind": "hf_repo", "repo": "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl"}


def test_classify_hf_single_file_both_spellings():
    want = {"kind": "hf_single_file", "repo": "org/repo", "filename": "model.safetensors"}
    assert classify_ref("org/repo::model.safetensors") == want
    assert classify_ref("org/repo/model.safetensors") == want


def test_classify_url_file():
    got = classify_ref("https://example.com/models/x.safetensors?token=abc")
    assert got["kind"] == "url_file" and got["filename"] == "x.safetensors"
    with pytest.raises(ValueError):
        classify_ref("https://example.com/page.html")


def test_classify_local_paths(tmp_path):
    d = tmp_path / "diffusers-model"
    d.mkdir()
    f = tmp_path / "single.safetensors"
    f.write_bytes(b"x")
    assert classify_ref(str(d))["kind"] == "local_dir"
    assert classify_ref(str(f))["kind"] == "local_file"


def test_classify_rejects_junk_and_missing_local_paths():
    for bad in ["", "   ", "/runpod-volume/models/checkpoints/nope.safetensors", "justaname"]:
        with pytest.raises(ValueError):
            classify_ref(bad)


def test_slug_and_local_path_are_deterministic(tmp_path):
    root = str(tmp_path)
    assert slug_for("org/repo") == "org--repo"
    assert slug_for("org/repo::model.safetensors") == "org--repo--model.safetensors"
    assert slug_for("org/repo/model.safetensors") == "org--repo--model.safetensors"
    p1 = local_path_for("org/repo", root=root)
    p2 = local_path_for("org/repo", root=root)
    assert p1 == p2 == os.path.join(root, "checkpoints", "org--repo")
    single = local_path_for("org/repo::model.safetensors", root=root)
    assert single == os.path.join(root, "checkpoints", "model.safetensors")
    lora = local_path_for("org/repo::l.safetensors", root=root, kind_hint="lora")
    assert lora == os.path.join(root, "loras", "l.safetensors")
    vae_dir = local_path_for("org/vae", root=root, kind_hint="vae")
    assert vae_dir == os.path.join(root, "vae", "org--vae")


def test_resolve_root_prefers_existing_root(tmp_path):
    root = str(tmp_path / "models")
    os.makedirs(root)
    assert resolve_root(model_root=root, fallback="/tmp/other") == root
    assert resolve_root(model_root=str(tmp_path / "missing"), fallback="/tmp/other") == "/tmp/other"


# --------------------------------------------------------------------------- #
# completeness checks
# --------------------------------------------------------------------------- #
def test_is_cached_requires_model_index_for_diffusers(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert is_cached(str(d), "hf_repo") is False
    (d / "model_index.json").write_text("{}")
    assert is_cached(str(d), "hf_repo") is True


def test_is_cached_rejects_tiny_or_missing_files(tmp_path):
    f = tmp_path / "m.safetensors"
    assert is_cached(str(f), "hf_single_file") is False
    f.write_bytes(b"0" * 10)
    assert is_cached(str(f), "hf_single_file") is False
    assert is_cached(str(f), "hf_single_file", min_bytes=5) is True
    assert is_cached(None, "hf_single_file") is False


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_roundtrip_and_corruption_tolerance(tmp_path):
    path = str(tmp_path / "registry.json")
    assert load_registry(path) == {"version": 1, "entries": {}}
    save_registry(path, {"version": 1, "entries": {"a": {"path": "b"}}})
    assert load_registry(path)["entries"]["a"]["path"] == "b"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ truncated")
    assert load_registry(path) == {"version": 1, "entries": {}}


# --------------------------------------------------------------------------- #
# cache-on-first-use
# --------------------------------------------------------------------------- #
def _fake_snapshot(repo, dest, token=None):
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "model_index.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    return dest


def _fake_file(repo, filename, dest, token=None):
    with open(dest, "wb") as fh:
        fh.write(b"0" * 64)
    return dest


def _fake_url(url, dest, token=None):
    with open(dest, "wb") as fh:
        fh.write(b"0" * 64)
    return dest


def test_ensure_diffusers_downloads_once_then_hits_cache(tmp_path):
    root = str(tmp_path)
    calls = []

    def snapshot(repo, dest, token=None):
        calls.append(repo)
        return _fake_snapshot(repo, dest, token)

    first = ensure("org/repo", root=root, snapshot_download=snapshot)
    assert first["cached"] is False
    assert first["path"] == os.path.join(root, "checkpoints", "org--repo")
    assert first["error"] is None
    assert first["download_seconds"] is not None

    second = ensure("org/repo", root=root, snapshot_download=snapshot)
    assert second["cached"] is True
    assert second["path"] == first["path"]
    assert calls == ["org/repo"]  # downloaded exactly once

    reg = load_registry(os.path.join(root, "registry.json"))
    assert reg["entries"]["org/repo"]["kind"] == "hf_repo"


def test_ensure_single_file_and_lora_and_vae_live_in_their_subdirs(tmp_path):
    root = str(tmp_path)
    single = ensure("org/repo::model.safetensors", root=root, file_download=_fake_file,
                    min_bytes=4)
    assert single["path"] == os.path.join(root, "checkpoints", "model.safetensors")
    assert single["cached"] is False and single["bytes"] == 64

    lora = ensure("org/lora::l.safetensors", root=root, kind_hint="lora",
                  file_download=_fake_file, min_bytes=4)
    assert lora["path"] == os.path.join(root, "loras", "l.safetensors")

    vae = ensure("org/vae::v.safetensors", root=root, kind_hint="vae",
                 file_download=_fake_file, min_bytes=4)
    assert vae["path"] == os.path.join(root, "vae", "v.safetensors")

    again = ensure("org/lora::l.safetensors", root=root, kind_hint="lora",
                   file_download=_fake_file, min_bytes=4)
    assert again["cached"] is True


def test_ensure_url_file(tmp_path):
    got = ensure("https://example.com/x.safetensors", root=str(tmp_path),
                 url_download=_fake_url, min_bytes=4)
    assert got["kind"] == "url_file"
    assert got["path"] == os.path.join(str(tmp_path), "checkpoints", "x.safetensors")
    assert got["cached"] is False


def test_ensure_reports_download_failure_instead_of_caching(tmp_path):
    root = str(tmp_path)

    def boom(repo, dest, token=None):
        os.makedirs(dest, exist_ok=True)  # partial dir, no model_index.json
        raise RuntimeError("network down")

    got = ensure("org/repo", root=root, snapshot_download=boom)
    assert got["cached"] is False and got["path"] is None
    assert "network down" in got["error"]
    # and the partial directory is not treated as cached on the next attempt
    got2 = ensure("org/repo", root=root, allow_download=False)
    assert got2["cached"] is False and "disabled" in got2["error"]


def test_ensure_download_without_result_is_an_error(tmp_path):
    def useless(repo, dest, token=None):
        return None

    got = ensure("org/repo", root=str(tmp_path), snapshot_download=useless)
    assert got["error"].startswith("download finished but")


def test_ensure_rejects_bad_reference(tmp_path):
    got = ensure("justaname", root=str(tmp_path))
    assert got["cached"] is False and "cannot interpret" in got["error"]


def test_ensure_local_paths_are_used_as_is(tmp_path):
    d = tmp_path / "local-model"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    got = ensure(str(d), root=str(tmp_path / "store"))
    assert got["kind"] == "local_dir" and got["cached"] is True and got["path"] == str(d)

    missing = ensure(str(tmp_path / "nope"), root=str(tmp_path / "store"))
    assert missing["cached"] is False and "does not exist" in missing["error"]


def test_ensure_dirs_creates_the_expected_layout(tmp_path):
    paths = ensure_dirs(str(tmp_path))
    for key in ("checkpoints", "loras", "vae"):
        assert os.path.isdir(paths[key])
    json.dumps(paths)  # the returned dict must stay JSON-friendly for logging
