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
    HF_CACHE_ROOT,
    classify_ref,
    ensure,
    ensure_dirs,
    hf_cache_candidates,
    hf_cache_model_dir,
    host_cache_lookup,
    host_cache_repos,
    is_cached,
    is_component_dir,
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


# --------------------------------------------------------------------------- #
# RunPod host model cache (kanban t_06eb5e83)
# --------------------------------------------------------------------------- #
def _prime_host_cache(cache_root, repo, files=("model_index.json",), rev="abc123",
                      write_ref=True, size=64):
    """Build a RunPod-style host cache entry; returns (snapshot_dir)."""
    org, name = repo.split("/", 1)
    model_dir = os.path.join(str(cache_root), "models--%s--%s" % (org, name))
    snap = os.path.join(model_dir, "snapshots", rev)
    os.makedirs(snap, exist_ok=True)
    for name_ in files:
        path = os.path.join(snap, name_)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"{}" if name_.endswith(".json") else b"0" * size)
    if write_ref:
        os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
        with open(os.path.join(model_dir, "refs", "main"), "w", encoding="utf-8") as fh:
            fh.write(rev)
    return snap


def test_host_cache_layout_matches_runpod_docs(tmp_path):
    cache = tmp_path / "hf-cache"
    assert hf_cache_model_dir("John6666/wai-nsfw-illustrious-sdxl-v150-sdxl", str(cache)) == os.path.join(
        str(cache), "models--John6666--wai-nsfw-illustrious-sdxl-v150-sdxl")
    assert HF_CACHE_ROOT == "/runpod-volume/huggingface-cache/hub"


def test_host_cache_candidates_prefer_refs_main(tmp_path):
    cache = tmp_path / "hf-cache"
    _prime_host_cache(cache, "org/repo", rev="bbbb", write_ref=False)
    _prime_host_cache(cache, "org/repo", rev="aaaa", write_ref=False)
    got = hf_cache_candidates("org/repo", str(cache))
    assert [os.path.basename(p) for p in got] == ["aaaa", "bbbb"]  # sorted fallback
    main = _prime_host_cache(cache, "org/repo", rev="cccc")
    assert hf_cache_candidates("org/repo", str(cache))[0] == main


def test_host_cache_candidates_absent_or_broken(tmp_path):
    cache = str(tmp_path / "hf-cache")
    assert hf_cache_candidates("org/repo", cache) == []  # nothing at all
    os.makedirs(os.path.join(cache, "models--org--repo"))  # no snapshots/ dir
    assert hf_cache_candidates("org/repo", cache) == []


def test_ensure_prefers_complete_host_cache_over_download(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    snap = _prime_host_cache(cache, "org/repo")
    downloads = []

    def snapshot_download(repo, dest, token=None):
        downloads.append(repo)
        return _fake_snapshot(repo, dest, token)

    got = ensure("org/repo", root=root, snapshot_download=snapshot_download,
                 cache_root=str(cache))
    assert got["path"] == snap
    assert got["source"] == "host_cache"
    assert got["cached"] is True
    assert got["download_seconds"] == 0.0
    assert downloads == []  # nothing was fetched


def test_ensure_ignores_incomplete_host_cache_and_falls_through(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    snap = _prime_host_cache(cache, "org/repo", files=())  # dir exists, empty
    assert os.path.isdir(snap)

    got = ensure("org/repo", root=root, snapshot_download=_fake_snapshot,
                 cache_root=str(cache))
    assert got["source"] == "download"
    assert got["path"] == os.path.join(root, "checkpoints", "org--repo")


def test_ensure_ignores_host_cache_with_dangling_refs_main(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    _prime_host_cache(cache, "org/repo", rev="real", write_ref=False)
    model_dir = os.path.join(str(cache), "models--org--repo")
    os.makedirs(os.path.join(model_dir, "refs"), exist_ok=True)
    with open(os.path.join(model_dir, "refs", "main"), "w", encoding="utf-8") as fh:
        fh.write("gone")  # hash without a snapshot directory

    got = ensure("org/repo", root=root, snapshot_download=_fake_snapshot,
                 cache_root=str(cache))
    # the stale ref must not shadow the usable snapshot next to it
    assert got["source"] == "host_cache"
    assert os.path.basename(got["path"]) == "real"


def test_ensure_prefers_host_cache_over_volume_copy(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    _fake_snapshot("org/repo", os.path.join(root, "checkpoints", "org--repo"))
    snap = _prime_host_cache(cache, "org/repo")

    got = ensure("org/repo", root=root, cache_root=str(cache))
    assert got["path"] == snap and got["source"] == "host_cache"


def test_ensure_host_cache_for_single_file_ref(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    snap = _prime_host_cache(cache, "org/repo", files=("model.safetensors",), size=64)
    got = ensure("org/repo::model.safetensors", root=root, min_bytes=4,
                 cache_root=str(cache))
    assert got["path"] == os.path.join(snap, "model.safetensors")
    assert got["source"] == "host_cache" and got["bytes"] == 64
    # a tiny stub in the cache is not a usable checkpoint
    tiny_cache = tmp_path / "tiny"
    _prime_host_cache(tiny_cache, "org/repo", files=("model.safetensors",), size=8)
    missed = host_cache_lookup("org/repo::model.safetensors", cache_root=str(tiny_cache))
    assert missed is None


def test_ensure_can_disable_host_cache_and_ignores_non_hf_refs(tmp_path):
    root, cache = str(tmp_path / "store"), tmp_path / "hf-cache"
    _prime_host_cache(cache, "org/repo")
    off = ensure("org/repo", root=root, snapshot_download=_fake_snapshot,
                 cache_root=str(cache), use_host_cache=False)
    assert off["source"] == "download"

    # URLs and volume paths are never served from the RunPod cache
    assert host_cache_lookup("https://example.com/x.safetensors", cache_root=str(cache)) is None
    assert host_cache_lookup("/runpod-volume/models/checkpoints/x.safetensors",
                             cache_root=str(cache)) is None
    assert host_cache_lookup("not a ref", cache_root=str(cache)) is None


def test_host_cache_repos_lists_preloaded_models(tmp_path):
    cache = tmp_path / "hf-cache"
    assert host_cache_repos(str(cache)) == []  # missing root is not an error
    _prime_host_cache(cache, "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl")
    _prime_host_cache(cache, "org/other")
    assert host_cache_repos(str(cache)) == [
        "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl", "org/other"]


def test_ensure_source_field_is_reported_for_every_path(tmp_path):
    root, cache = str(tmp_path / "store"), str(tmp_path / "hf-cache")
    assert ensure("justaname", root=root, cache_root=cache)["source"] is None
    assert ensure("org/repo", root=root, snapshot_download=_fake_snapshot,
                  cache_root=cache)["source"] == "download"
    _fake_snapshot("org/repo", os.path.join(root, "checkpoints", "org--repo"))
    assert ensure("org/repo", root=root, cache_root=cache)["source"] == "volume"
    d = tmp_path / "local"
    d.mkdir()
    (d / "model_index.json").write_text("{}")
    assert ensure(str(d), root=root, cache_root=cache)["source"] == "local"


# --------------------------------------------------------------------------- #
# standalone component repos (VAE) - kanban t_06eb5e83
# --------------------------------------------------------------------------- #
def _fake_component_dir(path, weights="diffusion_pytorch_model.safetensors"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    if weights:
        with open(os.path.join(path, weights), "wb") as fh:
            fh.write(b"0" * 64)
    return path


def test_pipeline_check_still_rejects_a_component_dir_without_the_hint(tmp_path):
    d = _fake_component_dir(str(tmp_path / "sdxl-vae"))
    assert is_cached(d, "hf_repo") is False          # a checkpoint needs model_index.json
    assert is_cached(d, "hf_repo", kind_hint="vae") is True


def test_is_component_dir_requires_config_and_weights(tmp_path):
    assert is_component_dir(None) is False
    assert is_component_dir(str(tmp_path / "nope")) is False
    bare = tmp_path / "bare-config"
    bare.mkdir()
    (bare / "config.json").write_text("{}")
    assert is_component_dir(str(bare)) is False       # config without weights
    assert is_component_dir(_fake_component_dir(str(tmp_path / "ok"))) is True
    assert is_component_dir(_fake_component_dir(str(tmp_path / "bin"),
                                                "diffusion_pytorch_model.bin")) is True


def test_standalone_vae_repo_downloads_once_and_is_then_a_cache_hit(tmp_path):
    """The reported bug: stabilityai/sdxl-vae downloaded, then was rejected as
    'not a usable model' because it ships no model_index.json."""
    root, cache = str(tmp_path / "store"), str(tmp_path / "hf-cache")
    calls = []

    def snapshot(repo, dest, token=None):
        calls.append(repo)
        _fake_component_dir(dest)
        return dest

    first = ensure("stabilityai/sdxl-vae", root=root, kind_hint="vae",
                   snapshot_download=snapshot, cache_root=cache)
    assert first["error"] is None
    assert first["path"] == os.path.join(root, "vae", "stabilityai--sdxl-vae")
    assert is_component_dir(first["path"]) is True

    second = ensure("stabilityai/sdxl-vae", root=root, kind_hint="vae",
                    snapshot_download=snapshot, cache_root=cache)
    assert second["cached"] is True and second["source"] == "volume"
    assert calls == ["stabilityai/sdxl-vae"]  # downloaded exactly once


def test_vae_inside_a_pipeline_repo_and_a_local_vae_file(tmp_path):
    """Variant 1: a full pipeline repo whose VAE lives in ./vae (skipped by the
    store, resolved by the handler); variant 3: a local .safetensors VAE."""
    root, cache = str(tmp_path / "store"), str(tmp_path / "hf-cache")
    pipeline = _fake_snapshot("org/sdxl-pipe", os.path.join(root, "vae", "org--sdxl-pipe"))
    _fake_component_dir(os.path.join(pipeline, "vae"))
    got = ensure("org/sdxl-pipe", root=root, kind_hint="vae", cache_root=cache)
    assert got["cached"] is True and got["source"] == "volume"
    assert os.path.isdir(os.path.join(got["path"], "vae"))  # handler adds subfolder

    local = tmp_path / "my-vae.safetensors"
    local.write_bytes(b"0" * 64)
    got_local = ensure(str(local), root=root, kind_hint="vae", min_bytes=4, cache_root=cache)
    assert got_local["kind"] == "local_file" and got_local["path"] == str(local)
    assert got_local["source"] == "local"


def test_vae_download_that_is_not_a_component_reports_a_clear_error(tmp_path):
    def junk(repo, dest, token=None):
        os.makedirs(dest, exist_ok=True)  # neither model_index.json nor config.json
        return dest

    got = ensure("org/not-a-vae", root=str(tmp_path), kind_hint="vae",
                 snapshot_download=junk, cache_root=str(tmp_path / "hf-cache"))
    assert got["path"] is None
    assert got["error"].startswith("download finished but")
