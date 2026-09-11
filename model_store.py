"""Cache-on-first-use model store on a RunPod network volume.

Layout (see pipeline_utils for the full rationale)::

    <root>/checkpoints/<slug>[/]      diffusers dir  or  single-file .safetensors
    <root>/loras/<slug>.safetensors
    <root>/vae/<slug>[/]
    <root>/registry.json              bookkeeping: ref -> local path, bytes, when

The store is deliberately boring: a model reference (HF repo id, HF
``repo::file``, direct URL, or an absolute path that already exists) maps to a
deterministic local path, so "is it cached?" is just an ``os.path`` check and a
half-finished download can never be mistaken for a cached one (downloads land in
a temp dir and are moved into place atomically).

Resolution order for an HF reference (kanban t_06eb5e83)::

    1. RunPod's host model cache   <-> /runpod-volume/huggingface-cache/hub/...
    2. this store on the volume    <-> /runpod-volume/models/...
    3. download (first use only)

Step 1 exists because RunPod's "Model caching" preloads one HF repo per endpoint
onto the worker host and schedules workers on hosts that already hold it; that
copy reads "significantly faster" than the same bytes streamed off a network
volume, and the download time is not billed. A cached copy is only used when it
is *complete* (model_index.json for a diffusers repo, plausible size for a
weight file) - an empty or half-written cache directory falls through to step 2
instead of producing a confusing diffusers error later.

Network access is injected through ``downloader`` callables so the resolver
logic stays testable without huggingface_hub installed.
"""
import json
import os
import shutil
import tempfile
import time

from pipeline_utils import (
    CHECKPOINTS_DIRNAME,
    LORAS_DIRNAME,
    MODEL_ROOT,
    REGISTRY_FILENAME,
    VAE_DIRNAME,
    VOLUME_ROOT,
)

#: Ignore the heavyweight duplicate formats that some HF repos also ship.
IGNORE_PATTERNS = ["*.ckpt", "*.gguf", "*.pth", "*.msgpack", "*.h5", "*.onnx", ".git*"]

MIN_SINGLE_FILE_BYTES = 100 * 1024 * 1024  # a real SDXL checkpoint is >2 GB


def volume_available(volume_root=VOLUME_ROOT):
    """True when a network volume is actually mounted."""
    return os.path.isdir(volume_root)


def resolve_root(model_root=MODEL_ROOT, fallback="/tmp/nsfw-models", volume_root=VOLUME_ROOT):
    """Use the volume when it is mounted, otherwise a non-persistent fallback dir."""
    if os.path.isdir(volume_root) or os.path.isdir(model_root):
        return model_root
    return fallback


def ensure_dirs(root):
    paths = {
        "root": root,
        "checkpoints": os.path.join(root, CHECKPOINTS_DIRNAME),
        "loras": os.path.join(root, LORAS_DIRNAME),
        "vae": os.path.join(root, VAE_DIRNAME),
        "registry": os.path.join(root, REGISTRY_FILENAME),
    }
    for key in ("checkpoints", "loras", "vae"):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def classify_ref(ref):
    """Describe what a model reference points at.

    Returns a dict with ``kind`` in {local_dir, local_file, hf_repo,
    hf_single_file, url_file} plus the pieces needed to fetch it later.
    """
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("empty model reference")

    if ref.startswith(("http://", "https://")):
        clean = ref.split("?", 1)[0].split("#", 1)[0]
        filename = os.path.basename(clean)
        if not filename.lower().endswith((".safetensors", ".ckpt", ".pt")):
            raise ValueError("URL reference must point at a single weight file: %s" % ref)
        return {"kind": "url_file", "url": ref, "filename": filename}

    if os.path.isdir(ref):
        return {"kind": "local_dir", "path": ref}
    if os.path.isfile(ref):
        return {"kind": "local_file", "path": ref}

    if "::" in ref:
        repo, filename = ref.split("::", 1)
        repo, filename = repo.strip(), filename.strip()
        if not repo or not filename:
            raise ValueError("bad 'repo::file' reference: %s" % ref)
        return {"kind": "hf_single_file", "repo": repo, "filename": filename}

    parts = ref.split("/")
    if len(parts) == 3 and parts[2].lower().endswith((".safetensors", ".ckpt", ".pt")):
        return {"kind": "hf_single_file", "repo": "/".join(parts[:2]), "filename": parts[2]}

    if len(parts) == 2 and all(parts):
        return {"kind": "hf_repo", "repo": ref}

    if ref.startswith("/"):
        # an absolute path that does not exist yet is a user error, not a repo id
        raise ValueError("local path does not exist: %s" % ref)

    raise ValueError(
        "cannot interpret model reference %r (use an HF repo id, 'repo::file.safetensors', "
        "a direct URL, or a local path)" % ref
    )


def slug_for(ref, info=None):
    """Filesystem-safe slug for a model reference."""
    info = info or classify_ref(ref)
    kind = info["kind"]
    if kind == "hf_single_file":
        return "%s--%s" % (info["repo"].replace("/", "--"), info["filename"])
    if kind == "hf_repo":
        return info["repo"].replace("/", "--")
    if kind == "url_file":
        return "url--" + info["filename"]
    return os.path.basename(info["path"])


def subdir_for(kind_hint):
    """Which store subdirectory a file kind belongs to."""
    hint = (kind_hint or "").lower()
    if hint == "lora":
        return LORAS_DIRNAME
    if hint == "vae":
        return VAE_DIRNAME
    return CHECKPOINTS_DIRNAME


def local_path_for(ref, root=None, kind_hint=None, info=None):
    """Deterministic local path for ``ref`` inside the store (may not exist yet)."""
    root = root or resolve_root()
    info = info or classify_ref(ref)
    paths = ensure_dirs(root)
    subdir = paths[subdir_for(kind_hint)]
    slug = slug_for(ref, info)
    if info["kind"] in ("hf_repo", "local_dir"):
        return os.path.join(subdir, slug)
    if info["kind"] == "local_file":
        return info["path"]
    filename = info["filename"]
    return os.path.join(subdir, filename if filename.lower().endswith(".safetensors") else slug + ".safetensors")


def is_component_dir(path):
    """A diffusers *component* directory (a bare VAE repo, or a ``vae/`` subfolder
    of a pipeline repo): a ``config.json`` next to some weights.

    Standalone component repos such as ``stabilityai/sdxl-vae`` ship no
    ``model_index.json``, so the pipeline check below rejected a perfectly good
    VAE - the download succeeded and was then reported as "not a usable model"
    (kanban t_06eb5e83).
    """
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(name.endswith((".safetensors", ".bin")) for name in names)


#: ``kind_hint`` values that name a pipeline component rather than a pipeline.
COMPONENT_KINDS = ("vae",)


def is_cached(path, kind, min_bytes=MIN_SINGLE_FILE_BYTES, kind_hint=None):
    """A cached entry must be *complete*: a diffusers dir with model_index.json
    (or ``config.json`` when the caller asked for a component), or a weight file
    of plausible size."""
    if not path:
        return False
    if kind in ("hf_repo", "local_dir"):
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return True
        return (kind_hint or "").lower() in COMPONENT_KINDS and is_component_dir(path)
    if not os.path.isfile(path):
        return False
    return os.path.getsize(path) >= min_bytes


# --------------------------------------------------------------------------- #
# RunPod host model cache ("Cached models" / Model caching)
# --------------------------------------------------------------------------- #
#: Where RunPod's model-caching feature exposes the preloaded repo inside the
#: worker container. Same mount path as a network volume, Hugging Face hub cache
#: layout, slashes in the repo id replaced by double dashes:
#:   <root>/models--{org}--{name}/snapshots/{commit-hash}/
HF_CACHE_ROOT = os.path.join(VOLUME_ROOT, "huggingface-cache", "hub")

REFS_MAIN = "main"


def hf_cache_model_dir(repo, cache_root=HF_CACHE_ROOT):
    """Cache directory RunPod uses for ``repo`` (may not exist yet)."""
    return os.path.join(cache_root, "models--" + (repo or "").replace("/", "--", 1))


def hf_cache_candidates(repo, cache_root=HF_CACHE_ROOT):
    """Snapshot dirs for ``repo`` in the host cache, most likely first.

    ``refs/main`` names the commit the cache was primed from, so that snapshot
    wins; any other snapshot found is a fallback (a cache primed off a branch
    other than main, or a refs file that was not written).
    """
    model_dir = hf_cache_model_dir(repo, cache_root)
    snapshots = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return []
    rev = ""
    try:
        with open(os.path.join(model_dir, "refs", REFS_MAIN), encoding="utf-8") as fh:
            rev = fh.read().strip()
    except OSError:
        pass
    try:
        rest = sorted(name for name in os.listdir(snapshots) if name != rev)
    except OSError:
        rest = []
    return [os.path.join(snapshots, name) for name in ([rev] if rev else []) + rest]


def host_cache_lookup(ref, info=None, cache_root=HF_CACHE_ROOT,
                      min_bytes=MIN_SINGLE_FILE_BYTES):
    """Resolve ``ref`` from RunPod's preloaded host model cache.

    Returns a store-style result dict (``source='host_cache'``) or ``None`` when
    the ref is not cached on this host, or when the cached copy is incomplete.
    Only HF references can be cached by RunPod, so anything else returns None.
    """
    try:
        info = info or classify_ref(ref)
    except ValueError:
        return None
    kind = info.get("kind")
    if kind not in ("hf_repo", "hf_single_file"):
        return None

    for snapshot in hf_cache_candidates(info["repo"], cache_root):
        if not os.path.isdir(snapshot):
            continue
        path = snapshot if kind == "hf_repo" else os.path.join(snapshot, info["filename"])
        if not is_cached(path, kind, min_bytes):
            continue
        return {"ref": ref, "kind": kind, "path": path, "cached": True,
                "source": "host_cache", "download_seconds": 0.0,
                "bytes": None if kind == "hf_repo" else os.path.getsize(path),
                "error": None}
    return None


def host_cache_repos(cache_root=HF_CACHE_ROOT):
    """Repo ids RunPod has preloaded on this host (``[]`` when there are none).

    Reported in the job metadata so a cold-start measurement can prove the
    worker really read the host cache and not the network volume.
    """
    try:
        names = os.listdir(cache_root)
    except OSError:
        return []
    return sorted(name[len("models--"):].replace("--", "/", 1)
                  for name in names if name.startswith("models--"))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def load_registry(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("bad registry shape")
        return data
    except (OSError, ValueError):
        return {"version": 1, "entries": {}}


def save_registry(path, registry):
    """Atomic write: a truncated registry must never survive on the volume."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def record(registry, ref, info):
    registry.setdefault("entries", {})[ref] = info
    return registry


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
def default_snapshot_download(repo, dest, token=None):
    """diffusers-style repo -> directory (huggingface_hub)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=4,
    )
    return dest


def default_file_download(repo, filename, dest, token=None):
    """single file from an HF repo -> dest (full local path)."""
    from huggingface_hub import hf_hub_download

    tmp_dir = tempfile.mkdtemp(prefix=".dl-", dir=os.path.dirname(dest))
    try:
        got = hf_hub_download(repo_id=repo, filename=filename, local_dir=tmp_dir, token=token)
        os.replace(got, dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return dest


def default_url_download(url, dest, token=None):
    """direct URL -> dest (streamed, with an atomic rename at the end)."""
    import urllib.request

    tmp = dest + ".partial"
    req = urllib.request.Request(url, headers={"User-Agent": "sdxl-serverless-worker/1"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    os.replace(tmp, dest)
    return dest


def ensure(ref, root=None, kind_hint=None, allow_download=True, token=None,
           snapshot_download=None, file_download=None, url_download=None,
           registry_path=None, min_bytes=MIN_SINGLE_FILE_BYTES,
           cache_root=HF_CACHE_ROOT, use_host_cache=True):
    """Make sure ``ref`` exists locally, downloading it on first use.

    Lookup order for an HF reference: RunPod's preloaded host cache, then this
    store on the volume, then a download.

    Returns a dict::

        {"ref", "kind", "path", "cached": bool, "source": str|None,
         "download_seconds": float|None, "bytes": int|None, "error": str|None}

    ``source`` is ``host_cache`` | ``volume`` | ``download`` | ``local`` | None.
    ``cached=True`` means nothing was downloaded on this call (no transfer was
    billed). A download failure is reported, not raised, so the caller can turn
    it into a proper RunPod job error.
    """
    root = root or resolve_root()
    try:
        info = classify_ref(ref)
    except ValueError as exc:
        return {"ref": ref, "kind": None, "path": None, "cached": False, "source": None,
                "download_seconds": None, "bytes": None, "error": str(exc)}

    kind = info["kind"]
    if kind in ("local_dir", "local_file"):
        ok = is_cached(info["path"], kind, min_bytes, kind_hint=kind_hint)
        return {"ref": ref, "kind": kind, "path": info["path"] if ok else None,
                "cached": ok, "source": "local" if ok else None,
                "download_seconds": 0.0 if ok else None,
                "bytes": os.path.getsize(info["path"]) if (ok and kind == "local_file") else None,
                "error": None if ok else "local path is not a usable model: %s" % info["path"]}

    if use_host_cache:
        hit = host_cache_lookup(ref, info=info, cache_root=cache_root, min_bytes=min_bytes)
        if hit:
            return hit

    paths = ensure_dirs(root)
    dest = local_path_for(ref, root=root, kind_hint=kind_hint, info=info)
    if is_cached(dest, kind, min_bytes, kind_hint=kind_hint):
        size = None if kind == "hf_repo" else os.path.getsize(dest)
        return {"ref": ref, "kind": kind, "path": dest, "cached": True, "source": "volume",
                "download_seconds": 0.0, "bytes": size, "error": None}

    if not allow_download:
        return {"ref": ref, "kind": kind, "path": None, "cached": False, "source": None,
                "download_seconds": None, "bytes": None,
                "error": "not cached and downloads are disabled"}

    snapshot_download = snapshot_download or default_snapshot_download
    file_download = file_download or default_file_download
    url_download = url_download or default_url_download

    t0 = time.time()
    try:
        if kind == "hf_repo":
            os.makedirs(dest, exist_ok=True)
            snapshot_download(info["repo"], dest, token)
        elif kind == "hf_single_file":
            file_download(info["repo"], info["filename"], dest, token)
        else:
            url_download(info["url"], dest, token)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as job error
        return {"ref": ref, "kind": kind, "path": None, "cached": False, "source": None,
                "download_seconds": round(time.time() - t0, 2), "bytes": None,
                "error": "download failed: %r" % (exc,)}

    if not is_cached(dest, kind, min_bytes, kind_hint=kind_hint):
        return {"ref": ref, "kind": kind, "path": None, "cached": False, "source": None,
                "download_seconds": round(time.time() - t0, 2), "bytes": None,
                "error": "download finished but %s is not a usable model" % dest}

    seconds = round(time.time() - t0, 2)
    size = None if kind == "hf_repo" else os.path.getsize(dest)
    try:
        reg_path = registry_path or paths["registry"]
        registry = load_registry(reg_path)
        record(registry, ref, {
            "kind": kind, "path": dest, "bytes": size,
            "download_seconds": seconds, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        save_registry(reg_path, registry)
    except OSError:
        pass  # bookkeeping is nice-to-have, never fatal

    return {"ref": ref, "kind": kind, "path": dest, "cached": False, "source": "download",
            "download_seconds": seconds, "bytes": size, "error": None}
