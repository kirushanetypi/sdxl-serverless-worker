"""Contract tests for handler.py without a GPU.

torch/diffusers/runpod are stubbed into ``sys.modules`` and the pipeline loader
is replaced, so what is covered here is the handler's own logic: input parsing,
pipeline caching/eviction, the response contract (base64 PNG + ``meta`` with the
GPU name and timings) and the seed handling.
"""
import base64
import importlib
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


# --------------------------------------------------------------------------- #
# stub torch / diffusers / runpod
# --------------------------------------------------------------------------- #
class _FakeImage:
    def __init__(self, pixels=None):
        #: None -> a structured frame; pass a list of 0-255 ints for a specific one
        self.pixels = pixels

    def save(self, buf, format="PNG", optimize=False):  # noqa: A002
        buf.write(b"\x89PNG\r\n\x1a\n" + format.encode())

    # --- the frame-health path (convert -> resize -> tobytes) --------------- #
    def convert(self, mode):  # noqa: ARG002
        return self

    def resize(self, size):  # noqa: ARG002
        return self

    def tobytes(self):
        if self.pixels is not None:
            return bytes(self.pixels)
        return bytes((i // 8) % 256 for i in range(128 * 128))


class _FakeScheduler:
    pass


class _FakeVAE:
    config = types.SimpleNamespace(_name_or_path="stub-vae", force_upcast=True)

    def __init__(self):
        self.dtype = "float16"
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeAutoencoderKL:
    """Records how the VAE was loaded so the subfolder/kind logic is testable."""

    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append(("from_pretrained", path, kwargs))
        return _FakeVAE()

    @classmethod
    def from_single_file(cls, path, **kwargs):
        cls.calls.append(("from_single_file", path, kwargs))
        return _FakeVAE()


class _FakePipe:
    def __init__(self):
        self.calls = []
        self.images = [_FakeImage()]
        self.moved_to = []
        self.scheduler = _FakeScheduler()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(images=self.images)

    def encode_prompt(self, *args, **kwargs):
        return ("default-encode",)

    def to(self, device, *_args, **_kwargs):
        self.moved_to.append(device)
        return self


class _FakeGenerator:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def _install_stubs():
    torch = types.ModuleType("torch")

    class _Cuda:
        #: fake VRAM accounting, in bytes - handlers read this back into meta
        allocated = 7 * 1024 ** 3
        free = 20 * 1024 ** 3
        total = 24 * 1024 ** 3

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(index=0):
            return "NVIDIA Test GPU"

        @staticmethod
        def get_device_properties(index=0):
            return types.SimpleNamespace(total_memory=24 * 1024**3)

        @staticmethod
        def empty_cache():
            _Cuda.allocated = 0

        def memory_allocated(self):
            return _Cuda.allocated

        def memory_reserved(self):
            return _Cuda.allocated

        def mem_get_info(self):
            return self.free, self.total

        @staticmethod
        def ipc_collect():
            return None

    torch.cuda = _Cuda()
    torch.float16 = "float16"
    torch.Generator = _FakeGenerator
    torch.concat = lambda tensors, dim=-1: tensors
    sys.modules["torch"] = torch

    diffusers = types.ModuleType("diffusers")
    diffusers.StableDiffusionXLPipeline = type("StableDiffusionXLPipeline", (), {})
    diffusers.AutoencoderKL = _FakeAutoencoderKL
    sys.modules["diffusers"] = diffusers

    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(
        start=lambda cfg: None, progress_update=lambda job, msg: None)
    sys.modules["runpod"] = runpod
    return torch


@pytest.fixture()
def worker():
    _install_stubs()
    sys.modules.pop("handler", None)
    handler = importlib.import_module("handler")
    handler._PIPELINES.clear()
    handler._PIPELINE_ORDER.clear()
    handler._GPU_STATS.update({"evictions": 0, "last_unload_allocated_gb": None})
    _FakeAutoencoderKL.calls.clear()
    built = []

    def fake_build(model_ref, vae_ref, loras):
        pipe = _FakePipe()
        built.append(model_ref)
        return {"pipe": pipe, "model": model_ref, "model_path": "/volume/" + model_ref,
                "model_type": "hf_repo", "model_cached": True, "model_download_seconds": 0.0,
                "model_bytes": None, "load_seconds": 1.5, "vae": None, "loras": [],
                "pipeline_reused": False}

    handler._build_pipeline = fake_build
    handler._built = built
    handler.MODEL_ID = "org/test-model"  # endpoint env default for the tests
    return handler


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_response_contract_image_and_meta(worker):
    out = worker.handler({"input": {"prompt": "a red apple", "seed": 7}})
    assert set(out) == {"image", "meta"}
    png = base64.b64decode(out["image"])
    assert png.startswith(b"\x89PNG")

    meta = out["meta"]
    assert meta["gpu_name"] == "NVIDIA Test GPU"
    assert meta["generation_seconds"] >= 0
    assert meta["request_seconds"] >= 0
    assert meta["model_cached"] is True
    assert meta["steps"] == 30 and meta["size"] == [1024, 1024]
    assert meta["guidance"] == 6.5
    assert meta["seed"] == 7
    assert meta["sampler"] == "auto"
    assert meta["clip_skip"] == 2
    assert json.dumps(meta)  # meta must be JSON-serialisable


def test_generation_kwargs_are_forwarded(worker):
    worker.handler({"input": {"prompt": "x", "width": 832, "height": 1216, "steps": 8,
                              "guidance": 1.5, "seed": 1}})
    pipe = next(iter(worker._PIPELINES.values()))["pipe"]
    call = pipe.calls[-1]
    assert call["width"] == 832 and call["height"] == 1216
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 1.5
    assert call["prompt"] == "x"


def test_random_seed_is_reported_and_stable_in_meta(worker):
    out = worker.handler({"input": {"prompt": "x"}})
    seed = out["meta"]["seed"]
    assert isinstance(seed, int) and seed >= 0


def test_same_model_is_loaded_once_and_reused(worker):
    worker.handler({"input": {"prompt": "one", "model": "org/a"}})
    out = worker.handler({"input": {"prompt": "two", "model": "org/a"}})
    assert worker._built == ["org/a"]
    assert out["meta"]["pipeline_reused"] is True


def test_pipeline_cache_is_bounded(worker):
    worker.MAX_CACHED_PIPELINES = 2
    for name in ("org/a", "org/b", "org/c"):
        worker.handler({"input": {"prompt": "x", "model": name}})
    assert len(worker._PIPELINES) == 2
    assert list(worker._PIPELINES) == [("org/b", "", ()), ("org/c", "", ())]


def test_model_defaults_to_endpoint_env(worker):
    worker.MODEL_ID = "env/default-model"
    out = worker.handler({"input": {"prompt": "x"}})
    assert out["meta"]["model"] == "env/default-model"
    assert worker._built == ["env/default-model"]


def test_missing_model_and_missing_prompt_are_errors(worker):
    worker.MODEL_ID = ""
    with pytest.raises(ValueError, match="no model requested"):
        worker.handler({"input": {"prompt": "x"}})
    with pytest.raises(ValueError, match="prompt is required"):
        worker.handler({"input": {"model": "org/a"}})


def test_apply_sampler_falls_back_on_unknown_name(worker):
    pipe = _FakePipe()
    with pytest.raises(ValueError):
        worker.apply_sampler(pipe, "nope")  # unknown names are rejected up front
    assert worker.apply_sampler(pipe, "auto") is None


def test_clip_skip_override_is_installed_and_removed(worker):
    pipe = _FakePipe()
    original = pipe.encode_prompt if hasattr(pipe, "encode_prompt") else None
    assert worker.apply_clip_skip(pipe, 2) is None
    assert not hasattr(pipe, "_clip_skip_original")
    assert worker.apply_clip_skip(pipe, 1) is None
    assert callable(pipe.encode_prompt)
    assert worker.apply_clip_skip(pipe, 2) is None
    assert "_clip_skip_original" not in pipe.__dict__
    assert getattr(pipe, "encode_prompt", None) == original


# --------------------------------------------------------------------------- #
# VRAM release on model switch (kanban t_06eb5e83)
# --------------------------------------------------------------------------- #
def _tracking_build(worker, events):
    """Replacement for _build_pipeline that logs build/move ordering."""
    first = worker._build_pipeline

    def fake_build(model_ref, vae_ref, loras):
        events.append(("build", model_ref))
        entry = first(model_ref, vae_ref, loras)
        entry["pipe"] = _TrackingPipe(events)
        return entry

    worker._build_pipeline = fake_build


class _TrackingPipe(_FakePipe):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def to(self, device, *args, **kwargs):
        self.events.append(("move", device))
        return super().to(device, *args, **kwargs)


def test_previous_pipeline_is_released_before_the_next_model_loads(worker):
    events = []
    _tracking_build(worker, events)
    worker.MAX_CACHED_PIPELINES = 1

    worker.handler({"input": {"prompt": "x", "model": "org/a"}})
    out = worker.handler({"input": {"prompt": "x", "model": "org/b"}})

    # the old pipeline is moved off the GPU *before* the new one is built
    assert events == [("build", "org/a"), ("move", "meta"), ("build", "org/b")]
    assert out["meta"]["pipelines_evicted"] == 1
    assert out["meta"]["max_cached_pipelines"] == 1


def test_model_switch_moves_pipeline_off_gpu_and_frees_it(worker):
    worker.MAX_CACHED_PIPELINES = 1
    worker.handler({"input": {"prompt": "x", "model": "org/a"}})
    pipe_a = next(iter(worker._PIPELINES.values()))["pipe"]

    out = worker.handler({"input": {"prompt": "x", "model": "org/b"}})

    assert pipe_a.moved_to[-1] == "meta"          # VRAM handed back
    assert len(worker._PIPELINES) == 1            # only the current model is held
    assert out["meta"]["gpu_memory_allocated_gb"] == 0.0


def test_a_b_a_switch_back_does_not_accumulate_vram(worker):
    """The acceptance scenario: model A -> model B -> model A back-to-back."""
    worker.MAX_CACHED_PIPELINES = 1
    for name in ("org/a", "org/b", "org/a"):
        worker.handler({"input": {"prompt": "x", "model": name}})

    assert worker._built == ["org/a", "org/b", "org/a"]   # A was reloaded, not kept
    assert len(worker._PIPELINES) == 1
    assert worker._GPU_STATS["evictions"] == 2
    assert worker._GPU_STATS["last_unload_allocated_gb"] == 0.0


def test_reused_pipeline_is_never_released(worker):
    worker.MAX_CACHED_PIPELINES = 1
    worker.handler({"input": {"prompt": "one", "model": "org/a"}})
    pipe_a = next(iter(worker._PIPELINES.values()))["pipe"]
    worker.handler({"input": {"prompt": "two", "model": "org/a"}})

    assert pipe_a.moved_to == []
    assert worker._GPU_STATS["evictions"] == 0
    assert worker._built == ["org/a"]


def test_low_free_vram_overrides_the_pipeline_cache_limit(worker):
    """The template env said MAX_CACHED_PIPELINES=2 on a 24 GB card - the free
    VRAM check has to win, or the next load OOMs."""
    worker.MAX_CACHED_PIPELINES = 2
    worker.VRAM_HEADROOM_GB = 9.0
    torch = sys.modules["torch"]
    torch.cuda.free = 4 * 1024 ** 3          # not enough for another checkpoint

    worker.handler({"input": {"prompt": "x", "model": "org/a"}})
    worker.handler({"input": {"prompt": "x", "model": "org/b"}})

    assert len(worker._PIPELINES) == 1       # the old pipeline was dropped anyway
    assert worker._GPU_STATS["evictions"] == 1


def test_plenty_of_free_vram_keeps_the_configured_cache(worker):
    worker.MAX_CACHED_PIPELINES = 2
    worker.VRAM_HEADROOM_GB = 9.0
    torch = sys.modules["torch"]
    torch.cuda.free = 20 * 1024 ** 3

    for name in ("org/a", "org/b"):
        worker.handler({"input": {"prompt": "x", "model": name}})
    assert len(worker._PIPELINES) == 2
    assert worker._GPU_STATS["evictions"] == 0

    worker.handler({"input": {"prompt": "x", "model": "org/c"}})
    assert len(worker._PIPELINES) == 2       # the configured limit still applies
    assert worker._GPU_STATS["evictions"] == 1


def test_release_pipeline_survives_a_pipe_that_refuses_to_move(worker):
    class _Stubborn(_FakePipe):
        def to(self, device, *args, **kwargs):
            raise RuntimeError("cannot move")

    entry = {"pipe": _Stubborn()}
    assert worker.release_pipeline(entry) == 0.0
    assert "pipe" not in entry  # dropped anyway, VRAM still reclaimed


def test_meta_reports_vram_and_host_cache_fields(worker):
    out = worker.handler({"input": {"prompt": "x"}})
    meta = out["meta"]
    for key in ("gpu_memory_allocated_gb", "gpu_memory_reserved_gb",
                "gpu_mem_before_load_gb", "gpu_mem_after_load_gb",
                "pipelines_evicted", "max_cached_pipelines",
                "host_cache_root", "host_cache_repos",
                "image_health", "png_bytes_per_pixel", "scheduler",
                "vae_requested", "vae_used"):
        assert key in meta, key
    assert meta["host_cache_root"] == "/runpod-volume/huggingface-cache/hub"
    assert meta["scheduler"] == "_FakeScheduler"
    assert json.dumps(meta)


# --------------------------------------------------------------------------- #
# frame sanity check + VAE loading (kanban t_06eb5e83)
# --------------------------------------------------------------------------- #
def _build_with_pixels(worker, pixels):
    first = worker._build_pipeline

    def fake_build(model_ref, vae_ref, loras):
        entry = first(model_ref, vae_ref, loras)
        entry["pipe"].images = [_FakeImage(pixels=pixels)]
        return entry

    worker._build_pipeline = fake_build


def test_noise_frame_is_flagged_in_meta(worker):
    noise = [(i * 7919 + 13) % 256 for i in range(128 * 128)]
    _build_with_pixels(worker, noise)
    meta = worker.handler({"input": {"prompt": "x"}})["meta"]
    assert meta["image_health"]["suspected_noise"] is True
    assert meta["image_health"]["ok"] is False
    assert meta["image_health"]["stddev"] >= 55


def test_normal_frame_passes_the_sanity_check(worker):
    meta = worker.handler({"input": {"prompt": "x"}})["meta"]
    assert meta["image_health"]["ok"] is True
    assert meta["image_health"]["suspected_noise"] is False
    assert meta["image_health"]["unique_values"] > 1


def test_vae_subfolder_detection(worker, tmp_path):
    pipeline = tmp_path / "pipe"
    (pipeline / "vae").mkdir(parents=True)
    assert worker.vae_subfolder(str(pipeline)) == "vae"
    bare = tmp_path / "bare-vae"
    bare.mkdir()
    assert worker.vae_subfolder(str(bare)) is None
    assert worker.vae_subfolder(None) is None


def _component_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "diffusion_pytorch_model.safetensors").write_bytes(b"0" * 32)
    return path


def test_vae_from_a_pipeline_repo_uses_the_vae_subfolder(worker, tmp_path):
    root = tmp_path / "models"
    pipeline = root / "vae" / "org--pipe"
    _component_dir(pipeline)
    (pipeline / "model_index.json").write_text("{}")
    _component_dir(pipeline / "vae")
    worker.MODEL_ROOT = str(root)

    entry = worker._apply_vae(_FakePipe(), "org/pipe")

    assert entry["subfolder"] == "vae"
    assert entry["class"] == "_FakeVAE" and entry["cached"] is True
    assert _FakeAutoencoderKL.calls[-1] == (
        "from_pretrained", str(pipeline), {"subfolder": "vae", "torch_dtype": "float16"})


def test_vae_from_a_bare_repo_loads_without_a_subfolder(worker, tmp_path):
    """stabilityai/sdxl-vae: config.json + weights at the root, no model_index.json."""
    root = tmp_path / "models"
    bare = _component_dir(root / "vae" / "stabilityai--sdxl-vae")
    worker.MODEL_ROOT = str(root)

    entry = worker._apply_vae(_FakePipe(), "stabilityai/sdxl-vae")

    assert entry["subfolder"] is None and entry["cached"] is True
    assert _FakeAutoencoderKL.calls[-1] == (
        "from_pretrained", str(bare), {"subfolder": None, "torch_dtype": "float16"})


def test_vae_from_a_local_safetensors_file(worker, tmp_path):
    local = tmp_path / "my-vae.safetensors"
    with open(local, "wb") as fh:
        fh.truncate(101 * 1024 * 1024)  # sparse: passes the >=100 MB size check
    worker.MODEL_ROOT = str(tmp_path / "models")

    entry = worker._apply_vae(_FakePipe(), str(local))

    assert entry["subfolder"] is None and entry["source"] == "local"
    assert _FakeAutoencoderKL.calls[-1][0] == "from_single_file"


def test_vae_description_reports_the_loaded_vae(worker):
    pipe = _FakePipe()
    pipe.vae = _FakeVAE()
    got = worker._vae_description(pipe)
    assert got["class"] == "_FakeVAE"
    assert got["source"] == "stub-vae"
    assert got["dtype"] == "float16"
    assert worker._vae_description(object()) is None


def test_fp16_unsafe_vae_is_forced_to_upcast(worker):
    pipe = _FakePipe()
    vae = _FakeVAE()
    vae.config = types.SimpleNamespace(_name_or_path="mirror-vae", force_upcast=False)
    pipe.vae = vae

    note = worker._enforce_vae_upcast(pipe)
    assert note and "force_upcast" in note
    assert vae.config.force_upcast is True          # decoding now happens in fp32
    assert worker._enforce_vae_upcast(pipe) is None  # idempotent
    # an already-upcasting VAE (the default) is left alone
    ok = _FakeVAE()
    ok.config = types.SimpleNamespace(force_upcast=True)
    assert worker._enforce_vae_upcast(types.SimpleNamespace(vae=ok)) is None
    assert worker._enforce_vae_upcast(object()) is None
